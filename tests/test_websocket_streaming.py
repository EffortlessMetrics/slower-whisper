"""Integration contract for the public revision-aware `/stream` endpoint.

Unit tests in ``test_streaming_ws.py`` own the legacy session object. This file
exercises the mounted FastAPI route with a ready process runtime, deterministic
speech decisions, real PCM wire messages, and no placeholder transcript path.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from transcription.config import AsrConfig, TranscriptionConfig
from transcription.service import create_app
from transcription.service_runtime import ASRRuntime, RuntimeProfile


class DeterministicEngine:
    """Small process-owned engine used only by the public route fixture."""

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.calls: list[int] = []
        self.model_load_attempts = [
            {
                "device": cfg.device,
                "compute_type": cfg.compute_type or "unknown",
                "outcome": "selected",
                "reason_code": "ok",
            }
        ]

    def transcribe_file(self, path: Path):
        import wave

        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
        self.calls.append(frames)
        return SimpleNamespace(segments=[SimpleNamespace(text=f"samples:{frames}")])

    def close(self) -> None:
        return None


class AlwaysSpeechClassifier:
    def is_speech(self, _audio: bytes, *, sample_rate: int) -> bool:
        assert sample_rate == 16_000
        return True


def runtime_profile() -> RuntimeProfile:
    return RuntimeProfile.from_config(
        TranscriptionConfig(
            model="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            task="transcribe",
            beam_size=3,
            vad_min_silence_ms=400,
            word_timestamps=False,
        )
    )


@pytest.fixture
def client() -> TestClient:
    """Run the route through a real ASRRuntime lifecycle."""
    runtime = ASRRuntime(runtime_profile(), engine_factory=DeterministicEngine)
    app = create_app(runtime_factory=lambda: runtime)
    app.state.streaming_speech_classifier_factory = AlwaysSpeechClassifier
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_audio_data() -> bytes:
    """Generate one second of mono signed-16 PCM at 16 kHz."""
    return b"\x00\x01" * 16_000


@pytest.fixture
def sample_audio_chunk(sample_audio_data: bytes) -> dict[str, Any]:
    return audio_message(sample_audio_data, 1)


def audio_message(audio: bytes, sequence: int) -> dict[str, Any]:
    return {
        "type": "AUDIO_CHUNK",
        "data": base64.b64encode(audio).decode("ascii"),
        "sequence": sequence,
    }


def start_message(**overrides: Any) -> dict[str, Any]:
    config = {
        "sample_rate": 16_000,
        "channels": 1,
        "audio_format": "pcm_s16le",
        "max_gap_sec": 0.5,
    }
    config.update(overrides)
    return {"type": "START_SESSION", "config": config}


def receive_until(websocket, expected: str, *, limit: int = 32) -> dict[str, Any]:
    observed: list[str | None] = []
    for _ in range(limit):
        message = websocket.receive_json()
        message_type = message.get("type")
        observed.append(message_type)
        if message_type == expected:
            return message
    raise AssertionError(f"missing {expected}; observed={observed}")


class TestWebSocketConnection:
    def test_ping_before_session(self, client: TestClient) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "PING", "timestamp": 12_345})
            message = websocket.receive_json()

        assert message["type"] == "PONG"
        assert message["payload"]["timestamp"] == 12_345
        assert "server_timestamp" in message["payload"]

    def test_graceful_empty_session(self, client: TestClient) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            started = websocket.receive_json()
            websocket.send_json({"type": "END_SESSION"})
            ended = receive_until(websocket, "SESSION_ENDED")

        assert started["type"] == "SESSION_STARTED"
        assert ended["stream_id"] == started["stream_id"]


class TestStartSession:
    def test_default_and_stable_custom_config(self, client: TestClient) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION"})
            default = websocket.receive_json()

        assert default["type"] == "SESSION_STARTED"
        assert default["event_id"] == 1
        assert default["stream_id"].startswith("str-")
        assert "session_id" in default["payload"]

        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message(max_gap_sec=0.25))
            custom = websocket.receive_json()

        assert custom["type"] == "SESSION_STARTED"

    def test_unearned_live_enrichment_is_rejected(self, client: TestClient) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message(enable_prosody=True))
            error = websocket.receive_json()

        assert error["type"] == "ERROR"
        assert error["payload"]["code"] == "streaming_audio_unsupported"
        assert error["payload"]["recoverable"] is False
        assert error["payload"]["context"]["mismatches"] == {
            "enable_prosody": True
        }

    def test_non_object_config_is_rejected(self, client: TestClient) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "START_SESSION", "config": "invalid"})
            error = websocket.receive_json()

        assert error["type"] == "ERROR"
        assert error["payload"]["code"] == "streaming_audio_unsupported"

    def test_duplicate_start_is_recoverable(self, client: TestClient) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            websocket.receive_json()
            websocket.send_json(start_message())
            error = websocket.receive_json()

        assert error["type"] == "ERROR"
        assert error["payload"]["code"] == "session_already_started"
        assert error["payload"]["recoverable"] is True


class TestAudioChunk:
    def test_audio_produces_real_replacement_event(
        self,
        client: TestClient,
        sample_audio_chunk: dict[str, Any],
    ) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            started = websocket.receive_json()
            websocket.send_json(sample_audio_chunk)
            partial = websocket.receive_json()

        assert partial["type"] == "PARTIAL"
        assert partial["stream_id"] == started["stream_id"]
        assert partial["payload"]["revision"] == 1
        assert partial["payload"]["text"] == "samples:16000"
        assert partial["payload"]["segment"]["text"] == "samples:16000"
        assert partial["payload"]["start_sample"] == 0
        assert partial["payload"]["end_sample"] == 16_000
        assert "[processing...]" not in str(partial)

    def test_audio_without_session_is_recoverable(
        self,
        client: TestClient,
        sample_audio_chunk: dict[str, Any],
    ) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(sample_audio_chunk)
            error = websocket.receive_json()

        assert error["payload"]["code"] == "no_session"
        assert error["payload"]["recoverable"] is True

    def test_sequence_and_encoding_validation(
        self,
        client: TestClient,
        sample_audio_chunk: dict[str, Any],
    ) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            websocket.receive_json()
            websocket.send_json(sample_audio_chunk)
            websocket.receive_json()
            websocket.send_json(sample_audio_chunk)
            duplicate = websocket.receive_json()
            websocket.send_json(
                {
                    "type": "AUDIO_CHUNK",
                    "data": "not-valid-base64!!!",
                    "sequence": 2,
                }
            )
            invalid_base64 = websocket.receive_json()
            websocket.send_json(
                {
                    "type": "AUDIO_CHUNK",
                    "data": base64.b64encode(b"\x00\x00").decode("ascii"),
                }
            )
            missing_sequence = websocket.receive_json()

        assert duplicate["payload"]["code"] == "invalid_audio_chunk"
        assert invalid_base64["payload"]["code"] == "invalid_audio_chunk"
        assert missing_sequence["payload"]["code"] == "invalid_audio_chunk"


class TestEndSession:
    def test_end_finalizes_before_stats(
        self,
        client: TestClient,
        sample_audio_chunk: dict[str, Any],
    ) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            websocket.receive_json()
            websocket.send_json(sample_audio_chunk)
            partial = websocket.receive_json()
            websocket.send_json({"type": "END_SESSION"})
            finalized = receive_until(websocket, "FINALIZED")
            ended = receive_until(websocket, "SESSION_ENDED")

        assert finalized["segment_id"] == partial["segment_id"]
        assert finalized["payload"]["revision"] == 2
        assert finalized["payload"]["final_reason"] == "end_of_stream"
        assert finalized["event_id"] < ended["event_id"]
        assert ended["payload"]["stats"]["chunks_received"] == 1
        assert ended["payload"]["stats"]["bytes_received"] == 32_000

    def test_end_without_session_is_recoverable(self, client: TestClient) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json({"type": "END_SESSION"})
            error = websocket.receive_json()

        assert error["payload"]["code"] == "no_session"


class TestControlAndEnvelope:
    def test_ping_during_session_and_event_ids_are_monotonic(
        self,
        client: TestClient,
        sample_audio_chunk: dict[str, Any],
    ) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            started = websocket.receive_json()
            websocket.send_json(sample_audio_chunk)
            partial = websocket.receive_json()
            websocket.send_json({"type": "PING", "timestamp": 1_000})
            pong = websocket.receive_json()

        assert pong["type"] == "PONG"
        assert pong["payload"]["timestamp"] == 1_000
        assert started["stream_id"] == partial["stream_id"] == pong["stream_id"]
        assert [started["event_id"], partial["event_id"], pong["event_id"]] == [
            1,
            2,
            3,
        ]
        assert partial["ts_audio_start"] == 0.0
        assert partial["ts_audio_end"] == 1.0
        assert partial["segment_id"]

    def test_ping_timestamp_is_current(self, client: TestClient) -> None:
        with client.websocket_connect("/stream") as websocket:
            client_timestamp = int(time.time() * 1_000)
            websocket.send_json({"type": "PING", "timestamp": client_timestamp})
            pong = websocket.receive_json()

        assert abs(pong["payload"]["server_timestamp"] - client_timestamp) < 10_000

    def test_tts_state_keeps_active_session_alive(self, client: TestClient) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            websocket.receive_json()
            websocket.send_json({"type": "TTS_STATE", "playing": True})
            websocket.send_json({"type": "PING", "timestamp": 12_345})
            pong = websocket.receive_json()

        assert pong["type"] == "PONG"


class TestErrorHandling:
    def test_invalid_message_is_recoverable(self, client: TestClient) -> None:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            websocket.receive_json()
            websocket.send_json({"type": "INVALID_TYPE"})
            invalid_type = websocket.receive_json()
            websocket.send_json({"config": {}})
            missing_type = websocket.receive_json()
            websocket.send_json({"type": "PING", "timestamp": 7})
            pong = websocket.receive_json()

        assert invalid_type["payload"]["code"] == "invalid_message_type"
        assert invalid_type["payload"]["recoverable"] is True
        assert missing_type["payload"]["code"] == "invalid_message_type"
        assert pong["type"] == "PONG"


class TestStreamConfigEndpoint:
    def test_get_stream_config_reports_only_earned_surface(
        self,
        client: TestClient,
    ) -> None:
        response = client.get("/stream/config")
        assert response.status_code == 200
        data = response.json()

        assert data["supported_audio_formats"] == ["pcm_s16le"]
        assert data["supported_sample_rates"] == [16_000]
        assert data["supported_channels"] == [1]
        assert data["optional_live_enrichment"] is False
        assert "START_SESSION" in data["message_types"]["client"]
        assert "PARTIAL" in data["message_types"]["server"]
        assert "FINALIZED" in data["message_types"]["server"]
