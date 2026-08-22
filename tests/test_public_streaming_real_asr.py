"""Actual `/stream` transaction through the process-owned incremental ASR path."""

from __future__ import annotations

import base64
import inspect
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from transcription.config import AsrConfig, TranscriptionConfig  # noqa: E402
from transcription.service import create_app  # noqa: E402
from transcription.service_runtime import ASRRuntime, RuntimeProfile  # noqa: E402


def pcm(samples: int, value: int = 11) -> bytes:
    return value.to_bytes(2, "little", signed=True) * samples


class DeterministicEngine:
    instances = 0
    closes = 0
    calls: list[int] = []

    def __init__(self, cfg: AsrConfig) -> None:
        type(self).instances += 1
        self.cfg = cfg
        self.model_load_attempts = [
            {
                "device": cfg.device,
                "compute_type": cfg.compute_type or "unknown",
                "outcome": "selected",
                "reason_code": "ok",
            }
        ]

    def transcribe_file(self, path: Path):
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            assert wav_file.getframerate() == 16_000
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
        type(self).calls.append(frames)
        return SimpleNamespace(segments=[SimpleNamespace(text=f"samples:{frames}")])

    def close(self) -> None:
        type(self).closes += 1


class FailingEngine(DeterministicEngine):
    def transcribe_file(self, path: Path):
        del path
        raise RuntimeError("private provider path /srv/models/tiny")


class VADDecision:
    def __init__(self, *, is_speech: bool, should_finalize: bool) -> None:
        self.is_speech = is_speech
        self.should_finalize = should_finalize
        self.speech_probability = 1.0 if is_speech else 0.0
        self.probability = self.speech_probability
        self.confidence = self.speech_probability

    def __iter__(self):
        yield self.is_speech
        yield self.should_finalize

    def __await__(self):
        async def resolve():
            return self

        return resolve().__await__()


class DeterministicVADProcessor:
    """Treat the first chunk as speech and the second as a finalizing silence."""

    instances: list["DeterministicVADProcessor"] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls = 0
        type(self).instances.append(self)

    def _decision(self) -> VADDecision:
        self.calls += 1
        return VADDecision(
            is_speech=self.calls == 1,
            should_finalize=self.calls >= 2,
        )

    def process_chunk(self, *_args: Any, **_kwargs: Any) -> VADDecision:
        return self._decision()

    def process_audio(self, *_args: Any, **_kwargs: Any) -> VADDecision:
        return self._decision()

    def process(self, *_args: Any, **_kwargs: Any) -> VADDecision:
        return self._decision()

    def reset(self) -> None:
        self.calls = 0


def profile() -> RuntimeProfile:
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


def app_with_engine(engine_factory):
    runtime = ASRRuntime(profile(), engine_factory=engine_factory)
    return create_app(runtime_factory=lambda: runtime), runtime


def event_type(payload: dict[str, Any]) -> str | None:
    for key in ("event_type", "event", "type"):
        value = payload.get(key)
        if isinstance(value, str) and value.upper() not in {
            "EVENT",
            "STREAM_EVENT",
            "MESSAGE",
        }:
            return value.upper()
    data = payload.get("data")
    return event_type(data) if isinstance(data, dict) else None


def event_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    return data if isinstance(data, dict) else payload


def receive_type(websocket, expected: str, *, limit: int = 64) -> dict[str, Any]:
    expected = expected.upper()
    observed: list[str | None] = []
    for _ in range(limit):
        payload = websocket.receive_json()
        observed.append(event_type(payload))
        if event_type(payload) == expected:
            return payload
    raise AssertionError(f"missing {expected}; observed={observed}")


def start_message(**overrides: Any) -> dict[str, Any]:
    config = {
        "sample_rate": 16_000,
        "audio_format": "pcm_s16le",
        "channels": 1,
    }
    config.update(overrides)
    return {"type": "START_SESSION", "config": config}


def audio_message(data: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "AUDIO_CHUNK",
        "audio": encoded,
        "data": {"audio": encoded},
    }


def install_vad(monkeypatch) -> None:
    DeterministicVADProcessor.instances.clear()
    monkeypatch.setattr(
        "transcription.streaming_ws.VADProcessor",
        DeterministicVADProcessor,
    )


def test_public_route_emits_real_revisions_and_no_placeholder_text(monkeypatch) -> None:
    DeterministicEngine.instances = 0
    DeterministicEngine.closes = 0
    DeterministicEngine.calls = []
    install_vad(monkeypatch)
    app, runtime = app_with_engine(DeterministicEngine)

    with TestClient(app) as client:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            receive_type(websocket, "SESSION_STARTED")

            websocket.send_json(audio_message(pcm(16_000)))
            partial = receive_type(websocket, "PARTIAL")
            partial_data = event_data(partial)

            websocket.send_json(audio_message(pcm(8_000, 0)))
            finalized = receive_type(websocket, "FINALIZED")
            final_data = event_data(finalized)

            websocket.send_json({"type": "END_SESSION"})
            receive_type(websocket, "SESSION_ENDED")

    assert partial_data["segment_id"] == final_data["segment_id"]
    assert partial_data["revision"] == 1
    assert final_data["revision"] == 2
    assert partial_data["text"] == final_data["text"] == "samples:16000"
    assert partial_data["start_sample"] == final_data["start_sample"] == 0
    assert partial_data["end_sample"] == final_data["end_sample"] == 16_000
    assert partial_data["final"] is False
    assert final_data["final"] is True
    assert final_data["final_reason"] == "vad_boundary"
    assert "[processing...]" not in str(partial)
    assert "[final segment]" not in str(finalized)
    assert DeterministicEngine.instances == 1
    assert DeterministicEngine.calls == [16_000]
    assert DeterministicEngine.closes == 1
    assert runtime.max_observed_inferences == 1


def test_public_route_rejects_unsupported_audio_before_engine_work(monkeypatch) -> None:
    DeterministicEngine.instances = 0
    DeterministicEngine.calls = []
    install_vad(monkeypatch)
    app, _runtime = app_with_engine(DeterministicEngine)

    with TestClient(app) as client:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message(sample_rate=8_000))
            error = receive_type(websocket, "ERROR")

    data = event_data(error)
    assert data["code"] == "runtime_not_ready"
    assert data["recoverable"] is False
    assert data["context"]["mismatches"] == {"sample_rate": 8_000}
    assert DeterministicEngine.instances == 1
    assert DeterministicEngine.calls == []


def test_public_route_maps_inference_failure_to_one_sanitized_terminal_error(
    monkeypatch,
) -> None:
    FailingEngine.instances = 0
    install_vad(monkeypatch)
    app, _runtime = app_with_engine(FailingEngine)

    with TestClient(app) as client:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            receive_type(websocket, "SESSION_STARTED")
            websocket.send_json(audio_message(pcm(16_000)))
            error = receive_type(websocket, "ERROR")

    data = event_data(error)
    assert data["code"] == "asr_inference_failed"
    assert data["message"] == "Streaming ASR failed"
    assert data["recoverable"] is False
    assert "private provider path" not in str(error)
    assert "[processing...]" not in str(error)


def test_vad_fake_is_compatible_with_sync_await_and_tuple_consumers() -> None:
    processor = DeterministicVADProcessor()
    first = processor.process_chunk(b"x")
    assert tuple(first) == (True, False)
    assert inspect.isawaitable(first)
