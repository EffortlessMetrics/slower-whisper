"""Actual `/stream` transaction through the process-owned incremental ASR path."""

from __future__ import annotations

import base64
import queue
import threading
import wave
from collections import deque
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
    wav_params: list[tuple[int, int, int]] = []

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
            type(self).wav_params.append(
                (
                    wav_file.getframerate(),
                    wav_file.getnchannels(),
                    wav_file.getsampwidth(),
                )
            )
        type(self).calls.append(frames)
        return SimpleNamespace(segments=[SimpleNamespace(text=f"samples:{frames}")])

    def close(self) -> None:
        type(self).closes += 1


class FailingEngine(DeterministicEngine):
    def transcribe_file(self, path: Path):
        del path
        raise RuntimeError("private provider path /srv/models/tiny")


class SequenceClassifier:
    def __init__(self, decisions: list[bool | BaseException]) -> None:
        self.decisions = deque(decisions)

    def is_speech(self, _audio: bytes, *, sample_rate: int) -> bool:
        assert sample_rate == 16_000
        decision = self.decisions.popleft()
        if isinstance(decision, BaseException):
            raise decision
        return decision


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
    value = payload.get("type")
    return value.upper() if isinstance(value, str) else None


def event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("payload")
    return data if isinstance(data, dict) else {}


def receive_json_bounded(websocket, *, timeout: float = 5.0) -> dict[str, Any]:
    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def receive() -> None:
        try:
            result.put((True, websocket.receive_json()))
        except BaseException as error:  # noqa: BLE001 - propagate into test thread
            result.put((False, error))

    thread = threading.Thread(target=receive, daemon=True)
    thread.start()
    try:
        succeeded, value = result.get(timeout=timeout)
    except queue.Empty as error:
        raise AssertionError("timed out waiting for WebSocket event") from error
    if not succeeded:
        raise value
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object, got {type(value).__name__}")
    return value


def receive_type(websocket, expected: str, *, limit: int = 64) -> dict[str, Any]:
    expected = expected.upper()
    observed: list[str | None] = []
    for _ in range(limit):
        payload = receive_json_bounded(websocket)
        observed.append(event_type(payload))
        if event_type(payload) == expected:
            return payload
    raise AssertionError(f"missing {expected}; observed={observed}")


def start_message(**overrides: Any) -> dict[str, Any]:
    config = {
        "sample_rate": 16_000,
        "audio_format": "pcm_s16le",
        "channels": 1,
        "max_gap_sec": 0.5,
    }
    config.update(overrides)
    return {"type": "START_SESSION", "config": config}


def audio_message(data: bytes, sequence: int) -> dict[str, Any]:
    return {
        "type": "AUDIO_CHUNK",
        "data": base64.b64encode(data).decode("ascii"),
        "sequence": sequence,
    }


def install_classifier(app, decisions: list[bool | BaseException]) -> None:
    app.state.streaming_speech_classifier_factory = lambda: SequenceClassifier(list(decisions))


def reset_engine() -> None:
    DeterministicEngine.instances = 0
    DeterministicEngine.closes = 0
    DeterministicEngine.calls = []
    DeterministicEngine.wav_params = []


def test_public_route_emits_real_revisions_and_no_placeholder_text() -> None:
    reset_engine()
    app, runtime = app_with_engine(DeterministicEngine)
    install_classifier(app, [True, False])

    with TestClient(app) as client:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            started = receive_type(websocket, "SESSION_STARTED")

            websocket.send_json(audio_message(pcm(16_000), 1))
            partial = receive_type(websocket, "PARTIAL")

            websocket.send_json(audio_message(pcm(8_000, 0), 2))
            finalized = receive_type(websocket, "FINALIZED")

            websocket.send_json({"type": "END_SESSION"})
            ended = receive_type(websocket, "SESSION_ENDED")

    partial_data = event_payload(partial)
    final_data = event_payload(finalized)
    assert partial["stream_id"] == finalized["stream_id"] == ended["stream_id"]
    assert [
        started["event_id"],
        partial["event_id"],
        finalized["event_id"],
        ended["event_id"],
    ] == [1, 2, 3, 4]
    assert partial["segment_id"] == finalized["segment_id"]
    assert partial_data["revision"] == 1
    assert final_data["revision"] == 2
    assert partial_data["text"] == final_data["text"] == "samples:16000"
    assert partial_data["start_sample"] == final_data["start_sample"] == 0
    assert partial_data["end_sample"] == final_data["end_sample"] == 16_000
    assert partial_data["final"] is False
    assert final_data["final"] is True
    assert final_data["final_reason"] == "vad_boundary"
    assert partial_data["segment"]["text"] == "samples:16000"
    assert "[processing...]" not in str(partial)
    assert "[final segment]" not in str(finalized)
    assert DeterministicEngine.instances == 1
    assert DeterministicEngine.calls == [16_000]
    assert DeterministicEngine.wav_params == [(16_000, 1, 2)]
    assert DeterministicEngine.closes == 1
    assert runtime.max_observed_inferences == 1


def test_public_route_rejects_unsupported_audio_before_engine_work() -> None:
    reset_engine()
    app, _runtime = app_with_engine(DeterministicEngine)
    install_classifier(app, [])

    with TestClient(app) as client:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message(sample_rate=8_000))
            error = receive_type(websocket, "ERROR")

    data = event_payload(error)
    assert data["code"] == "streaming_audio_unsupported"
    assert data["message"] == "Unsupported streaming audio configuration"
    assert data["recoverable"] is False
    assert data["context"]["mismatches"] == {"sample_rate": 8_000}
    assert DeterministicEngine.instances == 1
    assert DeterministicEngine.calls == []


def test_public_route_maps_inference_failure_to_one_sanitized_terminal_error() -> None:
    reset_engine()
    app, _runtime = app_with_engine(FailingEngine)
    install_classifier(app, [True])

    with TestClient(app) as client:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            receive_type(websocket, "SESSION_STARTED")
            websocket.send_json(audio_message(pcm(16_000), 1))
            error = receive_type(websocket, "ERROR")

    data = event_payload(error)
    assert data["code"] == "asr_inference_failed"
    assert data["message"] == "Streaming ASR failed"
    assert data["recoverable"] is False
    assert "private provider path" not in str(error)
    assert "[processing...]" not in str(error)


def test_public_route_sanitizes_unexpected_classifier_failure() -> None:
    reset_engine()
    app, _runtime = app_with_engine(DeterministicEngine)
    install_classifier(app, [RuntimeError("private classifier detail")])

    with TestClient(app) as client:
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(start_message())
            receive_type(websocket, "SESSION_STARTED")
            websocket.send_json(audio_message(pcm(16_000), 1))
            error = receive_type(websocket, "ERROR")

    data = event_payload(error)
    assert data["code"] == "asr_inference_failed"
    assert data["message"] == "Streaming ASR failed"
    assert data["context"] == {
        "phase": "process_audio",
        "violation": "unexpected_streaming_error",
    }
    assert "private classifier detail" not in str(error)
