"""Live FastAPI lifecycle, reuse, readiness, and error-contract tests."""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from transcription.config import AsrConfig, TranscriptionConfig  # noqa: E402
from transcription.exceptions import ASRInferenceError, ASRModelLoadError  # noqa: E402
from transcription.models import Transcript  # noqa: E402
from transcription.service import create_app  # noqa: E402
from transcription.service_runtime import ASRRuntime, RuntimeProfile, RuntimeState  # noqa: E402


class LifecycleEngine:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.model_load_attempts = [
            {
                "device": cfg.device,
                "compute_type": cfg.compute_type or "unknown",
                "outcome": "selected",
                "reason_code": "ok",
            }
        ]
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1

    def transcribe_file(self, path: Path) -> Transcript:
        return Transcript(file_name=path.name, language="en", segments=[])


def runtime_profile() -> RuntimeProfile:
    return RuntimeProfile.from_config(
        TranscriptionConfig(
            model="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            task="transcribe",
            beam_size=5,
            vad_min_silence_ms=500,
            word_timestamps=False,
        )
    )


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("h" * 160, *[0] * 160))
    return buffer.getvalue()


def post_transcribe(client: TestClient, **params):
    return client.post(
        "/transcribe",
        files={"audio": ("test.wav", wav_bytes(), "audio/wav")},
        params=params,
    )


def test_lifespan_owns_one_runtime_and_reuses_one_engine() -> None:
    engines: list[LifecycleEngine] = []

    def engine_factory(cfg: AsrConfig) -> LifecycleEngine:
        engine = LifecycleEngine(cfg)
        engines.append(engine)
        return engine

    runtime = ASRRuntime(runtime_profile(), engine_factory=engine_factory)
    app = create_app(runtime_factory=lambda: runtime)
    transcript = Transcript(file_name="test.wav", language="en", segments=[])

    with (
        patch("transcription.service_transcribe.validate_audio_format"),
        patch(
            "transcription.service_transcribe.transcribe_file",
            return_value=transcript,
        ) as transcribe,
        TestClient(app) as client,
    ):
        first = post_transcribe(client)
        second = post_transcribe(client)
        assert first.status_code == 200
        assert second.status_code == 200
        assert runtime.state == RuntimeState.READY
        assert len(engines) == 1
        assert transcribe.call_count == 2
        assert transcribe.call_args_list[0].kwargs["_engine"] is engines[0]
        assert transcribe.call_args_list[1].kwargs["_engine"] is engines[0]

    assert runtime.state == RuntimeState.STOPPED
    assert engines[0].close_count == 1


def test_startup_failure_keeps_liveness_but_blocks_readiness() -> None:
    def engine_factory(_cfg: AsrConfig):
        raise ASRModelLoadError("provider secret", context={"model": "tiny"})

    runtime = ASRRuntime(runtime_profile(), engine_factory=engine_factory)
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 503
    runtime_check = ready.json()["checks"]["runtime"]
    assert runtime_check["state"] == "failed"
    assert runtime_check["error"]["reason_code"] == "asr_model_load_failed"
    assert "provider secret" not in str(ready.json())


def test_profile_mismatch_is_rejected_before_inference() -> None:
    runtime = ASRRuntime(runtime_profile(), engine_factory=LifecycleEngine)
    app = create_app(runtime_factory=lambda: runtime)

    with (
        patch("transcription.service_transcribe.validate_audio_format"),
        patch("transcription.service_transcribe.transcribe_file") as transcribe,
        TestClient(app) as client,
    ):
        response = post_transcribe(client, model="base")

    assert response.status_code == 409
    body = response.json()
    assert body["error"]["type"] == "runtime_profile_mismatch"
    assert "model" in body["error"]["details"]["context"]["mismatches"]
    transcribe.assert_not_called()


def test_typed_inference_error_survives_http_boundary_without_provider_detail() -> None:
    runtime = ASRRuntime(runtime_profile(), engine_factory=LifecycleEngine)
    app = create_app(runtime_factory=lambda: runtime)

    with (
        patch("transcription.service_transcribe.validate_audio_format"),
        patch(
            "transcription.service_transcribe.transcribe_file",
            side_effect=ASRInferenceError(
                "provider path /private/model",
                context={"phase": "transcribe"},
            ),
        ),
        TestClient(app) as client,
    ):
        response = post_transcribe(client)

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["type"] == "asr_inference_failed"
    assert body["error"]["details"]["reason_code"] == "asr_inference_failed"
    assert "/private/model" not in str(body)


def test_deep_probe_distinguishes_silence(monkeypatch, tmp_path: Path) -> None:
    runtime = ASRRuntime(runtime_profile(), engine_factory=LifecycleEngine)
    app = create_app(runtime_factory=lambda: runtime)
    probe = tmp_path / "probe.wav"
    probe.write_bytes(b"fake engine does not read audio")
    monkeypatch.setenv("SLOWER_WHISPER_DEEP_PROBE_AUDIO", str(probe))

    with TestClient(app) as client:
        response = client.post("/health/deep")

    assert response.status_code == 200
    assert response.json()["outcome"] == "silence"
