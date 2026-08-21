"""REST reuse and typed-failure contract for the service-owned ASR runtime."""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from transcription.config import AsrConfig, TranscriptionConfig  # noqa: E402
from transcription.exceptions import (  # noqa: E402
    ASRInferenceError,
    ASRModelLoadError,
    ASROutputError,
)
from transcription.models import Transcript  # noqa: E402
from transcription.service import create_app  # noqa: E402
from transcription.service_runtime import ASRRuntime, RuntimeProfile  # noqa: E402


class RuntimeEngine:
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


def profile_config() -> TranscriptionConfig:
    return TranscriptionConfig(
        model="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        task="transcribe",
        beam_size=3,
        vad_min_silence_ms=400,
        word_timestamps=False,
    )


def make_app(*, engine_factory=None, max_concurrency: int = 1):
    created: list[RuntimeEngine] = []

    def default_factory(cfg: AsrConfig) -> RuntimeEngine:
        engine = RuntimeEngine(cfg)
        created.append(engine)
        return engine

    factory = engine_factory or default_factory
    runtime = ASRRuntime(
        RuntimeProfile.from_config(profile_config()),
        engine_factory=factory,
        max_concurrency=max_concurrency,
    )
    return create_app(runtime_factory=lambda: runtime), runtime, created


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(struct.pack("h" * 160, *[0] * 160))
    return buffer.getvalue()


def post_transcribe(client: TestClient, **params: Any):
    return client.post(
        "/transcribe",
        files={"audio": ("test.wav", wav_bytes(), "audio/wav")},
        params=params,
    )


def empty_result(
    audio_path: Path,
    _root: Path,
    _config: TranscriptionConfig,
    *,
    _engine: RuntimeEngine,
) -> Transcript:
    return Transcript(
        file_name=audio_path.name,
        language="en",
        segments=[],
        meta={
            "asr_backend": "faster-whisper",
            "asr_device": _engine.cfg.device,
            "asr_compute_type": _engine.cfg.compute_type,
            "asr_model_load_attempts": list(_engine.model_load_attempts),
        },
    )


def test_two_requests_reuse_one_runtime_engine() -> None:
    app, _runtime, created = make_app()
    injected_engines: list[RuntimeEngine] = []

    def transcribe(
        audio_path: Path,
        root: Path,
        config: TranscriptionConfig,
        *,
        _engine: RuntimeEngine,
    ) -> Transcript:
        injected_engines.append(_engine)
        return empty_result(audio_path, root, config, _engine=_engine)

    with (
        patch("transcription.service_transcribe.validate_audio_format"),
        patch(
            "transcription.service_transcribe.transcribe_file",
            side_effect=transcribe,
        ) as public_api,
        TestClient(app) as client,
    ):
        first = post_transcribe(client)
        second = post_transcribe(client)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["segments"] == []
    assert len(created) == 1
    assert public_api.call_count == 2
    assert injected_engines == [created[0], created[0]]
    assert created[0].close_count == 1


def test_profile_mismatch_is_rejected_before_file_save_or_inference() -> None:
    app, _runtime, _created = make_app()

    with (
        patch("transcription.service_transcribe.save_upload_file_streaming") as save,
        patch("transcription.service_transcribe.transcribe_file") as transcribe,
        TestClient(app) as client,
    ):
        response = post_transcribe(client, model="base")

    assert response.status_code == 409
    body = response.json()
    assert body["error"]["type"] == "runtime_profile_mismatch"
    assert body["error"]["details"]["reason_code"] == "runtime_profile_mismatch"
    assert "model" in body["error"]["details"]["context"]["mismatches"]
    save.assert_not_called()
    transcribe.assert_not_called()


def test_request_scoped_diarization_options_do_not_change_asr_profile() -> None:
    app, _runtime, _created = make_app()
    seen: list[TranscriptionConfig] = []

    def transcribe(
        audio_path: Path,
        root: Path,
        config: TranscriptionConfig,
        *,
        _engine: RuntimeEngine,
    ) -> Transcript:
        seen.append(config)
        return empty_result(audio_path, root, config, _engine=_engine)

    with (
        patch("transcription.service_transcribe.validate_audio_format"),
        patch(
            "transcription.service_transcribe.transcribe_file",
            side_effect=transcribe,
        ),
        TestClient(app) as client,
    ):
        response = post_transcribe(
            client,
            enable_diarization=True,
            diarization_device="cpu",
            min_speakers=1,
            max_speakers=3,
            overlap_threshold=0.5,
        )

    assert response.status_code == 200
    assert len(seen) == 1
    assert seen[0].enable_diarization is True
    assert seen[0].min_speakers == 1
    assert seen[0].max_speakers == 3
    assert seen[0].overlap_threshold == 0.5


def test_failed_startup_returns_typed_503_before_upload() -> None:
    def failing_factory(_cfg: AsrConfig) -> RuntimeEngine:
        raise ASRModelLoadError(
            "provider secret should stay local",
            context={
                "model": "tiny",
                "attempts": [
                    {
                        "device": "cpu",
                        "compute_type": "int8",
                        "outcome": "failed",
                        "reason_code": "asr_model_load_failed",
                    }
                ],
            },
        )

    app, _runtime, _created = make_app(engine_factory=failing_factory)
    with (
        patch("transcription.service_transcribe.save_upload_file_streaming") as save,
        TestClient(app) as client,
    ):
        response = post_transcribe(client)

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "runtime_not_ready"
    startup = body["error"]["details"]["context"]["startup_error"]
    assert startup["reason_code"] == "asr_model_load_failed"
    assert "provider secret" not in str(body)
    save.assert_not_called()


@pytest.mark.parametrize(
    ("error", "reason_code"),
    [
        (
            ASRInferenceError(
                "private provider path /srv/models/tiny",
                context={"phase": "transcribe"},
            ),
            "asr_inference_failed",
        ),
        (
            ASROutputError(
                "private backend output detail",
                context={"violation": "end_before_start"},
            ),
            "asr_output_invalid",
        ),
    ],
)
def test_typed_asr_failure_survives_http_without_provider_detail(
    error,
    reason_code: str,
) -> None:
    app, _runtime, _created = make_app()

    with (
        patch("transcription.service_transcribe.validate_audio_format"),
        patch(
            "transcription.service_transcribe.transcribe_file",
            side_effect=error,
        ),
        TestClient(app) as client,
    ):
        response = post_transcribe(client)

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["type"] == reason_code
    assert body["error"]["details"]["reason_code"] == reason_code
    assert str(error) not in str(body)


def test_genuine_silence_is_successful_empty_transcript() -> None:
    app, _runtime, _created = make_app()

    with (
        patch("transcription.service_transcribe.validate_audio_format"),
        patch(
            "transcription.service_transcribe.transcribe_file",
            side_effect=empty_result,
        ),
        TestClient(app) as client,
    ):
        response = post_transcribe(client)

    assert response.status_code == 200
    assert response.json()["segments"] == []
    assert response.json()["meta"]["asr_backend"] == "faster-whisper"


def test_service_mounts_one_batch_route_and_preserves_sse_route() -> None:
    app, _runtime, _created = make_app()
    paths = [getattr(route, "path", None) for route in app.routes]

    assert paths.count("/transcribe") == 1
    assert paths.count("/transcribe/stream") == 1
