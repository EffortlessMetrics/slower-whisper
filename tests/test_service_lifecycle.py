"""FastAPI lifecycle and runtime-grounded readiness tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from transcription.config import AsrConfig, TranscriptionConfig  # noqa: E402
from transcription.exceptions import ASRModelLoadError  # noqa: E402
from transcription.models import Transcript  # noqa: E402
from transcription.service import build_service_runtime, create_app  # noqa: E402
from transcription.service_runtime import ASRRuntime, RuntimeProfile, RuntimeState  # noqa: E402


class LifecycleEngine:
    instances = 0

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
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1

    def transcribe_file(self, path: Path) -> Transcript:
        return Transcript(file_name=path.name, language="en", segments=[])


def runtime_profile(
    *,
    device: str = "cpu",
    compute_type: str = "int8",
) -> RuntimeProfile:
    return RuntimeProfile.from_config(
        TranscriptionConfig(
            model="tiny",
            device=device,
            compute_type=compute_type,
            language="en",
            task="transcribe",
            beam_size=5,
            vad_min_silence_ms=500,
            word_timestamps=False,
        )
    )


def test_lifespan_starts_one_runtime_and_closes_it_once() -> None:
    LifecycleEngine.instances = 0
    engines: list[LifecycleEngine] = []

    def engine_factory(cfg: AsrConfig) -> LifecycleEngine:
        engine = LifecycleEngine(cfg)
        engines.append(engine)
        return engine

    runtime = ASRRuntime(runtime_profile(), engine_factory=engine_factory)
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert runtime.state == RuntimeState.READY
        assert LifecycleEngine.instances == 1

    assert runtime.state == RuntimeState.STOPPED
    assert engines[0].close_count == 1


def test_startup_failure_keeps_liveness_but_blocks_readiness() -> None:
    attempts = [
        {
            "device": "cpu",
            "compute_type": "int8",
            "outcome": "failed",
            "reason_code": "asr_model_load_failed",
        }
    ]

    def engine_factory(_cfg: AsrConfig):
        raise ASRModelLoadError(
            "provider secret",
            context={"model": "tiny", "attempts": attempts},
        )

    runtime = ASRRuntime(runtime_profile(), engine_factory=engine_factory)
    app = create_app(runtime_factory=lambda: runtime)

    with (
        patch(
            "transcription.service_health._check_ffmpeg",
            return_value={"status": "ok"},
        ),
        patch(
            "transcription.service_health._check_package_resources",
            return_value={"status": "ok", "resources": []},
        ),
        TestClient(app) as client,
    ):
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 503
    runtime_check = ready.json()["checks"]["runtime"]
    assert runtime_check["state"] == "failed"
    assert runtime_check["error"]["reason_code"] == "asr_model_load_failed"
    assert runtime_check["attempts"] == attempts
    assert "provider secret" not in str(ready.json())


def test_readiness_reports_actual_backend_selected_profile() -> None:
    def engine_factory(cfg: AsrConfig) -> LifecycleEngine:
        cfg.device = "cpu"
        cfg.compute_type = "int8"
        engine = LifecycleEngine(cfg)
        engine.model_load_attempts = [
            {
                "device": "cuda",
                "compute_type": "float16",
                "outcome": "failed",
                "reason_code": "asr_model_load_failed",
            },
            {
                "device": "cpu",
                "compute_type": "int8",
                "outcome": "selected",
                "reason_code": "ok",
            },
        ]
        return engine

    runtime = ASRRuntime(
        runtime_profile(device="cuda", compute_type="float16"),
        engine_factory=engine_factory,
    )
    app = create_app(runtime_factory=lambda: runtime)

    with (
        patch(
            "transcription.service_health._check_ffmpeg",
            return_value={"status": "ok"},
        ),
        patch(
            "transcription.service_health._check_package_resources",
            return_value={"status": "ok", "resources": []},
        ),
        TestClient(app) as client,
    ):
        ready = client.get("/health/ready")

    assert ready.status_code == 200
    runtime_check = ready.json()["checks"]["runtime"]
    assert runtime_check["profile"]["device"] == "cuda"
    assert runtime_check["selected"]["device"] == "cpu"
    assert runtime_check["selected"]["compute_type"] == "int8"
    assert [attempt["outcome"] for attempt in runtime_check["attempts"]] == [
        "failed",
        "selected",
    ]


def test_build_service_runtime_prints_resolved_preflight_to_stderr(
    monkeypatch,
    capsys,
) -> None:
    config = TranscriptionConfig(
        model="tiny",
        device="cpu",
        compute_type="int8",
    )
    monkeypatch.setenv("SLOWER_WHISPER_SERVICE_MAX_CONCURRENCY", "2")

    with patch.object(TranscriptionConfig, "from_env", return_value=config):
        runtime = build_service_runtime()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "[preflight] model=tiny device=cpu compute_type=int8 max_concurrency=2\n"
    )
    assert runtime.profile == RuntimeProfile.from_config(config)
    assert runtime.max_concurrency == 2
