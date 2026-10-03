"""Tests for runtime-grounded service health endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from transcription.config import AsrConfig, TranscriptionConfig  # noqa: E402
from transcription.models import Transcript  # noqa: E402
from transcription.service import create_app  # noqa: E402
from transcription.service_runtime import (  # noqa: E402
    ASRRuntime,
    RuntimeProfile,
    RuntimeState,
)


class HealthEngine:
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

    def transcribe_file(self, path: Path) -> Transcript:
        return Transcript(file_name=path.name, language="en", segments=[])


def health_runtime() -> ASRRuntime:
    profile = RuntimeProfile.from_config(
        TranscriptionConfig(
            model="tiny",
            device="cpu",
            compute_type="int8",
        )
    )
    return ASRRuntime(profile, engine_factory=HealthEngine)


@pytest.fixture
def client():
    """Create a live app with a deterministic ready runtime."""
    app = create_app(runtime_factory=health_runtime)
    with (
        patch(
            "transcription.service_health._check_ffmpeg",
            return_value={"status": "ok", "path": "/usr/bin/ffmpeg"},
        ),
        patch(
            "transcription.service_health._check_package_resources",
            return_value={"status": "ok", "resources": []},
        ),
        TestClient(app) as test_client,
    ):
        yield test_client


class TestHealthEndpoints:
    def test_legacy_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "slower-whisper-api"
        assert "version" in data
        assert "schema_version" in data

    def test_liveness_endpoint(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert data["service"] == "slower-whisper-api"

    def test_readiness_endpoint_reports_runtime_and_resources(self, client):
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["healthy"] is True

        checks = data["checks"]
        assert set(checks) == {
            "ffmpeg",
            "resources",
            "runtime",
            "faster_whisper",
            "cuda",
            "disk_space",
        }
        for check_result in checks.values():
            assert check_result["status"] in {"ok", "warning", "error"}

        runtime = checks["runtime"]
        assert runtime["status"] == "ok"
        assert runtime["state"] == "ready"
        assert runtime["ready"] is True
        assert runtime["profile"]["model"] == "tiny"
        assert runtime["selected"]["device"] == "cpu"
        assert runtime["selected"]["compute_type"] == "int8"
        assert runtime["attempts"][-1]["outcome"] == "selected"

    @pytest.mark.parametrize("failed_check", ["ffmpeg", "resources", "runtime"])
    def test_each_critical_failure_returns_503(self, failed_check):
        runtime = health_runtime()
        app = create_app(runtime_factory=lambda: runtime)

        ffmpeg = {"status": "ok", "path": "/usr/bin/ffmpeg"}
        resources = {"status": "ok", "resources": []}
        if failed_check == "ffmpeg":
            ffmpeg = {"status": "error", "message": "missing"}
        if failed_check == "resources":
            resources = {"status": "error", "message": "missing"}

        with (
            patch("transcription.service_health._check_ffmpeg", return_value=ffmpeg),
            patch(
                "transcription.service_health._check_package_resources",
                return_value=resources,
            ),
            TestClient(app) as test_client,
        ):
            if failed_check == "runtime":
                runtime._state = RuntimeState.FAILED
                runtime._engine = None
            response = test_client.get("/health/ready")

        assert response.status_code == 503
        assert response.json()["healthy"] is False
        assert response.json()["status"] == "degraded"

    def test_informational_check_failure_does_not_override_ready_runtime(self, client):
        with patch(
            "transcription.service_health._check_faster_whisper",
            return_value={"status": "error", "message": "import check failed"},
        ):
            response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["checks"]["runtime"]["status"] == "ok"

    def test_all_health_endpoints_have_request_id(self, client):
        for endpoint in ("/health", "/health/live", "/health/ready"):
            response = client.get(endpoint)
            request_id = response.headers["X-Request-ID"]
            assert len(request_id) == 36
            assert request_id.count("-") == 4


class TestHealthCheckHelpers:
    def test_check_ffmpeg_helper(self):
        from transcription.service_health import _check_ffmpeg

        result = _check_ffmpeg()
        assert result["status"] in {"ok", "error"}
        assert "path" in result or "message" in result

    def test_check_faster_whisper_helper(self):
        from transcription.service_health import _check_faster_whisper

        result = _check_faster_whisper()
        assert result["status"] in {"ok", "error"}

    def test_check_cuda_helper_cpu_mode(self):
        from transcription.service_health import _check_cuda

        result = _check_cuda("cpu")
        assert result["status"] == "ok"
        assert "CUDA not required" in result["message"]

    def test_check_cuda_helper_cuda_mode(self):
        from transcription.service_health import _check_cuda

        result = _check_cuda("cuda")
        assert result["status"] in {"ok", "warning", "error"}
        assert "device_count" in result or "message" in result

    def test_check_disk_space_helper(self):
        from transcription.service_health import _check_disk_space

        result = _check_disk_space()
        assert result["status"] in {"ok", "warning", "error"}
        if result["status"] in {"ok", "warning"}:
            assert isinstance(result["free_gb"], int | float)
            assert isinstance(result["total_gb"], int | float)
            assert isinstance(result["percent_used"], int | float)
        else:
            assert "message" in result

    def test_check_package_resources_helper(self):
        from transcription.service_health import _check_package_resources

        result = _check_package_resources()
        assert result["status"] in {"ok", "error"}
        assert "resources" in result or "message" in result
