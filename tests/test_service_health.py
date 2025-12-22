"""Tests for service health check endpoints."""

import pytest

# Skip all tests if API dependencies are not installed
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from transcription.service import app  # noqa: E402


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_legacy_health_endpoint(self, client):
        """Test legacy /health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "slower-whisper-api"
        assert "version" in data
        assert "schema_version" in data

    def test_liveness_endpoint(self, client):
        """Test /health/live endpoint returns 200 (always alive if running)."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert data["service"] == "slower-whisper-api"
        assert "version" in data
        assert "schema_version" in data

    def test_readiness_endpoint_structure(self, client):
        """Test /health/ready endpoint returns correct structure."""
        response = client.get("/health/ready")
        # Status code could be 200 or 503 depending on environment
        assert response.status_code in [200, 503]
        data = response.json()

        # Check top-level fields
        assert "status" in data
        assert data["status"] in ["ready", "degraded"]
        assert "healthy" in data
        assert isinstance(data["healthy"], bool)
        assert data["service"] == "slower-whisper-api"
        assert "version" in data
        assert "schema_version" in data
        assert "checks" in data

        # Check checks structure
        checks = data["checks"]
        assert "ffmpeg" in checks
        assert "faster_whisper" in checks
        assert "cuda" in checks
        assert "disk_space" in checks

        # Each check should have a status
        for _check_name, check_result in checks.items():
            assert "status" in check_result
            assert check_result["status"] in ["ok", "warning", "error"]

    def test_readiness_ffmpeg_check(self, client):
        """Test that ffmpeg check is included in readiness response."""
        response = client.get("/health/ready")
        data = response.json()
        ffmpeg_check = data["checks"]["ffmpeg"]

        assert "status" in ffmpeg_check
        # In CI/dev environment, ffmpeg should be available
        # But don't enforce this as a hard requirement
        if ffmpeg_check["status"] == "ok":
            assert "path" in ffmpeg_check
        elif ffmpeg_check["status"] == "error":
            assert "message" in ffmpeg_check

    def test_readiness_faster_whisper_check(self, client):
        """Test that faster-whisper check is included in readiness response."""
        response = client.get("/health/ready")
        data = response.json()
        whisper_check = data["checks"]["faster_whisper"]

        assert "status" in whisper_check
        # If it fails, should have a message
        if whisper_check["status"] == "error":
            assert "message" in whisper_check

    def test_readiness_cuda_check(self, client):
        """Test that CUDA check is included in readiness response."""
        response = client.get("/health/ready")
        data = response.json()
        cuda_check = data["checks"]["cuda"]

        assert "status" in cuda_check
        # CUDA check should always have a message explaining status
        if cuda_check["status"] in ["ok", "warning"]:
            assert "message" in cuda_check or "device_count" in cuda_check

    def test_readiness_disk_space_check(self, client):
        """Test that disk space check is included in readiness response."""
        response = client.get("/health/ready")
        data = response.json()
        disk_check = data["checks"]["disk_space"]

        assert "status" in disk_check
        # If successful, should have disk space details
        if disk_check["status"] in ["ok", "warning"]:
            assert "cache_root" in disk_check
            assert "free_gb" in disk_check
            assert "total_gb" in disk_check
            assert "percent_used" in disk_check
        elif disk_check["status"] == "error":
            assert "message" in disk_check

    def test_readiness_returns_503_when_degraded(self, client):
        """Test that readiness returns 503 when service is degraded."""
        response = client.get("/health/ready")
        data = response.json()

        # If any critical check (ffmpeg or faster_whisper) fails, should be 503
        ffmpeg_status = data["checks"]["ffmpeg"]["status"]
        whisper_status = data["checks"]["faster_whisper"]["status"]

        if ffmpeg_status == "error" or whisper_status == "error":
            assert response.status_code == 503
            assert data["healthy"] is False
            assert data["status"] == "degraded"

    def test_readiness_returns_200_when_healthy(self, client):
        """Test that readiness returns 200 when all critical checks pass."""
        response = client.get("/health/ready")
        data = response.json()

        # If both critical checks pass, should be 200
        ffmpeg_status = data["checks"]["ffmpeg"]["status"]
        whisper_status = data["checks"]["faster_whisper"]["status"]

        if ffmpeg_status == "ok" and whisper_status == "ok":
            assert response.status_code == 200
            assert data["healthy"] is True
            assert data["status"] == "ready"

    def test_all_health_endpoints_have_request_id(self, client):
        """Test that all health endpoints return X-Request-ID header."""
        endpoints = ["/health", "/health/live", "/health/ready"]
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert "X-Request-ID" in response.headers
            # Request ID should be a valid UUID format
            request_id = response.headers["X-Request-ID"]
            assert len(request_id) == 36  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
            assert request_id.count("-") == 4


class TestHealthCheckHelpers:
    """Test individual health check helper functions."""

    def test_check_ffmpeg_helper(self):
        """Test _check_ffmpeg helper function directly."""
        from transcription.service import _check_ffmpeg

        result = _check_ffmpeg()
        assert "status" in result
        assert result["status"] in ["ok", "error"]
        if result["status"] == "ok":
            assert "path" in result
        else:
            assert "message" in result

    def test_check_faster_whisper_helper(self):
        """Test _check_faster_whisper helper function directly."""
        from transcription.service import _check_faster_whisper

        result = _check_faster_whisper()
        assert "status" in result
        assert result["status"] in ["ok", "error"]
        if result["status"] == "error":
            assert "message" in result

    def test_check_cuda_helper_cpu_mode(self):
        """Test _check_cuda helper returns ok for cpu mode."""
        from transcription.service import _check_cuda

        result = _check_cuda("cpu")
        assert result["status"] == "ok"
        assert "message" in result
        assert "CUDA not required" in result["message"]

    def test_check_cuda_helper_cuda_mode(self):
        """Test _check_cuda helper checks CUDA for cuda mode."""
        from transcription.service import _check_cuda

        result = _check_cuda("cuda")
        assert "status" in result
        assert result["status"] in ["ok", "warning", "error"]
        # Should have either device info or a message
        has_device_info = "device_count" in result
        has_message = "message" in result
        assert has_device_info or has_message

    def test_check_disk_space_helper(self):
        """Test _check_disk_space helper function directly."""
        from transcription.service import _check_disk_space

        result = _check_disk_space()
        assert "status" in result
        assert result["status"] in ["ok", "warning", "error"]

        if result["status"] in ["ok", "warning"]:
            # Should have disk space metrics
            assert "cache_root" in result
            assert "free_gb" in result
            assert "total_gb" in result
            assert "used_gb" in result
            assert "percent_used" in result
            # Validate metric types
            assert isinstance(result["free_gb"], int | float)
            assert isinstance(result["total_gb"], int | float)
            assert isinstance(result["percent_used"], int | float)
        elif result["status"] == "error":
            assert "message" in result
