
from fastapi.testclient import TestClient
from transcription.service import app

class TestSecurityHeaders:
    """Test presence of new security headers."""

    def test_new_security_headers(self):
        client = TestClient(app)
        response = client.get("/health")

        # Existing
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

        # New
        assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert response.headers.get("Permissions-Policy") == "microphone=(), camera=(), geolocation=()"
