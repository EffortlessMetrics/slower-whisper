"""
Security regression tests for sensitive data leakage.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from transcription.service import app

@pytest.fixture
def client() -> TestClient:
    return TestClient(app)

@pytest.fixture
def sample_wav_bytes() -> bytes:
    # Minimal WAV header
    return b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

def test_transcribe_exception_leak(client: TestClient, sample_wav_bytes: bytes) -> None:
    """
    Test that an unexpected exception containing sensitive information
    is NOT leaked to the client in the error response.
    """
    sensitive_info = "/home/user/secrets/api_key.txt"

    with patch("transcription.service.transcribe_file") as mock_transcribe, \
         patch("transcription.service.validate_audio_format"):

        # Simulate a crash with sensitive path in the message
        mock_transcribe.side_effect = RuntimeError(f"File not found at {sensitive_info}")

        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "cpu"},
        )

        assert response.status_code == 500
        body = response.json()
        detail = body.get("detail", "") or body.get("error", {}).get("message", "")

        # Verify fix: Sensitive info should NOT be in the response
        assert sensitive_info not in str(detail), "Sensitive info leaked!"

        # Verify generic message
        assert "Unexpected error during transcription" in str(detail)

def test_enrich_exception_leak(client: TestClient, sample_wav_bytes: bytes) -> None:
    """
    Test that an unexpected exception containing sensitive information
    is NOT leaked to the client in the enrich endpoint.
    """
    sensitive_info = "DB_PASSWORD=secret123"

    with patch("transcription.service._enrich_transcript") as mock_enrich, \
         patch("transcription.service.load_transcript") as mock_load, \
         patch("transcription.service.validate_audio_format"):

        # Return a valid transcript object so validation passes
        mock_load.return_value = MagicMock(segments=[MagicMock()])

        mock_enrich.side_effect = ValueError(f"Connection failed: {sensitive_info}")

        response = client.post(
            "/enrich",
            files={
                "transcript": ("t.json", b'{"file_name": "x", "segments": [{"id":0, "start":0, "end":1, "text":"test"}]}', "application/json"),
                "audio": ("test.wav", sample_wav_bytes, "audio/wav")
            },
        )

        assert response.status_code == 500
        body = response.json()
        detail = body.get("detail", "") or body.get("error", {}).get("message", "")

        print(f"\nDEBUG: Response detail: {detail}")

        # Verify fix: Sensitive info should NOT be in the response
        assert sensitive_info not in str(detail), "Sensitive info leaked!"

        # Verify generic message
        assert "Unexpected error during enrichment" in str(detail)
