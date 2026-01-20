"""
Security regression tests for sensitive data leakage.

Ensures that exception details containing potentially sensitive information
(file paths, credentials, internal state) are not exposed in API responses.
Full details are logged server-side for debugging.
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
    # Minimal WAV header (44 bytes) - just enough to pass basic validation
    return (
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
        b"\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00"
        b"\x02\x00\x10\x00data\x00\x00\x00\x00"
    )


@pytest.fixture
def valid_transcript_json() -> bytes:
    return b'{"file_name": "x", "segments": [{"id":0, "start":0, "end":1, "text":"test"}]}'


class TestTranscribeEndpointLeaks:
    """Security tests for /transcribe endpoint exception handling."""

    def test_unexpected_exception_no_leak(
        self, client: TestClient, sample_wav_bytes: bytes
    ) -> None:
        """Unexpected exceptions must not leak sensitive paths to clients."""
        sensitive_path = "/home/user/secrets/api_key.txt"

        with (
            patch("transcription.service.transcribe_file") as mock_transcribe,
            patch("transcription.service.validate_audio_format"),
        ):
            mock_transcribe.side_effect = RuntimeError(f"File not found at {sensitive_path}")

            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
                params={"device": "cpu"},
            )

            assert response.status_code == 500
            detail = response.json().get("detail", "")

            assert sensitive_path not in detail, "Sensitive path leaked!"
            assert "Unexpected error during transcription" in detail

    def test_transcription_error_no_leak(self, client: TestClient, sample_wav_bytes: bytes) -> None:
        """TranscriptionError must not leak internal details."""
        from transcription.exceptions import TranscriptionError

        sensitive_detail = "Model path: /opt/models/whisper-secret-v3"

        with (
            patch("transcription.service.transcribe_file") as mock_transcribe,
            patch("transcription.service.validate_audio_format"),
        ):
            mock_transcribe.side_effect = TranscriptionError(sensitive_detail)

            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
                params={"device": "cpu"},
            )

            assert response.status_code == 500
            detail = response.json().get("detail", "")

            assert sensitive_detail not in detail, "Internal model path leaked!"
            assert detail == "Transcription failed"

    def test_configuration_error_no_leak(self, client: TestClient, sample_wav_bytes: bytes) -> None:
        """ConfigurationError must not leak system configuration details."""
        from transcription.exceptions import ConfigurationError

        sensitive_config = "CUDA_HOME=/usr/local/cuda-secret"

        with (
            patch("transcription.service.transcribe_file") as mock_transcribe,
            patch("transcription.service.validate_audio_format"),
        ):
            mock_transcribe.side_effect = ConfigurationError(sensitive_config)

            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
                params={"device": "cpu"},
            )

            assert response.status_code == 400
            detail = response.json().get("detail", "")

            assert sensitive_config not in detail, "System config leaked!"
            assert "Configuration error" in detail


class TestEnrichEndpointLeaks:
    """Security tests for /enrich endpoint exception handling."""

    def test_unexpected_exception_no_leak(
        self,
        client: TestClient,
        sample_wav_bytes: bytes,
        valid_transcript_json: bytes,
    ) -> None:
        """Unexpected exceptions must not leak credentials to clients."""
        sensitive_cred = "DB_PASSWORD=secret123"

        with (
            patch("transcription.service._enrich_transcript") as mock_enrich,
            patch("transcription.service.load_transcript") as mock_load,
            patch("transcription.service.validate_audio_format"),
        ):
            mock_load.return_value = MagicMock(segments=[MagicMock()])
            mock_enrich.side_effect = RuntimeError(f"Connection failed: {sensitive_cred}")

            response = client.post(
                "/enrich",
                files={
                    "transcript": ("t.json", valid_transcript_json, "application/json"),
                    "audio": ("test.wav", sample_wav_bytes, "audio/wav"),
                },
            )

            assert response.status_code == 500
            detail = response.json().get("detail", "")

            assert sensitive_cred not in detail, "Credentials leaked!"
            assert "Unexpected error during enrichment" in detail

    def test_enrichment_error_no_leak(
        self,
        client: TestClient,
        sample_wav_bytes: bytes,
        valid_transcript_json: bytes,
    ) -> None:
        """EnrichmentError must not leak internal processing details."""
        from transcription.exceptions import EnrichmentError

        sensitive_detail = "Torch cache: /home/user/.cache/torch/hub/secret"

        with (
            patch("transcription.service._enrich_transcript") as mock_enrich,
            patch("transcription.service.load_transcript") as mock_load,
            patch("transcription.service.validate_audio_format"),
        ):
            mock_load.return_value = MagicMock(segments=[MagicMock()])
            mock_enrich.side_effect = EnrichmentError(sensitive_detail)

            response = client.post(
                "/enrich",
                files={
                    "transcript": ("t.json", valid_transcript_json, "application/json"),
                    "audio": ("test.wav", sample_wav_bytes, "audio/wav"),
                },
            )

            assert response.status_code == 500
            detail = response.json().get("detail", "")

            assert sensitive_detail not in detail, "Cache path leaked!"
            assert detail == "Enrichment failed"


class TestValidationErrorMessages:
    """Tests for user-facing validation errors (400s) - safe but sanitized."""

    def test_json_parse_error_shows_location_not_content(
        self, client: TestClient, sample_wav_bytes: bytes
    ) -> None:
        """JSON parse errors show line/column but not raw exception."""
        # Invalid JSON with a "secret" that shouldn't appear in response
        invalid_json = b'{"file_name": "secret_project", "segments": [INVALID]}'

        response = client.post(
            "/enrich",
            files={
                "transcript": ("t.json", invalid_json, "application/json"),
                "audio": ("test.wav", sample_wav_bytes, "audio/wav"),
            },
        )

        assert response.status_code == 400
        detail = response.json().get("detail", "")

        # Should show location info
        assert "line" in detail.lower() or "Invalid transcript" in detail
        # Should not leak the raw exception message with context
        assert "INVALID" not in detail and "Expecting value" not in detail
