"""
Comprehensive tests for the FastAPI REST API service.

This module provides thorough testing of the service endpoints including:
- Request validation
- Response schema validation
- Error response formats (400, 404, 413, 422, 500)
- File upload handling
- Mocked transcription/enrichment to avoid actual processing

Tests are organized by functionality:
- TestErrorResponseFormat: Verify consistent error response structure
- TestMiddleware: Request ID, logging, timing
- TestFileSizeValidation: File size limits (413 errors)
- TestAudioValidation: Audio format validation
- TestTranscriptValidation: Transcript JSON structure validation
- TestTranscribeEndpointMocked: Full transcribe endpoint with mocks
- TestEnrichEndpointMocked: Full enrich endpoint with mocks
- TestExceptionHandlers: Custom exception handling
"""

from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if API dependencies are not installed
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from slower_whisper.pipeline.models import Segment, Transcript, Word  # noqa: E402
from slower_whisper.pipeline.service import app  # noqa: E402
from slower_whisper.pipeline.service_errors import create_error_response  # noqa: E402
from slower_whisper.pipeline.service_serialization import (  # noqa: E402
    _segment_to_dict,
    _transcript_to_dict,
    _word_to_dict,
)
from slower_whisper.pipeline.service_validation import (  # noqa: E402
    validate_file_size,
    validate_transcript_json,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_wav_bytes() -> bytes:
    """Create a minimal valid WAV file in memory (1 second of silence)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        samples = struct.pack("h" * 16000, *[0] * 16000)
        wav.writeframes(samples)
    return buffer.getvalue()


@pytest.fixture
def sample_wav_file(tmp_path: Path) -> Path:
    """Create a minimal valid WAV file on disk."""
    wav_path = tmp_path / "test.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        samples = struct.pack("h" * 16000, *[0] * 16000)
        wav.writeframes(samples)
    return wav_path


@pytest.fixture
def sample_transcript() -> Transcript:
    """Create a sample transcript for testing."""
    return Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=2.0, text="Hello world"),
            Segment(id=1, start=2.0, end=4.0, text="This is a test"),
        ],
        meta={"test": "value"},
    )


@pytest.fixture
def sample_transcript_json(sample_transcript: Transcript) -> bytes:
    """Create sample transcript JSON bytes."""
    data = {
        "schema_version": 2,
        "file_name": sample_transcript.file_name,
        "language": sample_transcript.language,
        "segments": [
            {"id": s.id, "start": s.start, "end": s.end, "text": s.text}
            for s in sample_transcript.segments
        ],
        "meta": sample_transcript.meta,
    }
    return json.dumps(data).encode("utf-8")


# =============================================================================
# Test Error Response Format
# =============================================================================


class TestErrorResponseFormat:
    """Tests for consistent error response structure."""

    def test_create_error_response_basic(self) -> None:
        """Test basic error response creation."""
        response = create_error_response(
            status_code=400,
            error_type="bad_request",
            message="Something went wrong",
        )

        assert response.status_code == 400
        body = json.loads(response.body)

        assert "error" in body
        assert body["error"]["type"] == "bad_request"
        assert body["error"]["message"] == "Something went wrong"
        assert body["error"]["status_code"] == 400
        # Backward compatibility field
        assert body["detail"] == "Something went wrong"

    def test_create_error_response_with_request_id(self) -> None:
        """Test error response includes request_id when provided."""
        response = create_error_response(
            status_code=500,
            error_type="internal_error",
            message="Internal error",
            request_id="abc-123",
        )

        body = json.loads(response.body)
        assert body["error"]["request_id"] == "abc-123"

    def test_create_error_response_with_details(self) -> None:
        """Test error response includes details when provided."""
        response = create_error_response(
            status_code=422,
            error_type="validation_error",
            message="Validation failed",
            details={"field": "audio", "reason": "Missing required file"},
        )

        body = json.loads(response.body)
        assert "details" in body["error"]
        assert body["error"]["details"]["field"] == "audio"

    def test_400_error_response_format(self, client: TestClient, sample_wav_bytes: bytes) -> None:
        """Test 400 error response has correct structure."""
        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "invalid_device"},
        )

        assert response.status_code == 400
        body = response.json()

        # Check structured error format
        assert "error" in body
        assert body["error"]["type"] == "bad_request"
        assert "message" in body["error"]
        assert body["error"]["status_code"] == 400
        # Backward compatibility
        assert "detail" in body

    def test_422_error_response_format(self, client: TestClient) -> None:
        """Test 422 validation error response has correct structure."""
        response = client.post("/transcribe", params={"model": "tiny"})

        assert response.status_code == 422
        body = response.json()

        assert "error" in body
        assert body["error"]["type"] == "validation_error"
        assert "details" in body["error"]
        assert "validation_errors" in body["error"]["details"]


# =============================================================================
# Test Middleware
# =============================================================================


class TestMiddleware:
    """Tests for request middleware."""

    def test_request_id_header_present(self, client: TestClient) -> None:
        """Test that X-Request-ID header is present in all responses."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_request_id_is_uuid_format(self, client: TestClient) -> None:
        """Test that request ID is a valid UUID."""
        response = client.get("/health")
        request_id = response.headers["X-Request-ID"]

        # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (36 chars)
        assert len(request_id) == 36
        parts = request_id.split("-")
        assert len(parts) == 5
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]

    def test_request_id_unique_per_request(self, client: TestClient) -> None:
        """Test that each request gets a unique request ID."""
        response1 = client.get("/health")
        response2 = client.get("/health")

        assert response1.headers["X-Request-ID"] != response2.headers["X-Request-ID"]

    def test_request_id_in_error_response(
        self, client: TestClient, sample_wav_bytes: bytes
    ) -> None:
        """Test that request ID is returned even for error responses."""
        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "invalid"},
        )

        assert response.status_code == 400
        assert "X-Request-ID" in response.headers


# =============================================================================
# Test Security Headers
# =============================================================================


class TestSecurityHeaders:
    """Tests for security headers middleware."""

    def test_security_headers_present(self, client: TestClient) -> None:
        """Test that security headers are present in responses."""
        response = client.get("/health")

        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"

        csp = response.headers.get("Content-Security-Policy", "")
        assert "default-src 'self'" in csp
        # Verify we allow CDNs for docs
        assert "cdn.jsdelivr.net" in csp
        assert "fastapi.tiangolo.com" in csp


# =============================================================================
# Test File Size Validation
# =============================================================================


class TestFileSizeValidation:
    """Tests for file size validation."""

    def test_validate_file_size_passes_for_small_file(self) -> None:
        """Test that small files pass validation."""
        small_content = b"x" * 1024  # 1KB
        validate_file_size(small_content, max_size_mb=1, file_type="audio")

    def test_validate_file_size_raises_for_large_file(self) -> None:
        """Test that large files raise HTTPException."""
        from fastapi import HTTPException

        # Create content larger than 1MB
        large_content = b"x" * (2 * 1024 * 1024)  # 2MB

        with pytest.raises(HTTPException) as exc_info:
            validate_file_size(large_content, max_size_mb=1, file_type="audio")

        assert exc_info.value.status_code == 413
        assert "Audio file too large" in str(exc_info.value.detail)
        assert "1 MB" in str(exc_info.value.detail)

    def test_file_size_limit_boundary(self) -> None:
        """Test file size at exactly the boundary."""
        from fastapi import HTTPException

        # Exactly 1MB should pass
        exactly_1mb = b"x" * (1024 * 1024)
        validate_file_size(exactly_1mb, max_size_mb=1, file_type="audio")

        # 1 byte over should fail
        over_1mb = b"x" * (1024 * 1024 + 1)
        with pytest.raises(HTTPException):
            validate_file_size(over_1mb, max_size_mb=1, file_type="audio")


# =============================================================================
# Test Transcript Validation
# =============================================================================


class TestTranscriptValidation:
    """Tests for transcript JSON validation."""

    def test_validate_transcript_json_valid(self) -> None:
        """Test that valid transcript JSON passes validation."""
        valid_json = json.dumps(
            {
                "file_name": "test.wav",
                "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello"}],
            }
        ).encode()
        validate_transcript_json(valid_json)

    def test_validate_transcript_json_invalid_json(self) -> None:
        """Test that invalid JSON raises HTTPException."""
        from fastapi import HTTPException

        invalid_json = b"{not valid json"
        with pytest.raises(HTTPException) as exc_info:
            validate_transcript_json(invalid_json)

        assert exc_info.value.status_code == 400
        assert "Invalid transcript JSON" in str(exc_info.value.detail)

    def test_validate_transcript_json_missing_file_name(self) -> None:
        """Test that missing file_name raises HTTPException."""
        from fastapi import HTTPException

        missing_field = json.dumps(
            {"segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello"}]}
        ).encode()

        with pytest.raises(HTTPException) as exc_info:
            validate_transcript_json(missing_field)

        assert exc_info.value.status_code == 400
        assert "file_name" in str(exc_info.value.detail)

    def test_validate_transcript_json_missing_segments(self) -> None:
        """Test that missing segments raises HTTPException."""
        from fastapi import HTTPException

        missing_segments = json.dumps({"file_name": "test.wav"}).encode()

        with pytest.raises(HTTPException) as exc_info:
            validate_transcript_json(missing_segments)

        assert exc_info.value.status_code == 400
        assert "segments" in str(exc_info.value.detail)

    def test_validate_transcript_json_segments_not_list(self) -> None:
        """Test that non-list segments raises HTTPException."""
        from fastapi import HTTPException

        invalid_segments = json.dumps({"file_name": "test.wav", "segments": "not a list"}).encode()

        with pytest.raises(HTTPException) as exc_info:
            validate_transcript_json(invalid_segments)

        assert exc_info.value.status_code == 400
        assert "must be a list" in str(exc_info.value.detail)

    def test_validate_transcript_json_empty_segments(self) -> None:
        """Test that empty segments raises HTTPException."""
        from fastapi import HTTPException

        empty_segments = json.dumps({"file_name": "test.wav", "segments": []}).encode()

        with pytest.raises(HTTPException) as exc_info:
            validate_transcript_json(empty_segments)

        assert exc_info.value.status_code == 400
        assert "no segments found" in str(exc_info.value.detail)


# =============================================================================
# Test Transcribe Endpoint with Mocks
# =============================================================================


class TestTranscribeEndpointMocked:
    """Tests for /transcribe endpoint with mocked transcription."""

    @patch("slower_whisper.pipeline.service_transcribe.transcribe_file")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_transcribe_success(
        self,
        mock_validate_audio: MagicMock,
        mock_transcribe: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
    ) -> None:
        """Test successful transcription returns correct response."""
        mock_transcribe.return_value = sample_transcript
        mock_validate_audio.return_value = None

        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"model": "tiny", "device": "cpu"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["schema_version"] == 2
        assert data["file_name"] == "test.wav"
        assert data["language"] == "en"
        assert len(data["segments"]) == 2
        assert data["segments"][0]["text"] == "Hello world"

    @patch("slower_whisper.pipeline.service_transcribe.transcribe_file")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_transcribe_response_schema(
        self,
        mock_validate_audio: MagicMock,
        mock_transcribe: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
    ) -> None:
        """Test transcription response has all required fields."""
        mock_transcribe.return_value = sample_transcript
        mock_validate_audio.return_value = None

        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "cpu"},
        )

        data = response.json()

        # Required top-level fields
        assert "schema_version" in data
        assert "file_name" in data
        assert "language" in data
        assert "meta" in data
        assert "segments" in data

        # Segment structure
        for segment in data["segments"]:
            assert "id" in segment
            assert "start" in segment
            assert "end" in segment
            assert "text" in segment

    @patch("slower_whisper.pipeline.service_transcribe.transcribe_file")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_transcribe_with_speakers_and_turns(
        self,
        mock_validate_audio: MagicMock,
        mock_transcribe: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
    ) -> None:
        """Test that speakers and turns are included when present."""
        sample_transcript.speakers = [
            {"id": "spk_0", "label": "Speaker 1", "total_speech_time": 2.0}
        ]
        sample_transcript.turns = [
            {"id": "turn_0", "speaker_id": "spk_0", "start": 0.0, "end": 2.0}
        ]
        mock_transcribe.return_value = sample_transcript
        mock_validate_audio.return_value = None

        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "cpu"},
        )

        data = response.json()
        assert "speakers" in data
        assert "turns" in data
        assert data["speakers"][0]["id"] == "spk_0"
        assert data["turns"][0]["speaker_id"] == "spk_0"

    @patch("slower_whisper.pipeline.service_transcribe.transcribe_file")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_transcribe_transcription_error(
        self,
        mock_validate_audio: MagicMock,
        mock_transcribe: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
    ) -> None:
        """Test that TranscriptionError returns 500."""
        from slower_whisper.pipeline.exceptions import TranscriptionError

        mock_transcribe.side_effect = TranscriptionError("Model failed to load")
        mock_validate_audio.return_value = None

        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "cpu"},
        )

        assert response.status_code == 500
        assert "Transcription failed" in response.json()["detail"]

    @patch("slower_whisper.pipeline.service_transcribe.transcribe_file")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_transcribe_configuration_error(
        self,
        mock_validate_audio: MagicMock,
        mock_transcribe: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
    ) -> None:
        """Test that ConfigurationError returns 400."""
        from slower_whisper.pipeline.exceptions import ConfigurationError

        mock_transcribe.side_effect = ConfigurationError("Invalid model")
        mock_validate_audio.return_value = None

        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "cpu"},
        )

        assert response.status_code == 400
        assert "Configuration error" in response.json()["detail"]

    def test_transcribe_all_parameters(self, client: TestClient, sample_wav_bytes: bytes) -> None:
        """Test that all query parameters are accepted."""
        with (
            patch("slower_whisper.pipeline.service_transcribe.transcribe_file") as mock_transcribe,
            patch("slower_whisper.pipeline.service_transcribe.validate_audio_format"),
        ):
            mock_transcribe.return_value = Transcript(
                file_name="test.wav", language="en", segments=[]
            )

            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
                params={
                    "model": "large-v3",
                    "language": "en",
                    "device": "cpu",
                    "compute_type": "float32",
                    "task": "transcribe",
                    "enable_diarization": False,
                    "diarization_device": "auto",
                    "min_speakers": 1,
                    "max_speakers": 5,
                    "overlap_threshold": 0.5,
                },
            )

            assert response.status_code == 200


# =============================================================================
# Test Enrich Endpoint with Mocks
# =============================================================================


class TestEnrichEndpointMocked:
    """Tests for /enrich endpoint with mocked enrichment."""

    @patch("slower_whisper.pipeline.service_enrich.enrich_transcript")
    @patch("slower_whisper.pipeline.api.load_transcript")
    @patch("slower_whisper.pipeline.service_enrich.validate_audio_format")
    def test_enrich_success(
        self,
        mock_validate_audio: MagicMock,
        mock_load: MagicMock,
        mock_enrich: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
        sample_transcript_json: bytes,
    ) -> None:
        """Test successful enrichment returns correct response."""
        mock_load.return_value = sample_transcript
        enriched = Transcript(
            file_name=sample_transcript.file_name,
            language=sample_transcript.language,
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=2.0,
                    text="Hello world",
                    audio_state={
                        "prosody": {"pitch": {"level": "high"}},
                        "rendering": "[audio: high pitch]",
                    },
                )
            ],
        )
        mock_enrich.return_value = enriched
        mock_validate_audio.return_value = None

        response = client.post(
            "/enrich",
            files={
                "transcript": ("transcript.json", sample_transcript_json, "application/json"),
                "audio": ("test.wav", sample_wav_bytes, "audio/wav"),
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["schema_version"] == 2
        assert len(data["segments"]) == 1
        assert data["segments"][0]["audio_state"] is not None
        assert data["segments"][0]["audio_state"]["prosody"]["pitch"]["level"] == "high"

    @patch("slower_whisper.pipeline.service_enrich.enrich_transcript")
    @patch("slower_whisper.pipeline.api.load_transcript")
    @patch("slower_whisper.pipeline.service_enrich.validate_audio_format")
    def test_enrich_with_all_options(
        self,
        mock_validate_audio: MagicMock,
        mock_load: MagicMock,
        mock_enrich: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
        sample_transcript_json: bytes,
    ) -> None:
        """Test enrichment with all optional parameters."""
        mock_load.return_value = sample_transcript
        mock_enrich.return_value = sample_transcript
        mock_validate_audio.return_value = None

        response = client.post(
            "/enrich",
            files={
                "transcript": ("transcript.json", sample_transcript_json, "application/json"),
                "audio": ("test.wav", sample_wav_bytes, "audio/wav"),
            },
            params={
                "enable_prosody": True,
                "enable_emotion": True,
                "enable_categorical_emotion": True,
                "device": "cpu",
            },
        )

        assert response.status_code == 200

    @patch("slower_whisper.pipeline.service_enrich.enrich_transcript")
    @patch("slower_whisper.pipeline.api.load_transcript")
    @patch("slower_whisper.pipeline.service_enrich.validate_audio_format")
    def test_enrich_enrichment_error(
        self,
        mock_validate_audio: MagicMock,
        mock_load: MagicMock,
        mock_enrich: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
        sample_transcript_json: bytes,
    ) -> None:
        """Test that EnrichmentError returns 500."""
        from slower_whisper.pipeline.exceptions import EnrichmentError

        mock_load.return_value = sample_transcript
        mock_enrich.side_effect = EnrichmentError("Feature extraction failed")
        mock_validate_audio.return_value = None

        response = client.post(
            "/enrich",
            files={
                "transcript": ("transcript.json", sample_transcript_json, "application/json"),
                "audio": ("test.wav", sample_wav_bytes, "audio/wav"),
            },
        )

        assert response.status_code == 500
        assert "Enrichment failed" in response.json()["detail"]

    @patch("slower_whisper.pipeline.service_enrich.enrich_transcript")
    @patch("slower_whisper.pipeline.api.load_transcript")
    @patch("slower_whisper.pipeline.service_enrich.validate_audio_format")
    def test_enrich_configuration_error(
        self,
        mock_validate_audio: MagicMock,
        mock_load: MagicMock,
        mock_enrich: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
        sample_transcript_json: bytes,
    ) -> None:
        """Test that ConfigurationError returns 400."""
        from slower_whisper.pipeline.exceptions import ConfigurationError

        mock_load.return_value = sample_transcript
        mock_enrich.side_effect = ConfigurationError("Invalid config")
        mock_validate_audio.return_value = None

        response = client.post(
            "/enrich",
            files={
                "transcript": ("transcript.json", sample_transcript_json, "application/json"),
                "audio": ("test.wav", sample_wav_bytes, "audio/wav"),
            },
        )

        assert response.status_code == 400
        assert "Configuration error" in response.json()["detail"]


# =============================================================================
# Test Exception Handlers
# =============================================================================


class TestExceptionHandlers:
    """Tests for custom exception handlers."""

    @patch("slower_whisper.pipeline.service_transcribe.transcribe_file")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_unhandled_exception_returns_500(
        self,
        mock_validate_audio: MagicMock,
        mock_transcribe: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
    ) -> None:
        """Test that unhandled exceptions return 500 with generic message."""
        # Simulate an unexpected exception during transcription
        mock_transcribe.side_effect = RuntimeError("Unexpected internal error")
        mock_validate_audio.return_value = None

        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "cpu"},
        )

        assert response.status_code == 500
        body = response.json()

        # Check that we get a 500 response with error details
        assert "error" in body or "detail" in body
        # The response should contain indication of error
        if "error" in body:
            assert body["error"]["status_code"] == 500

    def test_validation_error_structured_response(
        self, client: TestClient, sample_wav_bytes: bytes
    ) -> None:
        """Test that validation errors have structured format."""
        # Missing required audio file
        response = client.post("/transcribe")

        assert response.status_code == 422
        body = response.json()

        assert body["error"]["type"] == "validation_error"
        assert "validation_errors" in body["error"]["details"]

        # Check validation error structure
        errors = body["error"]["details"]["validation_errors"]
        assert len(errors) > 0
        assert "loc" in errors[0]
        assert "msg" in errors[0]
        assert "type" in errors[0]


# =============================================================================
# Test Transcript to Dict Helper
# =============================================================================


class TestTranscriptToDict:
    """Tests for _transcript_to_dict helper function."""

    def test_basic_transcript_conversion(self, sample_transcript: Transcript) -> None:
        """Test basic transcript conversion to dict."""
        result = _transcript_to_dict(sample_transcript)

        assert result["schema_version"] == 2
        assert result["file_name"] == "test.wav"
        assert result["language"] == "en"
        assert len(result["segments"]) == 2

    def test_segment_fields_preserved(self, sample_transcript: Transcript) -> None:
        """Test that all segment fields are preserved."""
        result = _transcript_to_dict(sample_transcript)

        segment = result["segments"][0]
        assert segment["id"] == 0
        assert segment["start"] == 0.0
        assert segment["end"] == 2.0
        assert segment["text"] == "Hello world"
        assert segment["speaker"] is None
        assert segment["tone"] is None
        assert segment["audio_state"] is None

    def test_optional_fields_excluded_when_none(self, sample_transcript: Transcript) -> None:
        """Test that speakers/turns are excluded when None."""
        result = _transcript_to_dict(sample_transcript)

        assert "speakers" not in result
        assert "turns" not in result

    def test_optional_fields_included_when_present(self, sample_transcript: Transcript) -> None:
        """Test that speakers/turns are included when present."""
        sample_transcript.speakers = [{"id": "spk_0"}]
        sample_transcript.turns = [{"id": "turn_0"}]

        result = _transcript_to_dict(sample_transcript)

        assert "speakers" in result
        assert "turns" in result
        assert result["speakers"] == [{"id": "spk_0"}]
        assert result["turns"] == [{"id": "turn_0"}]

    def test_audio_state_preserved(self) -> None:
        """Test that audio_state is preserved in segments."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="Hello",
                    audio_state={
                        "prosody": {"pitch": {"level": "high"}},
                        "rendering": "[audio: high pitch]",
                    },
                )
            ],
        )

        result = _transcript_to_dict(transcript)

        assert result["segments"][0]["audio_state"] is not None
        assert result["segments"][0]["audio_state"]["prosody"]["pitch"]["level"] == "high"

    def test_words_excluded_by_default(self) -> None:
        """Test that words are excluded from segments by default."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="Hello world",
                    words=[
                        Word(word="Hello", start=0.0, end=0.4, probability=0.95),
                        Word(word="world", start=0.5, end=1.0, probability=0.92),
                    ],
                )
            ],
        )

        result = _transcript_to_dict(transcript)

        # Words should NOT be in output by default
        assert "words" not in result["segments"][0]

    def test_words_included_when_requested(self) -> None:
        """Test that words are included when include_words=True."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="Hello world",
                    words=[
                        Word(word="Hello", start=0.0, end=0.4, probability=0.95),
                        Word(word="world", start=0.5, end=1.0, probability=0.92),
                    ],
                )
            ],
        )

        result = _transcript_to_dict(transcript, include_words=True)

        # Words should be included
        assert "words" in result["segments"][0]
        assert len(result["segments"][0]["words"]) == 2
        assert result["segments"][0]["words"][0]["word"] == "Hello"
        assert result["segments"][0]["words"][0]["probability"] == 0.95
        assert result["segments"][0]["words"][1]["word"] == "world"

    def test_words_not_included_when_segment_has_none(self) -> None:
        """Test that words field is absent when segment.words is None."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[Segment(id=0, start=0.0, end=1.0, text="Hello", words=None)],
        )

        result = _transcript_to_dict(transcript, include_words=True)

        # No words field should be present
        assert "words" not in result["segments"][0]


# =============================================================================
# Test Word Serialization Helpers
# =============================================================================


class TestWordSerialization:
    """Tests for word serialization helper functions."""

    def test_word_to_dict_from_dataclass(self) -> None:
        """Test _word_to_dict with Word dataclass."""
        word = Word(word="hello", start=0.0, end=0.5, probability=0.95)
        result = _word_to_dict(word)

        assert result["word"] == "hello"
        assert result["start"] == 0.0
        assert result["end"] == 0.5
        assert result["probability"] == 0.95
        assert "speaker" not in result  # Not included when None

    def test_word_to_dict_with_speaker(self) -> None:
        """Test _word_to_dict includes speaker when present."""
        word = Word(word="hello", start=0.0, end=0.5, probability=0.95, speaker="spk_0")
        result = _word_to_dict(word)

        assert result["speaker"] == "spk_0"

    def test_word_to_dict_from_dict(self) -> None:
        """Test _word_to_dict with dict input."""
        word_dict = {"word": "test", "start": 1.0, "end": 1.5, "probability": 0.8}
        result = _word_to_dict(word_dict)

        assert result == word_dict

    def test_segment_to_dict_without_words(self) -> None:
        """Test _segment_to_dict excludes words when include_words=False."""
        seg = Segment(
            id=0,
            start=0.0,
            end=1.0,
            text="Hello",
            words=[Word(word="Hello", start=0.0, end=1.0, probability=0.9)],
        )
        result = _segment_to_dict(seg, include_words=False)

        assert "words" not in result
        assert result["text"] == "Hello"

    def test_segment_to_dict_with_words(self) -> None:
        """Test _segment_to_dict includes words when include_words=True."""
        seg = Segment(
            id=0,
            start=0.0,
            end=1.0,
            text="Hello world",
            words=[
                Word(word="Hello", start=0.0, end=0.4, probability=0.95),
                Word(word="world", start=0.5, end=1.0, probability=0.92),
            ],
        )
        result = _segment_to_dict(seg, include_words=True)

        assert "words" in result
        assert len(result["words"]) == 2


# =============================================================================
# Test 404 Responses
# =============================================================================


class TestNotFoundResponses:
    """Tests for 404 Not Found responses."""

    def test_unknown_endpoint_returns_404(self, client: TestClient) -> None:
        """Test that unknown endpoints return 404."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

    def test_unknown_post_endpoint_returns_404(self, client: TestClient) -> None:
        """Test that POST to unknown endpoint returns 404."""
        response = client.post("/nonexistent-endpoint")
        assert response.status_code == 404


# =============================================================================
# Test Audio Validation Edge Cases
# =============================================================================


class TestAudioValidationEdgeCases:
    """Tests for audio validation edge cases."""

    def test_empty_audio_file(self, client: TestClient) -> None:
        """Test that empty audio file is rejected."""
        response = client.post(
            "/transcribe",
            files={"audio": ("empty.wav", b"", "audio/wav")},
            params={"device": "cpu"},
        )

        # Should fail validation
        assert response.status_code == 400

    def test_non_audio_file_rejected(self, client: TestClient) -> None:
        """Test that non-audio files are rejected."""
        # Try to upload a text file as audio
        response = client.post(
            "/transcribe",
            files={"audio": ("test.txt", b"This is not audio", "text/plain")},
            params={"device": "cpu"},
        )

        # Should fail validation
        assert response.status_code == 400


# =============================================================================
# Test Query Parameter Validation
# =============================================================================


class TestQueryParameterValidation:
    """Tests for query parameter validation."""

    def test_invalid_overlap_threshold_too_low(
        self, client: TestClient, sample_wav_bytes: bytes
    ) -> None:
        """Test that overlap_threshold below 0 is rejected."""
        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"overlap_threshold": -0.1},
        )

        assert response.status_code == 422

    def test_invalid_overlap_threshold_too_high(
        self, client: TestClient, sample_wav_bytes: bytes
    ) -> None:
        """Test that overlap_threshold above 1 is rejected."""
        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"overlap_threshold": 1.5},
        )

        assert response.status_code == 422

    def test_invalid_min_speakers_too_low(
        self, client: TestClient, sample_wav_bytes: bytes
    ) -> None:
        """Test that min_speakers below 1 is rejected."""
        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"min_speakers": 0},
        )

        assert response.status_code == 422

    def test_valid_translate_task(self, client: TestClient, sample_wav_bytes: bytes) -> None:
        """Test that 'translate' task is accepted."""
        with (
            patch("slower_whisper.pipeline.service_transcribe.transcribe_file") as mock_transcribe,
            patch("slower_whisper.pipeline.service_transcribe.validate_audio_format"),
        ):
            mock_transcribe.return_value = Transcript(
                file_name="test.wav", language="en", segments=[]
            )

            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
                params={"task": "translate", "device": "cpu"},
            )

            assert response.status_code == 200


# =============================================================================
# Test Content-Type Handling
# =============================================================================


class TestContentTypeHandling:
    """Tests for Content-Type handling in file uploads."""

    def test_audio_without_explicit_content_type(
        self, client: TestClient, sample_wav_bytes: bytes
    ) -> None:
        """Test audio upload without explicit content type."""
        with (
            patch("slower_whisper.pipeline.service_transcribe.transcribe_file") as mock_transcribe,
            patch("slower_whisper.pipeline.service_transcribe.validate_audio_format"),
        ):
            mock_transcribe.return_value = Transcript(
                file_name="test.wav", language="en", segments=[]
            )

            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", sample_wav_bytes)},
                params={"device": "cpu"},
            )

            # Should still work without explicit content type
            assert response.status_code == 200

    def test_json_transcript_content_type(
        self,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript_json: bytes,
    ) -> None:
        """Test transcript with JSON content type."""
        with (
            patch("slower_whisper.pipeline.service_enrich.enrich_transcript") as mock_enrich,
            patch("slower_whisper.pipeline.api.load_transcript") as mock_load,
            patch("slower_whisper.pipeline.service_enrich.validate_audio_format"),
        ):
            mock_load.return_value = Transcript(
                file_name="test.wav",
                language="en",
                segments=[Segment(id=0, start=0.0, end=1.0, text="Hello")],
            )
            mock_enrich.return_value = mock_load.return_value

            response = client.post(
                "/enrich",
                files={
                    "transcript": (
                        "transcript.json",
                        sample_transcript_json,
                        "application/json",
                    ),
                    "audio": ("test.wav", sample_wav_bytes, "audio/wav"),
                },
            )

            assert response.status_code == 200


# =============================================================================
# Test SSE Streaming Endpoint
# =============================================================================


class TestTranscribeStreamingEndpoint:
    """Tests for /transcribe/stream SSE endpoint."""

    @patch("slower_whisper.pipeline.asr_engine.TranscriptionEngine")
    @patch("slower_whisper.pipeline.audio_io.normalize_single")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_stream_returns_sse_content_type(
        self,
        mock_validate_audio: MagicMock,
        mock_normalize: MagicMock,
        mock_engine_class: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
    ) -> None:
        """Test that streaming endpoint returns text/event-stream content type."""
        # Setup mocks
        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = sample_transcript
        mock_engine_class.return_value = mock_engine
        mock_validate_audio.return_value = None

        def fake_normalize(src: Path, dst: Path) -> None:
            dst.write_bytes(sample_wav_bytes)

        mock_normalize.side_effect = fake_normalize

        response = client.post(
            "/transcribe/stream",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"model": "tiny", "device": "cpu"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    @patch("slower_whisper.pipeline.asr_engine.TranscriptionEngine")
    @patch("slower_whisper.pipeline.audio_io.normalize_single")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_stream_emits_segment_events(
        self,
        mock_validate_audio: MagicMock,
        mock_normalize: MagicMock,
        mock_engine_class: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
    ) -> None:
        """Test that streaming emits segment events."""
        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = sample_transcript
        mock_engine_class.return_value = mock_engine
        mock_validate_audio.return_value = None

        def fake_normalize(src: Path, dst: Path) -> None:
            dst.write_bytes(sample_wav_bytes)

        mock_normalize.side_effect = fake_normalize

        response = client.post(
            "/transcribe/stream",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "cpu"},
        )

        assert response.status_code == 200
        content = response.text

        # New envelope format uses data: {envelope_json}
        # Should contain PARTIAL or FINALIZED segment events
        assert '"type": "PARTIAL"' in content or '"type": "FINALIZED"' in content
        # Should contain SESSION_ENDED event
        assert '"type": "SESSION_ENDED"' in content

    @patch("slower_whisper.pipeline.asr_engine.TranscriptionEngine")
    @patch("slower_whisper.pipeline.audio_io.normalize_single")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_stream_done_event_contains_metadata(
        self,
        mock_validate_audio: MagicMock,
        mock_normalize: MagicMock,
        mock_engine_class: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
    ) -> None:
        """Test that done event contains expected metadata."""
        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = sample_transcript
        mock_engine_class.return_value = mock_engine
        mock_validate_audio.return_value = None

        def fake_normalize(src: Path, dst: Path) -> None:
            dst.write_bytes(sample_wav_bytes)

        mock_normalize.side_effect = fake_normalize

        response = client.post(
            "/transcribe/stream",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "cpu"},
        )

        content = response.text

        # Parse SSE events
        events = _parse_sse_events(content)
        session_ended_events = [e for e in events if e["type"] == "SESSION_ENDED"]

        assert len(session_ended_events) == 1
        ended_data = session_ended_events[0]["data"]
        assert "total_segments" in ended_data
        assert ended_data["total_segments"] == 2
        assert "language" in ended_data
        assert ended_data["language"] == "en"
        # Also verify stats are present in new format
        assert "stats" in ended_data

    def test_stream_invalid_device_returns_400(
        self, client: TestClient, sample_wav_bytes: bytes
    ) -> None:
        """Test that invalid device returns 400."""
        response = client.post(
            "/transcribe/stream",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "invalid"},
        )

        assert response.status_code == 400
        assert "Invalid device" in response.json()["detail"]

    def test_stream_invalid_task_returns_400(
        self, client: TestClient, sample_wav_bytes: bytes
    ) -> None:
        """Test that invalid task returns 400."""
        response = client.post(
            "/transcribe/stream",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"task": "invalid"},
        )

        assert response.status_code == 400
        assert "Invalid task" in response.json()["detail"]

    def test_stream_missing_audio_returns_422(self, client: TestClient) -> None:
        """Test that missing audio file returns 422."""
        response = client.post("/transcribe/stream", params={"device": "cpu"})

        assert response.status_code == 422

    @patch("slower_whisper.pipeline.asr_engine.TranscriptionEngine")
    @patch("slower_whisper.pipeline.audio_io.normalize_single")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_stream_segment_data_structure(
        self,
        mock_validate_audio: MagicMock,
        mock_normalize: MagicMock,
        mock_engine_class: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
    ) -> None:
        """Test that segment events have correct data structure."""
        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = sample_transcript
        mock_engine_class.return_value = mock_engine
        mock_validate_audio.return_value = None

        def fake_normalize(src: Path, dst: Path) -> None:
            dst.write_bytes(sample_wav_bytes)

        mock_normalize.side_effect = fake_normalize

        response = client.post(
            "/transcribe/stream",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "cpu"},
        )

        events = _parse_sse_events(response.text)
        # New format uses PARTIAL and FINALIZED for segments
        segment_events = [e for e in events if e["type"] in ("PARTIAL", "FINALIZED")]

        # With 2 segments: first is PARTIAL, last is FINALIZED
        assert len(segment_events) == 2

        # Check segment structure (data is the segment dict)
        first_segment = segment_events[0]["data"]
        assert "id" in first_segment
        assert "start" in first_segment
        assert "end" in first_segment
        assert "text" in first_segment
        assert first_segment["text"] == "Hello world"

        # Check envelope structure
        envelope = segment_events[0]["envelope"]
        assert "event_id" in envelope
        assert "stream_id" in envelope
        assert envelope["stream_id"].startswith("sse-")
        assert "ts_server" in envelope

    @patch("slower_whisper.pipeline.asr_engine.TranscriptionEngine")
    @patch("slower_whisper.pipeline.audio_io.normalize_single")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_stream_progress_events(
        self,
        mock_validate_audio: MagicMock,
        mock_normalize: MagicMock,
        mock_engine_class: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
    ) -> None:
        """Test that progress events are emitted."""
        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = sample_transcript
        mock_engine_class.return_value = mock_engine
        mock_validate_audio.return_value = None

        def fake_normalize(src: Path, dst: Path) -> None:
            dst.write_bytes(sample_wav_bytes)

        mock_normalize.side_effect = fake_normalize

        response = client.post(
            "/transcribe/stream",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "cpu"},
        )

        events = _parse_sse_events(response.text)
        # New format doesn't have explicit progress events; uses PARTIAL/FINALIZED
        # Verify we have segment events (PARTIAL or FINALIZED)
        segment_events = [e for e in events if e["type"] in ("PARTIAL", "FINALIZED")]

        # Should have at least one segment event
        assert len(segment_events) >= 1

        # Verify SESSION_ENDED contains stats with segment counts
        session_ended = [e for e in events if e["type"] == "SESSION_ENDED"]
        assert len(session_ended) == 1
        stats = session_ended[0]["data"].get("stats", {})
        # Stats should track segment counts
        assert "segments_finalized" in stats or "segments_partial" in stats

    @patch("slower_whisper.pipeline.audio_io.normalize_single")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_stream_handles_empty_audio(
        self,
        mock_validate_audio: MagicMock,
        mock_normalize: MagicMock,
        client: TestClient,
    ) -> None:
        """Test handling of empty audio file."""
        mock_validate_audio.side_effect = HTTPException(
            status_code=400,
            detail="Invalid audio file: file is empty.",
        )

        response = client.post(
            "/transcribe/stream",
            files={"audio": ("empty.wav", b"", "audio/wav")},
            params={"device": "cpu"},
        )

        assert response.status_code == 400

    @patch("slower_whisper.pipeline.asr_engine.TranscriptionEngine")
    @patch("slower_whisper.pipeline.audio_io.normalize_single")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_stream_with_word_timestamps(
        self,
        mock_validate_audio: MagicMock,
        mock_normalize: MagicMock,
        mock_engine_class: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
    ) -> None:
        """Test streaming with word timestamps enabled."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="Hello world",
                    words=[
                        Word(word="Hello", start=0.0, end=0.5, probability=0.95),
                        Word(word="world", start=0.5, end=1.0, probability=0.9),
                    ],
                )
            ],
        )

        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = transcript
        mock_engine_class.return_value = mock_engine
        mock_validate_audio.return_value = None

        def fake_normalize(src: Path, dst: Path) -> None:
            dst.write_bytes(sample_wav_bytes)

        mock_normalize.side_effect = fake_normalize

        response = client.post(
            "/transcribe/stream",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "cpu", "word_timestamps": True},
        )

        events = _parse_sse_events(response.text)
        # Single segment should be FINALIZED (last segment is always finalized)
        segment_events = [e for e in events if e["type"] == "FINALIZED"]

        assert len(segment_events) == 1
        segment_data = segment_events[0]["data"]
        assert "words" in segment_data
        assert len(segment_data["words"]) == 2

    @patch("slower_whisper.pipeline.asr_engine.TranscriptionEngine")
    @patch("slower_whisper.pipeline.audio_io.normalize_single")
    @patch("slower_whisper.pipeline.service_transcribe.validate_audio_format")
    def test_stream_cache_control_headers(
        self,
        mock_validate_audio: MagicMock,
        mock_normalize: MagicMock,
        mock_engine_class: MagicMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript: Transcript,
    ) -> None:
        """Test that appropriate caching headers are set."""
        mock_engine = MagicMock()
        mock_engine.transcribe_file.return_value = sample_transcript
        mock_engine_class.return_value = mock_engine
        mock_validate_audio.return_value = None

        def fake_normalize(src: Path, dst: Path) -> None:
            dst.write_bytes(sample_wav_bytes)

        mock_normalize.side_effect = fake_normalize

        response = client.post(
            "/transcribe/stream",
            files={"audio": ("test.wav", sample_wav_bytes, "audio/wav")},
            params={"device": "cpu"},
        )

        assert response.status_code == 200
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("x-accel-buffering") == "no"


def _parse_sse_events(content: str) -> list[dict]:
    """Parse SSE event stream into list of event dicts.

    Handles both old format (event: type\\ndata: json) and new envelope format
    (data: {envelope with type inside}).

    Args:
        content: Raw SSE text content

    Returns:
        List of dicts with 'type' and 'data' keys (normalized from envelope)
    """
    events = []

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("data:"):
            data_str = line[5:].strip()
            try:
                envelope = json.loads(data_str)
                # New envelope format: type is inside the JSON
                if isinstance(envelope, dict) and "type" in envelope:
                    # Normalize to expected test format
                    event_type = envelope.get("type", "")
                    payload = envelope.get("payload", {})

                    # Map new event types to test-friendly names
                    if event_type == "PARTIAL":
                        events.append(
                            {
                                "type": "PARTIAL",
                                "data": payload.get("segment", {}),
                                "envelope": envelope,
                            }
                        )
                    elif event_type == "FINALIZED":
                        events.append(
                            {
                                "type": "FINALIZED",
                                "data": payload.get("segment", {}),
                                "envelope": envelope,
                            }
                        )
                    elif event_type == "SESSION_ENDED":
                        events.append(
                            {
                                "type": "SESSION_ENDED",
                                "data": payload,
                                "envelope": envelope,
                            }
                        )
                    elif event_type == "ERROR":
                        events.append(
                            {
                                "type": "ERROR",
                                "data": payload,
                                "envelope": envelope,
                            }
                        )
                    else:
                        # Unknown type, store as-is
                        events.append(
                            {
                                "type": event_type,
                                "data": payload,
                                "envelope": envelope,
                            }
                        )
                else:
                    # Old format or unexpected structure
                    events.append({"type": "unknown", "data": envelope})
            except json.JSONDecodeError:
                # Not JSON, skip
                pass

    return events
