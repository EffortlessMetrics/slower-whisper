"""
Tests for the FastAPI service wrapper.

These tests use the FastAPI TestClient to verify endpoint behavior without
requiring a running server.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# Skip all tests if API dependencies are not installed
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from slower_whisper.pipeline.models import Segment, Transcript  # noqa: E402
from slower_whisper.pipeline.service import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""

    # Patch normalize_all to copy files instead of running ffmpeg
    # This allows tests to run in environments without ffmpeg
    def mock_normalize_all(paths):
        import shutil

        for src in paths.raw_dir.glob("*"):
            if src.is_file():
                dst = paths.norm_dir / f"{src.stem}.wav"
                if not dst.exists():
                    shutil.copy(src, dst)

    with patch("slower_whisper.pipeline.audio_io.normalize_all", side_effect=mock_normalize_all):
        yield TestClient(app)


@pytest.fixture
def sample_transcript() -> Transcript:
    """Create a sample transcript for testing."""
    return Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(
                id=0,
                start=0.0,
                end=2.0,
                text="Hello world",
            ),
            Segment(
                id=1,
                start=2.0,
                end=4.0,
                text="This is a test",
            ),
        ],
        meta={"test": "value"},
    )


@pytest.fixture
def sample_audio_wav(tmp_path: Path) -> Path:
    """
    Create a minimal valid WAV file for testing.

    This is a 1-second silent mono 16kHz WAV file.
    """
    import struct
    import wave

    wav_path = tmp_path / "test.wav"

    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(16000)  # 16kHz

        # Write 1 second of silence (16000 samples of 0)
        samples = struct.pack("h" * 16000, *[0] * 16000)
        wav.writeframes(samples)

    return wav_path


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, client):
        """Test that health endpoint returns 200 and correct status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["service"] == "slower-whisper-api"
        assert "version" in data
        assert "schema_version" in data


class TestTranscribeEndpoint:
    """Tests for the /transcribe endpoint."""

    @pytest.mark.requires_enrich  # Requires faster-whisper
    @pytest.mark.slow  # Transcription is slow
    def test_transcribe_minimal(self, client, sample_audio_wav):
        """Test transcription with minimal parameters."""
        with open(sample_audio_wav, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", f, "audio/wav")},
                params={"model": "tiny", "device": "cpu"},
            )

        assert response.status_code == 200
        data = response.json()

        assert "schema_version" in data
        assert "file_name" in data
        assert "language" in data
        assert "segments" in data
        assert isinstance(data["segments"], list)

    def test_transcribe_missing_audio(self, client):
        """Test that missing audio file returns 422."""
        response = client.post(
            "/transcribe",
            params={"model": "tiny", "device": "cpu"},
        )

        assert response.status_code == 422

    def test_transcribe_invalid_device(self, client, sample_audio_wav):
        """Test that invalid device parameter returns 400."""
        with open(sample_audio_wav, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", f, "audio/wav")},
                params={"model": "tiny", "device": "invalid"},
            )

        assert response.status_code == 400
        assert "Invalid device" in response.json()["detail"]

    def test_transcribe_invalid_task(self, client, sample_audio_wav):
        """Test that invalid task parameter returns 400."""
        with open(sample_audio_wav, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", f, "audio/wav")},
                params={"model": "tiny", "task": "invalid"},
            )

        assert response.status_code == 400
        assert "Invalid task" in response.json()["detail"]

    def test_transcribe_invalid_compute_type(self, client, sample_audio_wav):
        """Invalid compute_type should return a 400 error."""
        with open(sample_audio_wav, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", f, "audio/wav")},
                params={"compute_type": "fp99"},
            )

        assert response.status_code == 400
        assert "Invalid compute_type" in response.json()["detail"]

    def test_transcribe_with_diarization_flags(self, client, sample_audio_wav):
        """Diarization options should be accepted and reflected in metadata."""
        with open(sample_audio_wav, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", f, "audio/wav")},
                params={
                    "enable_diarization": True,
                    "diarization_device": "auto",
                    "min_speakers": 1,
                    "max_speakers": 2,
                    "overlap_threshold": 0.25,
                },
            )

        assert response.status_code == 200
        data = response.json()

        diar_meta = data["meta"].get("diarization", {})
        assert diar_meta.get("requested") is True
        status = diar_meta.get("status")
        assert status in {"ok", "error", "skipped"}
        if status == "skipped":
            # Missing HF_TOKEN or pyannote dependency
            assert diar_meta.get("error_type") in {"auth", "missing_dependency"}

    def test_transcribe_invalid_diarization_bounds(self, client, sample_audio_wav):
        """min_speakers > max_speakers should return a 400 error.

        Note: Detailed validation info (which parameter failed) is logged
        server-side for debugging but not exposed to clients for security.
        """
        with open(sample_audio_wav, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", f, "audio/wav")},
                params={
                    "enable_diarization": True,
                    "min_speakers": 3,
                    "max_speakers": 2,
                },
            )

        assert response.status_code == 400
        # Security: generic message, details logged server-side
        assert "Configuration error" in response.json()["detail"]

    def test_transcribe_invalid_diarization_device(self, client, sample_audio_wav):
        """Invalid diarization_device should return a 400 error."""
        with open(sample_audio_wav, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", f, "audio/wav")},
                params={"diarization_device": "tpu"},
            )

        assert response.status_code == 400
        assert "diarization_device" in response.json()["detail"]


class TestEnrichEndpoint:
    """Tests for the /enrich endpoint."""

    def test_enrich_missing_transcript(self, client, sample_audio_wav):
        """Test that missing transcript file returns 422."""
        with open(sample_audio_wav, "rb") as f:
            response = client.post(
                "/enrich",
                files={"audio": ("test.wav", f, "audio/wav")},
            )

        assert response.status_code == 422

    def test_enrich_missing_audio(self, client, sample_transcript, tmp_path):
        """Test that missing audio file returns 422."""
        # Create transcript JSON
        transcript_path = tmp_path / "transcript.json"
        from slower_whisper.pipeline.writers import write_json

        write_json(sample_transcript, transcript_path)

        with open(transcript_path, "rb") as f:
            response = client.post(
                "/enrich",
                files={"transcript": ("transcript.json", f, "application/json")},
            )

        assert response.status_code == 422

    def test_enrich_invalid_device(self, client, sample_transcript, sample_audio_wav, tmp_path):
        """Test that invalid device parameter returns 400."""
        # Create transcript JSON
        transcript_path = tmp_path / "transcript.json"
        from slower_whisper.pipeline.writers import write_json

        write_json(sample_transcript, transcript_path)

        with open(transcript_path, "rb") as t, open(sample_audio_wav, "rb") as a:
            response = client.post(
                "/enrich",
                files={
                    "transcript": ("transcript.json", t, "application/json"),
                    "audio": ("test.wav", a, "audio/wav"),
                },
                params={"device": "invalid"},
            )

        assert response.status_code == 400
        assert "Invalid device" in response.json()["detail"]

    def test_enrich_invalid_json(self, client, sample_audio_wav, tmp_path):
        """Test that invalid transcript JSON returns 400."""
        # Create invalid JSON
        invalid_json_path = tmp_path / "invalid.json"
        invalid_json_path.write_text("{invalid json")

        with open(invalid_json_path, "rb") as t, open(sample_audio_wav, "rb") as a:
            response = client.post(
                "/enrich",
                files={
                    "transcript": ("invalid.json", t, "application/json"),
                    "audio": ("test.wav", a, "audio/wav"),
                },
            )

        assert response.status_code == 400
        assert "Invalid transcript JSON" in response.json()["detail"]


def test_transcript_to_dict_includes_optional_fields(sample_transcript):
    """Helper serializer should emit diarization fields when present."""
    from slower_whisper.pipeline.service_serialization import _transcript_to_dict

    sample_transcript.speakers = [
        {"id": "spk_0", "label": None, "total_speech_time": 1.5, "num_segments": 1}
    ]
    sample_transcript.turns = [
        {
            "id": "turn_0",
            "speaker_id": "spk_0",
            "start": 0.0,
            "end": 2.0,
            "segment_ids": [0],
            "text": "Hello world",
        }
    ]

    payload = _transcript_to_dict(sample_transcript)

    assert payload["speakers"] == sample_transcript.speakers
    assert payload["turns"] == sample_transcript.turns
    assert payload["segments"]  # ensure base fields remain


class TestOpenAPIDocumentation:
    """Tests for OpenAPI documentation endpoints."""

    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()

        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

        # Check that our endpoints are documented
        assert "/health" in schema["paths"]
        assert "/transcribe" in schema["paths"]
        assert "/enrich" in schema["paths"]

    def test_swagger_ui(self, client):
        """Test that Swagger UI is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()

    def test_redoc(self, client):
        """Test that ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Integration tests for complete workflows."""

    @pytest.mark.requires_enrich
    @pytest.mark.slow
    def test_transcribe_then_enrich(self, client, sample_audio_wav, tmp_path):
        """Test complete workflow: transcribe then enrich."""
        # Step 1: Transcribe
        with open(sample_audio_wav, "rb") as f:
            transcribe_response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", f, "audio/wav")},
                params={"model": "tiny", "device": "cpu"},
            )

        assert transcribe_response.status_code == 200
        transcript_data = transcribe_response.json()

        # Save transcript to file
        transcript_path = tmp_path / "transcript.json"
        transcript_path.write_text(json.dumps(transcript_data))

        # Step 2: Enrich (skip because it requires enrichment dependencies)
        # This is a placeholder for the full workflow test
        # In practice, you would need prosody/emotion dependencies installed
        #
        # with open(transcript_path, "rb") as t, open(sample_audio_wav, "rb") as a:
        #     enrich_response = client.post(
        #         "/enrich",
        #         files={
        #             "transcript": ("transcript.json", t, "application/json"),
        #             "audio": ("test.wav", a, "audio/wav"),
        #         },
        #         params={"enable_prosody": True, "enable_emotion": False},
        #     )
        #
        # assert enrich_response.status_code == 200
        # enriched_data = enrich_response.json()
        # assert enriched_data["segments"][0].get("audio_state") is not None
