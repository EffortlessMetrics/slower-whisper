
from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from transcription.models import Segment, Transcript
from transcription.service import app

@pytest.fixture
def client() -> TestClient:
    return TestClient(app)

@pytest.fixture
def sample_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        samples = struct.pack("h" * 16000, *[0] * 16000)
        wav.writeframes(samples)
    return buffer.getvalue()

@pytest.fixture
def sample_transcript_json() -> bytes:
    data = {
        "schema_version": 2,
        "file_name": "test.wav",
        "language": "en",
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello"}],
    }
    return json.dumps(data).encode("utf-8")

class TestEnrichSecurity:
    """Tests for security improvements in enrich endpoint."""

    @patch("transcription.service_enrich.save_upload_file_streaming", new_callable=AsyncMock)
    @patch("transcription.service_enrich.enrich_transcript")
    @patch("transcription.api.load_transcript")
    @patch("transcription.service_enrich.validate_audio_format")
    def test_enrich_streams_transcript_to_disk(
        self,
        mock_validate_audio: MagicMock,
        mock_load: MagicMock,
        mock_enrich: MagicMock,
        mock_save_streaming: AsyncMock,
        client: TestClient,
        sample_wav_bytes: bytes,
        sample_transcript_json: bytes,
    ) -> None:
        """
        Verify that the transcript file is streamed to disk using save_upload_file_streaming
        instead of being read into memory with read().
        """
        # Setup mocks
        mock_load.return_value = Transcript(
            file_name="test.wav",
            language="en",
            segments=[Segment(id=0, start=0.0, end=1.0, text="Hello")],
        )
        mock_enrich.return_value = mock_load.return_value
        mock_validate_audio.return_value = None

        # We need to simulate the side effect of save_upload_file_streaming writing the file
        # because the code will try to read it back.
        async def side_effect(upload, dest, **kwargs):
            dest.write_bytes(sample_transcript_json if "transcript" in str(dest) else sample_wav_bytes)

        mock_save_streaming.side_effect = side_effect

        response = client.post(
            "/enrich",
            files={
                "transcript": ("transcript.json", sample_transcript_json, "application/json"),
                "audio": ("test.wav", sample_wav_bytes, "audio/wav"),
            },
        )

        assert response.status_code == 200

        # Verify save_upload_file_streaming was called twice (audio and transcript)
        assert mock_save_streaming.call_count == 2

        # Verify calls
        calls = mock_save_streaming.call_args_list

        # Extract file_types passed to calls
        file_types = [call.kwargs.get("file_type") for call in calls]

        # One should be "audio" and one should be "transcript"
        assert "audio" in file_types
        assert "transcript" in file_types
