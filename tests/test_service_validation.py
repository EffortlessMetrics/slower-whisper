
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from transcription.service_validation import validate_audio_format

class TestServiceValidation:
    """Test suite for service validation logic."""

    @pytest.fixture
    def mock_ffprobe_missing(self):
        """Mock subprocess.run to raise FileNotFoundError (simulating ffprobe missing)."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            yield

    def test_validate_audio_format_fallback_valid_wav(self, tmp_path, mock_ffprobe_missing):
        """Should accept valid WAV file with Python fallback."""
        path = tmp_path / "test.wav"
        # Minimal valid WAV header
        # RIFF + size + WAVE
        header = b"RIFF\x24\x00\x00\x00WAVE"
        path.write_bytes(header)

        validate_audio_format(path)  # Should not raise

    def test_validate_audio_format_fallback_invalid_wav(self, tmp_path, mock_ffprobe_missing):
        """Should reject invalid WAV file with Python fallback."""
        path = tmp_path / "test.wav"
        path.write_bytes(b"garbage")

        with pytest.raises(HTTPException) as exc:
            validate_audio_format(path)
        assert "Invalid WAV file" in exc.value.detail

    def test_validate_audio_format_fallback_valid_flac(self, tmp_path, mock_ffprobe_missing):
        """Should accept valid FLAC file with Python fallback."""
        path = tmp_path / "test.flac"
        path.write_bytes(b"fLaC") # Minimal FLAC marker

        validate_audio_format(path)

    def test_validate_audio_format_fallback_invalid_flac(self, tmp_path, mock_ffprobe_missing):
        """Should reject invalid FLAC file with Python fallback."""
        path = tmp_path / "test.flac"
        path.write_bytes(b"garbage")

        # This test is EXPECTED TO FAIL before the fix
        with pytest.raises(HTTPException) as exc:
            validate_audio_format(path)
        assert "Invalid FLAC file" in exc.value.detail

    def test_validate_audio_format_fallback_valid_ogg(self, tmp_path, mock_ffprobe_missing):
        """Should accept valid OGG file with Python fallback."""
        path = tmp_path / "test.ogg"
        path.write_bytes(b"OggS") # Minimal OGG marker

        validate_audio_format(path)

    def test_validate_audio_format_fallback_invalid_ogg(self, tmp_path, mock_ffprobe_missing):
        """Should reject invalid OGG file with Python fallback."""
        path = tmp_path / "test.ogg"
        path.write_bytes(b"garbage")

        # This test is EXPECTED TO FAIL before the fix
        with pytest.raises(HTTPException) as exc:
            validate_audio_format(path)
        assert "Invalid OGG file" in exc.value.detail

    def test_validate_audio_format_fallback_valid_wma(self, tmp_path, mock_ffprobe_missing):
        """Should accept valid WMA file with Python fallback."""
        path = tmp_path / "test.wma"
        # WMA GUID
        path.write_bytes(b"\x30\x26\xb2\x75\x8e\x66\xcf\x11\xa6\xd9\x00\xaa\x00\x62\xce\x6c")

        validate_audio_format(path)

    def test_validate_audio_format_fallback_invalid_wma(self, tmp_path, mock_ffprobe_missing):
        """Should reject invalid WMA file with Python fallback."""
        path = tmp_path / "test.wma"
        path.write_bytes(b"garbage")

        # This test is EXPECTED TO FAIL before the fix
        with pytest.raises(HTTPException) as exc:
            validate_audio_format(path)
        assert "Invalid WMA file" in exc.value.detail
