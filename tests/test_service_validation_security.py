"""Security tests for service validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from transcription.service_validation import validate_audio_format


class TestAudioValidationSecurity:
    """Security tests for audio format validation."""

    def setup_method(self):
        self.mock_run_patcher = patch("subprocess.run")
        self.mock_run = self.mock_run_patcher.start()
        # Simulate ffprobe not found
        self.mock_run.side_effect = FileNotFoundError

    def teardown_method(self):
        self.mock_run_patcher.stop()

    def test_reject_invalid_ogg_header(self, tmp_path: Path):
        """Test rejection of OGG file with invalid header."""
        path = tmp_path / "malicious.ogg"
        path.write_bytes(b"BAD_HEADER")

        with pytest.raises(HTTPException) as exc_info:
            validate_audio_format(path)

        assert exc_info.value.status_code == 400
        assert "Invalid OGG file" in str(exc_info.value.detail)

    def test_reject_invalid_flac_header(self, tmp_path: Path):
        """Test rejection of FLAC file with invalid header."""
        path = tmp_path / "malicious.flac"
        path.write_bytes(b"BAD_HEADER")

        with pytest.raises(HTTPException) as exc_info:
            validate_audio_format(path)

        assert exc_info.value.status_code == 400
        assert "Invalid FLAC file" in str(exc_info.value.detail)

    def test_reject_invalid_wma_header(self, tmp_path: Path):
        """Test rejection of WMA file with invalid header."""
        path = tmp_path / "malicious.wma"
        path.write_bytes(b"BAD_HEADER_00000")  # Need some bytes but not GUID

        with pytest.raises(HTTPException) as exc_info:
            validate_audio_format(path)

        assert exc_info.value.status_code == 400
        assert "Invalid WMA file" in str(exc_info.value.detail)

    def test_accept_valid_ogg_header(self, tmp_path: Path):
        """Test acceptance of OGG file with valid header."""
        path = tmp_path / "valid.ogg"
        path.write_bytes(b"OggS" + b"\x00" * 20)

        validate_audio_format(path)

    def test_accept_valid_flac_header(self, tmp_path: Path):
        """Test acceptance of FLAC file with valid header."""
        path = tmp_path / "valid.flac"
        path.write_bytes(b"fLaC" + b"\x00" * 20)

        validate_audio_format(path)

    def test_accept_valid_wma_header(self, tmp_path: Path):
        """Test acceptance of WMA file with valid header."""
        path = tmp_path / "valid.wma"
        wma_guid = b"\x30\x26\xB2\x75\x8E\x66\xCF\x11\xA6\xD9\x00\xAA\x00\x62\xCE\x6C"
        path.write_bytes(wma_guid + b"\x00" * 20)

        validate_audio_format(path)

    def test_path_resolution(self, tmp_path: Path):
        """Test that path is resolved before passing to subprocess (mock check)."""
        # Restore mock to verify call args
        self.mock_run.side_effect = None
        self.mock_run.return_value.returncode = 0
        self.mock_run.return_value.stdout = "format_name=wav\n10.0"

        path = tmp_path / "test.wav"
        path.touch()

        validate_audio_format(path)

        # Verify call used resolved path
        args, _ = self.mock_run.call_args
        cmd_list = args[0]
        # The last argument should be the path
        assert cmd_list[-1] == str(path.resolve())
