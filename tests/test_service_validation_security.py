import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from transcription.service_validation import validate_audio_format


class TestServiceValidationSecurity(unittest.TestCase):
    def test_unsafe_path_rejected(self):
        """Test that paths with unsafe characters are rejected."""
        unsafe_paths = ["file|unsafe.wav", "$(cmd).wav", ";rm -rf.wav", "`touch pwn`.wav"]

        for path_str in unsafe_paths:
            unsafe_path = Path(path_str)
            try:
                validate_audio_format(unsafe_path)
                self.fail(f"VULNERABILITY: Unsafe path '{path_str}' passed validation!")
            except HTTPException as e:
                self.assertEqual(e.status_code, 400)
                self.assertEqual(e.detail, "Invalid audio file path.")
            except Exception as e:
                self.fail(f"Unexpected exception for path '{path_str}': {type(e)} {e}")

    def test_flac_magic_bytes_check(self):
        """Test that FLAC files without proper magic bytes are rejected when ffprobe is missing."""
        # Create a fake flac file with invalid content
        fake_flac = Path("fake.flac")
        with open(fake_flac, "wb") as f:
            f.write(b"NOT A FLAC FILE")

        try:
            # Mock subprocess.run to raise FileNotFoundError (simulating missing ffprobe)
            with patch("subprocess.run", side_effect=FileNotFoundError):
                validate_audio_format(fake_flac)
                self.fail("VULNERABILITY: Invalid FLAC file passed validation!")
        except HTTPException as e:
            self.assertEqual(e.status_code, 400)
            self.assertIn("Invalid FLAC file", e.detail)
        finally:
            if fake_flac.exists():
                fake_flac.unlink()

    def test_ogg_magic_bytes_check(self):
        """Test that OGG files without proper magic bytes are rejected when ffprobe is missing."""
        fake_ogg = Path("fake.ogg")
        with open(fake_ogg, "wb") as f:
            f.write(b"NOT AN OGG FILE")

        try:
            with patch("subprocess.run", side_effect=FileNotFoundError):
                validate_audio_format(fake_ogg)
                self.fail("VULNERABILITY: Invalid OGG file passed validation!")
        except HTTPException as e:
            self.assertEqual(e.status_code, 400)
            self.assertIn("Invalid OGG file", e.detail)
        finally:
            if fake_ogg.exists():
                fake_ogg.unlink()

    def test_wma_magic_bytes_check(self):
        """Test that WMA files without proper GUID are rejected when ffprobe is missing."""
        fake_wma = Path("fake.wma")
        with open(fake_wma, "wb") as f:
            f.write(b"NOT A WMA FILE" * 2)  # Write enough bytes but wrong content

        try:
            with patch("subprocess.run", side_effect=FileNotFoundError):
                validate_audio_format(fake_wma)
                self.fail("VULNERABILITY: Invalid WMA file passed validation!")
        except HTTPException as e:
            self.assertEqual(e.status_code, 400)
            self.assertIn("Invalid WMA file", e.detail)
        finally:
            if fake_wma.exists():
                fake_wma.unlink()


if __name__ == "__main__":
    unittest.main()
