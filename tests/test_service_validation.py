from pathlib import Path

import pytest
from fastapi import HTTPException

from transcription.service_validation import validate_audio_format


def test_validate_audio_format_invalid_path():
    """Ensure validate_audio_format correctly rejects unsafe paths and raises HTTPException 400."""
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_format(Path("invalid|path.wav"))

    assert exc_info.value.status_code == 400
    assert "Invalid audio file path." in str(exc_info.value.detail)
