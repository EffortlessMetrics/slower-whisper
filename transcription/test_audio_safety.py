import pytest
from pathlib import Path
from fastapi import HTTPException
from transcription.service_validation import validate_audio_format

def test_validate_audio_format_option_injection():
    # Attempt to inject an option by prefixing with '-'
    unsafe_path = Path("-i")

    with pytest.raises(HTTPException) as exc_info:
        validate_audio_format(unsafe_path)

    assert exc_info.value.status_code == 400
