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

def test_validate_audio_format_shell_injection():
    unsafe_path = Path("file; rm -rf /")

    with pytest.raises(HTTPException) as exc_info:
        validate_audio_format(unsafe_path)

    assert exc_info.value.status_code == 400

def test_validate_audio_format_safe_file(tmp_path):
    safe_path = tmp_path / "safe_file.wav"
    # This might raise something else but it won't raise 400 from ValueError
    try:
        validate_audio_format(safe_path)
    except HTTPException as e:
        # We expect either 400 because the file is missing/empty, or success if we patched it.
        # Let's just make sure it's not the ValueError HTTP 400
        assert "Invalid audio file path" not in e.detail

def test_validate_audio_format_subprocess_timeout(tmp_path, mocker):
    safe_path = tmp_path / "timeout_file.wav"
    safe_path.touch()

    # Mock subprocess.run to raise TimeoutExpired
    import subprocess
    mocker.patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=10)
    )

    with pytest.raises(HTTPException) as exc_info:
        validate_audio_format(safe_path)

    assert exc_info.value.status_code == 400
    assert "timeout" in exc_info.value.detail.lower()

def test_validate_audio_format_invalid_ffprobe_output(tmp_path, mocker):
    safe_path = tmp_path / "bad_output.wav"
    safe_path.touch()

    # Mock subprocess.run to return non-zero exit code
    import subprocess
    mock_result = subprocess.CompletedProcess(args=["ffprobe"], returncode=1, stdout="")
    mocker.patch("subprocess.run", return_value=mock_result)

    with pytest.raises(HTTPException) as exc_info:
        validate_audio_format(safe_path)

    assert exc_info.value.status_code == 400
    assert "invalid audio file format" in exc_info.value.detail.lower()
