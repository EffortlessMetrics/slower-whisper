
from pathlib import Path
from unittest.mock import MagicMock, patch

from transcription.service_validation import validate_audio_format


def test_validate_audio_format_uses_absolute_path():
    """Test that validate_audio_format uses absolute path for ffprobe to prevent option injection."""
    with patch("subprocess.run") as mock_run:
        # Mock successful execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "wav\n10.0"
        mock_run.return_value = mock_result

        path = Path("test.wav")
        validate_audio_format(path)

        # Check arguments
        args = mock_run.call_args[0][0]

        # Verify no '--'
        assert "--" not in args, "'--' should not be used as it breaks ffprobe"

        # Verify the filename argument is absolute
        filename_arg = args[-1]
        print(f"Filename arg: {filename_arg}")

        assert Path(filename_arg).is_absolute(), "Path passed to ffprobe should be absolute"
        assert filename_arg == str(path.resolve())
