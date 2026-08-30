from unittest.mock import patch

import pytest

from transcription.cli import main


class TestSamplesGenerateUX:
    """Test UX interactions for samples generate command."""

    @pytest.fixture
    def mock_generate(self):
        """Mock generate_synthetic_2speaker to avoid dependency issues."""
        with patch("transcription.samples.generate_synthetic_2speaker") as mock:
            yield mock

    def test_generate_default(self, mock_generate, capsys):
        """Default generation should suggest default transcribe command."""
        exit_code = main(["samples", "generate"])
        assert exit_code == 0

        captured = capsys.readouterr()
        # Expect implicit root (CWD)
        assert "uv run slower-whisper transcribe --enable-diarization" in captured.out
        assert "--root" not in captured.out

    def test_generate_custom_raw_audio(self, mock_generate, capsys, tmp_path):
        """Generation into raw_audio dir should suggest parent as root."""
        output_dir = tmp_path / "project" / "raw_audio"

        exit_code = main(["samples", "generate", "--output", str(output_dir)])
        assert exit_code == 0

        captured = capsys.readouterr()
        # Expect explicit root
        project_root = output_dir.parent
        expected_cmd = f"uv run slower-whisper transcribe --root {project_root}"
        assert expected_cmd in captured.out

    def test_generate_custom_invalid_dir(self, mock_generate, capsys, tmp_path):
        """Generation into non-raw_audio dir should warn user."""
        output_dir = tmp_path / "some_folder"

        exit_code = main(["samples", "generate", "--output", str(output_dir)])
        assert exit_code == 0

        captured = capsys.readouterr()
        # Expect warning about raw_audio requirement
        assert "looks for files in a 'raw_audio' directory" in captured.out
        # Should suggest moving or renaming
        assert "move the file or rename" in captured.out
