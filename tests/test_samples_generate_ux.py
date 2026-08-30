"""
UX and interaction tests for samples generate command.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from transcription.cli import main


class TestSamplesGenerateUX:
    """Test UX interactions for samples generate command."""

    @pytest.fixture
    def mock_generate(self):
        """Mock generate_synthetic_2speaker to control behavior."""
        with patch("transcription.samples.generate_synthetic_2speaker") as mock:
            yield mock

    def test_generate_success_no_conflict(self, mock_generate, tmp_path):
        """Normal generate without conflicts should proceed."""
        output_dir = tmp_path / "raw_audio"
        # Ensure file does not exist

        exit_code = main(["samples", "generate", "--output", str(output_dir)])

        assert exit_code == 0
        mock_generate.assert_called_once()
        # Verify the path passed to generate
        args, _ = mock_generate.call_args
        assert args[0] == output_dir / "synthetic_2speaker.wav"

    def test_generate_conflict_non_interactive_fails(self, mock_generate, tmp_path):
        """Non-interactive generate with conflicts should fail."""
        output_dir = tmp_path / "raw_audio"
        output_dir.mkdir(parents=True)
        (output_dir / "synthetic_2speaker.wav").touch()

        # Patch isatty to False
        with patch("sys.stdin.isatty", return_value=False):
            # Also patch sys.stderr to capture output if needed, but return code is enough
            exit_code = main(["samples", "generate", "--output", str(output_dir)])

        assert exit_code == 1
        # Should not have called generate because of conflict
        mock_generate.assert_not_called()

    def test_generate_conflict_interactive_abort(self, mock_generate, tmp_path, capsys):
        """Interactive generate with conflicts should prompt and abort if user says no."""
        output_dir = tmp_path / "raw_audio"
        output_dir.mkdir(parents=True)
        (output_dir / "synthetic_2speaker.wav").touch()

        # Patch isatty to True and input to 'n'
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value="n"):
                exit_code = main(["samples", "generate", "--output", str(output_dir)])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Aborted." in captured.out
        mock_generate.assert_not_called()

    def test_generate_conflict_interactive_confirm(self, mock_generate, tmp_path):
        """Interactive generate with conflicts should prompt and proceed if user says yes."""
        output_dir = tmp_path / "raw_audio"
        output_dir.mkdir(parents=True)
        (output_dir / "synthetic_2speaker.wav").touch()

        # Patch isatty to True and input to 'y'
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value="y"):
                exit_code = main(["samples", "generate", "--output", str(output_dir)])

        assert exit_code == 0
        mock_generate.assert_called_once()

    def test_generate_force_skips_check(self, mock_generate, tmp_path):
        """--force should skip conflict check."""
        output_dir = tmp_path / "raw_audio"
        output_dir.mkdir(parents=True)
        (output_dir / "synthetic_2speaker.wav").touch()

        exit_code = main(["samples", "generate", "--output", str(output_dir), "--force"])

        assert exit_code == 0
        mock_generate.assert_called_once()
