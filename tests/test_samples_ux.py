"""
UX and interaction tests for sample operations.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from pathlib import Path

from transcription.cli import main
from transcription.exceptions import SampleExistsError


class TestSamplesCopyUX:
    """Test UX interactions for samples copy command."""

    @pytest.fixture
    def mock_copy(self):
        """Mock copy_sample_to_project to control behavior."""
        with patch("transcription.samples.copy_sample_to_project") as mock:
            yield mock

    def test_copy_success_no_conflict(self, mock_copy, tmp_path):
        """Normal copy without conflicts should proceed."""
        mock_copy.return_value = [Path("file1.wav")]

        exit_code = main(["samples", "copy", "mini_diarization", "--root", str(tmp_path)])

        assert exit_code == 0
        # verify called with overwrite=False (default from args.force=False)
        mock_copy.assert_called_once()
        assert mock_copy.call_args.kwargs["overwrite"] is False

    def test_copy_conflict_non_interactive_fails(self, mock_copy, tmp_path):
        """Non-interactive copy with conflicts should fail."""
        # Simulate conflict
        mock_copy.side_effect = SampleExistsError("Conflict", [Path("file1.wav")])

        # Patch isatty to False
        with patch("sys.stdin.isatty", return_value=False):
            exit_code = main(["samples", "copy", "mini_diarization", "--root", str(tmp_path)])

        assert exit_code == 1

    def test_copy_conflict_interactive_abort(self, mock_copy, tmp_path, capsys):
        """Interactive copy with conflicts should prompt and abort if user says no."""
        mock_copy.side_effect = SampleExistsError("Conflict", [Path("file1.wav")])

        # Patch isatty to True and input to 'n'
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value="n"):
                exit_code = main(["samples", "copy", "mini_diarization", "--root", str(tmp_path)])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Aborted." in captured.out
        # Should verify it wasn't called a second time
        assert mock_copy.call_count == 1

    def test_copy_conflict_interactive_confirm(self, mock_copy, tmp_path):
        """Interactive copy with conflicts should prompt and retry if user says yes."""
        # First call raises error, second call succeeds
        mock_copy.side_effect = [
            SampleExistsError("Conflict", [Path("file1.wav")]),
            [Path("file1.wav")]
        ]

        # Patch isatty to True and input to 'y'
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", return_value="y"):
                exit_code = main(["samples", "copy", "mini_diarization", "--root", str(tmp_path)])

        assert exit_code == 0
        assert mock_copy.call_count == 2
        # First call overwrite=False
        assert mock_copy.call_args_list[0].kwargs["overwrite"] is False
        # Second call overwrite=True
        assert mock_copy.call_args_list[1].kwargs["overwrite"] is True

    def test_copy_force_skips_check(self, mock_copy, tmp_path):
        """--force should pass overwrite=True immediately."""
        mock_copy.return_value = [Path("file1.wav")]

        exit_code = main(["samples", "copy", "mini_diarization", "--root", str(tmp_path), "--force"])

        assert exit_code == 0
        mock_copy.assert_called_once()
        assert mock_copy.call_args.kwargs["overwrite"] is True
