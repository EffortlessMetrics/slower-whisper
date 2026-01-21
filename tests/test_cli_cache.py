"""Tests for cache command confirmation prompt and --force flag."""

from unittest.mock import MagicMock, patch

import pytest

from transcription.cli import main


@pytest.fixture
def mock_cache_paths():
    """Mock CachePaths and related dependencies for cache command tests."""
    with (
        patch("transcription.cache.CachePaths") as mock_cls,
        patch("transcription.samples.get_samples_cache_dir") as mock_samples_dir,
    ):
        mock_paths = MagicMock()
        mock_paths.whisper_root.exists.return_value = True
        mock_paths.emotion_root.exists.return_value = False
        mock_paths.diarization_root.exists.return_value = False
        mock_paths.hf_home.exists.return_value = False
        mock_paths.torch_home.exists.return_value = False
        mock_cls.from_env.return_value.ensure_dirs.return_value = mock_paths
        mock_samples_dir.return_value = MagicMock(exists=MagicMock(return_value=False))
        yield mock_paths


class TestCacheClearConfirmation:
    """Tests for interactive confirmation on cache --clear."""

    def test_interactive_prompt_yes_proceeds(self, mock_cache_paths, capsys):
        """Answering 'y' to prompt clears the cache."""
        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("builtins.input", return_value="y") as mock_input,
            patch("sys.stdin.isatty", return_value=True),
        ):
            exit_code = main(["cache", "--clear", "whisper"])

            assert exit_code == 0
            assert mock_input.called
            assert mock_rmtree.called
            captured = capsys.readouterr()
            assert "Cleared Whisper cache" in captured.out

    def test_interactive_prompt_no_aborts(self, mock_cache_paths, capsys):
        """Answering 'n' to prompt aborts without clearing."""
        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("builtins.input", return_value="n") as mock_input,
            patch("sys.stdin.isatty", return_value=True),
        ):
            exit_code = main(["cache", "--clear", "whisper"])

            assert exit_code == 0
            assert mock_input.called
            assert not mock_rmtree.called
            captured = capsys.readouterr()
            assert "Aborted" in captured.out

    def test_force_flag_skips_prompt(self, mock_cache_paths, capsys):
        """--force skips the confirmation prompt entirely."""
        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("builtins.input") as mock_input,
            patch("sys.stdin.isatty", return_value=True),
        ):
            exit_code = main(["cache", "--clear", "whisper", "--force"])

            assert exit_code == 0
            assert not mock_input.called
            assert mock_rmtree.called

    def test_short_force_flag_works(self, mock_cache_paths, capsys):
        """-f short flag works same as --force."""
        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("builtins.input") as mock_input,
            patch("sys.stdin.isatty", return_value=True),
        ):
            exit_code = main(["cache", "--clear", "whisper", "-f"])

            assert exit_code == 0
            assert not mock_input.called
            assert mock_rmtree.called

    def test_yes_flag_alias_works(self, mock_cache_paths, capsys):
        """-y alias works same as --force."""
        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("builtins.input") as mock_input,
            patch("sys.stdin.isatty", return_value=True),
        ):
            exit_code = main(["cache", "--clear", "whisper", "-y"])

            assert exit_code == 0
            assert not mock_input.called
            assert mock_rmtree.called


class TestCacheClearNonInteractive:
    """Tests for non-interactive (piped/scripted) cache --clear behavior."""

    def test_non_interactive_without_force_fails(self, mock_cache_paths, capsys):
        """Non-interactive mode without --force returns error."""
        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("builtins.input") as mock_input,
            patch("sys.stdin.isatty", return_value=False),
        ):
            exit_code = main(["cache", "--clear", "whisper"])

            assert exit_code == 1
            assert not mock_input.called
            assert not mock_rmtree.called
            captured = capsys.readouterr()
            assert "requires --force" in captured.err

    def test_non_interactive_with_force_succeeds(self, mock_cache_paths, capsys):
        """Non-interactive mode with --force proceeds normally."""
        with (
            patch("shutil.rmtree") as mock_rmtree,
            patch("sys.stdin.isatty", return_value=False),
        ):
            exit_code = main(["cache", "--clear", "whisper", "--force"])

            assert exit_code == 0
            assert mock_rmtree.called
