from unittest.mock import MagicMock, patch

from transcription.cli import main

# Since CachePaths is imported inside the function, we need to mock it where it's defined
# or mock the module it comes from.
# In `transcription/cli.py`, it does: `from .cache import CachePaths`
# So we can patch `transcription.cache.CachePaths`.


def test_cache_clear_interactive_prompt_yes(capsys):
    """Test that cache clear prompts and proceeds on 'y'."""
    with patch("shutil.rmtree") as mock_rmtree:
        with patch("builtins.input", return_value="y") as mock_input:
            with patch("sys.stdin.isatty", return_value=True):
                with patch("transcription.cache.CachePaths") as MockCachePaths:
                    mock_paths = MagicMock()
                    mock_paths.whisper_root.exists.return_value = True
                    MockCachePaths.from_env.return_value.ensure_dirs.return_value = mock_paths

                    exit_code = main(["cache", "--clear", "whisper"])

                    assert exit_code == 0
                    assert mock_input.called
                    assert mock_rmtree.called
                    captured = capsys.readouterr()
                    assert "Cleared Whisper cache" in captured.out


def test_cache_clear_interactive_prompt_no(capsys):
    """Test that cache clear prompts and aborts on 'n'."""
    with patch("shutil.rmtree") as mock_rmtree:
        with patch("builtins.input", return_value="n") as mock_input:
            with patch("sys.stdin.isatty", return_value=True):
                with patch("transcription.cache.CachePaths") as MockCachePaths:
                    mock_paths = MagicMock()
                    MockCachePaths.from_env.return_value.ensure_dirs.return_value = mock_paths

                    exit_code = main(["cache", "--clear", "whisper"])

                    assert exit_code == 0
                    assert mock_input.called
                    assert not mock_rmtree.called
                    captured = capsys.readouterr()
                    assert "Aborted" in captured.out


def test_cache_clear_force_flag(capsys):
    """Test that --force skips prompt."""
    with patch("shutil.rmtree") as mock_rmtree:
        with patch("builtins.input") as mock_input:
            # works even if interactive
            with patch("sys.stdin.isatty", return_value=True):
                with patch("transcription.cache.CachePaths") as MockCachePaths:
                    mock_paths = MagicMock()
                    mock_paths.whisper_root.exists.return_value = True
                    MockCachePaths.from_env.return_value.ensure_dirs.return_value = mock_paths

                    exit_code = main(["cache", "--clear", "whisper", "--force"])

                    assert exit_code == 0
                    assert not mock_input.called
                    assert mock_rmtree.called


def test_cache_clear_non_interactive_no_force(capsys):
    """Test that non-interactive usage without force fails."""
    with patch("shutil.rmtree") as mock_rmtree:
        with patch("builtins.input") as mock_input:
            with patch("sys.stdin.isatty", return_value=False):
                with patch("transcription.cache.CachePaths") as MockCachePaths:
                    MockCachePaths.from_env.return_value.ensure_dirs.return_value = MagicMock()

                    exit_code = main(["cache", "--clear", "whisper"])

                    assert exit_code == 1
                    assert not mock_input.called
                    assert not mock_rmtree.called
                    captured = capsys.readouterr()
                    assert "requires --force" in captured.err


def test_cache_clear_non_interactive_with_force(capsys):
    """Test that non-interactive usage with force succeeds."""
    with patch("shutil.rmtree") as mock_rmtree:
        with patch("sys.stdin.isatty", return_value=False):
            with patch("transcription.cache.CachePaths") as MockCachePaths:
                mock_paths = MagicMock()
                mock_paths.whisper_root.exists.return_value = True
                MockCachePaths.from_env.return_value.ensure_dirs.return_value = mock_paths

                exit_code = main(["cache", "--clear", "whisper", "--force"])

                assert exit_code == 0
                assert mock_rmtree.called
