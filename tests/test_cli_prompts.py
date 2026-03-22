from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transcription.cli import main
from transcription.speaker_identity import Speaker, SpeakerRegistry, _handle_delete


class MockArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=SpeakerRegistry)
    speaker = Speaker(
        id="test_id",
        name="Test Speaker",
        embedding=np.array([1.0, 2.0]),
        metadata={},
        created_at="",
        updated_at="",
    )
    registry.get_speaker.return_value = speaker
    return registry


class TestCliPrompts:
    def test_cache_clear_keyboard_interrupt(self, tmp_path: Path):
        with (
            patch("sys.argv", ["slower-whisper", "cache", "--clear", "all"]),
            patch("builtins.input", side_effect=KeyboardInterrupt()),
            patch("sys.stdin.isatty", return_value=True),
            patch("transcription.cli._get_cache_size", return_value=1024),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = main()
            assert result == 130

    def test_samples_copy_overwrite_keyboard_interrupt(self, tmp_path: Path):
        from transcription.exceptions import SampleExistsError

        def mock_copy(*args, **kwargs):
            if not kwargs.get("overwrite", False):
                raise SampleExistsError("Files exist", [Path("test.wav")])
            return []

        with (
            patch(
                "sys.argv", ["slower-whisper", "samples", "copy", "smoke", "--root", str(tmp_path)]
            ),
            patch("transcription.samples.copy_sample_to_project", side_effect=mock_copy),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=KeyboardInterrupt()),
        ):
            result = main()
            assert result == 130

    def test_samples_copy_overwrite_yes(self, tmp_path: Path):
        from transcription.exceptions import SampleExistsError

        def mock_copy(*args, **kwargs):
            if not kwargs.get("overwrite", False):
                raise SampleExistsError("Files exist", [Path("test.wav")])
            return []

        with (
            patch(
                "sys.argv", ["slower-whisper", "samples", "copy", "smoke", "--root", str(tmp_path)]
            ),
            patch("transcription.samples.copy_sample_to_project", side_effect=mock_copy),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
        ):
            result = main()
            assert result == 0

    def test_samples_copy_overwrite_no(self, tmp_path: Path):
        from transcription.exceptions import SampleExistsError

        def mock_copy(*args, **kwargs):
            if not kwargs.get("overwrite", False):
                raise SampleExistsError("Files exist", [Path("test.wav")])
            return []

        with (
            patch(
                "sys.argv", ["slower-whisper", "samples", "copy", "smoke", "--root", str(tmp_path)]
            ),
            patch("transcription.samples.copy_sample_to_project", side_effect=mock_copy),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="n"),
        ):
            result = main()
            assert result == 0

    def test_cache_clear_yes(self, tmp_path: Path):
        with (
            patch("sys.argv", ["slower-whisper", "cache", "--clear", "all"]),
            patch("builtins.input", return_value="y"),
            patch("sys.stdin.isatty", return_value=True),
            patch("transcription.cli._get_cache_size", return_value=1024),
            patch("pathlib.Path.exists", return_value=True),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            result = main()
            assert result == 0
            mock_rmtree.assert_called()

    def test_cache_clear_no(self, tmp_path: Path):
        with (
            patch("sys.argv", ["slower-whisper", "cache", "--clear", "all"]),
            patch("builtins.input", return_value="n"),
            patch("sys.stdin.isatty", return_value=True),
            patch("transcription.cli._get_cache_size", return_value=1024),
            patch("pathlib.Path.exists", return_value=True),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            result = main()
            assert result == 0
            mock_rmtree.assert_not_called()

    def test_samples_copy_overwrite_keyboard_interrupt_retry(self, tmp_path: Path):
        from transcription.exceptions import SampleExistsError

        copy_calls = 0

        def mock_copy(*args, **kwargs):
            nonlocal copy_calls
            copy_calls += 1
            if copy_calls == 1:
                raise SampleExistsError("Files exist", [Path("test.wav")])
            return []

        with (
            patch(
                "sys.argv", ["slower-whisper", "samples", "copy", "smoke", "--root", str(tmp_path)]
            ),
            patch("transcription.samples.copy_sample_to_project", side_effect=mock_copy),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=KeyboardInterrupt()),
        ):
            result = main()
            assert result == 130

    def test_samples_copy_overwrite_non_interactive(self, tmp_path: Path):
        from transcription.exceptions import SampleExistsError

        def mock_copy(*args, **kwargs):
            if not kwargs.get("overwrite", False):
                raise SampleExistsError("Files exist", [Path("test.wav")])
            return []

        with (
            patch(
                "sys.argv", ["slower-whisper", "samples", "copy", "smoke", "--root", str(tmp_path)]
            ),
            patch("transcription.samples.copy_sample_to_project", side_effect=mock_copy),
            patch("sys.stdin.isatty", return_value=False),
        ):
            result = main()
            assert result == 1

    def test_cache_clear_non_interactive(self, tmp_path: Path):
        with (
            patch("sys.argv", ["slower-whisper", "cache", "--clear", "all"]),
            patch("sys.stdin.isatty", return_value=False),
            patch("transcription.cli._get_cache_size", return_value=1024),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = main()
            assert result == 1

    def test_samples_copy_overwrite_invalid_then_yes(self, tmp_path: Path):
        from transcription.exceptions import SampleExistsError

        copy_calls = 0

        def mock_copy(*args, **kwargs):
            nonlocal copy_calls
            copy_calls += 1
            if copy_calls == 1:
                raise SampleExistsError("Files exist", [Path("test.wav")])
            return []

        with (
            patch(
                "sys.argv", ["slower-whisper", "samples", "copy", "smoke", "--root", str(tmp_path)]
            ),
            patch("transcription.samples.copy_sample_to_project", side_effect=mock_copy),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=["invalid", "y"]),
        ):
            result = main()
            assert result == 0

    def test_handle_delete_keyboard_interrupt(self, mock_registry):
        args = MockArgs(speaker_id="test_id", force=False)
        with patch("builtins.input", side_effect=KeyboardInterrupt()):
            with patch("sys.stdin.isatty", return_value=True):
                result = _handle_delete(mock_registry, args)
                assert result == 130

    def test_handle_delete_yes(self, mock_registry):
        args = MockArgs(speaker_id="test_id", force=False)
        with patch("builtins.input", return_value="y"):
            with patch("sys.stdin.isatty", return_value=True):
                result = _handle_delete(mock_registry, args)
                assert result == 0
                mock_registry.delete_speaker.assert_called_once_with("test_id")

    def test_handle_delete_no(self, mock_registry):
        args = MockArgs(speaker_id="test_id", force=False)
        with patch("builtins.input", return_value="n"):
            with patch("sys.stdin.isatty", return_value=True):
                result = _handle_delete(mock_registry, args)
                assert result == 0
                mock_registry.delete_speaker.assert_not_called()

    def test_handle_delete_not_found(self):
        registry = MagicMock(spec=SpeakerRegistry)
        registry.get_speaker.return_value = None
        args = MockArgs(speaker_id="test_id", force=False)
        result = _handle_delete(registry, args)
        assert result == 1

    def test_handle_delete_non_interactive(self, mock_registry):
        args = MockArgs(speaker_id="test_id", force=False)
        with patch("sys.stdin.isatty", return_value=False):
            result = _handle_delete(mock_registry, args)
            assert result == 1
            mock_registry.delete_speaker.assert_not_called()
