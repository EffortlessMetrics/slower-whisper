from pathlib import Path
from unittest.mock import MagicMock, patch

from transcription.cli import _handle_cache_command, _handle_samples_command
from transcription.exceptions import SampleExistsError
from transcription.speaker_identity import Speaker, SpeakerRegistry, _handle_delete


class DummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@patch("sys.stdin.isatty", return_value=True)
@patch("builtins.input", side_effect=KeyboardInterrupt)
@patch("transcription.cli._get_cache_size", return_value=100)
def test_handle_cache_command_keyboard_interrupt(
    mock_get_size, mock_input, mock_isatty, tmp_path, capsys
):
    args = DummyArgs(clear="all", show=False, force=False)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with patch("transcription.cli.Path.home", return_value=tmp_path):
        result = _handle_cache_command(args)

    assert result == 0
    captured = capsys.readouterr()
    assert "Aborted." in captured.out


@patch("sys.stdin.isatty", return_value=True)
@patch("builtins.input", side_effect=KeyboardInterrupt)
def test_handle_samples_command_keyboard_interrupt(mock_input, mock_isatty, tmp_path, capsys):
    args = DummyArgs(
        generate=None,
        download="librispeech",
        force=False,
        copy_to=tmp_path,
        dataset_name=None,
        limit=1,
    )

    with patch("transcription.samples.copy_sample_to_project") as mock_copy:
        mock_copy.side_effect = SampleExistsError("test", existing_files=[Path("test.wav")])

        args = DummyArgs(samples_action="copy", force=False, root=tmp_path, dataset="librispeech")
        result = _handle_samples_command(args)

    assert result == 0
    captured = capsys.readouterr()
    assert "Aborted." in captured.out


@patch("sys.stdin.isatty", return_value=True)
@patch("builtins.input", side_effect=KeyboardInterrupt)
def test_handle_delete_keyboard_interrupt(mock_input, mock_isatty, capsys):
    from datetime import datetime

    registry = MagicMock(spec=SpeakerRegistry)
    registry.get_speaker.return_value = Speaker(
        id="SPK1",
        name="Test Speaker",
        embedding=[],
        metadata={},
        created_at=datetime.now(),
        updated_at=datetime.now(),
        sample_count=0,
    )
    args = DummyArgs(speaker_id="SPK1", force=False)

    result = _handle_delete(registry, args)

    assert result == 0
    captured = capsys.readouterr()
    assert "Aborted." in captured.out
