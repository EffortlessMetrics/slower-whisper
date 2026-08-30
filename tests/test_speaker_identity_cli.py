import argparse
from unittest.mock import MagicMock, patch

from transcription.speaker_identity import _handle_delete


def test_handle_speakers_delete_interrupt(capsys):
    """Ctrl+C at delete prompt gracefully aborts and returns 130."""
    args = argparse.Namespace(speaker_id="speaker_1", force=False)

    mock_registry = MagicMock()
    mock_speaker = MagicMock()
    mock_speaker.name = "Alice"
    mock_speaker.id = "speaker_1"
    mock_registry.get_speaker.return_value = mock_speaker

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("builtins.input", side_effect=KeyboardInterrupt()) as mock_input,
    ):
        exit_code = _handle_delete(mock_registry, args)

        assert exit_code == 130
        assert mock_input.called
        assert not mock_registry.delete_speaker.called

        captured = capsys.readouterr()
        assert "Aborted." in captured.out


def test_handle_speakers_delete_confirm_yes(capsys):
    """Confirming yes deletes the speaker and returns 0."""
    args = argparse.Namespace(speaker_id="speaker_1", force=False)
    mock_registry = MagicMock()
    mock_speaker = MagicMock()
    mock_speaker.name = "Alice"
    mock_speaker.id = "speaker_1"
    mock_registry.get_speaker.return_value = mock_speaker

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("builtins.input", return_value="y") as mock_input,
    ):
        exit_code = _handle_delete(mock_registry, args)

        assert exit_code == 0
        assert mock_input.called
        assert mock_registry.delete_speaker.called


def test_handle_speakers_delete_confirm_no(capsys):
    """Confirming no aborts and returns 0."""
    args = argparse.Namespace(speaker_id="speaker_1", force=False)
    mock_registry = MagicMock()
    mock_speaker = MagicMock()
    mock_speaker.name = "Alice"
    mock_speaker.id = "speaker_1"
    mock_registry.get_speaker.return_value = mock_speaker

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("builtins.input", return_value="n") as mock_input,
    ):
        exit_code = _handle_delete(mock_registry, args)

        assert exit_code == 0
        assert mock_input.called
        assert not mock_registry.delete_speaker.called
