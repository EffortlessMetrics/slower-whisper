"""Tests for UX interactions in speaker identity CLI."""

from unittest.mock import MagicMock, patch

import pytest

from transcription.speaker_identity import Speaker, SpeakerRegistry


@pytest.fixture
def mock_registry():
    """Mock SpeakerRegistry for testing."""
    mock = MagicMock(spec=SpeakerRegistry)
    # Give the mock a realistic speaker to return
    # Assuming Speaker has name and id attributes
    speaker = MagicMock(spec=Speaker)
    speaker.id = "test_id"
    speaker.name = "Alice"
    mock.get_speaker.return_value = speaker
    return mock


class TestSpeakerIdentityCliUX:
    """Test UX and interactions for speaker identity commands."""

    def test_delete_interactive_abort(self, mock_registry, capsys):
        """Interactive delete should prompt and abort if user says no."""
        with (
            patch("transcription.speaker_identity.SpeakerRegistry", return_value=mock_registry),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="n"),
        ):
            from argparse import Namespace

            from transcription.speaker_identity import _handle_delete

            args = Namespace(speaker_id="test_id", force=False)
            exit_code = _handle_delete(mock_registry, args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Aborted." in captured.out
        mock_registry.delete_speaker.assert_not_called()

    def test_delete_interactive_confirm(self, mock_registry, capsys):
        """Interactive delete should prompt and delete if user says yes."""
        with (
            patch("transcription.speaker_identity.SpeakerRegistry", return_value=mock_registry),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
        ):
            from argparse import Namespace

            from transcription.speaker_identity import _handle_delete

            args = Namespace(speaker_id="test_id", force=False)
            exit_code = _handle_delete(mock_registry, args)

        assert exit_code == 0
        mock_registry.delete_speaker.assert_called_once_with("test_id")
