from unittest.mock import MagicMock, patch

from transcription.color_utils import Colors
from transcription.speaker_identity import Speaker, SpeakerRegistry, _handle_delete


class MockArgs:
    def __init__(self, speaker_id, force=False):
        self.speaker_id = speaker_id
        self.force = force


def test_handle_delete_speaker_not_found():
    registry = MagicMock(spec=SpeakerRegistry)
    registry.get_speaker.return_value = None
    args = MockArgs("spk-1")

    assert _handle_delete(registry, args) == 1


def test_handle_delete_non_interactive_no_force():
    registry = MagicMock(spec=SpeakerRegistry)
    speaker = MagicMock(spec=Speaker)
    speaker.id = "spk-1"
    speaker.name = "John"
    registry.get_speaker.return_value = speaker

    args = MockArgs("spk-1", force=False)

    with patch("sys.stdin.isatty", return_value=False):
        assert _handle_delete(registry, args) == 1


@patch("builtins.input")
def test_handle_delete_interactive_abort(mock_input):
    mock_input.return_value = "n"
    registry = MagicMock(spec=SpeakerRegistry)
    speaker = MagicMock(spec=Speaker)
    speaker.id = "spk-1"
    speaker.name = "John"
    registry.get_speaker.return_value = speaker

    args = MockArgs("spk-1", force=False)

    with patch("sys.stdin.isatty", return_value=True):
        assert _handle_delete(registry, args) == 0
        registry.delete_speaker.assert_not_called()

        warning = Colors.red("This cannot be undone.")
        mock_input.assert_called_once_with(f"Delete speaker 'John' (spk-1)? {warning} [y/N] ")


@patch("builtins.input")
def test_handle_delete_interactive_confirm(mock_input):
    mock_input.return_value = "y"
    registry = MagicMock(spec=SpeakerRegistry)
    speaker = MagicMock(spec=Speaker)
    speaker.id = "spk-1"
    speaker.name = "John"
    registry.get_speaker.return_value = speaker

    args = MockArgs("spk-1", force=False)

    with patch("sys.stdin.isatty", return_value=True):
        assert _handle_delete(registry, args) == 0
        registry.delete_speaker.assert_called_once_with("spk-1")

        warning = Colors.red("This cannot be undone.")
        mock_input.assert_called_once_with(f"Delete speaker 'John' (spk-1)? {warning} [y/N] ")
