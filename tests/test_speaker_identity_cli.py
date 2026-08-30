from unittest.mock import MagicMock, patch

import pytest

from transcription.color_utils import Colors
from transcription.speaker_identity import Speaker, SpeakerRegistry, _handle_delete


class MockArgs:
    def __init__(self, speaker_id, force=False):
        self.speaker_id = speaker_id
        self.force = force


@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=SpeakerRegistry)
    return registry


def test_handle_delete_not_found(mock_registry, capsys):
    mock_registry.get_speaker.return_value = None
    args = MockArgs("unknown_id")

    result = _handle_delete(mock_registry, args)

    assert result == 1
    assert "Speaker not found: unknown_id" in capsys.readouterr().err


def test_handle_delete_force(mock_registry, capsys):
    mock_speaker = MagicMock(spec=Speaker)
    mock_speaker.name = "Alice"
    mock_speaker.id = "known_id"
    mock_registry.get_speaker.return_value = mock_speaker

    args = MockArgs("known_id", force=True)

    result = _handle_delete(mock_registry, args)

    assert result == 0
    assert mock_registry.delete_speaker.called
    assert "Deleted speaker 'Alice' (known_id)" in capsys.readouterr().out


def test_handle_delete_interactive_yes(mock_registry, capsys):
    mock_speaker = MagicMock(spec=Speaker)
    mock_speaker.name = "Bob"
    mock_speaker.id = "known_id"
    mock_registry.get_speaker.return_value = mock_speaker

    args = MockArgs("known_id", force=False)

    with patch("sys.stdin.isatty", return_value=True):
        with patch("builtins.input", return_value="y") as mock_input:
            result = _handle_delete(mock_registry, args)

            assert result == 0
            assert mock_registry.delete_speaker.called
            assert mock_input.called
            prompt_str = mock_input.call_args[0][0]
            assert "Delete speaker 'Bob' (known_id)?" in prompt_str
            assert Colors.red("This cannot be undone.") in prompt_str


def test_handle_delete_interactive_no(mock_registry, capsys):
    mock_speaker = MagicMock(spec=Speaker)
    mock_speaker.name = "Charlie"
    mock_speaker.id = "known_id"
    mock_registry.get_speaker.return_value = mock_speaker

    args = MockArgs("known_id", force=False)

    with patch("sys.stdin.isatty", return_value=True):
        with patch("builtins.input", return_value="n"):
            result = _handle_delete(mock_registry, args)

            assert result == 0
            assert not mock_registry.delete_speaker.called
            assert "Aborted." in capsys.readouterr().out


def test_handle_delete_non_interactive_no_force(mock_registry, capsys):
    mock_speaker = MagicMock(spec=Speaker)
    mock_registry.get_speaker.return_value = mock_speaker

    args = MockArgs("known_id", force=False)

    with patch("sys.stdin.isatty", return_value=False):
        result = _handle_delete(mock_registry, args)

        assert result == 1
        assert not mock_registry.delete_speaker.called
        assert "Error: Delete requires --force in non-interactive mode." in capsys.readouterr().err
