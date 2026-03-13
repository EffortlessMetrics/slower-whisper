import argparse
import builtins
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from transcription.speaker_identity import _handle_delete, Speaker

def test_speaker_delete_keyboard_interrupt() -> None:
    """Speaker delete aborts cleanly on KeyboardInterrupt."""
    args = argparse.Namespace(speaker_id="mock-id", force=False)

    mock_registry = MagicMock()
    mock_speaker = Speaker(id="mock-id", name="Alice", embedding=None)
    mock_registry.get_speaker.return_value = mock_speaker

    with patch("sys.stdin.isatty", return_value=True), \
         patch("builtins.input", side_effect=KeyboardInterrupt) as mock_input:

        assert _handle_delete(mock_registry, args) == 130
        mock_input.assert_called_once()
        mock_registry.delete_speaker.assert_not_called()
