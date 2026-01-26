import argparse
import os
from unittest.mock import patch
from transcription.cli import _handle_cache_command

def test_cache_clear_confirmation_is_colored():
    """Verify that the cache clear confirmation prompt uses red text for the warning."""
    # Mock args
    args = argparse.Namespace(
        show=False,
        clear="all",
        force=False
    )

    # Mock sys.stdin.isatty to return True
    with patch("sys.stdin.isatty", return_value=True), \
         patch("builtins.input", return_value="n") as mock_input, \
         patch("transcription.cli._get_cache_size", return_value=1024), \
         patch.dict("os.environ", {"FORCE_COLOR": "1"}): # Force colors

        _handle_cache_command(args)

    # Check the prompt
    assert mock_input.called, "input() was not called"

    call_args = mock_input.call_args
    prompt = call_args[0][0]

    # Check if "This cannot be undone." is colored red (ANSI code \033[31m)
    expected_red_start = "\033[31m"
    warning_text = "This cannot be undone."

    assert expected_red_start + warning_text in prompt, \
        f"Prompt warning should be red. Got: {repr(prompt)}"
