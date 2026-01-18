"""Tests for transcription/color_utils.py"""

import os
from unittest.mock import MagicMock, patch

from transcription.color_utils import Colors


def test_colors_constants():
    assert Colors.GREEN == "\033[0;32m"
    assert Colors.NC == "\033[0m"


def test_colors_colorize_enabled():
    """Test colorize when enabled."""
    with patch.object(Colors, "_is_enabled", return_value=True):
        assert Colors.colorize("text", Colors.RED) == f"{Colors.RED}text{Colors.NC}"
        assert Colors.success("text") == f"{Colors.GREEN}text{Colors.NC}"
        assert Colors.failure("text") == f"{Colors.RED}text{Colors.NC}"
        assert Colors.warning("text") == f"{Colors.YELLOW}text{Colors.NC}"
        assert Colors.info("text") == f"{Colors.BLUE}text{Colors.NC}"
        assert Colors.bold("text") == f"{Colors.BOLD}text{Colors.NC}"


def test_colors_colorize_disabled():
    """Test colorize when disabled."""
    with patch.object(Colors, "_is_enabled", return_value=False):
        assert Colors.colorize("text", Colors.RED) == "text"
        assert Colors.success("text") == "text"


def test_is_enabled_no_color_env():
    """Test _is_enabled respects NO_COLOR."""
    with patch.dict(os.environ, {"NO_COLOR": "1"}):
        assert Colors._is_enabled() is False


def test_is_enabled_tty():
    """Test _is_enabled checks isatty."""
    with patch.dict(os.environ, {}, clear=True):  # Ensure NO_COLOR is cleared
        mock_stream = MagicMock()

        # TTY
        mock_stream.isatty.return_value = True
        assert Colors._is_enabled(mock_stream) is True

        # Not TTY
        mock_stream.isatty.return_value = False
        assert Colors._is_enabled(mock_stream) is False

        # Missing isatty
        del mock_stream.isatty
        assert Colors._is_enabled(mock_stream) is False
