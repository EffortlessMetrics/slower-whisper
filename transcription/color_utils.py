"""
ANSI color codes for terminal output.

Provides simple color constants and helper functions for CLI output styling.
Respects NO_COLOR environment variable and checks for TTY.
"""

import os
import sys
from typing import TextIO


class Colors:
    """ANSI color codes and helpers."""

    # Colors
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    MAGENTA = "\033[0;35m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Reset
    NC = "\033[0m"  # No Color

    @classmethod
    def _is_enabled(cls, stream: TextIO = sys.stdout) -> bool:
        """Check if colors should be enabled for the given stream."""
        # Respect NO_COLOR standard (https://no-color.org/)
        if os.environ.get("NO_COLOR"):
            return False

        # Check if stream is a TTY
        # For non-standard streams (like StringIO in tests), isatty might not exist
        if not hasattr(stream, "isatty") or not stream.isatty():
            return False

        # Windows handling (if colorama not present/initialized) - overly simplified check
        if sys.platform == "win32":
            return False

        return True

    @classmethod
    def colorize(cls, text: str, color_code: str, stream: TextIO = sys.stdout) -> str:
        """Wrap text in color code if colors are enabled."""
        if cls._is_enabled(stream):
            return f"{color_code}{text}{cls.NC}"
        return text

    @classmethod
    def success(cls, text: str) -> str:
        return cls.colorize(text, cls.GREEN)

    @classmethod
    def failure(cls, text: str) -> str:
        return cls.colorize(text, cls.RED)

    @classmethod
    def warning(cls, text: str) -> str:
        return cls.colorize(text, cls.YELLOW)

    @classmethod
    def info(cls, text: str) -> str:
        return cls.colorize(text, cls.BLUE)

    @classmethod
    def bold(cls, text: str) -> str:
        return cls.colorize(text, cls.BOLD)
