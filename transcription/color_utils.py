"""
Lightweight ANSI color utilities for the CLI.

Designed to be zero-dependency and safe for non-interactive environments.
Respects NO_COLOR environment variable.
"""

import os
import sys
from typing import ClassVar


class Colors:
    """ANSI color codes for terminal output."""

    # Standard colors
    RESET: ClassVar[str] = "\033[0m"
    BOLD: ClassVar[str] = "\033[1m"
    DIM: ClassVar[str] = "\033[2m"
    ITALIC: ClassVar[str] = "\033[3m"
    UNDERLINE: ClassVar[str] = "\033[4m"

    RED: ClassVar[str] = "\033[31m"
    GREEN: ClassVar[str] = "\033[32m"
    YELLOW: ClassVar[str] = "\033[33m"
    BLUE: ClassVar[str] = "\033[34m"
    MAGENTA: ClassVar[str] = "\033[35m"
    CYAN: ClassVar[str] = "\033[36m"
    WHITE: ClassVar[str] = "\033[37m"

    # Bright colors
    BRIGHT_RED: ClassVar[str] = "\033[91m"
    BRIGHT_GREEN: ClassVar[str] = "\033[92m"
    BRIGHT_YELLOW: ClassVar[str] = "\033[93m"
    BRIGHT_BLUE: ClassVar[str] = "\033[94m"
    BRIGHT_MAGENTA: ClassVar[str] = "\033[95m"
    BRIGHT_CYAN: ClassVar[str] = "\033[96m"
    BRIGHT_WHITE: ClassVar[str] = "\033[97m"

    @classmethod
    def should_use_color(cls) -> bool:
        """
        Determine if colors should be used.
        Returns False if NO_COLOR is set, TERM is dumb, or stdout is not a TTY.
        """
        if os.getenv("NO_COLOR"):
            return False
        if os.getenv("TERM") == "dumb":
            return False
        # Check if running in a TTY (interactive terminal)
        # We use sys.stdout.isatty() but also check if we are forced to color
        # via FORCE_COLOR (common convention)
        if os.getenv("FORCE_COLOR"):
            return True
        return sys.stdout.isatty()

    @classmethod
    def _should_use_color(cls) -> bool:
        return cls.should_use_color()

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not cls.should_use_color():
            return text
        return f"{color}{text}{cls.RESET}"

    @classmethod
    def green(cls, text: str) -> str:
        return cls.colorize(text, cls.GREEN)

    @classmethod
    def red(cls, text: str) -> str:
        return cls.colorize(text, cls.RED)

    @classmethod
    def yellow(cls, text: str) -> str:
        return cls.colorize(text, cls.YELLOW)

    @classmethod
    def blue(cls, text: str) -> str:
        return cls.colorize(text, cls.BLUE)

    @classmethod
    def cyan(cls, text: str) -> str:
        return cls.colorize(text, cls.CYAN)

    @classmethod
    def magenta(cls, text: str) -> str:
        return cls.colorize(text, cls.MAGENTA)

    @classmethod
    def bold(cls, text: str) -> str:
        return cls.colorize(text, cls.BOLD)

    @classmethod
    def dim(cls, text: str) -> str:
        return cls.colorize(text, cls.DIM)


class Symbols:
    """Unicode symbols for CLI output."""

    CHECK: ClassVar[str] = "✓"
    CROSS: ClassVar[str] = "✗"
    WARNING: ClassVar[str] = "⚠"
    INFO: ClassVar[str] = "ℹ"
    DOT: ClassVar[str] = "•"
    ARROW: ClassVar[str] = "→"
