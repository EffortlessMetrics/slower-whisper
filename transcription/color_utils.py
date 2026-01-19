"""
Lightweight ANSI color utilities for the CLI.
Respects NO_COLOR environment variable and TTY status.
"""
import os
import sys

def should_use_colors() -> bool:
    """
    Determine if colors should be used in output.
    Returns True if:
    - stdout is a TTY
    - NO_COLOR env var is not set
    - TERM is not "dumb"
    """
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("TERM") == "dumb":
        return False
    # If we are piping to a file, we usually don't want colors, unless forced (not handled here)
    return sys.stdout.isatty()

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    @classmethod
    def wrap(cls, text: str, color: str) -> str:
        if not should_use_colors():
            return text
        return f"{color}{text}{cls.RESET}"

    @classmethod
    def red(cls, text: str) -> str:
        return cls.wrap(text, cls.RED)

    @classmethod
    def green(cls, text: str) -> str:
        return cls.wrap(text, cls.GREEN)

    @classmethod
    def yellow(cls, text: str) -> str:
        return cls.wrap(text, cls.YELLOW)

    @classmethod
    def blue(cls, text: str) -> str:
        return cls.wrap(text, cls.BLUE)

    @classmethod
    def cyan(cls, text: str) -> str:
        return cls.wrap(text, cls.CYAN)

    @classmethod
    def magenta(cls, text: str) -> str:
        return cls.wrap(text, cls.MAGENTA)

    @classmethod
    def gray(cls, text: str) -> str:
        return cls.wrap(text, cls.GRAY)

    @classmethod
    def bold(cls, text: str) -> str:
        return cls.wrap(text, cls.BOLD)

def print_error(msg: str) -> None:
    """Print an error message to stderr in red."""
    print(Colors.red(f"Error: {msg}"), file=sys.stderr)

def print_warning(msg: str) -> None:
    """Print a warning message to stderr in yellow."""
    print(Colors.yellow(f"Warning: {msg}"), file=sys.stderr)

def print_success(msg: str) -> None:
    """Print a success message to stdout in green."""
    print(Colors.green(msg))

def print_header(msg: str) -> None:
    """Print a header message in bold cyan."""
    print(Colors.bold(Colors.cyan(msg)))
