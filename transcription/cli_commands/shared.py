"""Shared utilities for CLI commands."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_progress_logging(show_progress: bool) -> None:
    """
    Configure logging to show/hide progress messages based on --progress flag.

    When show_progress=True, sets root logger to INFO to show file counters.
    When show_progress=False, sets to WARNING to hide progress messages.

    Args:
        show_progress: Whether to show progress indicators (file counters).
    """
    level = logging.INFO if show_progress else logging.WARNING

    # Configure basic format first (only works on first call)
    logging.basicConfig(format="%(message)s")

    # Always set level directly (works on subsequent calls)
    logging.getLogger().setLevel(level)


def get_cache_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except (OSError, PermissionError):
                pass
    return total


def format_size(bytes_size: int) -> str:
    """Format bytes as human-readable string."""
    size = float(bytes_size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def default_export_path(input_path: Path, fmt: str) -> Path:
    """Derive default export path from input path and format."""
    suffix_map = {"csv": ".csv", "html": ".html", "vtt": ".vtt", "textgrid": ".TextGrid"}
    suffix = suffix_map.get(fmt.lower(), f".{fmt}")
    return input_path.with_suffix(suffix)
