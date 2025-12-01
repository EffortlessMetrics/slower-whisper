"""
Module-level warning filters for optional dependencies.

This module suppresses informational warnings from optional backend libraries
that don't affect functionality (e.g., torchcodec FFmpeg binding warnings).
"""

from __future__ import annotations

import warnings


def suppress_optional_dependency_warnings() -> None:
    """
    Suppress non-critical warnings from optional dependencies.

    Applies to:
    - torchcodec: FFmpeg version mismatch warnings
    - torchaudio: Backend loading messages

    Call this before importing pyannote.audio or transformers.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")
    warnings.filterwarnings("ignore", message=".*FFmpeg.*")
