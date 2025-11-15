"""
Local transcription pipeline package.

Provides:
- Domain models (Segment, Transcript)
- Audio normalization utilities (ffmpeg)
- ASR engine wrapper around faster-whisper
- Writers for JSON/TXT/SRT
- Pipeline orchestration and CLI entrypoint
"""

# Version of the transcription pipeline; included in JSON metadata.
# Must be defined before other imports to avoid circular imports
__version__ = "1.0.0"

from .config import AppConfig, AsrConfig, Paths
from .models import Segment, Transcript


def __getattr__(name):
    """Lazy import for run_pipeline to avoid circular imports."""
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AppConfig",
    "AsrConfig",
    "Paths",
    "Segment",
    "Transcript",
    "run_pipeline",
]
