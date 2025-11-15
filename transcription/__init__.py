"""
Local transcription pipeline package.

Provides:
- Domain models (Segment, Transcript)
- Audio normalization utilities (ffmpeg)
- ASR engine wrapper around faster-whisper
- Writers for JSON/TXT/SRT
- Pipeline orchestration and CLI entrypoint
"""

from .models import Segment, Transcript
from .config import AppConfig, AsrConfig, Paths
from .pipeline import run_pipeline

__all__ = [
    "Segment",
    "Transcript",
    "AppConfig",
    "AsrConfig",
    "Paths",
    "run_pipeline",
]

# Version of the transcription pipeline; included in JSON metadata.
__version__ = "1.0.0"
