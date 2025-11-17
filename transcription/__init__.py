"""
Local transcription pipeline package.

Public API:
    - transcribe_directory: Transcribe all audio in a project
    - transcribe_file: Transcribe a single file
    - enrich_directory: Enrich all transcripts with audio features
    - enrich_transcript: Enrich a single transcript
    - load_transcript: Load transcript from JSON
    - save_transcript: Save transcript to JSON

Configuration:
    - TranscriptionConfig: Stage 1 configuration
    - EnrichmentConfig: Stage 2 configuration

Models:
    - Transcript: Complete transcript with segments
    - Segment: Single transcribed segment

Exceptions:
    - SlowerWhisperError: Base exception for this library
    - TranscriptionError: Raised when transcription fails
    - EnrichmentError: Raised when audio enrichment fails
    - ConfigurationError: Raised when configuration is invalid
"""

# Version of the transcription pipeline; included in JSON metadata.
# Must be defined before other imports to avoid circular imports
__version__ = "1.0.0"

# Public API exports
from .api import (
    enrich_directory,
    enrich_transcript,
    load_transcript,
    save_transcript,
    transcribe_directory,
    transcribe_file,
)

# Backward compatibility: keep legacy exports
from .config import AppConfig, AsrConfig, EnrichmentConfig, Paths, TranscriptionConfig

# Exception classes
from .exceptions import (
    ConfigurationError,
    EnrichmentError,
    SlowerWhisperError,
    TranscriptionError,
)
from .models import Segment, Transcript


def __getattr__(name):
    """Lazy import for run_pipeline to avoid circular imports."""
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Public API functions
    "transcribe_directory",
    "transcribe_file",
    "enrich_directory",
    "enrich_transcript",
    "load_transcript",
    "save_transcript",
    # Public configuration
    "TranscriptionConfig",
    "EnrichmentConfig",
    # Models
    "Segment",
    "Transcript",
    # Exceptions
    "SlowerWhisperError",
    "TranscriptionError",
    "EnrichmentError",
    "ConfigurationError",
    # Legacy (for backward compatibility)
    "AppConfig",
    "AsrConfig",
    "Paths",
    "run_pipeline",
]
