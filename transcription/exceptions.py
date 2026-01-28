"""Custom exception classes for slower-whisper library."""

from pathlib import Path


class SlowerWhisperError(Exception):
    """Base error for this library."""


class TranscriptionError(SlowerWhisperError):
    """Raised when transcription fails."""


class EnrichmentError(SlowerWhisperError):
    """Raised when audio enrichment fails."""


class ConfigurationError(SlowerWhisperError):
    """Raised when configuration is invalid."""


class SampleExistsError(SlowerWhisperError):
    """Raised when sample files already exist in the target directory."""

    def __init__(self, message: str, existing_files: list[Path]):
        super().__init__(message)
        self.existing_files = existing_files
