"""Custom exception classes for slower-whisper library."""


class SlowerWhisperError(Exception):
    """Base error for this library."""


class TranscriptionError(SlowerWhisperError):
    """Raised when transcription fails."""


class EnrichmentError(SlowerWhisperError):
    """Raised when audio enrichment fails."""


class ConfigurationError(SlowerWhisperError):
    """Raised when configuration is invalid."""
