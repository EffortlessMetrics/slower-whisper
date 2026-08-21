"""Custom exception classes for slower-whisper library."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar


class SlowerWhisperError(Exception):
    """Base error for this library."""


class TranscriptionError(SlowerWhisperError):
    """Raised when transcription fails.

    ``reason_code`` is stable machine-readable identity. ``context`` contains
    only deliberately selected, non-sensitive fields suitable for remote error
    responses. Provider exceptions remain available through exception chaining.
    """

    default_reason_code: ClassVar[str] = "transcription_error"

    def __init__(
        self,
        message: str,
        *,
        reason_code: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code or self.default_reason_code
        self.context = dict(context or {})

    def public_details(self) -> dict[str, Any]:
        """Return the stable, non-sensitive portion of this error contract."""
        return {
            "reason_code": self.reason_code,
            "context": dict(self.context),
        }


class ASRError(TranscriptionError):
    """Base class for typed automatic-speech-recognition failures."""


class ASRUnavailableError(ASRError):
    """Raised when the configured ASR backend or dependency is unavailable."""

    default_reason_code = "asr_backend_unavailable"


class ASRModelLoadError(ASRError):
    """Raised when no supported runtime configuration can load the ASR model."""

    default_reason_code = "asr_model_load_failed"


class ASRInferenceError(ASRError):
    """Raised when an initialized ASR backend fails during inference."""

    default_reason_code = "asr_inference_failed"


class ASROutputError(ASRError):
    """Raised when an ASR backend returns malformed or invalid output."""

    default_reason_code = "asr_output_invalid"


class EnrichmentError(SlowerWhisperError):
    """Raised when audio enrichment fails."""


class ConfigurationError(SlowerWhisperError):
    """Raised when configuration is invalid."""


class SampleExistsError(SlowerWhisperError):
    """Raised when sample files already exist in the target directory."""

    def __init__(self, message: str, existing_files: list[Path]):
        super().__init__(message)
        self.existing_files = existing_files
