"""Validation logic for transcription configuration.

This module contains pure validation functions and constants used throughout
the configuration system. These validators are dependency-light and can be
used by both public configs (TranscriptionConfig, EnrichmentConfig) and legacy
configs (AppConfig, AsrConfig).

Functions in this module:
- validate_model_name: Validates Whisper model names
- validate_compute_type: Validates compute type options
- auto_derive_compute_type: Auto-selects compute type based on device
- validate_diarization_settings: Validates speaker diarization parameters

Constants:
- WhisperTask: Type alias for valid Whisper tasks
- ALLOWED_COMPUTE_TYPES: Valid compute type options
- ALLOWED_WHISPER_MODELS: Valid Whisper model names
"""

from __future__ import annotations

from typing import Literal

from .exceptions import ConfigurationError

# ============================================================================
# Type Aliases and Constants
# ============================================================================

WhisperTask = Literal["transcribe", "translate"]

ALLOWED_COMPUTE_TYPES: set[str] = {
    "float16",
    "float32",
    "int16",
    "int8",
    "int8_float16",
    "int8_float32",
}

ALLOWED_WHISPER_MODELS: set[str] = {
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
}


# ============================================================================
# Validation Functions
# ============================================================================


def validate_model_name(model_name: str) -> str:
    """
    Validate that model_name is a supported Whisper model.

    Args:
        model_name: Requested Whisper model name.

    Returns:
        The validated model name (unchanged).

    Raises:
        ConfigurationError: If model_name is not one of the allowed values.
    """
    if model_name not in ALLOWED_WHISPER_MODELS:
        allowed = ", ".join(sorted(ALLOWED_WHISPER_MODELS))
        raise ConfigurationError(f"Invalid model name '{model_name}'. Must be one of: {allowed}")

    return model_name


def validate_compute_type(compute_type: str | None) -> str | None:
    """
    Validate that compute_type is supported by faster-whisper.

    Args:
        compute_type: Requested compute type (case-insensitive) or None/"auto" for auto.

    Returns:
        Normalized compute type (lowercase) or None if not provided.

    Raises:
        ConfigurationError: If compute_type is not one of the allowed values.
    """
    if compute_type is None:
        return None

    if not isinstance(compute_type, str):
        raise ConfigurationError("compute_type must be a string or null.")

    normalized = compute_type.strip().lower()
    if not normalized or normalized in {"auto", "none", "null"}:
        return None

    if normalized not in ALLOWED_COMPUTE_TYPES:
        allowed = ", ".join(sorted(ALLOWED_COMPUTE_TYPES))
        raise ConfigurationError(
            f"Invalid compute_type '{compute_type}'. Must be one of: {allowed}"
        )

    return normalized


def auto_derive_compute_type(device: str, compute_type: str | None) -> str:
    """
    Auto-derive compute_type based on device if not explicitly provided.

    If compute_type is None/"auto", selects sensible defaults:
    - "int8" for CPU (optimal for CPU inference)
    - "float16" for CUDA (uses Tensor Cores on modern GPUs)

    Args:
        device: Target device ("cpu" or "cuda").
        compute_type: Requested compute type or None/"auto" for auto-selection.

    Returns:
        Normalized compute type (lowercase, validated).

    Raises:
        ConfigurationError: If compute_type is not supported by faster-whisper.
    """
    normalized = validate_compute_type(compute_type)
    if normalized is None:
        # Use sensible defaults: int8 for CPU, float16 for CUDA
        return "int8" if device == "cpu" else "float16"

    return normalized


def validate_diarization_settings(
    min_speakers: int | None,
    max_speakers: int | None,
    overlap_threshold: float | None = None,
) -> None:
    """
    Validate diarization-related configuration values.

    Ensures provided speaker bounds are positive integers, overlap thresholds
    are within [0.0, 1.0], and that min_speakers does not exceed max_speakers
    when both are set.

    Raises:
        ConfigurationError: If any constraint is violated.
    """
    if min_speakers is not None:
        if isinstance(min_speakers, bool) or not isinstance(min_speakers, int):
            raise ConfigurationError("min_speakers must be a positive integer when provided.")
        if min_speakers < 1:
            raise ConfigurationError("min_speakers must be a positive integer when provided.")

    if max_speakers is not None:
        if isinstance(max_speakers, bool) or not isinstance(max_speakers, int):
            raise ConfigurationError("max_speakers must be a positive integer when provided.")
        if max_speakers < 1:
            raise ConfigurationError("max_speakers must be a positive integer when provided.")

    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise ConfigurationError(
            f"min_speakers ({min_speakers}) cannot be greater than max_speakers ({max_speakers})."
        )

    if overlap_threshold is not None:
        if isinstance(overlap_threshold, bool) or not isinstance(
            overlap_threshold,
            int | float,
        ):
            raise ConfigurationError("overlap_threshold must be between 0.0 and 1.0.")
        if not 0.0 <= float(overlap_threshold) <= 1.0:
            raise ConfigurationError("overlap_threshold must be between 0.0 and 1.0.")
