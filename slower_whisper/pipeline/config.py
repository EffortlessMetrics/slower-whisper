"""Configuration management for transcription and enrichment pipelines.

This module is a stable façade for backward compatibility. All configuration
classes and utilities have been extracted to specialized modules:

- transcription.legacy_config: Legacy internal configs (Paths, AsrConfig, AppConfig)
- transcription.transcription_config: Public API for Stage 1 (TranscriptionConfig)
- transcription.enrichment_config: Public API for Stage 2 (EnrichmentConfig)
- transcription.config_validation: Validation helpers and constants
- transcription.config_merge: Generic merge helpers

This façade re-exports all public items from their new modules to maintain
existing import patterns without breaking changes.
"""

from __future__ import annotations

# Re-export merge helpers (used by tests and CLI)
from .config_merge import _merge_configs, _merge_enrich_configs

# Re-export validation helpers and constants
from .config_validation import (
    ALLOWED_COMPUTE_TYPES,
    ALLOWED_WHISPER_MODELS,
    WhisperTask,
    auto_derive_compute_type,
    validate_compute_type,
    validate_diarization_settings,
    validate_model_name,
)

# Re-export public API config classes
from .enrichment_config import EnrichmentConfig

# Re-export legacy config classes
from .legacy_config import AppConfig, AsrConfig, Paths
from .transcription_config import TranscriptionConfig

__all__ = [
    # Legacy config classes
    "Paths",
    "AsrConfig",
    "AppConfig",
    # Public API config classes
    "TranscriptionConfig",
    "EnrichmentConfig",
    # Validation constants
    "WhisperTask",
    "ALLOWED_COMPUTE_TYPES",
    "ALLOWED_WHISPER_MODELS",
    # Validation functions
    "validate_model_name",
    "validate_compute_type",
    "auto_derive_compute_type",
    "validate_diarization_settings",
    # Merge helpers
    "_merge_configs",
    "_merge_enrich_configs",
]
