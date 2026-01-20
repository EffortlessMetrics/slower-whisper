"""Configuration merge logic for transcription and enrichment pipelines.

This module provides generic merge functionality for dataclasses with
_source_fields tracking. The merge strategy ensures proper config layering:

- If override._source_fields exists and is non-empty → only override those fields
- Else → fallback to value comparison (backward compatibility)

This is critical for implementing correct precedence: CLI > file > env > defaults.
When a config file sets a value that happens to equal the default (e.g., device="cuda"),
the merge logic must use the file value, not fall back to the environment variable.
"""

import dataclasses
from typing import Any


def _merge_by_source_fields(base: Any, override: Any, *, cls_type: type) -> Any:
    """
    Generic merge helper for dataclasses with _source_fields tracking.

    This function implements proper config layering by checking which fields
    were explicitly set in the override config (via _source_fields attribute).
    Only explicitly-set fields override the base config, preventing the bug where
    a file value that equals the default would be ignored.

    The merge strategy:
    - If field was explicitly set in override (in _source_fields): use override value
    - Otherwise: keep base value

    This correctly handles the case where:
    1. Env sets device="cpu"
    2. File explicitly sets device="cuda" (which happens to equal the default)
    3. Result is "cuda" (file overrides env) ✓

    And also handles:
    1. Env sets device="cpu"
    2. File doesn't mention device (so device="cuda" is just a default)
    3. Result is "cpu" (env value preserved) ✓

    Args:
        base: Base configuration (lower precedence).
        override: Override configuration (higher precedence).
        cls_type: The dataclass type to instantiate for the result.

    Returns:
        New instance of cls_type with merged values.
    """
    # Check if override has _source_fields (set by from_file/from_env)
    # If not, or if it's empty, fall back to comparing values (for backward compatibility)
    source_fields = getattr(override, "_source_fields", None)

    kwargs: dict[str, Any] = {}

    # Iterate over all init fields of the dataclass
    for field in dataclasses.fields(cls_type):
        if not field.init:
            continue

        field_name = field.name

        if source_fields is None or not source_fields:
            # No source tracking - fall back to value comparison
            # This shouldn't happen in normal CLI usage, but provides safety
            base_value = getattr(base, field_name)
            override_value = getattr(override, field_name)
            kwargs[field_name] = override_value if override_value != base_value else base_value
        else:
            # Use source_fields to determine which values to override
            if field_name in source_fields:
                kwargs[field_name] = getattr(override, field_name)
            else:
                kwargs[field_name] = getattr(base, field_name)

    return cls_type(**kwargs)


def _merge_configs(base, override):
    """
    Merge two TranscriptionConfig instances, with override taking precedence.

    This is a thin wrapper around _merge_by_source_fields for TranscriptionConfig.

    Args:
        base: Base configuration (lower precedence).
        override: Override configuration (higher precedence).

    Returns:
        New TranscriptionConfig with merged values.
    """
    from .transcription_config import TranscriptionConfig

    return _merge_by_source_fields(base, override, cls_type=TranscriptionConfig)


def _merge_enrich_configs(base, override):
    """
    Merge two EnrichmentConfig instances, with override taking precedence.

    This is a thin wrapper around _merge_by_source_fields for EnrichmentConfig.

    Args:
        base: Base configuration (lower precedence).
        override: Override configuration (higher precedence).

    Returns:
        New EnrichmentConfig with merged values.
    """
    from .enrichment_config import EnrichmentConfig

    return _merge_by_source_fields(base, override, cls_type=EnrichmentConfig)
