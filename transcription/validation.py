"""Transcript validation helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "transcript-v2.schema.json"


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON with a helpful error on failure."""
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data
    except FileNotFoundError as exc:
        raise ConfigurationError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:  # noqa: B904
        raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc


def _require_jsonschema() -> Any:
    try:
        import jsonschema  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        raise ConfigurationError(
            "jsonschema is required for validation. Install with `uv pip install jsonschema` "
            "or add it to your environment."
        ) from exc
    return jsonschema


def validate_transcript_json(
    transcript_path: Path, schema_path: Path | None = None
) -> tuple[bool, list[str]]:
    """Validate a transcript JSON against the v2 schema.

    Returns:
        (is_valid, errors) where errors is a list of strings.
    """
    jsonschema = _require_jsonschema()

    schema = load_json(schema_path or DEFAULT_SCHEMA_PATH)
    instance = load_json(transcript_path)

    validator = jsonschema.Draft7Validator(schema)
    errors = [f"{err.message} at {err.json_path}" for err in validator.iter_errors(instance)]
    return (len(errors) == 0, errors)


def validate_many(transcript_paths: Iterable[Path], schema_path: Path | None = None) -> list[str]:
    """Validate multiple transcript files, returning aggregated error strings."""
    failures: list[str] = []
    for path in transcript_paths:
        ok, errs = validate_transcript_json(path, schema_path=schema_path)
        if not ok:
            prefix = f"{path}: "
            failures.extend(prefix + err for err in errs)
    return failures
