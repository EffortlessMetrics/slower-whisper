"""
Dossier validation - schema and semantic validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _get_schema_path() -> Path:
    """Get path to the dossier schema file."""
    return Path(__file__).parent.parent / "schemas" / "pr-dossier-v2.schema.json"


def schema_exists() -> bool:
    """
    Check if the schema file exists.

    Returns:
        True if schema file exists, False otherwise
    """
    return _get_schema_path().exists()


def require_schema(mode: str = "warn") -> None:
    """
    Require the schema file to exist.

    Args:
        mode: "strict" to raise an error, "warn" to print warning to stderr

    Raises:
        FileNotFoundError: If mode is "strict" and schema file doesn't exist
    """
    schema_path = _get_schema_path()
    if not schema_path.exists():
        msg = f"Schema file not found: {schema_path}. Cannot perform schema validation."
        if mode == "strict":
            raise FileNotFoundError(msg)
        else:
            print(f"Warning: {msg}", file=sys.stderr)


def validate_dossier_schema(dossier: dict[str, Any]) -> list[str]:
    """
    Validate dossier against JSON Schema.

    Args:
        dossier: The dossier dict to validate

    Returns:
        List of validation errors (empty if valid)

    Note:
        If the schema file does not exist, this returns a hard error.
        Schema validation cannot proceed without the schema file.
    """
    errors = []

    try:
        import jsonschema

        schema_path = _get_schema_path()
        if schema_path.exists():
            with open(schema_path) as f:
                schema = json.load(f)
            validator = jsonschema.Draft7Validator(schema)
            errors.extend([f"Schema: {e.message}" for e in validator.iter_errors(dossier)])
        else:
            # Schema file missing - this is a hard error
            errors.append(
                f"Schema file not found: {schema_path}. Cannot perform schema validation."
            )

    except ImportError:
        # jsonschema not available, skip schema validation
        pass
    except Exception as e:
        errors.append(f"Schema validation error: {e}")

    return errors


def validate_dossier_semantic(dossier: dict[str, Any]) -> list[str]:
    """
    Semantic validation of dossier (cross-field consistency).

    Args:
        dossier: The dossier dict to validate

    Returns:
        List of semantic validation errors
    """
    errors = []

    # Check benchmarks have baseline if metrics present
    benchmarks = dossier.get("evidence", {}).get("benchmarks", {})
    if benchmarks.get("metrics") and not benchmarks.get("baseline_commit"):
        errors.append("Semantic: benchmarks.metrics present but no baseline_commit specified")

    # Check findings have anchors
    for i, finding in enumerate(dossier.get("findings", [])):
        has_anchor = finding.get("commit") or finding.get("detected_by")
        if not has_anchor:
            desc = finding.get("description", f"finding {i}")[:50]
            errors.append(f"Semantic: finding '{desc}' has no anchor (commit or detected_by)")

    # Check DevLT has bands if minutes present
    cost = dossier.get("cost", {})
    devlt = cost.get("devlt", {})

    # Handle both flat and nested devlt structure
    if isinstance(devlt, dict):
        author = devlt.get("author", {})
        if isinstance(author, dict):
            if author.get("lb_minutes") and not author.get("band"):
                errors.append("Semantic: devlt.author has minutes but no band")
        elif devlt.get("author_minutes") and not devlt.get("author_band"):
            errors.append("Semantic: author_minutes specified without author_band")

    # Check wall_clock consistency
    wall_clock = cost.get("wall_clock", {})
    if wall_clock.get("days") and wall_clock.get("days") < 0:
        errors.append("Semantic: wall_clock.days cannot be negative")

    # Check outcome is valid
    valid_outcomes = {"shipped", "deferred", "rejected", "invalid_measurement", "pending"}
    outcome = dossier.get("outcome")
    if outcome and outcome not in valid_outcomes:
        errors.append(f"Semantic: invalid outcome '{outcome}', must be one of {valid_outcomes}")

    # Check intent type is valid
    valid_types = {"feature", "hardening", "mechanization", "perf/bench", "refactor", "unknown"}
    intent_type = dossier.get("intent", {}).get("type")
    if intent_type and intent_type not in valid_types:
        errors.append(f"Semantic: invalid intent.type '{intent_type}'")

    return errors


def validate_dossier(dossier: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Full validation of dossier (schema + semantic).

    Args:
        dossier: The dossier dict to validate

    Returns:
        (is_valid, list_of_errors)

    Note:
        Schema validation requires the schema file to exist. If the schema file
        is missing, validation will fail with an error indicating the missing file.
        Use schema_exists() to check availability, or require_schema() to enforce
        presence before calling this function.
    """
    errors = []

    # Schema validation
    schema_errors = validate_dossier_schema(dossier)
    errors.extend(schema_errors)

    # Semantic validation
    semantic_errors = validate_dossier_semantic(dossier)
    errors.extend(semantic_errors)

    return (len(errors) == 0, errors)


def validate_dossier_file(path: Path) -> tuple[bool, list[str]]:
    """
    Validate a dossier JSON file.

    Args:
        path: Path to the dossier JSON file

    Returns:
        (is_valid, list_of_errors)
    """
    try:
        with open(path) as f:
            dossier = json.load(f)
        return validate_dossier(dossier)
    except json.JSONDecodeError as e:
        return (False, [f"JSON parse error: {e}"])
    except FileNotFoundError:
        return (False, [f"File not found: {path}"])
    except Exception as e:
        return (False, [f"Error reading file: {e}"])
