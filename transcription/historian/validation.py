"""
Dossier validation - schema and semantic validation.

Strict mode (default for analyze/publish):
    Missing schema file is a hard error, not a degraded warning.
    This prevents silent failures where validation appears to pass
    but no actual schema checking occurred.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class SchemaNotFoundError(FileNotFoundError):
    """Raised when the schema file is missing and strict mode is enabled."""

    def __init__(self, schema_path: Path) -> None:
        self.schema_path = schema_path
        super().__init__(
            f"Schema file not found: {schema_path}\n"
            f"  This is required for validation in analyze/publish modes.\n"
            f"  Expected location: {schema_path.resolve()}\n"
            f"  Ensure the package is properly installed or the schema file exists."
        )


class ValidationError(Exception):
    """Raised when validation fails in strict mode."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        error_list = "\n  - ".join(errors)
        super().__init__(f"Validation failed with {len(errors)} error(s):\n  - {error_list}")


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


def require_schema(*, strict: bool = True) -> Path:
    """
    Require the schema file to exist.

    Args:
        strict: If True (default), raise SchemaNotFoundError when missing.
                If False, print warning to stderr and return Path anyway.

    Returns:
        Path to the schema file

    Raises:
        SchemaNotFoundError: If strict=True and schema file doesn't exist
    """
    schema_path = _get_schema_path()
    if not schema_path.exists():
        if strict:
            raise SchemaNotFoundError(schema_path)
        else:
            print(
                f"Warning: Schema file not found: {schema_path}. Validation will be incomplete.",
                file=sys.stderr,
            )
    return schema_path


def _format_json_path(path: list[Any]) -> str:
    """Format a JSON path from jsonschema into a readable dotted path."""
    if not path:
        return "<root>"
    parts = []
    for item in path:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            if parts:
                parts.append(f".{item}")
            else:
                parts.append(str(item))
    return "".join(parts)


def _format_schema_error(error: Any) -> str:
    """Format a jsonschema validation error with context."""
    path = _format_json_path(list(error.absolute_path))
    validator = error.validator

    # Provide actionable error messages based on error type
    if validator == "required":
        missing = error.message.replace("'", "")  # e.g., "'pr_number' is a required property"
        return f"Missing required field at {path}: {missing}"

    elif validator == "type":
        expected = error.schema.get("type", "unknown")
        actual = type(error.instance).__name__
        return f"Type mismatch at {path}: expected {expected}, got {actual} (value: {error.instance!r})"

    elif validator == "enum":
        allowed = error.schema.get("enum", [])
        return f"Invalid value at {path}: got {error.instance!r}, must be one of {allowed}"

    elif validator == "const":
        expected = error.schema.get("const")
        return f"Invalid value at {path}: expected {expected!r}, got {error.instance!r}"

    elif validator == "format":
        fmt = error.schema.get("format", "unknown")
        return f"Invalid format at {path}: expected {fmt}, got {error.instance!r}"

    elif validator == "additionalProperties":
        return f"Unexpected field at {path}: {error.message}"

    elif validator == "minLength":
        return f"Field at {path} is too short: {error.message}"

    elif validator == "maxLength":
        return f"Field at {path} is too long: {error.message}"

    elif validator == "minimum" or validator == "maximum":
        return f"Value at {path} out of range: {error.message}"

    else:
        # Generic fallback with path context
        return f"Schema error at {path}: {error.message}"


def validate_dossier_schema(
    dossier: dict[str, Any],
    *,
    strict: bool = True,
) -> list[str]:
    """
    Validate dossier against JSON Schema.

    Args:
        dossier: The dossier dict to validate
        strict: If True (default), missing schema file raises SchemaNotFoundError.
                If False, returns error message but continues.

    Returns:
        List of validation errors (empty if valid)

    Raises:
        SchemaNotFoundError: If strict=True and schema file doesn't exist
        ValidationError: Never raised by this function (returns errors as list)

    Note:
        In strict mode (default), missing schema is a hard error, not a warning.
        This prevents silent degradation where validation "passes" without
        actually checking the schema.
    """
    errors: list[str] = []

    schema_path = _get_schema_path()

    if not schema_path.exists():
        if strict:
            raise SchemaNotFoundError(schema_path)
        else:
            errors.append(
                f"Schema file not found: {schema_path}. "
                "Cannot perform schema validation. "
                "To fix: ensure the package is properly installed."
            )
            return errors

    try:
        import jsonschema
    except ImportError:
        msg = (
            "jsonschema package not installed. "
            "Cannot perform schema validation. "
            "To fix: pip install jsonschema"
        )
        if strict:
            errors.append(msg)
        return errors

    try:
        with open(schema_path) as f:
            schema = json.load(f)

        validator = jsonschema.Draft7Validator(schema)

        # Collect all errors with detailed formatting
        for error in validator.iter_errors(dossier):
            errors.append(_format_schema_error(error))

    except json.JSONDecodeError as e:
        errors.append(f"Schema file is malformed JSON: {e}")
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


def validate_dossier(
    dossier: dict[str, Any],
    *,
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """
    Full validation of dossier (schema + semantic).

    Args:
        dossier: The dossier dict to validate
        strict: If True (default), missing schema file raises SchemaNotFoundError.
                If False, continues with degraded validation (not recommended).

    Returns:
        (is_valid, list_of_errors)

    Raises:
        SchemaNotFoundError: If strict=True and schema file doesn't exist

    Note:
        In analyze/publish modes, strict=True is the default to prevent
        silent degradation where validation "passes" without the schema.
        Use strict=False only for debugging or when intentionally skipping
        schema validation.
    """
    errors: list[str] = []

    # Schema validation (may raise in strict mode)
    schema_errors = validate_dossier_schema(dossier, strict=strict)
    errors.extend(schema_errors)

    # Semantic validation
    semantic_errors = validate_dossier_semantic(dossier)
    errors.extend(semantic_errors)

    return (len(errors) == 0, errors)


def validate_dossier_strict(dossier: dict[str, Any]) -> None:
    """
    Validate dossier in strict mode, raising on any error.

    This is the recommended validation function for analyze/publish pipelines.
    It fails fast and loudly when validation fails.

    Args:
        dossier: The dossier dict to validate

    Raises:
        SchemaNotFoundError: If schema file doesn't exist
        ValidationError: If dossier fails validation
    """
    valid, errors = validate_dossier(dossier, strict=True)
    if not valid:
        raise ValidationError(errors)


def validate_dossier_file(
    path: Path,
    *,
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validate a dossier JSON file.

    Args:
        path: Path to the dossier JSON file
        strict: If True (default), missing schema raises SchemaNotFoundError.

    Returns:
        (is_valid, list_of_errors)

    Raises:
        SchemaNotFoundError: If strict=True and schema file doesn't exist
    """
    try:
        with open(path) as f:
            dossier = json.load(f)
        return validate_dossier(dossier, strict=strict)
    except json.JSONDecodeError as e:
        return (False, [f"JSON parse error at {path}: {e}"])
    except FileNotFoundError:
        return (False, [f"Dossier file not found: {path}"])
    except SchemaNotFoundError:
        raise  # Re-raise schema errors
    except Exception as e:
        return (False, [f"Error reading file {path}: {e}"])
