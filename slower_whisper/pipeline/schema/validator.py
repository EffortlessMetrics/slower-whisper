"""Schema validator wrapping jsonschema with graceful fallback.

This module provides the SchemaValidator class for validating data against
JSON schemas. The jsonschema import is optional with graceful fallback,
allowing the library to function even when jsonschema is not installed.
"""

from __future__ import annotations

import json
from typing import Any

from .exceptions import SchemaError, SchemaNotFoundError, SchemaValidationError
from .registry import SchemaRegistry, get_default_registry


class SchemaValidator:
    """Validator for JSON schema validation with optional jsonschema support.

    The SchemaValidator wraps the jsonschema library for validating data
    against JSON schemas. When jsonschema is not installed, validation
    is skipped with a warning rather than failing.

    Example:
        >>> from transcription.schema.validator import SchemaValidator
        >>> validator = SchemaValidator()
        >>> data = {"schema_version": 2, "file": "test.wav", "language": "en", "segments": []}
        >>> validator.validate(data, "transcript-v2")
    """

    def __init__(self, registry: SchemaRegistry | None = None) -> None:
        """Initialize the schema validator.

        Args:
            registry: Optional SchemaRegistry instance. If None, uses the
                default global registry.
        """
        self._registry = registry or get_default_registry()
        self._jsonschema_available = self._check_jsonschema()
        self._validator_class = None
        self._Draft7Validator = None

        if self._jsonschema_available:
            try:
                from jsonschema import Draft7Validator
                from jsonschema.validators import validator_for

                self._Draft7Validator = Draft7Validator
                self._validator_class = validator_for
            except ImportError:
                self._jsonschema_available = False

    @staticmethod
    def _check_jsonschema() -> bool:
        """Check if jsonschema is available.

        Returns:
            True if jsonschema can be imported, False otherwise.
        """
        try:
            import jsonschema  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        """Check if jsonschema validation is available.

        Returns:
            True if jsonschema is installed and validation can be performed,
            False if validation will be skipped.
        """
        return self._jsonschema_available

    def validate(
        self,
        data: dict[str, Any] | str,
        schema_name: str,
        raise_on_error: bool = True,
    ) -> bool:
        """Validate data against a schema.

        Args:
            data: The data to validate, either as a dict or JSON string.
            schema_name: The name of the schema to validate against.
            raise_on_error: If True, raises SchemaValidationError on failure.
                If False, returns False on validation failure.

        Returns:
            True if validation passes (or jsonschema is unavailable),
            False if validation fails and raise_on_error is False.

        Raises:
            SchemaNotFoundError: If the schema does not exist.
            SchemaValidationError: If validation fails and raise_on_error is True.
            SchemaError: If the data cannot be parsed.

        Example:
            >>> validator = SchemaValidator()
            >>> data = {"schema_version": 2, "file": "test.wav", "language": "en", "segments": []}
            >>> is_valid = validator.validate(data, "transcript-v2")
        """
        # Parse data if it's a JSON string
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                raise SchemaError(f"Failed to parse data as JSON: {e}") from e

        # Get schema from registry
        try:
            schema_info = self._registry.get_schema(schema_name)
        except SchemaNotFoundError:
            raise
        except Exception as e:
            raise SchemaError(f"Failed to load schema '{schema_name}': {e}") from e

        # Skip validation if jsonschema is not available
        if not self._jsonschema_available:
            return True

        # Perform validation
        assert self._Draft7Validator is not None  # guarded by _jsonschema_available
        errors = list(self._Draft7Validator(schema_info.content).iter_errors(data))

        if errors:
            error_details = [
                {
                    "path": " -> ".join(str(p) for p in error.path),
                    "message": error.message,
                    "validator": error.validator,
                }
                for error in errors
            ]

            if raise_on_error:
                raise SchemaValidationError(schema_name, error_details)
            return False

        return True

    def validate_and_get_errors(
        self,
        data: dict[str, Any] | str,
        schema_name: str,
    ) -> list[dict[str, Any]]:
        """Validate data and return all validation errors.

        This method always returns errors (empty list if valid) rather
        than raising an exception.

        Args:
            data: The data to validate, either as a dict or JSON string.
            schema_name: The name of the schema to validate against.

        Returns:
            List of validation error dictionaries. Each dict contains:
            - path: JSON path to the error location
            - message: Human-readable error message
            - validator: Name of the validator that failed

        Example:
            >>> validator = SchemaValidator()
            >>> errors = validator.validate_and_get_errors(data, "transcript-v2")
            >>> for error in errors:
            ...     print(f"{error['path']}: {error['message']}")
        """
        # Parse data if it's a JSON string
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                # Return parsing error as a validation error
                return [
                    {
                        "path": "",
                        "message": "Invalid JSON data",
                        "validator": "parse",
                    }
                ]

        # Get schema from registry
        try:
            schema_info = self._registry.get_schema(schema_name)
        except Exception:
            # Return schema loading error as a validation error
            return [
                {
                    "path": "",
                    "message": f"Failed to load schema '{schema_name}'",
                    "validator": "schema_load",
                }
            ]

        # Return empty list if jsonschema is not available
        if not self._jsonschema_available:
            return []

        # Perform validation and collect errors
        assert self._Draft7Validator is not None  # guarded by _jsonschema_available
        errors = list(self._Draft7Validator(schema_info.content).iter_errors(data))

        return [
            {
                "path": " -> ".join(str(p) for p in error.path),
                "message": error.message,
                "validator": error.validator,
            }
            for error in errors
        ]

    def format_errors(self, errors: list[dict[str, Any]]) -> str:
        """Format validation errors into a human-readable string.

        Args:
            errors: List of error dictionaries from validate_and_get_errors.

        Returns:
            Formatted error message string.

        Example:
            >>> validator = SchemaValidator()
            >>> errors = validator.validate_and_get_errors(data, "transcript-v2")
            >>> message = validator.format_errors(errors)
            >>> print(message)
        """
        if not errors:
            return "No validation errors."

        lines = ["Validation errors:"]
        for i, error in enumerate(errors, 1):
            path_str = error.get("path", "root")
            if path_str:
                lines.append(f"  {i}. [{path_str}] {error['message']}")
            else:
                lines.append(f"  {i}. {error['message']}")

        return "\n".join(lines)


# Global validator instance for convenience
_default_validator: SchemaValidator | None = None


def get_default_validator() -> SchemaValidator:
    """Get or create the default global schema validator.

    Returns:
        The global SchemaValidator instance.

    Example:
        >>> from transcription.schema.validator import get_default_validator
        >>> validator = get_default_validator()
        >>> validator.validate(data, "transcript-v2")
    """
    global _default_validator
    if _default_validator is None:
        _default_validator = SchemaValidator()
    return _default_validator


def validate_data(
    data: dict[str, Any] | str,
    schema_name: str,
    raise_on_error: bool = True,
) -> bool:
    """Convenience function to validate data against a schema.

    Args:
        data: The data to validate, either as a dict or JSON string.
        schema_name: The name of the schema to validate against.
        raise_on_error: If True, raises SchemaValidationError on failure.

    Returns:
        True if validation passes, False if validation fails and
        raise_on_error is False.

    Raises:
        SchemaNotFoundError: If the schema does not exist.
        SchemaValidationError: If validation fails and raise_on_error is True.
        SchemaError: If the data cannot be parsed.

    Example:
        >>> from transcription.schema.validator import validate_data
        >>> data = {"schema_version": 2, "file": "test.wav", "language": "en", "segments": []}
        >>> validate_data(data, "transcript-v2")
    """
    validator = get_default_validator()
    return validator.validate(data, schema_name, raise_on_error=raise_on_error)
