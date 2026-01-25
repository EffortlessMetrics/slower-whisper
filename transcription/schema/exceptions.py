"""Custom exception classes for schema operations.

This module defines exceptions raised by the schema locking infrastructure,
including schema validation, migration, and integrity verification errors.
"""

from typing import Any


class SchemaError(Exception):
    """Base exception for all schema-related errors.

    All custom schema exceptions inherit from this class, allowing
    for generic exception handling of schema operations.
    """


class SchemaNotFoundError(SchemaError):
    """Raised when a requested schema file cannot be found.

    Attributes:
        schema_name: The name of the schema that was not found.
    """

    def __init__(self, schema_name: str) -> None:
        self.schema_name = schema_name
        super().__init__(f"Schema not found: {schema_name}")


class SchemaValidationError(SchemaError):
    """Raised when data fails validation against a schema.

    Attributes:
        schema_name: The name of the schema used for validation.
        errors: List of validation error details.
    """

    def __init__(self, schema_name: str, errors: list[dict[str, Any]] | None = None) -> None:
        self.schema_name = schema_name
        self.errors = errors or []
        error_count = len(self.errors)
        super().__init__(f"Validation failed for schema '{schema_name}': {error_count} error(s)")


class SchemaIntegrityError(SchemaError):
    """Raised when schema integrity verification fails.

    This occurs when a schema file's computed hash does not match
    the expected hash in the lockfile, indicating potential tampering
    or unintended modification.

    Attributes:
        schema_name: The name of the schema with integrity issues.
        expected_hash: The expected SHA256 hash from the lockfile.
        actual_hash: The computed SHA256 hash of the schema file.
    """

    def __init__(self, schema_name: str, expected_hash: str, actual_hash: str) -> None:
        self.schema_name = schema_name
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"Schema integrity check failed for '{schema_name}': "
            f"expected hash {expected_hash[:16]}..., got {actual_hash[:16]}..."
        )


class SchemaMigrationError(SchemaError):
    """Raised when a schema migration fails.

    Attributes:
        from_version: The source schema version.
        to_version: The target schema version.
        reason: Description of why the migration failed.
    """

    def __init__(self, from_version: str, to_version: str, reason: str) -> None:
        self.from_version = from_version
        self.to_version = to_version
        self.reason = reason
        super().__init__(f"Migration from {from_version} to {to_version} failed: {reason}")


class SchemaVersionError(SchemaError):
    """Raised when an unsupported or unknown schema version is encountered.

    Attributes:
        version: The unsupported schema version.
        available_versions: List of available schema versions.
    """

    def __init__(self, version: str, available_versions: list[str] | None = None) -> None:
        self.version = version
        self.available_versions = available_versions or []
        versions_str = ", ".join(self.available_versions)
        super().__init__(
            f"Unsupported schema version: {version}. Available versions: {versions_str}"
        )


class LockfileError(SchemaError):
    """Raised when lockfile operations fail.

    Attributes:
        reason: Description of the lockfile error.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Lockfile error: {reason}")


class SchemaRegistryError(SchemaError):
    """Raised when schema registry operations fail.

    Attributes:
        reason: Description of the registry error.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Schema registry error: {reason}")
