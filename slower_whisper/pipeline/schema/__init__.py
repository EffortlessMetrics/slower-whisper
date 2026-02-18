"""Schema locking infrastructure for slower-whisper.

This package provides tools for schema validation, integrity verification,
and migration between schema versions. It supports SHA256 hash-based
integrity checks and optional jsonschema validation.

Public API:
    - get_schema(): Get a schema by name
    - validate_data(): Validate data against a schema
    - get_schema_version(): Get the version of a schema
    - verify_schema_integrity(): Verify schema hash matches expected value

Example:
    >>> from transcription.schema import get_schema, validate_data
    >>> schema = get_schema("transcript-v2")
    >>> print(f"Schema version: {schema.version}")
    >>> data = {"schema_version": 2, "file": "test.wav", "language": "en", "segments": []}
    >>> validate_data(data, "transcript-v2")
"""

from typing import Any

# Re-export exceptions for public API
from .exceptions import (
    LockfileError,
    SchemaError,
    SchemaIntegrityError,
    SchemaMigrationError,
    SchemaNotFoundError,
    SchemaRegistryError,
    SchemaValidationError,
    SchemaVersionError,
)

# Re-export integrity classes and functions
from .integrity import (
    SchemaLockEntry,
    SchemaLockfile,
    create_lockfile,
    load_lockfile,
    save_lockfile,
    update_lockfile,
    verify_all_schema_integrity,
    verify_all_schema_integrity_strict,
    verify_schema_integrity,
    verify_schema_integrity_strict,
)

# Re-export migration classes and functions
from .migration import (
    MigrationPath,
    MigrationRegistry,
    auto_migrate,
    detect_schema_version,
    get_default_migration_registry,
    migrate_v1_to_v2,
)

# Re-export registry classes and functions
from .registry import (
    SchemaInfo,
    SchemaRegistry,
    get_default_registry,
)

# Re-export validator classes and functions
from .validator import (
    SchemaValidator,
    get_default_validator,
    validate_data,
)

__all__ = [
    # Exceptions
    "SchemaError",
    "SchemaNotFoundError",
    "SchemaValidationError",
    "SchemaIntegrityError",
    "SchemaMigrationError",
    "SchemaVersionError",
    "LockfileError",
    "SchemaRegistryError",
    # Registry
    "SchemaRegistry",
    "SchemaInfo",
    "get_default_registry",
    # Validator
    "SchemaValidator",
    "get_default_validator",
    "validate_data",
    # Migration
    "MigrationPath",
    "MigrationRegistry",
    "auto_migrate",
    "detect_schema_version",
    "get_default_migration_registry",
    "migrate_v1_to_v2",
    # Integrity
    "SchemaLockfile",
    "SchemaLockEntry",
    "load_lockfile",
    "save_lockfile",
    "create_lockfile",
    "update_lockfile",
    "verify_schema_integrity",
    "verify_schema_integrity_strict",
    "verify_all_schema_integrity",
    "verify_all_schema_integrity_strict",
]


# Convenience functions for common operations
def get_schema(name: str) -> SchemaInfo:
    """Get a schema by name using the default registry.

    This is a convenience wrapper around SchemaRegistry.get_schema().

    Args:
        name: The schema name (e.g., "transcript-v2", "pr-dossier-v2").

    Returns:
        SchemaInfo containing the schema metadata and content.

    Raises:
        SchemaNotFoundError: If the schema does not exist.
        SchemaRegistryError: If the schema cannot be loaded.
        SchemaVersionError: If the schema version is unexpected.

    Example:
        >>> from transcription.schema import get_schema
        >>> schema = get_schema("transcript-v2")
        >>> print(f"Version: {schema.version}, Hash: {schema.hash}")
    """
    return get_default_registry().get_schema(name)


def get_schema_version(name: str) -> int:
    """Get the version number of a schema.

    This is a convenience wrapper around SchemaRegistry.get_schema_version().

    Args:
        name: The schema name.

    Returns:
        The schema version number.

    Raises:
        SchemaNotFoundError: If the schema does not exist.

    Example:
        >>> from transcription.schema import get_schema_version
        >>> version = get_schema_version("transcript-v2")
        >>> print(version)  # 2
    """
    return get_default_registry().get_schema_version(name)


def get_schema_hash(name: str) -> str:
    """Get the SHA256 hash of a schema.

    This is a convenience wrapper around SchemaRegistry.get_schema_hash().

    Args:
        name: The schema name.

    Returns:
        The hexadecimal SHA256 hash string.

    Raises:
        SchemaNotFoundError: If the schema does not exist.

    Example:
        >>> from transcription.schema import get_schema_hash
        >>> hash_value = get_schema_hash("transcript-v2")
        >>> print(hash_value)
    """
    return get_default_registry().get_schema_hash(name)


def list_schemas() -> list[str]:
    """List all available schema names.

    This is a convenience wrapper around SchemaRegistry.list_schemas().

    Returns:
        List of schema names available in the schemas directory.

    Example:
        >>> from transcription.schema import list_schemas
        >>> schemas = list_schemas()
        >>> print(schemas)  # ['transcript-v2', 'pr-dossier-v2']
    """
    return get_default_registry().list_schemas()


def migrate_data(
    data: dict[str, Any],
    from_version: str,
    to_version: str = "v2",
) -> dict[str, Any]:
    """Migrate data between schema versions.

    This is a convenience wrapper around MigrationRegistry.migrate().

    Args:
        data: The data to migrate.
        from_version: Source schema version.
        to_version: Target schema version (default: "v2").

    Returns:
        The migrated data.

    Raises:
        SchemaMigrationError: If migration fails.

    Example:
        >>> from transcription.schema import migrate_data
        >>> migrated = migrate_data(data, "v1", "v2")
    """
    return get_default_migration_registry().migrate(data, from_version, to_version)


# Add convenience functions to __all__
__all__.extend(
    [
        "get_schema",
        "get_schema_version",
        "get_schema_hash",
        "list_schemas",
        "migrate_data",
    ]
)
