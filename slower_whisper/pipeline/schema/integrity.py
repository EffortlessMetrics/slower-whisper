"""Schema integrity verification and lockfile management.

This module provides functions for verifying schema integrity using SHA256
hashes and managing schema lockfiles. The lockfile stores expected hashes
for all schemas, enabling detection of unintended modifications.

The lockfile format is JSON with the following structure:
{
    "version": "1.0.0",
    "schemas": {
        "transcript-v2": {
            "version": 2,
            "hash": "abc123...",
            "path": "transcript-v2.schema.json"
        },
        ...
    }
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .exceptions import LockfileError, SchemaIntegrityError
from .registry import SchemaRegistry, get_default_registry


@dataclass
class SchemaLockEntry:
    """Entry for a single schema in the lockfile.

    Attributes:
        name: The schema name.
        version: The schema version number.
        hash: The SHA256 hash of the schema content.
        path: Relative path to the schema file.
        updated_at: ISO timestamp of when the hash was recorded.
    """

    name: str
    version: int
    hash: str
    path: str
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of this lock entry.
        """
        return {
            "name": self.name,
            "version": self.version,
            "hash": self.hash,
            "path": self.path,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaLockEntry:
        """Create from dictionary.

        Args:
            data: Dictionary containing lock entry data.

        Returns:
            A SchemaLockEntry instance.
        """
        return cls(
            name=data["name"],
            version=data["version"],
            hash=data["hash"],
            path=data["path"],
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class SchemaLockfile:
    """Schema lockfile containing hash information for all schemas.

    Attributes:
        version: The lockfile format version.
        schemas: Dictionary mapping schema names to SchemaLockEntry.
        created_at: ISO timestamp when lockfile was created.
        updated_at: ISO timestamp when lockfile was last updated.
    """

    version: str = "1.0.0"
    schemas: dict[str, SchemaLockEntry] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def add_schema(self, entry: SchemaLockEntry) -> None:
        """Add or update a schema entry.

        Args:
            entry: The schema lock entry to add.
        """
        self.schemas[entry.name] = entry
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def get_schema_entry(self, name: str) -> SchemaLockEntry | None:
        """Get a schema entry by name.

        Args:
            name: The schema name.

        Returns:
            The SchemaLockEntry if found, None otherwise.
        """
        return self.schemas.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of this lockfile.
        """
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schemas": {name: entry.to_dict() for name, entry in self.schemas.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaLockfile:
        """Create from dictionary.

        Args:
            data: Dictionary containing lockfile data.

        Returns:
            A SchemaLockfile instance.
        """
        schemas_data = data.get("schemas", {})
        schemas = {
            name: SchemaLockEntry.from_dict(entry_data) for name, entry_data in schemas_data.items()
        }

        return cls(
            version=data.get("version", "1.0.0"),
            schemas=schemas,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


def load_lockfile(path: Path | str) -> SchemaLockfile:
    """Load a schema lockfile from disk.

    Args:
        path: Path to the lockfile.

    Returns:
        The loaded SchemaLockfile.

    Raises:
        LockfileError: If the lockfile cannot be loaded or parsed.

    Example:
        >>> from transcription.schema.integrity import load_lockfile
        >>> lockfile = load_lockfile("schema-lock.json")
    """
    lockfile_path = Path(path)

    if not lockfile_path.exists():
        raise LockfileError(f"Lockfile not found: {lockfile_path}")

    try:
        with open(lockfile_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise LockfileError(f"Invalid JSON in lockfile: {e}") from e
    except OSError as e:
        raise LockfileError(f"Failed to read lockfile: {e}") from e

    try:
        return SchemaLockfile.from_dict(data)
    except (KeyError, TypeError) as e:
        raise LockfileError(f"Invalid lockfile format: {e}") from e


def save_lockfile(lockfile: SchemaLockfile, path: Path | str) -> None:
    """Save a schema lockfile to disk.

    Args:
        lockfile: The SchemaLockfile to save.
        path: Path where the lockfile should be saved.

    Raises:
        LockfileError: If the lockfile cannot be saved.

    Example:
        >>> from transcription.schema.integrity import save_lockfile, SchemaLockfile
        >>> lockfile = SchemaLockfile()
        >>> save_lockfile(lockfile, "schema-lock.json")
    """
    lockfile_path = Path(path)

    # Update timestamp
    lockfile.updated_at = datetime.utcnow().isoformat() + "Z"

    try:
        # Create parent directories if needed
        lockfile_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lockfile_path, "w", encoding="utf-8") as f:
            json.dump(lockfile.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")  # Add trailing newline
    except OSError as e:
        raise LockfileError(f"Failed to save lockfile: {e}") from e


def verify_schema_integrity(
    schema_name: str,
    expected_hash: str,
    registry: SchemaRegistry | None = None,
) -> bool:
    """Verify that a schema's hash matches the expected value.

    Args:
        schema_name: The name of the schema to verify.
        expected_hash: The expected SHA256 hash.
        registry: Optional SchemaRegistry instance. If None, uses default.

    Returns:
        True if the hash matches, False otherwise.

    Raises:
        SchemaNotFoundError: If the schema does not exist.

    Example:
        >>> from transcription.schema.integrity import verify_schema_integrity
        >>> is_valid = verify_schema_integrity("transcript-v2", "abc123...")
    """
    if registry is None:
        registry = get_default_registry()

    actual_hash = registry.get_schema_hash(schema_name)
    return actual_hash == expected_hash


def verify_all_schema_integrity(
    lockfile: SchemaLockfile,
    registry: SchemaRegistry | None = None,
) -> dict[str, bool]:
    """Verify integrity of all schemas in a lockfile.

    Args:
        lockfile: The SchemaLockfile containing expected hashes.
        registry: Optional SchemaRegistry instance. If None, uses default.

    Returns:
        Dictionary mapping schema names to verification results.

    Example:
        >>> from transcription.schema.integrity import load_lockfile, verify_all_schema_integrity
        >>> lockfile = load_lockfile("schema-lock.json")
        >>> results = verify_all_schema_integrity(lockfile)
        >>> for name, is_valid in results.items():
        ...     print(f"{name}: {'OK' if is_valid else 'FAILED'}")
    """
    if registry is None:
        registry = get_default_registry()

    results = {}
    for name, entry in lockfile.schemas.items():
        try:
            results[name] = verify_schema_integrity(name, entry.hash, registry)
        except Exception:
            results[name] = False

    return results


def verify_schema_integrity_strict(
    schema_name: str,
    expected_hash: str,
    registry: SchemaRegistry | None = None,
) -> None:
    """Verify schema integrity, raising an exception on mismatch.

    Args:
        schema_name: The name of the schema to verify.
        expected_hash: The expected SHA256 hash.
        registry: Optional SchemaRegistry instance. If None, uses default.

    Raises:
        SchemaIntegrityError: If the hash does not match.
        SchemaNotFoundError: If the schema does not exist.

    Example:
        >>> from transcription.schema.integrity import verify_schema_integrity_strict
        >>> verify_schema_integrity_strict("transcript-v2", "abc123...")
    """
    if registry is None:
        registry = get_default_registry()

    actual_hash = registry.get_schema_hash(schema_name)

    if actual_hash != expected_hash:
        raise SchemaIntegrityError(schema_name, expected_hash, actual_hash)


def verify_all_schema_integrity_strict(
    lockfile: SchemaLockfile,
    registry: SchemaRegistry | None = None,
) -> None:
    """Verify all schemas in a lockfile, raising on first failure.

    Args:
        lockfile: The SchemaLockfile containing expected hashes.
        registry: Optional SchemaRegistry instance. If None, uses default.

    Raises:
        SchemaIntegrityError: If any schema hash does not match.

    Example:
        >>> from transcription.schema.integrity import load_lockfile, verify_all_schema_integrity_strict
        >>> lockfile = load_lockfile("schema-lock.json")
        >>> verify_all_schema_integrity_strict(lockfile)
    """
    if registry is None:
        registry = get_default_registry()

    for name, entry in lockfile.schemas.items():
        verify_schema_integrity_strict(name, entry.hash, registry)


def create_lockfile(
    registry: SchemaRegistry | None = None,
    schema_names: list[str] | None = None,
) -> SchemaLockfile:
    """Create a new lockfile from the current schema registry.

    Args:
        registry: Optional SchemaRegistry instance. If None, uses default.
        schema_names: Optional list of schema names to include. If None,
            includes all available schemas.

    Returns:
        A new SchemaLockfile with current schema hashes.

    Example:
        >>> from transcription.schema.integrity import create_lockfile, save_lockfile
        >>> lockfile = create_lockfile()
        >>> save_lockfile(lockfile, "schema-lock.json")
    """
    if registry is None:
        registry = get_default_registry()

    lockfile = SchemaLockfile()

    # Determine which schemas to include
    if schema_names is None:
        schema_names = registry.list_schemas()

    # Add each schema to the lockfile
    for name in schema_names:
        try:
            schema_info = registry.get_schema(name)
            entry = SchemaLockEntry(
                name=name,
                version=schema_info.version,
                hash=schema_info.hash,
                path=str(schema_info.path.relative_to(registry._schemas_dir)),
            )
            lockfile.add_schema(entry)
        except Exception:
            # Skip schemas that can't be loaded
            continue

    return lockfile


def update_lockfile(
    lockfile: SchemaLockfile,
    schema_names: list[str] | None = None,
    registry: SchemaRegistry | None = None,
) -> SchemaLockfile:
    """Update an existing lockfile with current schema hashes.

    Args:
        lockfile: The existing SchemaLockfile to update.
        schema_names: Optional list of schema names to update. If None,
            updates all available schemas.
        registry: Optional SchemaRegistry instance. If None, uses default.

    Returns:
        The updated SchemaLockfile.

    Example:
        >>> from transcription.schema.integrity import load_lockfile, update_lockfile, save_lockfile
        >>> lockfile = load_lockfile("schema-lock.json")
        >>> lockfile = update_lockfile(lockfile)
        >>> save_lockfile(lockfile, "schema-lock.json")
    """
    if registry is None:
        registry = get_default_registry()

    # Determine which schemas to update
    if schema_names is None:
        schema_names = registry.list_schemas()

    # Update each schema in the lockfile
    for name in schema_names:
        try:
            schema_info = registry.get_schema(name)
            entry = SchemaLockEntry(
                name=name,
                version=schema_info.version,
                hash=schema_info.hash,
                path=str(schema_info.path.relative_to(registry._schemas_dir)),
            )
            lockfile.add_schema(entry)
        except Exception:
            # Skip schemas that can't be loaded
            continue

    return lockfile
