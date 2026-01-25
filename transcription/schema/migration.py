"""Migration utilities for schema version transitions.

This module provides tools for migrating data between schema versions,
with specific support for v1 to v2 transcript migration. The module
includes version detection, migration path resolution, and automated
migration execution.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .exceptions import SchemaMigrationError, SchemaVersionError


@dataclass
class MigrationPath:
    """Represents a migration path between schema versions.

    Attributes:
        from_version: Source schema version.
        to_version: Target schema version.
        migrate_func: Function that performs the migration.
        description: Human-readable description of the migration.
    """

    from_version: str
    to_version: str
    migrate_func: Callable[[dict[str, Any]], dict[str, Any]]
    description: str


class MigrationRegistry:
    """Registry for available schema migrations.

    The MigrationRegistry tracks available migration paths between
    schema versions and can execute migrations automatically.

    Example:
        >>> from transcription.schema.migration import MigrationRegistry
        >>> registry = MigrationRegistry()
        >>> migrated = registry.migrate(data, "v1", "v2")
    """

    def __init__(self) -> None:
        """Initialize the migration registry."""
        self._migrations: dict[tuple[str, str], MigrationPath] = {}
        self._register_builtin_migrations()

    def _register_builtin_migrations(self) -> None:
        """Register built-in migrations."""
        # Register v1 to v2 migration
        self.register(
            MigrationPath(
                from_version="v1",
                to_version="v2",
                migrate_func=migrate_v1_to_v2,
                description="Migrate transcript from v1 to v2 format",
            )
        )

    def register(self, path: MigrationPath) -> None:
        """Register a migration path.

        Args:
            path: The MigrationPath to register.

        Raises:
            SchemaMigrationError: If a migration for this path already exists.
        """
        key = (path.from_version, path.to_version)
        if key in self._migrations:
            raise SchemaMigrationError(
                path.from_version,
                path.to_version,
                "Migration path already registered",
            )
        self._migrations[key] = path

    def get_migration(self, from_version: str, to_version: str) -> MigrationPath | None:
        """Get a migration path between versions.

        Args:
            from_version: Source schema version.
            to_version: Target schema version.

        Returns:
            The MigrationPath if available, None otherwise.

        Example:
            >>> registry = MigrationRegistry()
            >>> path = registry.get_migration("v1", "v2")
            >>> if path:
            ...     print(path.description)
        """
        return self._migrations.get((from_version, to_version))

    def migrate(
        self,
        data: dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> dict[str, Any]:
        """Migrate data from one schema version to another.

        Args:
            data: The data to migrate.
            from_version: Source schema version.
            to_version: Target schema version.

        Returns:
            The migrated data.

        Raises:
            SchemaMigrationError: If no migration path exists or migration fails.

        Example:
            >>> registry = MigrationRegistry()
            >>> migrated = registry.migrate(data, "v1", "v2")
        """
        if from_version == to_version:
            # No migration needed
            return copy.deepcopy(data)

        path = self.get_migration(from_version, to_version)
        if path is None:
            raise SchemaMigrationError(
                from_version,
                to_version,
                "No migration path available",
            )

        try:
            return path.migrate_func(data)
        except Exception as e:
            raise SchemaMigrationError(
                from_version,
                to_version,
                f"Migration function failed: {e}",
            ) from e

    def list_migrations(self) -> list[MigrationPath]:
        """List all registered migrations.

        Returns:
            List of MigrationPath objects.

        Example:
            >>> registry = MigrationRegistry()
            >>> for path in registry.list_migrations():
            ...     print(f"{path.from_version} -> {path.to_version}")
        """
        return list(self._migrations.values())


def detect_schema_version(data: dict[str, Any]) -> str:
    """Detect the schema version of transcript data.

    Args:
        data: The transcript data to analyze.

    Returns:
        The detected schema version ("v1" or "v2").

    Raises:
        SchemaVersionError: If the version cannot be determined.

    Example:
        >>> version = detect_schema_version(data)
        >>> print(version)  # "v2"
    """
    # Check for explicit schema_version field (v2+)
    if "schema_version" in data:
        version = data["schema_version"]
        if isinstance(version, int):
            return f"v{version}"
        elif isinstance(version, str):
            if version.startswith("v"):
                return version
            return f"v{version}"

    # Check for v1 indicators
    # v1 had "file" field but no "schema_version"
    if "file" in data and "segments" in data:
        # v1 format
        return "v1"

    # Cannot determine version
    raise SchemaVersionError(
        "unknown",
        ["v1", "v2"],
    )


def migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate transcript data from v1 to v2 format.

    This function converts v1 transcript data to v2 format by:
    1. Adding schema_version field set to 2
    2. Renaming "file" to "file_name" (keeping "file" for backward compat)
    3. Ensuring all required v2 fields are present
    4. Converting optional fields to v2 structure

    Args:
        data: The v1 transcript data.

    Returns:
        The migrated v2 transcript data.

    Example:
        >>> v1_data = {"file": "test.wav", "language": "en", "segments": []}
        >>> v2_data = migrate_v1_to_v2(v1_data)
        >>> print(v2_data["schema_version"])  # 2
    """
    # Create a deep copy to avoid modifying the original
    migrated = copy.deepcopy(data)

    # Set schema version
    migrated["schema_version"] = 2

    # Handle file field: v1 used "file", v2 uses "file_name"
    # Keep both for backward compatibility
    if "file" in migrated and "file_name" not in migrated:
        migrated["file_name"] = migrated["file"]

    # Ensure required fields exist
    if "segments" not in migrated:
        migrated["segments"] = []

    if "language" not in migrated:
        migrated["language"] = "en"

    # Ensure optional fields are properly typed
    optional_fields = ["meta", "annotations", "speakers", "turns", "speaker_stats", "chunks"]
    for field in optional_fields:
        if field not in migrated:
            migrated[field] = None

    # Migrate segments if needed
    if "segments" in migrated:
        for segment in migrated["segments"]:
            # Ensure segment has required fields
            if "id" not in segment:
                segment["id"] = 0
            if "start" not in segment:
                segment["start"] = 0.0
            if "end" not in segment:
                segment["end"] = 0.0
            if "text" not in segment:
                segment["text"] = ""

            # Ensure optional segment fields exist
            optional_segment_fields = ["speaker", "tone", "audio_state", "words"]
            for field in optional_segment_fields:
                if field not in segment:
                    segment[field] = None

    # Migrate meta.diarization if present
    if "meta" in migrated and isinstance(migrated["meta"], dict):
        meta = migrated["meta"]
        if "diarization" in meta:
            diarization = meta["diarization"]
            if isinstance(diarization, dict):
                # Ensure diarization has required fields
                if "requested" not in diarization:
                    diarization["requested"] = False
                if "status" not in diarization:
                    diarization["status"] = "disabled"

    return migrated


def auto_migrate(data: dict[str, Any], target_version: str = "v2") -> dict[str, Any]:
    """Automatically migrate data to the target schema version.

    This function detects the current version of the data and migrates
    it to the target version using available migration paths.

    Args:
        data: The data to migrate.
        target_version: The target schema version (default: "v2").

    Returns:
        The migrated data.

    Raises:
        SchemaMigrationError: If migration fails.
        SchemaVersionError: If version cannot be determined.

    Example:
        >>> from transcription.schema.migration import auto_migrate
        >>> migrated = auto_migrate(data, "v2")
    """
    current_version = detect_schema_version(data)

    if current_version == target_version:
        return copy.deepcopy(data)

    registry = MigrationRegistry()
    return registry.migrate(data, current_version, target_version)


# Global registry instance for convenience
_default_registry: MigrationRegistry | None = None


def get_default_migration_registry() -> MigrationRegistry:
    """Get or create the default global migration registry.

    Returns:
        The global MigrationRegistry instance.

    Example:
        >>> from transcription.schema.migration import get_default_migration_registry
        >>> registry = get_default_migration_registry()
        >>> migrated = registry.migrate(data, "v1", "v2")
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = MigrationRegistry()
    return _default_registry
