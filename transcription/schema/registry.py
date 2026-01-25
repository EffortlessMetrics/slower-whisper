"""Schema registry with hash tracking and version management.

This module provides the SchemaRegistry class for tracking loaded JSON schemas,
computing SHA256 hashes for integrity verification, and managing schema versions.

The registry caches loaded schemas to avoid repeated file I/O and provides
change detection by comparing computed hashes against stored values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import (
    SchemaNotFoundError,
    SchemaRegistryError,
    SchemaVersionError,
)


@dataclass
class SchemaInfo:
    """Metadata about a registered schema.

    Attributes:
        name: The schema name (e.g., "transcript-v2").
        version: The schema version number.
        path: Filesystem path to the schema JSON file.
        hash: SHA256 hash of the schema content.
        content: The loaded schema dictionary.
    """

    name: str
    version: int
    path: Path
    hash: str
    content: dict[str, Any]


class SchemaRegistry:
    """Registry for managing JSON schemas with hash tracking.

    The SchemaRegistry loads schema JSON files from the schemas directory,
    computes SHA256 hashes for integrity verification, and caches loaded
    schemas to avoid repeated file I/O operations.

    Example:
        >>> from transcription.schema.registry import SchemaRegistry
        >>> registry = SchemaRegistry()
        >>> schema_info = registry.get_schema("transcript-v2")
        >>> print(f"Schema hash: {schema_info.hash}")
    """

    # Default schemas directory relative to this module
    DEFAULT_SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"

    # Known schema names and their expected versions
    KNOWN_SCHEMAS = {
        "transcript-v2": 2,
        "pr-dossier-v2": 2,
    }

    def __init__(self, schemas_dir: Path | str | None = None) -> None:
        """Initialize the schema registry.

        Args:
            schemas_dir: Optional path to the schemas directory. If None,
                uses the default schemas directory (transcription/schemas/).

        Raises:
            SchemaRegistryError: If the schemas directory does not exist.
        """
        if schemas_dir is None:
            self._schemas_dir = self.DEFAULT_SCHEMAS_DIR
        else:
            self._schemas_dir = Path(schemas_dir)

        if not self._schemas_dir.exists():
            raise SchemaRegistryError(f"Schemas directory does not exist: {self._schemas_dir}")

        # Cache for loaded schemas
        self._cache: dict[str, SchemaInfo] = {}

    @staticmethod
    def compute_hash(content: dict[str, Any] | str | bytes) -> str:
        """Compute SHA256 hash of schema content.

        Args:
            content: The schema content as a dict, JSON string, or bytes.

        Returns:
            Hexadecimal SHA256 hash string.

        Example:
            >>> schema = {"type": "object", "properties": {}}
            >>> hash_value = SchemaRegistry.compute_hash(schema)
            >>> print(hash_value)
            'a1b2c3d4...'
        """
        if isinstance(content, dict):
            # Normalize JSON for consistent hashing
            json_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
            content_bytes = json_str.encode("utf-8")
        elif isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        return hashlib.sha256(content_bytes).hexdigest()

    def get_schema(self, name: str) -> SchemaInfo:
        """Get a schema by name, loading and caching if necessary.

        Args:
            name: The schema name (e.g., "transcript-v2").

        Returns:
            SchemaInfo containing the schema metadata and content.

        Raises:
            SchemaNotFoundError: If the schema file does not exist.
            SchemaRegistryError: If the schema file cannot be parsed.
            SchemaVersionError: If the schema version is unexpected.

        Example:
            >>> registry = SchemaRegistry()
            >>> info = registry.get_schema("transcript-v2")
            >>> print(f"Version: {info.version}")
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Construct schema file path
        schema_path = self._schemas_dir / f"{name}.schema.json"

        if not schema_path.exists():
            raise SchemaNotFoundError(name)

        # Load and parse schema
        try:
            with open(schema_path, encoding="utf-8") as f:
                content = json.load(f)
        except json.JSONDecodeError as e:
            raise SchemaRegistryError(f"Failed to parse schema file {schema_path}: {e}") from e
        except OSError as e:
            raise SchemaRegistryError(f"Failed to read schema file {schema_path}: {e}") from e

        # Extract schema version
        # For JSON Schema files, the version is typically in the title or
        # in the const value of the schema_version property definition
        version = content.get("schema_version", 1)

        # If schema_version is not at top level, try to extract from title or properties
        if version == 1:
            # Try extracting from title (e.g., "Schema v2")
            title = content.get("title", "")
            if "v2" in title.lower():
                version = 2
            elif "v1" in title.lower():
                version = 1
            else:
                # Try extracting from schema_version property definition
                properties = content.get("properties", {})
                schema_version_prop = properties.get("schema_version", {})
                const_val = schema_version_prop.get("const")
                if const_val is not None:
                    version = int(const_val)

        # Ensure version is an integer
        if isinstance(version, int):
            pass  # Already an integer
        elif isinstance(version, str):
            try:
                version = int(version)
            except ValueError as e:
                raise SchemaVersionError(
                    str(version),
                    [f"{k} v{v}" for k, v in self.KNOWN_SCHEMAS.items()],
                ) from e
        else:
            raise SchemaVersionError(
                str(version),
                [f"{k} v{v}" for k, v in self.KNOWN_SCHEMAS.items()],
            )

        # Validate version for known schemas
        if name in self.KNOWN_SCHEMAS:
            expected_version = self.KNOWN_SCHEMAS[name]
            if version != expected_version:
                raise SchemaVersionError(
                    f"{name} v{version}",
                    [f"{name} v{expected_version}"],
                )

        # Compute hash and create SchemaInfo
        schema_hash = self.compute_hash(content)
        schema_info = SchemaInfo(
            name=name,
            version=version,
            path=schema_path,
            hash=schema_hash,
            content=content,
        )

        # Cache and return
        self._cache[name] = schema_info
        return schema_info

    def get_schema_version(self, name: str) -> int:
        """Get the version number of a schema.

        Args:
            name: The schema name.

        Returns:
            The schema version number.

        Raises:
            SchemaNotFoundError: If the schema does not exist.

        Example:
            >>> registry = SchemaRegistry()
            >>> version = registry.get_schema_version("transcript-v2")
            >>> print(version)  # 2
        """
        schema_info = self.get_schema(name)
        return schema_info.version

    def get_schema_hash(self, name: str) -> str:
        """Get the SHA256 hash of a schema.

        Args:
            name: The schema name.

        Returns:
            The hexadecimal SHA256 hash string.

        Raises:
            SchemaNotFoundError: If the schema does not exist.

        Example:
            >>> registry = SchemaRegistry()
            >>> hash_value = registry.get_schema_hash("transcript-v2")
            >>> print(hash_value)
            'a1b2c3d4...'
        """
        schema_info = self.get_schema(name)
        return schema_info.hash

    def list_schemas(self) -> list[str]:
        """List all available schema names.

        Returns:
            List of schema names available in the schemas directory.

        Example:
            >>> registry = SchemaRegistry()
            >>> schemas = registry.list_schemas()
            >>> print(schemas)  # ['transcript-v2', 'pr-dossier-v2']
        """
        schemas = []
        for path in self._schemas_dir.glob("*.schema.json"):
            # Remove .schema suffix from filename stem
            name = path.stem
            if name.endswith(".schema"):
                name = name[:-7]  # Remove ".schema" suffix
            schemas.append(name)
        return sorted(schemas)

    def clear_cache(self) -> None:
        """Clear the schema cache.

        This forces subsequent calls to get_schema() to reload
        schema files from disk.

        Example:
            >>> registry = SchemaRegistry()
            >>> registry.clear_cache()
        """
        self._cache.clear()

    def has_changed(self, name: str, expected_hash: str) -> bool:
        """Check if a schema has changed compared to an expected hash.

        Args:
            name: The schema name.
            expected_hash: The expected SHA256 hash.

        Returns:
            True if the schema hash differs from expected, False otherwise.

        Raises:
            SchemaNotFoundError: If the schema does not exist.

        Example:
            >>> registry = SchemaRegistry()
            >>> if registry.has_changed("transcript-v2", "abc123..."):
            ...     print("Schema has been modified!")
        """
        actual_hash = self.get_schema_hash(name)
        return actual_hash != expected_hash

    def get_all_schema_info(self) -> dict[str, SchemaInfo]:
        """Get info for all available schemas.

        Returns:
            Dictionary mapping schema names to SchemaInfo objects.

        Example:
            >>> registry = SchemaRegistry()
            >>> all_info = registry.get_all_schema_info()
            >>> for name, info in all_info.items():
            ...     print(f"{name}: v{info.version}")
        """
        result = {}
        for name in self.list_schemas():
            result[name] = self.get_schema(name)
        return result


# Global registry instance for convenience
_default_registry: SchemaRegistry | None = None


def get_default_registry() -> SchemaRegistry:
    """Get or create the default global schema registry.

    Returns:
        The global SchemaRegistry instance.

    Example:
        >>> from transcription.schema.registry import get_default_registry
        >>> registry = get_default_registry()
        >>> schema = registry.get_schema("transcript-v2")
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = SchemaRegistry()
    return _default_registry
