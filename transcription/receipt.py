"""Receipt contract for provenance tracking.

This module provides a standardized receipt schema for capturing provenance
information in transcript and benchmark outputs. Receipts enable:

- Reproducibility: Config hash and git commit allow recreating runs
- Traceability: run_id uniquely identifies each execution
- Versioning: tool_version and schema_version track compatibility

Contract fields (all required unless noted):
- tool_version: Package version (e.g., "2.1.0")
- schema_version: JSON schema version (int, e.g., 2)
- model: ASR model name (e.g., "large-v3")
- device: Resolved device (e.g., "cuda", "cpu")
- compute_type: Compute type used (e.g., "float16", "int8")
- config_hash: SHA-256 hash of normalized config (first 12 chars)
- run_id: Unique identifier for this execution (format: run-YYYYMMDD-HHMMSS-XXXXXX)
- created_at: ISO 8601 timestamp when receipt was created
- git_commit: Optional short git commit hash (7-12 chars)

Example receipt:
    {
        "tool_version": "2.1.0",
        "schema_version": 2,
        "model": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "config_hash": "a1b2c3d4e5f6",
        "run_id": "run-20260128-143052-x7k9p2",
        "created_at": "2024-01-15T10:30:00Z",
        "git_commit": "abc1234"
    }
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .ids import generate_run_id as _generate_run_id
from .ids import is_valid_run_id

# Required fields in a receipt (for validation)
RECEIPT_REQUIRED_FIELDS = frozenset(
    {
        "tool_version",
        "schema_version",
        "model",
        "device",
        "compute_type",
        "config_hash",
        "run_id",
        "created_at",
    }
)

# Receipt schema version for the receipt contract itself
RECEIPT_CONTRACT_VERSION = 1


def get_tool_version() -> str:
    """Get the current tool version from package metadata.

    Returns:
        Version string (e.g., "2.1.0") or "0.0.0-dev" if not installed.
    """
    try:
        from . import __version__

        return __version__
    except ImportError:
        return "0.0.0-dev"


def get_git_commit() -> str | None:
    """Get the current short git commit hash if in a git repository.

    Returns:
        Short commit hash (e.g., "abc1234") or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=None,  # Use current working directory
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute a deterministic hash from a configuration dictionary.

    The config is normalized by sorting keys and using consistent JSON
    serialization to ensure the same config always produces the same hash.

    Args:
        config: Configuration dictionary to hash.

    Returns:
        First 12 characters of the SHA-256 hash.
    """
    # Normalize by sorting keys and using consistent serialization
    normalized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    full_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return full_hash[:12]


def generate_run_id() -> str:
    """Generate a unique run identifier.

    Format: `run-YYYYMMDD-HHMMSS-XXXXXX` where XXXXXX is 6 random alphanumeric chars.

    Returns:
        A unique run_id string (e.g., "run-20260128-143052-x7k9p2").
    """
    return _generate_run_id()


@dataclass
class Receipt:
    """Provenance receipt for transcript and benchmark outputs.

    This dataclass captures all the information needed to understand
    how a transcript or benchmark result was produced.

    Attributes:
        tool_version: Package version (e.g., "2.1.0")
        schema_version: JSON schema version (int)
        model: ASR model name
        device: Resolved device (cuda/cpu)
        compute_type: Compute type used
        config_hash: Hash of normalized config
        run_id: Unique execution identifier
        created_at: ISO 8601 timestamp
        git_commit: Optional git commit hash
    """

    tool_version: str
    schema_version: int
    model: str
    device: str
    compute_type: str
    config_hash: str
    run_id: str = field(default_factory=generate_run_id)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert receipt to a JSON-serializable dictionary.

        Returns:
            Dictionary with all receipt fields. git_commit is only
            included if it has a value.
        """
        result: dict[str, Any] = {
            "tool_version": self.tool_version,
            "schema_version": self.schema_version,
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "config_hash": self.config_hash,
            "run_id": self.run_id,
            "created_at": self.created_at,
        }
        if self.git_commit is not None:
            result["git_commit"] = self.git_commit
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Receipt:
        """Create a Receipt from a dictionary.

        Args:
            data: Dictionary with receipt fields.

        Returns:
            Receipt instance.

        Raises:
            KeyError: If required fields are missing.
        """
        return cls(
            tool_version=data["tool_version"],
            schema_version=data["schema_version"],
            model=data["model"],
            device=data["device"],
            compute_type=data["compute_type"],
            config_hash=data["config_hash"],
            run_id=data.get("run_id", generate_run_id()),
            created_at=data.get(
                "created_at",
                datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            ),
            git_commit=data.get("git_commit"),
        )


def build_receipt(
    *,
    model: str,
    device: str,
    compute_type: str,
    config: dict[str, Any] | None = None,
    schema_version: int | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
    include_git_commit: bool = True,
) -> Receipt:
    """Build a receipt with the given parameters.

    This is the primary factory function for creating receipts. It handles
    version detection, config hashing, and optional git commit lookup.

    Args:
        model: ASR model name (e.g., "large-v3")
        device: Resolved device (e.g., "cuda", "cpu")
        compute_type: Compute type (e.g., "float16", "int8")
        config: Optional config dict for hashing. If None, uses model/device/compute_type.
        schema_version: Override schema version. If None, uses SCHEMA_VERSION from models.
        run_id: Override run_id. If None, generates a new UUID4.
        created_at: Override created_at. If None, uses current UTC time.
        include_git_commit: Whether to look up and include git commit.

    Returns:
        Receipt instance with all fields populated.

    Example:
        >>> receipt = build_receipt(
        ...     model="large-v3",
        ...     device="cuda",
        ...     compute_type="float16",
        ... )
        >>> print(receipt.to_dict())
    """
    from .models import SCHEMA_VERSION

    # Build config for hashing if not provided
    if config is None:
        config = {
            "model": model,
            "device": device,
            "compute_type": compute_type,
        }

    # Compute deterministic hash
    config_hash = compute_config_hash(config)

    # Get tool version
    tool_version = get_tool_version()

    # Get git commit if requested
    git_commit = get_git_commit() if include_git_commit else None

    # Use provided or default values
    actual_schema_version = schema_version if schema_version is not None else SCHEMA_VERSION
    actual_run_id = run_id if run_id is not None else generate_run_id()
    actual_created_at = (
        created_at
        if created_at is not None
        else datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )

    return Receipt(
        tool_version=tool_version,
        schema_version=actual_schema_version,
        model=model,
        device=device,
        compute_type=compute_type,
        config_hash=config_hash,
        run_id=actual_run_id,
        created_at=actual_created_at,
        git_commit=git_commit,
    )


def validate_receipt(data: dict[str, Any]) -> list[str]:
    """Validate that a receipt dictionary has all required fields.

    Args:
        data: Dictionary to validate as a receipt.

    Returns:
        List of validation error messages. Empty list if valid.
    """
    errors: list[str] = []

    # Check required fields
    missing = RECEIPT_REQUIRED_FIELDS - set(data.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")

    # Validate types for present fields
    if "tool_version" in data and not isinstance(data["tool_version"], str):
        errors.append("tool_version must be a string")

    if "schema_version" in data and not isinstance(data["schema_version"], int):
        errors.append("schema_version must be an integer")

    if "model" in data and not isinstance(data["model"], str):
        errors.append("model must be a string")

    if "device" in data and not isinstance(data["device"], str):
        errors.append("device must be a string")

    if "compute_type" in data and not isinstance(data["compute_type"], str):
        errors.append("compute_type must be a string")

    if "config_hash" in data:
        if not isinstance(data["config_hash"], str):
            errors.append("config_hash must be a string")
        elif len(data["config_hash"]) != 12:
            errors.append("config_hash must be exactly 12 characters")

    if "run_id" in data:
        if not isinstance(data["run_id"], str):
            errors.append("run_id must be a string")
        elif not is_valid_run_id(data["run_id"]):
            # Allow legacy UUID format for backward compatibility
            import uuid

            try:
                uuid.UUID(data["run_id"])
            except ValueError:
                errors.append("run_id must be in format 'run-YYYYMMDD-HHMMSS-XXXXXX' or valid UUID")

    if "created_at" in data and not isinstance(data["created_at"], str):
        errors.append("created_at must be a string")

    # git_commit is optional but must be string if present
    if "git_commit" in data and data["git_commit"] is not None:
        if not isinstance(data["git_commit"], str):
            errors.append("git_commit must be a string or null")

    return errors
