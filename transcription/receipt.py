"""Receipt contract for transcript and benchmark provenance."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ._build_info import BUILD_ID, SOURCE_COMMIT
from .ids import generate_run_id as _generate_run_id
from .ids import is_valid_run_id

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
RECEIPT_CONTRACT_VERSION = 1
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")


def get_tool_version() -> str:
    """Get the installed package version or a source-tree fallback."""
    try:
        from . import __version__

        return __version__
    except ImportError:
        return "0.0.0-dev"


def get_git_commit() -> str | None:
    """Return only the source identity embedded in this package artifact.

    This compatibility name is retained for the existing receipt field. It no
    longer invokes git or inspects the caller's working directory.
    """
    if isinstance(SOURCE_COMMIT, str):
        candidate = SOURCE_COMMIT.strip().lower()
        if _COMMIT_PATTERN.fullmatch(candidate):
            return candidate
    return None


def get_build_id() -> str | None:
    if isinstance(BUILD_ID, str):
        candidate = BUILD_ID.strip()
        if candidate:
            return candidate
    return None


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute a deterministic 12-character SHA-256 projection."""
    normalized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    full_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return full_hash[:12]


def generate_run_id() -> str:
    return _generate_run_id()


@dataclass
class Receipt:
    """Provenance for one transcript or benchmark result."""

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
    build_id: str | None = None
    model_revision: str | None = None
    runtime_attempts: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
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
        if self.build_id is not None:
            result["build_id"] = self.build_id
        if self.model_revision is not None:
            result["model_revision"] = self.model_revision
        if self.runtime_attempts is not None:
            result["runtime_attempts"] = [dict(item) for item in self.runtime_attempts]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Receipt:
        attempts = data.get("runtime_attempts")
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
            build_id=data.get("build_id"),
            model_revision=data.get("model_revision"),
            runtime_attempts=(
                [dict(item) for item in attempts]
                if isinstance(attempts, list)
                else None
            ),
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
    model_revision: str | None = None,
    runtime_attempts: list[dict[str, Any]] | None = None,
) -> Receipt:
    """Build a receipt from actual runtime values and stable config."""
    from .models import SCHEMA_VERSION

    if config is None:
        config = {
            "model": model,
            "device": device,
            "compute_type": compute_type,
        }

    return Receipt(
        tool_version=get_tool_version(),
        schema_version=schema_version if schema_version is not None else SCHEMA_VERSION,
        model=model,
        device=device,
        compute_type=compute_type,
        config_hash=compute_config_hash(config),
        run_id=run_id if run_id is not None else generate_run_id(),
        created_at=(
            created_at
            if created_at is not None
            else datetime.now(UTC).isoformat().replace("+00:00", "Z")
        ),
        git_commit=get_git_commit() if include_git_commit else None,
        build_id=get_build_id(),
        model_revision=model_revision,
        runtime_attempts=(
            [dict(item) for item in runtime_attempts]
            if runtime_attempts is not None
            else None
        ),
    )


def validate_receipt(data: dict[str, Any]) -> list[str]:
    """Validate the receipt's required and optional public fields."""
    errors: list[str] = []
    missing = RECEIPT_REQUIRED_FIELDS - set(data.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")

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
            import uuid

            try:
                uuid.UUID(data["run_id"])
            except ValueError:
                errors.append(
                    "run_id must be in format 'run-YYYYMMDD-HHMMSS-XXXXXX' or valid UUID"
                )

    if "created_at" in data and not isinstance(data["created_at"], str):
        errors.append("created_at must be a string")

    commit = data.get("git_commit")
    if commit is not None:
        if not isinstance(commit, str):
            errors.append("git_commit must be a string or null")
        elif not _COMMIT_PATTERN.fullmatch(commit):
            errors.append("git_commit must be a hexadecimal source revision")

    for field_name in ("build_id", "model_revision"):
        value = data.get(field_name)
        if value is not None and not isinstance(value, str):
            errors.append(f"{field_name} must be a string or null")

    attempts = data.get("runtime_attempts")
    if attempts is not None:
        if not isinstance(attempts, list):
            errors.append("runtime_attempts must be an array or null")
        elif not all(isinstance(item, dict) for item in attempts):
            errors.append("runtime_attempts entries must be objects")

    return errors
