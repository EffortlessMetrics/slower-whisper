"""Receipt contract for transcript provenance and runtime evidence.

Receipts identify the installed package artifact and the actual selected ASR
runtime. Runtime provenance is package-local: this module never invokes git and
never inspects the caller's working directory or environment for source identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .build_info import get_build_id, get_source_commit
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
RECEIPT_VOLATILE_FIELDS = frozenset({"run_id", "created_at"})
RECEIPT_CONTRACT_VERSION = 1

_ATTEMPT_FIELDS = ("device", "compute_type", "outcome", "reason_code")
_ALLOWED_ATTEMPT_OUTCOMES = frozenset({"failed", "selected"})
_GIT_COMMIT_PATTERN = r"^[0-9a-f]{7,64}$"
_BUILD_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_REASON_CODE_PATTERN = re.compile(r"^[a-z0-9_]+$")


def get_tool_version() -> str:
    """Return the installed package version, or a source-tree development value."""
    try:
        from . import __version__

        return __version__
    except ImportError:
        return "0.0.0-dev"


def get_git_commit() -> str | None:
    """Return the trusted package source commit embedded during artifact build.

    The historical function name is retained for API compatibility. It no longer
    shells out to git and cannot cite the repository containing the caller's cwd.
    """
    return get_source_commit()


def _normalize_config_value(value: object) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("receipt config values must be finite")
        return value
    if isinstance(value, Enum):
        return _normalize_config_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_config_value(asdict(value))
    if isinstance(value, Mapping):
        items: list[tuple[str, object]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("receipt config keys must be strings")
            items.append((key, item))
        return {
            key: _normalize_config_value(item)
            for key, item in sorted(items, key=lambda entry: entry[0])
        }
    if isinstance(value, list | tuple):
        return [_normalize_config_value(item) for item in value]
    if isinstance(value, set | frozenset):
        normalized_items = [_normalize_config_value(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _normalize_config_value(to_dict())
    raise TypeError(
        f"receipt config value {type(value).__name__!r} is not canonically serializable"
    )


def compute_config_hash(config: Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 projection of canonical config data."""
    normalized = _normalize_config_value(config)
    serialized = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def generate_run_id() -> str:
    """Generate a unique run identifier."""
    return _generate_run_id()


def normalize_model_load_attempts(
    attempts: Sequence[object] | None,
) -> list[dict[str, str]]:
    """Project backend attempts onto the bounded public receipt contract."""
    normalized: list[dict[str, str]] = []
    for raw_attempt in attempts or ():
        if not isinstance(raw_attempt, Mapping):
            continue
        attempt: dict[str, str] = {}
        for field_name in _ATTEMPT_FIELDS:
            value = raw_attempt.get(field_name)
            if value is None or isinstance(value, bool):
                continue
            candidate = str(value).strip()
            if candidate:
                attempt[field_name] = candidate
        if not all(field_name in attempt for field_name in _ATTEMPT_FIELDS):
            continue
        if attempt["outcome"] not in _ALLOWED_ATTEMPT_OUTCOMES:
            continue
        if _REASON_CODE_PATTERN.fullmatch(attempt["reason_code"]) is None:
            continue
        normalized.append(attempt)
    return normalized


@dataclass
class Receipt:
    """Provenance receipt attached to a successful transcript."""

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
    backend: str | None = None
    model_revision: str | None = None
    model_load_attempts: list[dict[str, str]] = field(default_factory=list)
    git_commit: str | None = None
    build_id: str | None = None

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
        if self.backend is not None:
            result["backend"] = self.backend
        if self.model_revision is not None:
            result["model_revision"] = self.model_revision
        if self.model_load_attempts:
            result["model_load_attempts"] = [dict(attempt) for attempt in self.model_load_attempts]
        if self.git_commit is not None:
            result["git_commit"] = self.git_commit
        if self.build_id is not None:
            result["build_id"] = self.build_id
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Receipt:
        raw_attempts = data.get("model_load_attempts")
        attempts = (
            raw_attempts
            if isinstance(raw_attempts, Sequence)
            and not isinstance(raw_attempts, str | bytes | bytearray)
            else None
        )
        return cls(
            tool_version=str(data["tool_version"]),
            schema_version=int(data["schema_version"]),
            model=str(data["model"]),
            device=str(data["device"]),
            compute_type=str(data["compute_type"]),
            config_hash=str(data["config_hash"]),
            run_id=str(data.get("run_id", generate_run_id())),
            created_at=str(
                data.get(
                    "created_at",
                    datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                )
            ),
            backend=_optional_string(data.get("backend")),
            model_revision=_optional_string(data.get("model_revision")),
            model_load_attempts=normalize_model_load_attempts(attempts),
            git_commit=_optional_string(data.get("git_commit")),
            build_id=_optional_string(data.get("build_id")),
        )


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate or None


def build_receipt(
    *,
    model: str,
    device: str,
    compute_type: str,
    config: Mapping[str, Any] | None = None,
    schema_version: int | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
    include_git_commit: bool = True,
    include_build_id: bool = True,
    backend: str | None = None,
    model_revision: str | None = None,
    model_load_attempts: Sequence[Mapping[str, Any]] | None = None,
) -> Receipt:
    """Build a receipt from actual runtime values and package-local identity."""
    from .models import SCHEMA_VERSION

    receipt_config: Mapping[str, Any]
    if config is None:
        generated_config: dict[str, Any] = {
            "model": model,
            "device": device,
            "compute_type": compute_type,
        }
        if backend is not None:
            generated_config["backend"] = backend
        if model_revision is not None:
            generated_config["model_revision"] = model_revision
        receipt_config = generated_config
    else:
        receipt_config = config

    return Receipt(
        tool_version=get_tool_version(),
        schema_version=(schema_version if schema_version is not None else SCHEMA_VERSION),
        model=model,
        device=device,
        compute_type=compute_type,
        config_hash=compute_config_hash(receipt_config),
        run_id=run_id if run_id is not None else generate_run_id(),
        created_at=(
            created_at
            if created_at is not None
            else datetime.now(UTC).isoformat().replace("+00:00", "Z")
        ),
        backend=_optional_string(backend),
        model_revision=_optional_string(model_revision),
        model_load_attempts=normalize_model_load_attempts(model_load_attempts),
        git_commit=get_git_commit() if include_git_commit else None,
        build_id=get_build_id() if include_build_id else None,
    )


def receipt_stable_projection(data: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only per-run volatile fields from a receipt dictionary."""
    return {key: value for key, value in data.items() if key not in RECEIPT_VOLATILE_FIELDS}


def validate_receipt(data: Mapping[str, Any]) -> list[str]:
    """Validate the in-memory receipt contract without requiring jsonschema."""
    import uuid

    errors: list[str] = []
    missing = RECEIPT_REQUIRED_FIELDS - set(data.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")

    string_fields = ("tool_version", "model", "device", "compute_type")
    for field_name in string_fields:
        if field_name in data and not isinstance(data[field_name], str):
            errors.append(f"{field_name} must be a string")

    if "schema_version" in data and not isinstance(data["schema_version"], int):
        errors.append("schema_version must be an integer")

    config_hash = data.get("config_hash")
    if config_hash is not None:
        if not isinstance(config_hash, str):
            errors.append("config_hash must be a string")
        elif re.fullmatch(r"[0-9a-f]{12}", config_hash) is None:
            errors.append("config_hash must be exactly 12 characters of lowercase hexadecimal")

    run_id = data.get("run_id")
    if run_id is not None:
        if not isinstance(run_id, str):
            errors.append("run_id must be a string")
        elif not is_valid_run_id(run_id):
            try:
                uuid.UUID(run_id)
            except ValueError:
                errors.append("run_id must be in format 'run-YYYYMMDD-HHMMSS-XXXXXX' or valid UUID")

    if "created_at" in data and not isinstance(data["created_at"], str):
        errors.append("created_at must be a string")

    for field_name in ("backend", "model_revision"):
        if field_name in data and not isinstance(data[field_name], str):
            errors.append(f"{field_name} must be a string")

    git_commit = data.get("git_commit")
    if git_commit is not None:
        if not isinstance(git_commit, str):
            errors.append("git_commit must be a string or null")
        elif re.fullmatch(_GIT_COMMIT_PATTERN, git_commit) is None:
            errors.append("git_commit must be a 7-64 character lowercase hexadecimal revision")

    build_id = data.get("build_id")
    if build_id is not None:
        if not isinstance(build_id, str):
            errors.append("build_id must be a string or null")
        elif re.fullmatch(_BUILD_ID_PATTERN, build_id) is None:
            errors.append("build_id has an invalid format")

    attempts = data.get("model_load_attempts")
    if attempts is not None:
        if not isinstance(attempts, list):
            errors.append("model_load_attempts must be an array")
        else:
            normalized_attempts = normalize_model_load_attempts(attempts)
            if len(normalized_attempts) != len(attempts):
                errors.append("model_load_attempts contains an invalid attempt")

    return errors
