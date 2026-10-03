"""Idempotent receipt attachment for generation-complete transcript metadata.

Direct file, bytes, and REST transcription attach the canonical receipt while
building generation metadata. Older directory and CLI orchestration can reach
the JSON writer with the same authoritative generation fields but no receipt.
This module closes that serialization seam without guessing from incomplete
metadata or consulting the caller's repository/environment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .receipt import build_receipt, normalize_model_load_attempts, validate_receipt

_GENERATION_FIELDS = frozenset(
    {
        "model_name",
        "device",
        "compute_type",
        "beam_size",
        "vad_min_silence_ms",
        "task",
    }
)


def _nonempty_string(*values: object, default: str | None = None) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _created_at(meta: Mapping[str, Any]) -> str | None:
    generated_at = meta.get("generated_at")
    if isinstance(generated_at, str) and generated_at.strip():
        return generated_at.strip()
    return None


def _attempts(
    meta: Mapping[str, Any],
    *,
    device: str,
    compute_type: str,
) -> list[dict[str, str]]:
    raw_attempts = meta.get("asr_model_load_attempts")
    attempts = normalize_model_load_attempts(
        raw_attempts
        if isinstance(raw_attempts, Sequence)
        and not isinstance(raw_attempts, str | bytes | bytearray)
        else None
    )
    if attempts:
        return attempts
    return [
        {
            "device": device,
            "compute_type": compute_type,
            "outcome": "selected",
            "reason_code": "ok",
        }
    ]


def ensure_generation_receipt(
    meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return copied metadata with one canonical receipt when evidence is complete.

    Existing receipts are validated and preserved byte-for-byte. Metadata that
    does not contain every generation field used by the canonical config hash is
    copied unchanged; unknown is preferable to a guessed provenance receipt.
    """
    copied = dict(meta or {})
    existing = copied.get("receipt")
    if existing is not None:
        if not isinstance(existing, Mapping):
            raise ValueError("meta.receipt must be an object")
        errors = validate_receipt(existing)
        if errors:
            raise ValueError(f"meta.receipt is invalid: {'; '.join(errors)}")
        copied["receipt"] = dict(existing)
        return copied

    if not _GENERATION_FIELDS.issubset(copied):
        return copied

    model = _nonempty_string(copied.get("asr_model"), copied.get("model_name"))
    device = _nonempty_string(copied.get("asr_device"), copied.get("device"))
    compute_type = _nonempty_string(
        copied.get("asr_compute_type"),
        copied.get("compute_type"),
    )
    if model is None or device is None or compute_type is None:
        return copied

    backend = _nonempty_string(
        copied.get("asr_backend"),
        default="faster-whisper",
    )
    model_revision = _nonempty_string(copied.get("asr_model_revision"))
    attempts = _attempts(
        copied,
        device=device,
        compute_type=compute_type,
    )
    receipt_config = {
        "model": model,
        "backend": backend,
        "model_revision": model_revision,
        "device": device,
        "compute_type": compute_type,
        "beam_size": copied["beam_size"],
        "vad_min_silence_ms": copied["vad_min_silence_ms"],
        "language_hint": copied.get("language_hint"),
        "task": copied["task"],
    }
    receipt = build_receipt(
        model=model,
        backend=backend,
        model_revision=model_revision,
        device=device,
        compute_type=compute_type,
        model_load_attempts=attempts,
        config=receipt_config,
        created_at=_created_at(copied),
    )
    copied["receipt"] = receipt.to_dict()
    if "asr_model_load_attempts" in copied:
        copied["asr_model_load_attempts"] = [dict(attempt) for attempt in attempts]
    return copied
