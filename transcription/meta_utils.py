"""Shared helpers for metadata and canonical provenance attachment."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .receipt import build_receipt, normalize_model_load_attempts

if TYPE_CHECKING:
    from .models import Transcript

logger = logging.getLogger(__name__)


def coalesce_runtime_value(*values: Any, default: str | None = None) -> str | None:
    """Pick the first non-empty runtime metadata value."""
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                return candidate
            continue
        if isinstance(value, int | float):
            return str(value)
    return default


def _receipt_attempts(
    asr_meta: Mapping[str, Any],
    *,
    device: str,
    compute_type: str,
) -> list[dict[str, str]]:
    raw_attempts = asr_meta.get("asr_model_load_attempts")
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


def build_generation_metadata(
    transcript: Transcript,
    *,
    duration_sec: float,
    model_name: str,
    config_device: str | None,
    config_compute_type: str | None,
    beam_size: int,
    vad_min_silence_ms: int,
    language_hint: str | None,
    task: str,
    pipeline_version: str,
    root: str | Path,
    runtime_device_candidates: tuple[str | None, ...] = (),
    runtime_compute_candidates: tuple[str | None, ...] = (),
) -> dict[str, Any]:
    """Merge generation metadata and attach one canonical provenance receipt.

    Runtime values emitted by the selected ASR engine outrank configured values.
    The receipt contains no caller path and reads source/build identity only from
    package-local build metadata.
    """
    asr_meta = transcript.meta or {}

    actual_model = (
        coalesce_runtime_value(
            asr_meta.get("asr_model"),
            default=model_name,
        )
        or model_name
    )
    actual_backend = (
        coalesce_runtime_value(
            asr_meta.get("asr_backend"),
            default="faster-whisper",
        )
        or "faster-whisper"
    )
    actual_model_revision = coalesce_runtime_value(
        asr_meta.get("asr_model_revision"),
    )
    actual_device = (
        coalesce_runtime_value(
            asr_meta.get("asr_device"),
            *runtime_device_candidates,
            default=config_device,
        )
        or "unknown"
    )
    actual_compute_type = (
        coalesce_runtime_value(
            asr_meta.get("asr_compute_type"),
            *runtime_compute_candidates,
            default=config_compute_type,
        )
        or "unknown"
    )
    attempts = _receipt_attempts(
        asr_meta,
        device=actual_device,
        compute_type=actual_compute_type,
    )

    receipt_config = {
        "model": actual_model,
        "backend": actual_backend,
        "model_revision": actual_model_revision,
        "device": actual_device,
        "compute_type": actual_compute_type,
        "beam_size": beam_size,
        "vad_min_silence_ms": vad_min_silence_ms,
        "language_hint": language_hint,
        "task": task,
    }
    receipt = build_receipt(
        model=actual_model,
        backend=actual_backend,
        model_revision=actual_model_revision,
        device=actual_device,
        compute_type=actual_compute_type,
        model_load_attempts=attempts,
        config=receipt_config,
    )

    base_meta = {
        "generated_at": receipt.created_at,
        "audio_file": transcript.file_name,
        "audio_duration_sec": duration_sec,
        "model_name": actual_model,
        "device": actual_device,
        "compute_type": actual_compute_type,
        "beam_size": beam_size,
        "vad_min_silence_ms": vad_min_silence_ms,
        "language_hint": language_hint,
        "task": task,
        "pipeline_version": pipeline_version,
        "root": str(root),
        "receipt": receipt.to_dict(),
    }

    merged_meta = asr_meta.copy()
    if "asr_model_load_attempts" in merged_meta:
        merged_meta["asr_model_load_attempts"] = [dict(attempt) for attempt in attempts]
    merged_meta.update(base_meta)
    return merged_meta
