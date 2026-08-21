"""Shared helpers for metadata handling across pipeline and API."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .receipt import build_receipt

if TYPE_CHECKING:
    from .models import Transcript

logger = logging.getLogger(__name__)


def coalesce_runtime_value(*values: Any, default: str | None = None) -> str | None:
    """Pick the first non-empty, non-boolean runtime metadata value."""
    for val in values:
        if val is None or isinstance(val, bool):
            continue
        if isinstance(val, str):
            candidate = val.strip()
            if candidate:
                return candidate
            continue
        if isinstance(val, int | float):
            return str(val)
    return default


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
    """Attach run metadata and a truthful provenance receipt.

    Runtime-selected values emitted by the ASR engine outrank requested config.
    The receipt hash excludes output location and volatile timestamps so an
    equivalent runtime/configuration projects to the same stable hash.
    """
    asr_meta = transcript.meta or {}

    actual_device = coalesce_runtime_value(
        asr_meta.get("asr_device"),
        *runtime_device_candidates,
        default=config_device,
    ) or "unknown"
    actual_compute_type = coalesce_runtime_value(
        asr_meta.get("asr_compute_type"),
        *runtime_compute_candidates,
        default=config_compute_type,
    ) or "unknown"

    generated_at = datetime.now(UTC).isoformat()
    base_meta = {
        "generated_at": generated_at,
        "audio_file": transcript.file_name,
        "audio_duration_sec": duration_sec,
        "model_name": model_name,
        "device": actual_device,
        "compute_type": actual_compute_type,
        "beam_size": beam_size,
        "vad_min_silence_ms": vad_min_silence_ms,
        "language_hint": language_hint,
        "task": task,
        "pipeline_version": pipeline_version,
        "root": str(root),
    }

    attempts_value = asr_meta.get("asr_model_load_attempts")
    runtime_attempts = (
        [dict(item) for item in attempts_value if isinstance(item, dict)]
        if isinstance(attempts_value, list)
        else None
    )
    receipt_config = {
        "model": model_name,
        "device": actual_device,
        "compute_type": actual_compute_type,
        "beam_size": beam_size,
        "vad_min_silence_ms": vad_min_silence_ms,
        "language": language_hint,
        "task": task,
        "backend": asr_meta.get("asr_backend", "faster-whisper"),
    }
    receipt = build_receipt(
        model=model_name,
        device=actual_device,
        compute_type=actual_compute_type,
        config=receipt_config,
        runtime_attempts=runtime_attempts,
    )

    merged_meta = asr_meta.copy()
    merged_meta.update(base_meta)
    merged_meta["receipt"] = receipt.to_dict()
    return merged_meta
