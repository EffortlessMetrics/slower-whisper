"""Shared helpers for metadata handling across pipeline and API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import Transcript

logger = logging.getLogger(__name__)


def coalesce_runtime_value(*values: Any, default: str | None = None) -> str | None:
    """
    Pick the first non-empty runtime metadata value.

    Treats None and whitespace-only strings as missing. Booleans are ignored to
    avoid accidentally returning True/False as device names. Numeric values are
    converted to strings.
    """
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
    """
    Merge ASR metadata with run-level details into a final transcript.meta dict.

    Prefers runtime values emitted by the ASR engine (asr_device/asr_compute_type)
    and falls back through provided runtime candidates before using the configured
    device/compute_type.
    """
    asr_meta = transcript.meta or {}

    actual_device = coalesce_runtime_value(
        asr_meta.get("asr_device"),
        *runtime_device_candidates,
        default=config_device,
    )
    actual_compute_type = coalesce_runtime_value(
        asr_meta.get("asr_compute_type"),
        *runtime_compute_candidates,
        default=config_compute_type,
    )

    base_meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
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

    merged_meta = asr_meta.copy()
    merged_meta.update(base_meta)
    return merged_meta
