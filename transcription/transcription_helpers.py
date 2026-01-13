"""
Small shared helpers used by the public API and pipeline.

These functions are intentionally dependency-light and safe to import early.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any

from .config import TranscriptionConfig
from .models import Transcript

logger = logging.getLogger(__name__)


def _get_wav_duration_seconds(path: Path) -> float:
    """Return WAV duration in seconds, tolerating unreadable files."""
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
        return frames / float(rate) if rate else 0.0
    except Exception as exc:  # noqa: BLE001 - want the raw error for debugging
        logger.warning("Could not read duration for %s: %s", path.name, exc)
        return 0.0


def _neutral_audio_state(error: str | None = None) -> dict[str, Any]:
    """Create a minimal audio_state placeholder used when enrichment fails."""
    extraction_status = {
        "prosody": "skipped",
        "emotion_dimensional": "skipped",
        "emotion_categorical": "skipped",
        "errors": [error] if error else [],
    }
    return {
        "prosody": {
            "pitch": {"level": "unknown", "mean_hz": None, "std_hz": None, "contour": "unknown"},
            "energy": {"level": "unknown", "db_rms": None, "variation": "unknown"},
            "rate": {"level": "unknown", "syllables_per_sec": None, "words_per_sec": None},
            "pauses": {"count": 0, "longest_ms": 0, "density": "unknown"},
        },
        "emotion": {
            "valence": {"level": "neutral", "score": 0.5},
            "arousal": {"level": "medium", "score": 0.5},
            "dominance": {"level": "neutral", "score": 0.5},
        },
        "rendering": "[audio: neutral]",
        "extraction_status": extraction_status,
    }


def _turns_have_metadata(turns: Any) -> bool:
    """Return True when turns are present and each has metadata/meta."""
    if not turns:
        return False

    for turn in turns:
        meta = None
        if isinstance(turn, dict):
            meta = turn.get("metadata") or turn.get("meta")
        elif hasattr(turn, "metadata"):
            meta = getattr(turn, "metadata", None)
        elif hasattr(turn, "meta"):
            meta = getattr(turn, "meta", None)

        if not meta:
            return False
    return True


def _maybe_build_chunks(transcript: Transcript, config: TranscriptionConfig) -> Transcript:
    """Attach RAG-friendly chunks when enabled."""
    if not getattr(config, "enable_chunking", False):
        return transcript

    from .chunking import ChunkingConfig, build_chunks

    chunk_cfg = ChunkingConfig(
        target_duration_s=config.chunk_target_duration_s,
        max_duration_s=config.chunk_max_duration_s,
        target_tokens=config.chunk_target_tokens,
        pause_split_threshold_s=config.chunk_pause_split_threshold_s,
    )
    build_chunks(transcript, chunk_cfg)
    return transcript
