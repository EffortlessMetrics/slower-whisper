"""Turn-aware chunking helpers (v1.3 preview).

These helpers slice transcripts into stable, RAG-friendly chunks while
respecting conversational turns when available.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .models import Chunk, Transcript
from .turn_helpers import turn_to_dict


@dataclass
class ChunkingConfig:
    """Configuration for chunk generation.

    Attributes:
        target_duration_s: Soft target duration before considering a split.
        max_duration_s: Hard maximum duration for a chunk.
        target_tokens: Soft target token budget for a chunk.
        pause_split_threshold_s: Pause length that triggers a split when near target.
    """

    target_duration_s: float = 30.0
    max_duration_s: float = 45.0
    target_tokens: int = 400
    pause_split_threshold_s: float = 1.5


def _speaker_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        raw = value.get("id")
        return str(raw) if raw is not None else None
    return str(value)


def _estimate_tokens(text: str) -> int:
    words = text.split()
    if not words:
        return 0
    return max(1, int(len(words) * 1.3))


def _normalize_turns(turns: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert turns to normalized dicts using turn_to_dict helper.

    Ensures consistent field access for chunking logic.
    """
    normalized: list[dict[str, Any]] = []
    for idx, turn in enumerate(turns):
        try:
            turn_dict = turn_to_dict(turn)
        except TypeError:
            # Unsupported type, skip this turn
            continue

        # Normalize fields for chunking
        turn_id = str(turn_dict.get("id") or f"turn_{idx}")
        start = float(turn_dict.get("start", 0.0))
        end = float(turn_dict.get("end", 0.0))
        text = str(turn_dict.get("text", ""))
        segment_ids = list(turn_dict.get("segment_ids", []) or [])
        speaker_id = _speaker_id(turn_dict.get("speaker_id"))

        normalized.append(
            {
                "id": turn_id,
                "start": start,
                "end": end,
                "text": text,
                "segment_ids": segment_ids,
                "speaker_id": speaker_id,
            }
        )
    return normalized


def _finalize_chunk(idx: int, units: list[dict[str, Any]]) -> Chunk:
    start = units[0]["start"]
    end = units[-1]["end"]
    text = " ".join(u["text"].strip() for u in units if u.get("text", "").strip())
    segment_ids: list[int] = []
    turn_ids: list[str] = []
    speaker_ids_set: set[str] = set()
    for unit in units:
        segment_ids.extend(unit.get("segment_ids", []))
        if unit.get("id"):
            turn_ids.append(str(unit["id"]))
        speaker = unit.get("speaker_id")
        if speaker:
            speaker_ids_set.add(speaker)

    return Chunk(
        id=f"chunk_{idx}",
        start=start,
        end=end,
        segment_ids=segment_ids,
        turn_ids=turn_ids,
        speaker_ids=sorted(speaker_ids_set),
        token_count_estimate=_estimate_tokens(text),
        text=text,
    )


def _segment_units(transcript: Transcript) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for seg in transcript.segments:
        units.append(
            {
                "id": f"seg_{seg.id}",
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "segment_ids": [seg.id],
                "speaker_id": _speaker_id(seg.speaker),
            }
        )
    return units


def build_chunks(transcript: Transcript, config: ChunkingConfig) -> list[Chunk]:
    """Build stable, turn-aware chunks for retrieval.

    The transcript is mutated with a `chunks` attribute for convenience.
    """

    # Prefer existing turn structure; otherwise fall back to segments.
    if transcript.turns is None:
        try:
            from .turns import build_turns as _build_turns

            transcript = _build_turns(transcript)
        except Exception:
            transcript.turns = transcript.turns or []

    turn_units = _normalize_turns(transcript.turns or []) if transcript.turns else []
    units = turn_units if turn_units else _segment_units(transcript)

    chunks: list[Chunk] = []
    current: list[dict[str, Any]] = []
    chunk_start = 0.0
    token_estimate = 0

    for idx, unit in enumerate(units):
        unit_tokens = _estimate_tokens(unit.get("text", ""))

        if current:
            gap_prev = max(
                0.0,
                unit.get("start", 0.0) - current[-1].get("end", unit.get("start", 0.0)),
            )
            current_duration = max(current[-1].get("end", chunk_start) - chunk_start, 0.0)
            if gap_prev >= config.pause_split_threshold_s and (
                current_duration >= config.target_duration_s
                or token_estimate >= config.target_tokens
            ):
                chunks.append(_finalize_chunk(len(chunks), current))
                current = []
                token_estimate = 0

        if not current:
            chunk_start = unit.get("start", 0.0)

        prospective_duration = max(unit.get("end", chunk_start) - chunk_start, 0.0)
        prospective_tokens = token_estimate + unit_tokens
        over_hard = (
            prospective_duration > config.max_duration_s
            or prospective_tokens > config.target_tokens
        )
        if over_hard and current:
            chunks.append(_finalize_chunk(len(chunks), current))
            current = []
            token_estimate = 0
            chunk_start = unit.get("start", 0.0)
            prospective_duration = max(unit.get("end", chunk_start) - chunk_start, 0.0)
            prospective_tokens = unit_tokens

        current.append(unit)
        token_estimate = prospective_tokens

        if idx + 1 < len(units):
            next_start = units[idx + 1].get("start", unit.get("end", chunk_start))
            gap_next = max(0.0, next_start - unit.get("end", next_start))
            current_duration = max(unit.get("end", chunk_start) - chunk_start, 0.0)
            hit_soft_limit = (
                current_duration >= config.target_duration_s
                or token_estimate >= config.target_tokens
            )
            if hit_soft_limit and gap_next >= config.pause_split_threshold_s:
                chunks.append(_finalize_chunk(len(chunks), current))
                current = []
                token_estimate = 0

    if current:
        chunks.append(_finalize_chunk(len(chunks), current))

    transcript.chunks = chunks  # type: ignore[assignment]
    return chunks
