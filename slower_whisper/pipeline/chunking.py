"""Turn-aware chunking helpers (v1.3 preview).

These helpers slice transcripts into stable, RAG-friendly chunks while
respecting conversational turns when available.

Key features:
- Configurable turn_affinity (0.0-1.0) controls how strongly chunks prefer
  to stay within speaker turns vs pure duration/token-based chunking.
- Cross-turn penalty discourages splits that would break mid-turn.
- Detection of rapid turn-taking and overlapping speech patterns.
- Chunk metadata includes turn boundary crossing information.

Usage:
    from transcription.chunking import ChunkingConfig, build_chunks

    config = ChunkingConfig(
        target_duration_s=30.0,
        max_duration_s=45.0,
        turn_affinity=0.7,  # Prefer turn boundaries
        cross_turn_penalty=1.0,
    )
    chunks = build_chunks(transcript, config)

    for chunk in chunks:
        print(f"Chunk {chunk.id}: {len(chunk.turn_ids)} turns")
        if chunk.crosses_turn_boundary:
            print(f"  Warning: crosses {chunk.turn_boundary_count} turn boundaries")
        if chunk.has_rapid_turn_taking:
            print("  Contains rapid turn-taking patterns")
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .models import Chunk, Transcript
from .turn_helpers import turn_to_dict

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunk generation.

    Attributes:
        target_duration_s: Soft target duration before considering a split.
        max_duration_s: Hard maximum duration for a chunk.
        target_tokens: Soft target token budget for a chunk.
        pause_split_threshold_s: Pause length that triggers a split when near target.
        turn_affinity: Controls how strongly chunks prefer to stay within speaker turns.
            Range 0.0-1.0 where:
            - 0.0: No turn preference (pure duration/token-based chunking)
            - 0.5: Moderate preference (default) - split at turn boundaries when near soft limits
            - 1.0: Strong preference - always split at turn boundaries when possible
        cross_turn_penalty: Multiplicative penalty applied when evaluating splits that
            would cross mid-turn. Higher values discourage mid-turn splits.
            Range 0.0-2.0 where:
            - 0.0: No penalty (ignore turn boundaries)
            - 1.0: Default - moderate discouragement of mid-turn splits
            - 2.0: Strong penalty - heavily prefer turn boundaries
        min_turn_gap_s: Minimum gap between turns to consider them distinct.
            Used to detect rapid turn-taking scenarios.
    """

    target_duration_s: float = 30.0
    max_duration_s: float = 45.0
    target_tokens: int = 400
    pause_split_threshold_s: float = 1.5
    turn_affinity: float = 0.5
    cross_turn_penalty: float = 1.0
    min_turn_gap_s: float = 0.3

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.turn_affinity <= 1.0:
            raise ValueError(f"turn_affinity must be in [0.0, 1.0], got {self.turn_affinity}")
        if not 0.0 <= self.cross_turn_penalty <= 2.0:
            raise ValueError(
                f"cross_turn_penalty must be in [0.0, 2.0], got {self.cross_turn_penalty}"
            )
        if self.min_turn_gap_s < 0.0:
            raise ValueError(f"min_turn_gap_s must be >= 0.0, got {self.min_turn_gap_s}")


def _speaker_id(value: Any) -> str | None:
    """Extract speaker ID from various speaker formats.

    Note: This is a thin wrapper around get_speaker_id() for backward
    compatibility. New code should use get_speaker_id() directly.
    """
    from .speaker_id import get_speaker_id

    return get_speaker_id(value)


def _estimate_tokens(text: str) -> int:
    """Estimate token count for text using word-based heuristic.

    Uses a 1.3x multiplier on word count to approximate tokenizer output,
    which typically produces more tokens than whitespace-split words
    due to subword tokenization.

    Returns:
        Estimated token count, minimum 1 for non-empty text.
    """
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


def _detect_overlapping_speech(units: list[dict[str, Any]]) -> bool:
    """Detect if any units have overlapping time ranges.

    Overlapping speech occurs when a segment/turn starts before the previous one ends,
    indicating multiple speakers talking simultaneously.

    Args:
        units: List of normalized unit dicts with start/end times.

    Returns:
        True if any overlap detected, False otherwise.
    """
    for i in range(1, len(units)):
        prev_end = units[i - 1].get("end", 0.0)
        curr_start = units[i].get("start", 0.0)
        if curr_start < prev_end:
            return True
    return False


def _detect_rapid_turn_taking(units: list[dict[str, Any]], min_turn_gap_s: float) -> bool:
    """Detect rapid turn-taking patterns within units.

    Rapid turn-taking occurs when speaker transitions happen with minimal gaps,
    indicating a fast-paced conversation or interruption patterns.

    Args:
        units: List of normalized unit dicts with start/end/speaker_id.
        min_turn_gap_s: Minimum gap threshold to consider turns distinct.

    Returns:
        True if rapid turn-taking detected, False otherwise.
    """
    for i in range(1, len(units)):
        prev_speaker = units[i - 1].get("speaker_id")
        curr_speaker = units[i].get("speaker_id")
        # Only check gaps between different speakers
        if prev_speaker != curr_speaker and prev_speaker and curr_speaker:
            prev_end = units[i - 1].get("end", 0.0)
            curr_start = units[i].get("start", 0.0)
            gap = curr_start - prev_end
            if gap < min_turn_gap_s:
                return True
    return False


def _count_turn_boundaries(units: list[dict[str, Any]]) -> int:
    """Count speaker turn transitions within a list of units.

    Args:
        units: List of normalized unit dicts with speaker_id.

    Returns:
        Number of speaker changes (turn boundaries) in the units.
    """
    if len(units) <= 1:
        return 0

    count = 0
    for i in range(1, len(units)):
        prev_speaker = units[i - 1].get("speaker_id")
        curr_speaker = units[i].get("speaker_id")
        if prev_speaker != curr_speaker and prev_speaker and curr_speaker:
            count += 1
    return count


def _finalize_chunk(idx: int, units: list[dict[str, Any]], config: ChunkingConfig) -> Chunk:
    """Create a Chunk from a list of turn/segment units.

    Aggregates segment IDs, turn IDs, speaker IDs, and text from all units.
    Also computes turn-aware metadata including boundary crossings and
    rapid turn-taking detection.

    Args:
        idx: Chunk index (used for generating chunk ID).
        units: List of normalized turn or segment dicts with start/end/text/segment_ids.
        config: ChunkingConfig with min_turn_gap_s for rapid turn-taking detection.

    Returns:
        Chunk dataclass with aggregated metadata and text.
    """
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

    # Compute turn-aware metadata
    turn_boundary_count = _count_turn_boundaries(units)
    crosses_turn_boundary = turn_boundary_count > 0
    has_rapid_turn_taking = _detect_rapid_turn_taking(units, config.min_turn_gap_s)
    has_overlapping_speech = _detect_overlapping_speech(units)

    return Chunk(
        id=f"chunk_{idx}",
        start=start,
        end=end,
        segment_ids=segment_ids,
        turn_ids=turn_ids,
        speaker_ids=sorted(speaker_ids_set),
        token_count_estimate=_estimate_tokens(text),
        text=text,
        crosses_turn_boundary=crosses_turn_boundary,
        turn_boundary_count=turn_boundary_count,
        has_rapid_turn_taking=has_rapid_turn_taking,
        has_overlapping_speech=has_overlapping_speech,
    )


def _segment_units(transcript: Transcript) -> list[dict[str, Any]]:
    """Convert transcript segments to normalized unit dicts for chunking.

    Each unit dict has: id, start, end, text, segment_ids, speaker_id.
    Used as fallback when transcript has no turns.

    Args:
        transcript: Transcript with segments to convert.

    Returns:
        List of normalized segment unit dicts.
    """
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


def _is_turn_boundary(current_unit: dict[str, Any], next_unit: dict[str, Any]) -> bool:
    """Check if there is a speaker change between two units.

    Args:
        current_unit: The current unit dict with speaker_id.
        next_unit: The next unit dict with speaker_id.

    Returns:
        True if speakers differ (turn boundary), False otherwise.
    """
    current_speaker = current_unit.get("speaker_id")
    next_speaker = next_unit.get("speaker_id")
    # A boundary exists if both have speakers and they differ
    if current_speaker and next_speaker:
        return bool(current_speaker != next_speaker)
    return False


def _compute_split_score(
    current_duration: float,
    token_estimate: int,
    gap: float,
    is_turn_boundary: bool,
    config: ChunkingConfig,
) -> float:
    """Compute a score indicating how favorable a split point is.

    Higher scores indicate better split points. The score considers:
    - Duration relative to target (higher score when near/above target)
    - Token count relative to target (higher score when near/above target)
    - Gap duration (longer gaps are better split points)
    - Turn boundary bonus (controlled by turn_affinity)
    - Cross-turn penalty (applied when NOT at a turn boundary)

    Args:
        current_duration: Current chunk duration in seconds.
        token_estimate: Current chunk token estimate.
        gap: Gap duration in seconds to next unit.
        is_turn_boundary: True if this is a speaker change point.
        config: ChunkingConfig with turn_affinity and cross_turn_penalty.

    Returns:
        Split score where higher values indicate better split points.
        Score >= 1.0 typically indicates a good split point.
    """
    # Base score from duration and tokens (0.0 to 1.0 each)
    duration_score = min(1.0, current_duration / config.target_duration_s)
    token_score = min(1.0, token_estimate / config.target_tokens)

    # Gap contribution (0.0 to 0.5 - pauses are natural break points)
    gap_score = min(0.5, gap / (config.pause_split_threshold_s * 2))

    # Combine base scores
    base_score = max(duration_score, token_score) + gap_score

    # Turn boundary adjustments
    if is_turn_boundary:
        # Bonus for splitting at turn boundaries
        # At turn_affinity=1.0, this can add up to 0.5 to the score
        turn_bonus = config.turn_affinity * 0.5
        base_score += turn_bonus
    else:
        # Penalty for NOT being at a turn boundary (if we have speaker info)
        # This discourages mid-turn splits
        # cross_turn_penalty of 1.0 reduces score by up to 0.25
        penalty = config.cross_turn_penalty * 0.25 * config.turn_affinity
        base_score -= penalty

    return base_score


def build_chunks(transcript: Transcript, config: ChunkingConfig) -> list[Chunk]:
    """Build stable, turn-aware chunks for retrieval.

    This function creates chunks from a transcript while respecting speaker turn
    boundaries. The chunking behavior is controlled by the config parameters:

    - turn_affinity (0.0-1.0): How strongly to prefer splitting at turn boundaries.
      At 0.0, purely duration/token-based. At 1.0, strongly prefers turn boundaries.
    - cross_turn_penalty (0.0-2.0): Penalty for creating chunks that cross mid-turn.
      Higher values discourage splitting within a speaker's turn.
    - min_turn_gap_s: Minimum gap to consider turns distinct (for rapid turn-taking).

    The algorithm:
    1. Normalizes transcript into units (turns or segments as fallback)
    2. Greedily builds chunks, evaluating each potential split point
    3. Splits when hard limits are exceeded OR when a favorable split point is found
    4. Tracks turn-aware metadata (boundary crossings, rapid turn-taking, overlaps)

    The transcript is mutated with a `chunks` attribute for convenience.

    Args:
        transcript: Transcript with segments and optionally turns.
        config: ChunkingConfig controlling chunk sizes and turn preferences.

    Returns:
        List of Chunk objects with turn-aware metadata.
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

    if not units:
        transcript.chunks = []
        return []

    chunks: list[Chunk] = []
    current: list[dict[str, Any]] = []
    chunk_start = 0.0
    token_estimate = 0

    for idx, unit in enumerate(units):
        unit_tokens = _estimate_tokens(unit.get("text", ""))

        # Check if we should split before adding this unit
        if current:
            gap_prev = max(
                0.0,
                unit.get("start", 0.0) - current[-1].get("end", unit.get("start", 0.0)),
            )
            current_duration = max(current[-1].get("end", chunk_start) - chunk_start, 0.0)

            # Check for turn boundary at this position
            is_boundary = _is_turn_boundary(current[-1], unit)

            # Compute split score considering turn affinity
            split_score = _compute_split_score(
                current_duration,
                token_estimate,
                gap_prev,
                is_boundary,
                config,
            )

            # Traditional split conditions (pause + soft limit)
            pause_triggered = gap_prev >= config.pause_split_threshold_s and (
                current_duration >= config.target_duration_s
                or token_estimate >= config.target_tokens
            )

            # Turn-aware split: high affinity + at boundary + reached threshold
            turn_triggered = (
                config.turn_affinity >= 0.7
                and is_boundary
                and (
                    current_duration >= config.target_duration_s * 0.5
                    or token_estimate >= config.target_tokens * 0.5
                )
            )

            # Score-based split for moderate affinity
            score_triggered = split_score >= 1.0 and current_duration >= config.target_duration_s

            if pause_triggered or turn_triggered or score_triggered:
                chunks.append(_finalize_chunk(len(chunks), current, config))
                current = []
                token_estimate = 0

        if not current:
            chunk_start = unit.get("start", 0.0)

        prospective_duration = max(unit.get("end", chunk_start) - chunk_start, 0.0)
        prospective_tokens = token_estimate + unit_tokens

        # Hard limit check - must split if exceeded
        over_hard = (
            prospective_duration > config.max_duration_s
            or prospective_tokens > config.target_tokens
        )
        if over_hard and current:
            chunks.append(_finalize_chunk(len(chunks), current, config))
            current = []
            token_estimate = 0
            chunk_start = unit.get("start", 0.0)
            prospective_tokens = unit_tokens

        current.append(unit)
        token_estimate = prospective_tokens

        # Look ahead: should we split after this unit?
        if idx + 1 < len(units):
            next_unit = units[idx + 1]
            next_start = next_unit.get("start", unit.get("end", chunk_start))
            gap_next = max(0.0, next_start - unit.get("end", next_start))
            current_duration = max(unit.get("end", chunk_start) - chunk_start, 0.0)

            is_boundary = _is_turn_boundary(unit, next_unit)

            split_score = _compute_split_score(
                current_duration,
                token_estimate,
                gap_next,
                is_boundary,
                config,
            )

            hit_soft_limit = (
                current_duration >= config.target_duration_s
                or token_estimate >= config.target_tokens
            )

            # Traditional pause-based split
            pause_split = hit_soft_limit and gap_next >= config.pause_split_threshold_s

            # Turn boundary split with high affinity
            turn_split = (
                config.turn_affinity >= 0.7
                and is_boundary
                and current_duration >= config.target_duration_s * 0.6
            )

            # Score-based split
            score_split = split_score >= 1.2 and hit_soft_limit

            if pause_split or turn_split or score_split:
                chunks.append(_finalize_chunk(len(chunks), current, config))
                current = []
                token_estimate = 0

    if current:
        chunks.append(_finalize_chunk(len(chunks), current, config))

    transcript.chunks = chunks  # type: ignore[assignment]
    return chunks
