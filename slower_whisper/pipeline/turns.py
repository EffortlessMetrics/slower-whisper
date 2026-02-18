"""
Conversational turn structure for slower-whisper.

This module builds the `turns[]` array by grouping contiguous segments
by speaker, enabling conversation-level analysis.

**Status**: v1.1 - Basic turn grouping (v1.1), rich metadata deferred to v1.2

**Turn definition**:
A turn is a contiguous sequence of segments attributed to the same speaker,
bounded by speaker changes or long pauses.

**v1.1 scope** (this module):
- Group segments by speaker into turns
- Basic turn metadata (speaker, start, end, segment_ids, text)

**v1.2 scope** (future enhancement):
- Turn-level enrichment (question_count, interruptions, pauses, disfluency)
- Cross-turn interaction patterns
- Sentiment/emotion aggregation per turn
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .models import Turn
from .speaker_id import get_speaker_id

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .models import Segment, Transcript


def _coerce_speaker_id(speaker: Any) -> str | None:
    """Normalize speaker objects into an ID string.

    Note: This is a thin wrapper around get_speaker_id() for backward
    compatibility. New code should use get_speaker_id() directly.
    """
    return get_speaker_id(speaker)


def build_turns(
    transcript: Transcript,
    pause_threshold: float | None = None,
) -> Transcript:
    """
    Build conversational turns from speaker-attributed segments.

    **v1.1 logic** (basic grouping):
    1. Group contiguous segments with same speaker_id
    2. Split on speaker changes
    3. Optionally split on long pauses (>= pause_threshold)

    **v1.2 enhancement** (planned):
    - Populate metadata.question_count (regex scan for '?')
    - Detect interruptions (rapid speaker changes)
    - Compute pause statistics within turn
    - Aggregate emotion/sentiment per turn

    **Mutates transcript in-place** by setting transcript.turns and returns it.

    Args:
        transcript: Transcript with speaker-attributed segments.
                   Segments with speaker=None are skipped.
        pause_threshold: Minimum pause duration (seconds) to split turns for same speaker.
                        If None (default), only speaker changes trigger turn splits.
                        If set (e.g., 2.0), a gap >= pause_threshold between segments
                        will start a new turn even if the speaker remains the same.
                        Must be >= 0.0 when provided.

    Returns:
        Updated transcript with transcript.turns populated.
        If no segments have speaker labels, turns = [].

    Raises:
        ValueError: If pause_threshold is negative.

    Example:
        >>> from transcription.models import Transcript, Segment
        >>> seg0 = Segment(id=0, start=0.0, end=2.0, text="Hello", speaker={"id": "spk_0", "confidence": 0.9})
        >>> seg1 = Segment(id=1, start=2.1, end=4.0, text="world", speaker={"id": "spk_0", "confidence": 0.9})
        >>> transcript = Transcript(file_name="test.wav", language="en", segments=[seg0, seg1])
        >>> result = build_turns(transcript)
        >>> len(result.turns)
        1
        >>> result.turns[0]["speaker_id"]
        'spk_0'
        >>> result.turns[0]["text"]
        'Hello world'
    """
    # Validate pause_threshold
    if pause_threshold is not None and pause_threshold < 0.0:
        raise ValueError(f"pause_threshold must be >= 0.0, got {pause_threshold}")

    # Filter segments with known speakers and normalize speaker IDs
    speaker_segments: list[tuple[Segment, str]] = []
    for seg in transcript.segments:
        speaker_id = _coerce_speaker_id(seg.speaker)
        if speaker_id is None:
            continue
        speaker_segments.append((seg, speaker_id))

    if not speaker_segments:
        # No speakers assigned → no turns
        transcript.turns = []
        return transcript

    turns: list[Turn | dict[str, Any]] = []
    current_turn_segments: list[Segment] = []
    current_speaker_id: str | None = None

    for segment, speaker_id in speaker_segments:
        if speaker_id != current_speaker_id:
            # Speaker change → finalize current turn and start new
            if current_turn_segments:
                assert current_speaker_id is not None
                turns.append(_finalize_turn(len(turns), current_turn_segments, current_speaker_id))
            current_turn_segments = [segment]
            current_speaker_id = speaker_id
        else:
            # Same speaker → check for pause-based split
            should_split_on_pause = False
            if pause_threshold is not None and current_turn_segments:
                # Calculate gap between previous segment end and current segment start
                previous_segment = current_turn_segments[-1]
                gap = segment.start - previous_segment.end
                if gap >= pause_threshold:
                    should_split_on_pause = True

            if should_split_on_pause:
                # Long pause → finalize current turn and start new with same speaker
                assert current_speaker_id is not None
                turns.append(_finalize_turn(len(turns), current_turn_segments, current_speaker_id))
                current_turn_segments = [segment]
            else:
                # Continue current turn
                current_turn_segments.append(segment)

    # Finalize last turn
    if current_turn_segments and current_speaker_id is not None:
        turns.append(_finalize_turn(len(turns), current_turn_segments, current_speaker_id))

    transcript.turns = turns
    return transcript


def _finalize_turn(turn_id: int, segments: list[Segment], speaker_id: str) -> dict[str, Any]:
    """
    Finalize a turn by building its dict representation.

    Args:
        turn_id: Sequential turn ID (0, 1, 2, ...).
        segments: List of Segment objects in this turn.
        speaker_id: Speaker ID for this turn.

    Returns:
        Turn dict with id, speaker_id, start, end, segment_ids, text.
    """
    segment_ids = [seg.id for seg in segments]
    texts = [seg.text.strip() for seg in segments if seg.text.strip()]  # Skip empty text
    start = segments[0].start
    end = segments[-1].end

    return {
        "id": f"turn_{turn_id}",
        "speaker_id": speaker_id,
        "start": start,
        "end": end,
        "segment_ids": segment_ids,
        "text": " ".join(texts),
    }


def aggregate_turn_stats(turns: list[Turn]) -> dict[str, Any]:
    """
    Compute conversation-level statistics from turns.

    **Planned metrics** (v1.2):
    - total_turns: Number of turns
    - avg_turn_duration: Mean turn length in seconds
    - turn_switches: Number of speaker changes
    - interruption_rate: Turns with rapid switches / total turns
    - question_density: Total questions / total speech time

    Args:
        turns: List of Turn objects from build_turns().

    Returns:
        Dict with conversation-level aggregates.

    Raises:
        NotImplementedError: v1.2 feature - deferred.
    """
    # Import turn_to_dict locally to avoid circular imports
    from .turn_helpers import turn_to_dict

    # Handle empty turns list
    if not turns:
        return {
            "total_turns": 0,
            "avg_turn_duration": 0.0,
            "turn_switches": 0,
            "interruption_rate": 0.0,
            "question_density": 0.0,
        }

    # Handle single turn
    if len(turns) == 1:
        turn_dict = turn_to_dict(turns[0])
        duration = turn_dict.get("end", 0.0) - turn_dict.get("start", 0.0)

        # Get question count from metadata if available
        metadata = turn_dict.get("metadata") or {}
        question_count = metadata.get("question_count", 0)

        # Calculate question density (questions per minute)
        question_density = 0.0
        if duration > 0:
            question_density = (question_count / duration) * 60.0

        return {
            "total_turns": 1,
            "avg_turn_duration": duration,
            "turn_switches": 0,
            "interruption_rate": 0.0,
            "question_density": question_density,
        }

    # Process multiple turns
    total_turns = len(turns)
    total_duration = 0.0
    turn_switches = 0
    interruption_count = 0
    total_questions = 0
    total_speech_time = 0.0

    prev_speaker_id: str | None = None

    for turn in turns:
        turn_dict = turn_to_dict(turn)

        # Calculate turn duration
        start = turn_dict.get("start", 0.0)
        end = turn_dict.get("end", 0.0)
        duration = end - start
        total_duration += duration
        total_speech_time += duration

        # Get speaker ID
        current_speaker_id = turn_dict.get("speaker_id")

        # Count speaker switches (transition from one speaker to another)
        if prev_speaker_id is not None and current_speaker_id != prev_speaker_id:
            turn_switches += 1
        prev_speaker_id = current_speaker_id

        # Get metadata if available
        metadata = turn_dict.get("metadata") or {}

        # Count interruptions
        if metadata.get("interruption_started_here", False):
            interruption_count += 1

        # Count questions
        question_count = metadata.get("question_count", 0)
        total_questions += question_count

    # Calculate averages and rates
    avg_turn_duration = total_duration / total_turns if total_turns > 0 else 0.0
    interruption_rate = interruption_count / total_turns if total_turns > 0 else 0.0

    # Calculate question density (questions per minute)
    question_density = 0.0
    if total_speech_time > 0:
        question_density = (total_questions / total_speech_time) * 60.0

    return {
        "total_turns": total_turns,
        "avg_turn_duration": avg_turn_duration,
        "turn_switches": turn_switches,
        "interruption_rate": interruption_rate,
        "question_density": question_density,
    }
