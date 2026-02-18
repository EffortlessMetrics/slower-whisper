"""
LLM rendering utilities for slower-whisper transcripts.

This module provides functions to convert structured Transcript objects
into text blocks optimized for LLM consumption, following the patterns
documented in docs/LLM_PROMPT_PATTERNS.md.
"""

from dataclasses import asdict, is_dataclass
from typing import Any, cast

from .models import Segment, Transcript
from .speaker_id import get_speaker_id


def _coerce_speaker_id(value: Any) -> str | None:
    """Extract a speaker ID string from a speaker object/dict.

    Note: This is a thin wrapper around get_speaker_id() for backward
    compatibility. New code should use get_speaker_id() directly.
    """
    return get_speaker_id(value)


def _resolve_speaker_label(
    speaker: dict[str, Any] | str | None, speaker_labels: dict[str, str] | None
) -> str | None:
    """Return a display label for a speaker, falling back to the raw ID."""
    speaker_id = _coerce_speaker_id(speaker)
    if speaker_id is None:
        return None
    if speaker_labels:
        return speaker_labels.get(speaker_id, speaker_id)
    return speaker_id


def _extract_audio_descriptors(rendering: str) -> list[str]:
    """Extract comma-separated descriptors from '[audio: ...]' format.

    Args:
        rendering: A rendering string like "[audio: high pitch, loud volume]"

    Returns:
        List of descriptor strings, e.g. ["high pitch", "loud volume"]
        Returns empty list if format doesn't match.
    """
    if rendering.startswith("[audio:") and rendering.endswith("]"):
        desc = rendering[7:-1].strip()  # Remove "[audio:" and "]"
        return [d.strip() for d in desc.split(",") if d.strip()]
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce dataclasses or objects with to_dict into plain dicts."""
    if isinstance(value, dict):
        return value
    # Reject classes (types) early - we only work with instances
    if isinstance(value, type):
        raise TypeError(f"Unsupported object type: {type(value)}")
    to_dict_fn = getattr(value, "to_dict", None)
    if callable(to_dict_fn):
        result: dict[str, Any] = to_dict_fn()
        return result
    if is_dataclass(value):
        return asdict(cast(Any, value))
    raise TypeError(f"Unsupported object type: {type(value)}")


def render_segment(
    segment: Segment,
    include_audio_cues: bool = True,
    include_timestamps: bool = False,
    speaker_labels: dict[str, str] | None = None,
) -> str:
    """
    Render a single segment as text for LLM consumption.

    Args:
        segment: The segment to render
        include_audio_cues: If True and audio_state exists, include rendering
        include_timestamps: If True, include [HH:MM:SS] timestamp prefix
        speaker_labels: Optional mapping from speaker IDs to human-readable labels
            (e.g., {"spk_0": "Agent", "spk_1": "Customer"})

    Returns:
        Rendered segment text, e.g.:
        "[Agent | calm tone, low pitch, slow rate] Hello, how can I help you?"
        or
        "[00:01:23] Hello, how can I help you?"
    """
    parts: list[str] = []

    # Timestamp prefix (if requested)
    if include_timestamps:
        minutes, seconds = divmod(int(segment.start), 60)
        hours, minutes = divmod(minutes, 60)
        parts.append(f"[{hours:02d}:{minutes:02d}:{seconds:02d}]")

    # Speaker + audio cues
    cues: list[str] = []
    speaker_label = _resolve_speaker_label(segment.speaker, speaker_labels)
    if speaker_label:
        cues.append(speaker_label)

    if include_audio_cues and segment.audio_state and segment.audio_state.get("rendering"):
        descriptors = _extract_audio_descriptors(segment.audio_state["rendering"])
        if descriptors:
            cues.append(", ".join(descriptors))

    if cues:
        parts.append(f"[{' | '.join(cues)}]")

    # Text content
    parts.append(segment.text.strip())

    return " ".join(parts)


def _render_turn_dict(
    turn_obj: dict[str, Any] | Any,
    transcript: Transcript,
    include_audio_cues: bool = True,
    include_timestamps: bool = False,
    speaker_labels: dict[str, str] | None = None,
    segment_index: dict[int, Segment] | None = None,
) -> str:
    """
    Render a turn dictionary from transcript.turns as text.

    Internal helper that works with the dict representation used in JSON.

    Args:
        turn_dict: Turn dictionary with keys: speaker_id, start, end, segment_ids, text
        transcript: Parent transcript (needed to lookup segments if enrichment needed)
        include_audio_cues: If True, include audio cues from segments
        include_timestamps: If True, include timestamp for turn start
        speaker_labels: Optional mapping from speaker IDs to human-readable labels
        segment_index: Optional pre-built mapping from segment ID to Segment for O(1) lookup.
            If not provided, falls back to linear search (O(n) per lookup).

    Returns:
        Rendered turn text
    """
    turn_dict: dict[str, Any]
    if isinstance(turn_obj, dict):
        turn_dict = turn_obj
    elif isinstance(turn_obj, type):
        # Reject classes (types) early - we only work with instances
        raise TypeError(f"Unsupported turn type: {type(turn_obj)}")
    elif callable(getattr(turn_obj, "to_dict", None)):
        turn_dict = turn_obj.to_dict()
    elif is_dataclass(turn_obj):
        turn_dict = asdict(cast(Any, turn_obj))
    else:
        raise TypeError(f"Unsupported turn type: {type(turn_obj)}")

    parts: list[str] = []

    # Timestamp prefix (if requested)
    if include_timestamps:
        start_time = turn_dict["start"]
        minutes, seconds = divmod(int(start_time), 60)
        hours, minutes = divmod(minutes, 60)
        parts.append(f"[{hours:02d}:{minutes:02d}:{seconds:02d}]")

    # Speaker + aggregated audio cues
    cues: list[str] = []
    speaker_label = _resolve_speaker_label(turn_dict.get("speaker_id"), speaker_labels)
    if speaker_label:
        cues.append(speaker_label)

    # Aggregate audio cues from segments in this turn
    if include_audio_cues:
        segment_ids = turn_dict.get("segment_ids", [])
        audio_descriptors: set[str] = set()

        for seg_id in segment_ids:
            # Find segment by ID - use index for O(1) lookup if available
            if segment_index is not None:
                segment = segment_index.get(seg_id)
            else:
                segment = next((s for s in transcript.segments if s.id == seg_id), None)
            if segment and segment.audio_state and segment.audio_state.get("rendering"):
                for desc in _extract_audio_descriptors(segment.audio_state["rendering"]):
                    audio_descriptors.add(desc)

        if audio_descriptors:
            cues.append(", ".join(sorted(audio_descriptors)))

    if cues:
        parts.append(f"[{' | '.join(cues)}]")

    # Text content from turn
    turn_text_raw = turn_dict.get("text", "")
    turn_text = str(turn_text_raw).strip()
    if turn_text:
        parts.append(turn_text)

    return " ".join(parts)


def render_conversation_for_llm(
    transcript: Transcript,
    mode: str = "turns",
    include_audio_cues: bool = True,
    include_timestamps: bool = False,
    include_metadata: bool = True,
    speaker_labels: dict[str, str] | None = None,
) -> str:
    """
    Render a complete transcript into text optimized for LLM consumption.

    This function follows the patterns documented in docs/LLM_PROMPT_PATTERNS.md,
    providing a clean text representation of the conversation with optional
    speaker labels, audio cues, and timestamps.

    Args:
        transcript: The Transcript object to render
        mode: Rendering mode:
            - "turns": Render by speaker turns (recommended if turns[] exists)
            - "segments": Render individual segments
        include_audio_cues: If True and audio enrichment exists, include
            prosody/emotion cues like "[high pitch, fast rate]"
        include_timestamps: If True, prefix each turn/segment with timestamp
        include_metadata: If True, prepend conversation metadata header
        speaker_labels: Optional mapping from speaker IDs to human-readable labels
            (e.g., {"spk_0": "Agent", "spk_1": "Customer"})

    Returns:
        Rendered conversation text ready for LLM context.

    Example output (turns mode, with audio cues):
        ```
        Conversation: customer_support_call_2025.wav (en)
        Duration: 00:05:23 | Speakers: 2 | Turns: 12

        [spk_0 | calm tone, moderate pitch] Hello, this is Alex from support.
        [spk_1 | frustrated tone, high pitch, fast rate] Hi, I'm having issues...
        [spk_0 | empathetic tone, slow rate] I understand that's frustrating...
        ```

    Usage:
        >>> from transcription import load_transcript
        >>> from transcription.llm_utils import render_conversation_for_llm
        >>>
        >>> transcript = load_transcript("whisper_json/meeting.json")
        >>> context = render_conversation_for_llm(transcript)
        >>> # Pass context to your LLM for analysis
    """
    output = []

    # Metadata header
    if include_metadata:
        duration_secs = int(transcript.segments[-1].end) if transcript.segments else 0
        minutes, seconds = divmod(duration_secs, 60)
        hours, minutes = divmod(minutes, 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        num_speakers = len(transcript.speakers) if transcript.speakers else "unknown"
        num_turns = len(transcript.turns) if transcript.turns else "N/A"

        header = (
            f"Conversation: {transcript.file_name} ({transcript.language})\n"
            f"Duration: {duration_str} | Speakers: {num_speakers} | Turns: {num_turns}\n"
        )
        output.append(header)

    # Render content
    if mode == "turns" and transcript.turns:
        # Turn-based rendering (recommended)
        # Build segment index once for O(1) lookups instead of O(n) per segment
        segment_index = {seg.id: seg for seg in transcript.segments}
        for turn_dict in transcript.turns:
            rendered = _render_turn_dict(
                turn_dict,
                transcript,
                include_audio_cues=include_audio_cues,
                include_timestamps=include_timestamps,
                speaker_labels=speaker_labels,
                segment_index=segment_index,
            )
            if rendered:
                output.append(rendered)
    else:
        # Segment-based rendering (fallback or explicit request)
        for segment in transcript.segments:
            rendered = render_segment(
                segment,
                include_audio_cues=include_audio_cues,
                include_timestamps=include_timestamps,
                speaker_labels=speaker_labels,
            )
            if rendered:
                output.append(rendered)

    return "\n".join(output)


def render_conversation_compact(
    transcript: Transcript,
    max_tokens: int | None = None,
    speaker_labels: dict[str, str] | None = None,
) -> str:
    """
    Render conversation in compact format for token-constrained LLM contexts.

    This mode:
    - Omits timestamps and audio cues
    - Uses minimal speaker prefixes
    - Optionally truncates to fit token budget

    Args:
        transcript: The Transcript object to render
        max_tokens: Approximate token limit (uses rough 4-char = 1 token heuristic)
        speaker_labels: Optional mapping from speaker IDs to human-readable labels
            (e.g., {"spk_0": "Agent", "spk_1": "Customer"})

    Returns:
        Compact conversation text, e.g.:
        ```
        Agent: Hello, this is Alex from support.
        Customer: Hi, I'm having issues with my account.
        Agent: I can help with that. What's happening?
        ```
    """
    lines: list[str] = []

    # Use turns if available, else segments
    if transcript.turns:
        for turn_item in transcript.turns:
            turn_dict = _as_dict(turn_item)
            speaker_label = (
                _resolve_speaker_label(turn_dict.get("speaker_id"), speaker_labels) or "unknown"
            )
            turn_text_raw = turn_dict.get("text", "")
            turn_text = str(turn_text_raw).strip()
            if turn_text:
                lines.append(f"{speaker_label}: {turn_text}")
    else:
        for segment in transcript.segments:
            speaker_label = _resolve_speaker_label(segment.speaker, speaker_labels) or "unknown"
            lines.append(f"{speaker_label}: {segment.text.strip()}")

    result = "\n".join(lines)

    # Truncate if token budget specified
    if max_tokens:
        # Rough heuristic: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        if len(result) > max_chars:
            result = result[:max_chars] + "\n[...truncated]"

    return result


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.s for readability."""
    secs = max(float(seconds), 0.0)
    minutes, secs = divmod(secs, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{secs:04.1f}"


def _format_time_range(start: float, end: float) -> str:
    """Render a start/end time window."""
    return f"{_format_timestamp(start)}-{_format_timestamp(end)}"


def _collect_audio_descriptors(
    segment_ids: list[Any], segment_index: dict[Any, Segment]
) -> list[str]:
    """Return sorted audio descriptors aggregated from a set of segments."""
    descriptors: set[str] = set()
    for seg_id in segment_ids:
        segment = segment_index.get(seg_id)
        if segment and segment.audio_state and segment.audio_state.get("rendering"):
            for desc in _extract_audio_descriptors(segment.audio_state["rendering"]):
                descriptors.add(desc)
    return sorted(descriptors)


def to_turn_view(
    transcript: Transcript,
    turns: list[dict[str, Any] | Any] | None = None,
    include_audio_state: bool = True,
    include_timestamps: bool = True,
    speaker_labels: dict[str, str] | None = None,
) -> str:
    """
    Render turn-level view with analytics metadata for LLM prompts.

    Args:
        transcript: Transcript containing segments/turns.
        turns: Optional turn list; defaults to transcript.turns.
        include_audio_state: If True, aggregate audio cues from segments.
        include_timestamps: If True, prefix each turn with start/end time window.
        speaker_labels: Optional mapping of speaker IDs to human-readable labels.

    Returns:
        Multi-line string with one line per turn including speaker, metadata, and text.
    """
    turn_list = turns if turns is not None else transcript.turns
    if not turn_list:
        return render_conversation_for_llm(
            transcript,
            mode="segments",
            include_audio_cues=include_audio_state,
            include_timestamps=include_timestamps,
            include_metadata=False,
            speaker_labels=speaker_labels,
        )

    segment_index = {seg.id: seg for seg in transcript.segments or []}
    lines: list[str] = []

    for turn in turn_list:
        turn_dict = _as_dict(turn)
        meta = turn_dict.get("metadata") or turn_dict.get("meta") or {}
        cues: list[str] = []

        if include_timestamps:
            start = float(turn_dict.get("start", 0.0))
            end = float(turn_dict.get("end", start))
            cues.append(_format_time_range(start, end))

        speaker_label = _resolve_speaker_label(turn_dict.get("speaker_id"), speaker_labels)
        if speaker_label:
            cues.append(speaker_label)

        if meta:
            if "question_count" in meta:
                cues.append(f"question_count={int(meta.get('question_count') or 0)}")
            if meta.get("interruption_started_here"):
                cues.append("interruption_started_here=true")
            if meta.get("avg_pause_ms") is not None:
                cues.append(f"avg_pause_ms={int(meta.get('avg_pause_ms') or 0)}")
            if meta.get("disfluency_ratio") is not None:
                cues.append(f"disfluency={float(meta.get('disfluency_ratio') or 0.0):.2f}")

        if include_audio_state:
            seg_ids = turn_dict.get("segment_ids") or []
            descriptors = _collect_audio_descriptors(seg_ids, segment_index)
            if descriptors:
                cues.append(f"audio={', '.join(descriptors)}")

        text = str(turn_dict.get("text") or "").strip()
        prefix = f"[{' | '.join(cues)}]" if cues else ""

        if prefix or text:
            lines.append(f"{prefix} {text}".strip())

    return "\n".join(lines)


def to_speaker_summary(
    transcript: Transcript,
    speaker_stats: list[dict[str, Any] | Any] | None = None,
    speaker_labels: dict[str, str] | None = None,
) -> str:
    """
    Render per-speaker analytics into a concise summary block.

    Args:
        transcript: Transcript holding analytics (used as default source).
        speaker_stats: Optional explicit stats list; defaults to transcript.speaker_stats.
        speaker_labels: Optional mapping of speaker IDs to human-readable labels.

    Returns:
        Multi-line string describing talk time, interruptions, and questions per speaker.
    """
    stats_source = speaker_stats if speaker_stats is not None else (transcript.speaker_stats or [])
    if not stats_source:
        return (
            "Speaker stats summary:\n- Analytics not available (enable_speaker_stats to populate)"
        )

    lines = ["Speaker stats summary:"]

    for stat in stats_source:
        stat_dict = _as_dict(stat)
        speaker_label = _resolve_speaker_label(stat_dict.get("speaker_id"), speaker_labels)
        display_name = speaker_label or (stat_dict.get("speaker_id") or "unknown")

        total_talk = float(stat_dict.get("total_talk_time") or 0.0)
        num_turns = int(stat_dict.get("num_turns") or 0)
        avg_turn = float(stat_dict.get("avg_turn_duration") or 0.0)
        initiated = int(stat_dict.get("interruptions_initiated") or 0)
        received = int(stat_dict.get("interruptions_received") or 0)
        questions = int(stat_dict.get("question_turns") or 0)

        prosody = stat_dict.get("prosody_summary") or {}
        sentiment = stat_dict.get("sentiment_summary") or {}
        extras: list[str] = []

        pitch_val = prosody.get("pitch_median_hz")
        if pitch_val is not None:
            extras.append(f"pitch~={float(pitch_val):.0f}Hz")
        energy_val = prosody.get("energy_median_db")
        if energy_val is not None:
            extras.append(f"energy~={float(energy_val):.1f}dB")

        pos = sentiment.get("positive")
        neg = sentiment.get("negative")
        if pos is not None or neg is not None:
            extras.append(f"sentiment+={float(pos or 0.0):.2f}/-={float(neg or 0.0):.2f}")

        extras_suffix = f"; {'; '.join(extras)}" if extras else ""

        lines.append(
            f"- {display_name}: {total_talk:.1f}s across {num_turns} turns "
            f"(avg {avg_turn:.1f}s); {initiated} interruptions started, "
            f"{received} received; {questions} question turns{extras_suffix}"
        )

    return "\n".join(lines)
