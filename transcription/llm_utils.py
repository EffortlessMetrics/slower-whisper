"""
LLM rendering utilities for slower-whisper transcripts.

This module provides functions to convert structured Transcript objects
into text blocks optimized for LLM consumption, following the patterns
documented in docs/LLM_PROMPT_PATTERNS.md.
"""

from typing import Any

from transcription.models import Segment, Transcript


def _coerce_speaker_id(value: Any) -> str | None:
    """Extract a speaker ID string from a speaker object/dict."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        raw_id = value.get("id")
        return str(raw_id) if raw_id is not None else None
    return str(value)


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
        # Extract just the descriptors from rendering like "[audio: high pitch, loud volume]"
        rendering = segment.audio_state["rendering"]
        # Strip "[audio: " and trailing "]"
        if rendering.startswith("[audio:") and rendering.endswith("]"):
            audio_desc = rendering[7:-1].strip()  # Remove "[audio:" and "]"
            cues.append(audio_desc)

    if cues:
        parts.append(f"[{' | '.join(cues)}]")

    # Text content
    parts.append(segment.text.strip())

    return " ".join(parts)


def _render_turn_dict(
    turn_dict: dict[str, Any],
    transcript: Transcript,
    include_audio_cues: bool = True,
    include_timestamps: bool = False,
    speaker_labels: dict[str, str] | None = None,
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

    Returns:
        Rendered turn text
    """
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
            # Find segment by ID
            segment = next((s for s in transcript.segments if s.id == seg_id), None)
            if segment and segment.audio_state and segment.audio_state.get("rendering"):
                rendering = segment.audio_state["rendering"]
                if rendering.startswith("[audio:") and rendering.endswith("]"):
                    desc = rendering[7:-1].strip()
                    # Split comma-separated descriptors and collect unique ones
                    for d in desc.split(","):
                        audio_descriptors.add(d.strip())

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
        for turn_dict in transcript.turns:
            rendered = _render_turn_dict(
                turn_dict,
                transcript,
                include_audio_cues=include_audio_cues,
                include_timestamps=include_timestamps,
                speaker_labels=speaker_labels,
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
        for turn_dict in transcript.turns:
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
