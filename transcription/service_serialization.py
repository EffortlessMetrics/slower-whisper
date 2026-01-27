"""Serialization helpers for transcripts and segments."""

from __future__ import annotations

from typing import Any

from .models import SCHEMA_VERSION, Transcript


def _word_to_dict(word: Any) -> dict[str, Any]:
    """
    Convert a Word object to a JSON-serializable dictionary.

    Handles both Word dataclass instances and dict representations.
    Uses the canonical Word.to_dict() when available, otherwise
    extracts fields manually for compatibility.

    Args:
        word: Word object (dataclass or dict) to serialize

    Returns:
        Dictionary with word, start, end, probability, and optional speaker
    """
    # Use canonical to_dict() if available
    if hasattr(word, "to_dict"):
        result: dict[str, Any] = word.to_dict()
        return result

    # Handle dict representation directly
    if isinstance(word, dict):
        return word

    # Manual extraction for other object types
    out: dict[str, Any] = {}
    for key in ("word", "start", "end", "probability", "speaker"):
        val = getattr(word, key, None)
        if val is not None:
            out[key] = val

    # Some models use 'text' instead of 'word'
    if "word" not in out and hasattr(word, "text"):
        out["word"] = word.text

    return out


def _segment_to_dict(seg: Any, *, include_words: bool) -> dict[str, Any]:
    """
    Convert a Segment object to a JSON-serializable dictionary.

    Args:
        seg: Segment object to serialize
        include_words: If True, include word-level timestamps in output

    Returns:
        Dictionary representation of the segment
    """
    d: dict[str, Any] = {
        "id": seg.id,
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "speaker": seg.speaker,
        "tone": seg.tone,
        "audio_state": seg.audio_state,
    }

    # Include words only when requested and present
    if include_words and getattr(seg, "words", None):
        d["words"] = [_word_to_dict(w) for w in seg.words]

    return d


def _transcript_to_dict(
    transcript: Transcript,
    *,
    include_words: bool = False,
) -> dict[str, Any]:
    """
    Convert a Transcript dataclass to a JSON-serializable dictionary.

    Args:
        transcript: Transcript object to serialize
        include_words: If True, include word-level timestamps in segments

    Returns:
        Dictionary representation suitable for JSON response
    """
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "file_name": transcript.file_name,
        "language": transcript.language,
        "meta": transcript.meta or {},
        "segments": [
            _segment_to_dict(seg, include_words=include_words) for seg in transcript.segments
        ],
    }

    # Include optional diarization fields when present (v1.1+)
    if transcript.speakers is not None:
        data["speakers"] = transcript.speakers
    if transcript.turns is not None:
        data["turns"] = transcript.turns

    return data
