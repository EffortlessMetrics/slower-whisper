"""Consolidated speaker ID extraction utility.

This module provides the canonical way to extract speaker IDs from
various speaker representations (dict, string, dataclass, etc.).

All modules that need to extract speaker IDs should use get_speaker_id()
from this module instead of implementing their own extraction logic.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def get_speaker_id(speaker: Any) -> str | None:
    """
    Extract speaker ID from a speaker dict/object.

    This is the canonical utility for extracting speaker IDs across the
    codebase. It handles multiple formats gracefully:

    - None → None
    - str → str (already an ID)
    - dict with "id" key → str(dict["id"])
    - Dataclass with speaker_id attribute → str(speaker_id)
    - Other → str(speaker)

    Args:
        speaker: Speaker object, dict, ID string, or None

    Returns:
        Speaker ID string or None if not found/applicable

    Examples:
        >>> get_speaker_id(None)
        None
        >>> get_speaker_id("spk_0")
        'spk_0'
        >>> get_speaker_id({"id": "spk_1", "confidence": 0.9})
        'spk_1'
        >>> get_speaker_id({"label": "Alice"})  # No "id" key
        None
    """
    if speaker is None:
        return None

    if isinstance(speaker, str):
        return speaker

    if isinstance(speaker, dict):
        raw_id = speaker.get("id")
        return str(raw_id) if raw_id is not None else None

    # Handle dataclasses with speaker_id attribute
    # Note: is_dataclass() returns True for both instances and types,
    # so we check it's not a type to ensure we have an instance.
    # mypy can't narrow `Any` through is_dataclass(), so we ignore unreachable.
    if is_dataclass(speaker) and not isinstance(speaker, type):
        data = asdict(speaker)  # type: ignore[unreachable]
        speaker_id = data.get("speaker_id") or data.get("id")
        return str(speaker_id) if speaker_id is not None else None

    # Handle objects with speaker_id attribute
    speaker_id = getattr(speaker, "speaker_id", None)
    if speaker_id is not None:
        return str(speaker_id)

    # Fallback: convert to string
    return str(speaker)


def get_speaker_label_or_id(speaker: Any, fallback: str = "unknown") -> str:
    """
    Extract a display label for a speaker, with fallback.

    This is an enhanced version that also checks for "label" key
    in speaker dicts, useful for export/display contexts.

    Args:
        speaker: Speaker object, dict, ID string, or None
        fallback: Value to return if no ID can be extracted

    Returns:
        Speaker label/ID string, or fallback if not found

    Examples:
        >>> get_speaker_label_or_id({"id": "spk_0", "label": "Alice"})
        'Alice'
        >>> get_speaker_label_or_id({"id": "spk_0"})
        'spk_0'
        >>> get_speaker_label_or_id(None)
        'unknown'
    """
    if speaker is None:
        return fallback

    if isinstance(speaker, str):
        return speaker

    if isinstance(speaker, dict):
        # Prefer label over id for display
        label = speaker.get("label")
        if label:
            return str(label)
        raw_id = speaker.get("id")
        if raw_id is not None:
            return str(raw_id)
        return fallback

    # Handle dataclasses
    # Note: is_dataclass() returns True for both instances and types,
    # so we check it's not a type to ensure we have an instance.
    # mypy can't narrow `Any` through is_dataclass(), so we ignore unreachable.
    if is_dataclass(speaker) and not isinstance(speaker, type):
        data = asdict(speaker)  # type: ignore[unreachable]
        speaker_id = data.get("id") or data.get("speaker_id")
        return str(speaker_id) if speaker_id is not None else fallback

    # Handle objects with speaker_id attribute
    speaker_id = getattr(speaker, "speaker_id", None)
    if speaker_id is not None:
        return str(speaker_id)

    return str(speaker)
