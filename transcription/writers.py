"""Output writers for transcripts (JSON, TXT, SRT formats).

This module provides functions to write transcripts to various formats:

- write_json: Structured JSON output (schema v2)
- write_txt: Plain text output
- write_srt: SRT subtitle format

All writers handle schema versioning, metadata serialization, and graceful
degradation for complex objects. JSON output is the canonical format and
includes full metadata, enrichment features, and speaker information.
"""

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .models import SCHEMA_VERSION, Chunk, Segment, Transcript, Word

logger = logging.getLogger(__name__)


def _to_dict(obj: object) -> dict[str, Any] | object:
    """
    Convert an object to a dict if possible.

    Conversion rules:
    - dict: returned as-is
    - classes (types): returned unchanged (we only convert instances)
    - object with to_dict(): calls to_dict()
    - dataclass instance: converted via asdict()
    - other: returned unchanged

    Returns:
        dict[str, Any] if conversion succeeded, otherwise the original object.
    """
    if isinstance(obj, dict):
        return obj
    # Skip classes (types) - we only convert instances
    if isinstance(obj, type):
        return obj
    if hasattr(obj, "to_dict"):
        result: dict[str, Any] = obj.to_dict()
        return result
    # Note: mypy's is_dataclass TypeGuard doesn't narrow away type after isinstance check
    if is_dataclass(obj):
        return dict(asdict(obj))  # type: ignore[arg-type]
    return obj


def write_json(transcript: Transcript, out_path: Path) -> None:
    """
    Write transcript to JSON with a stable schema for downstream processing.

    Schema v2 includes:
    - audio_state field for segments (v1.0+)
    - speakers and turns arrays (v1.1+, optional)
    """

    raw_meta: Any = transcript.meta
    if raw_meta is None:
        meta_out: dict[str, Any] = {}
    elif isinstance(raw_meta, dict):
        meta_out = dict(raw_meta)
        diar_meta = meta_out.get("diarization")
        if diar_meta is not None and not isinstance(diar_meta, dict):
            meta_out["diarization"] = _to_dict(diar_meta)
    else:
        converted = _to_dict(raw_meta)
        if isinstance(converted, dict):
            meta_out = converted
        else:
            logger.warning(
                "Unexpected transcript.meta type %r could not be converted to dict; "
                "using empty dict",
                type(raw_meta).__name__,
            )
            meta_out = {}

    data = {
        "schema_version": SCHEMA_VERSION,
        "file": transcript.file_name,
        "language": transcript.language,
        "meta": meta_out,
        "segments": [
            {
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "speaker": s.speaker,
                "tone": s.tone,
                "audio_state": s.audio_state,
                # Word-level timestamps (v1.8+) - only include if present
                **({"words": [w.to_dict() for w in s.words]} if s.words is not None else {}),
            }
            for s in transcript.segments
        ],
    }

    # Add v1.1+ fields if present
    if getattr(transcript, "annotations", None) is not None:
        data["annotations"] = transcript.annotations
    if transcript.speakers is not None:
        data["speakers"] = transcript.speakers
    if transcript.turns is not None:
        data["turns"] = [_to_dict(t) for t in transcript.turns]
    if getattr(transcript, "speaker_stats", None) is not None:
        data["speaker_stats"] = [_to_dict(s) for s in transcript.speaker_stats or []]
    if getattr(transcript, "chunks", None) is not None:
        data["chunks"] = [_to_dict(c) for c in transcript.chunks or []]

    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_transcript_from_json(json_path: Path) -> Transcript:
    """
    Load transcript from JSON file and reconstruct Transcript objects.

    This function gracefully handles both old JSON files (without audio_state)
    and new ones (with audio_state), ensuring backward compatibility. It also
    accepts both the canonical "file" key and API-style "file_name" to keep
    REST responses loadable.

    Args:
        json_path: Path to the JSON file to load.

    Returns:
        Transcript object with all segments reconstructed.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If required fields (id, start, end, text) are missing.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    for seg_data in data.get("segments", []):
        # Parse word-level timestamps if present (v1.8+)
        words: list[Word] | None = None
        raw_words = seg_data.get("words")
        if raw_words is not None:
            words = [Word.from_dict(w) for w in raw_words]

        segment = Segment(
            id=seg_data["id"],
            start=seg_data["start"],
            end=seg_data["end"],
            text=seg_data["text"],
            speaker=seg_data.get("speaker"),
            tone=seg_data.get("tone"),
            audio_state=seg_data.get("audio_state"),  # Gracefully handles missing field
            words=words,  # Word-level timestamps (v1.8+)
        )
        segments.append(segment)

    chunk_entries = data.get("chunks")
    chunks = None
    if chunk_entries is not None:
        parsed: list[Chunk | dict] = []
        for chunk in chunk_entries:
            if isinstance(chunk, dict):
                parsed.append(Chunk.from_dict(chunk))
            else:
                parsed.append(chunk)
        chunks = parsed

    transcript = Transcript(
        # Prefer canonical "file", but fall back to "file_name" for API responses.
        file_name=data.get("file") or data.get("file_name", ""),
        language=data.get("language", ""),
        segments=segments,
        meta=data.get("meta"),
        annotations=data.get("annotations"),
        speakers=data.get("speakers"),  # v1.1+ speaker metadata (optional)
        turns=data.get("turns"),  # v1.1+ turn structure (optional)
        speaker_stats=data.get("speaker_stats"),  # v1.2+ speaker aggregates (optional)
        chunks=chunks,  # v1.3+ turn-aware chunks (optional)
    )

    return transcript


def write_txt(transcript: Transcript, out_path: Path) -> None:
    """
    Write a human-readable, timestamped text transcript.
    """
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# File: {transcript.file_name}\n")
        f.write(f"# Language: {transcript.language}\n\n")
        for s in transcript.segments:
            f.write(f"{s.start:8.2f}–{s.end:8.2f}: {s.text}\n")


def _fmt_srt_ts(t: float) -> str:
    """
    Format seconds as SRT timestamp (HH:MM:SS,mmm).
    """
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t * 1000) % 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(transcript: Transcript, out_path: Path) -> None:
    """
    Write an SRT subtitle file for the transcript.
    """
    with out_path.open("w", encoding="utf-8") as f:
        for idx, s in enumerate(transcript.segments, start=1):
            f.write(f"{idx}\n")
            f.write(f"{_fmt_srt_ts(s.start)} --> {_fmt_srt_ts(s.end)}\n")
            f.write(s.text + "\n\n")
