import json
from pathlib import Path

from .models import SCHEMA_VERSION, Segment, Transcript


def write_json(transcript: Transcript, out_path: Path) -> None:
    """
    Write transcript to JSON with a stable schema for downstream processing.

    Schema v2 includes:
    - audio_state field for segments (v1.0+)
    - speakers and turns arrays (v1.1+, optional)
    """
    data = {
        "schema_version": SCHEMA_VERSION,
        "file": transcript.file_name,
        "language": transcript.language,
        "meta": transcript.meta or {},
        "segments": [
            {
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "speaker": s.speaker,
                "tone": s.tone,
                "audio_state": s.audio_state,
            }
            for s in transcript.segments
        ],
    }

    # Add v1.1+ fields if present
    if transcript.speakers is not None:
        data["speakers"] = transcript.speakers
    if transcript.turns is not None:
        data["turns"] = transcript.turns

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
        segment = Segment(
            id=seg_data["id"],
            start=seg_data["start"],
            end=seg_data["end"],
            text=seg_data["text"],
            speaker=seg_data.get("speaker"),
            tone=seg_data.get("tone"),
            audio_state=seg_data.get("audio_state"),  # Gracefully handles missing field
        )
        segments.append(segment)

    transcript = Transcript(
        # Prefer canonical "file", but fall back to "file_name" for API responses.
        file_name=data.get("file") or data.get("file_name", ""),
        language=data.get("language", ""),
        segments=segments,
        meta=data.get("meta"),
        speakers=data.get("speakers"),  # v1.1+ speaker metadata (optional)
        turns=data.get("turns"),  # v1.1+ turn structure (optional)
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
