import json
from pathlib import Path
from .models import Transcript, SCHEMA_VERSION


def write_json(transcript: Transcript, out_path: Path) -> None:
    """
    Write transcript to JSON with a stable schema for downstream processing.
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
            }
            for s in transcript.segments
        ],
    }
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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


def _fmt_vtt_ts(t: float) -> str:
    """
    Format seconds as WebVTT timestamp (HH:MM:SS.mmm).
    """
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t * 1000) % 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def write_vtt(transcript: Transcript, out_path: Path) -> None:
    """
    Write a WebVTT subtitle file for the transcript.

    WebVTT is a web-friendly subtitle format with better browser support
    than SRT. It's ideal for HTML5 video players.
    """
    with out_path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for idx, s in enumerate(transcript.segments, start=1):
            # Optional: include speaker and tone as cue identifiers
            cue_id = f"{idx}"
            if s.speaker:
                cue_id += f" [{s.speaker}]"
            if s.tone:
                cue_id += f" ({s.tone})"

            f.write(f"{cue_id}\n")
            f.write(f"{_fmt_vtt_ts(s.start)} --> {_fmt_vtt_ts(s.end)}\n")
            f.write(s.text + "\n\n")
