"""Transcript export helpers for CSV/HTML/VTT/TextGrid."""

from __future__ import annotations

import csv
import hashlib
import html
from pathlib import Path
from typing import Any

from .models import Transcript
from .speaker_id import get_speaker_label_or_id
from .turn_helpers import turn_to_dict

ExportFormat = str
SUPPORTED_EXPORT_FORMATS: set[ExportFormat] = {"csv", "html", "vtt", "textgrid"}


def _coerce_speaker_id(value: Any) -> str:
    """Return a speaker identifier or placeholder.

    Note: This is a thin wrapper around get_speaker_label_or_id() for backward
    compatibility. New code should use get_speaker_label_or_id() directly.
    """
    return get_speaker_label_or_id(value, fallback="unknown")


def _collect_segments(transcript: Transcript, unit: str = "segments") -> list[dict[str, Any]]:
    """Normalize transcript content into a list of rows."""
    rows: list[dict[str, Any]] = []

    if unit == "turns" and transcript.turns:
        for idx, turn in enumerate(transcript.turns):
            try:
                turn_dict = turn_to_dict(turn)
            except TypeError:
                # Fallback for unsupported types
                turn_dict = {"id": str(idx)}
            rows.append(
                {
                    "id": turn_dict.get("id", f"turn_{idx}"),
                    "start": float(turn_dict.get("start", 0.0)),
                    "end": float(turn_dict.get("end", 0.0)),
                    "speaker": _coerce_speaker_id(turn_dict.get("speaker_id")),
                    "text": str(turn_dict.get("text", "")).strip(),
                }
            )
        return rows

    for seg in transcript.segments:
        rows.append(
            {
                "id": getattr(seg, "id", len(rows)),
                "start": float(getattr(seg, "start", 0.0)),
                "end": float(getattr(seg, "end", 0.0)),
                "speaker": _coerce_speaker_id(getattr(seg, "speaker", None)),
                "text": str(getattr(seg, "text", "")).strip(),
            }
        )
    return rows


def export_csv(transcript: Transcript, output_path: Path, unit: str = "segments") -> None:
    """Write transcript rows to CSV."""
    rows = _collect_segments(transcript, unit=unit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "start", "end", "speaker", "text"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for WebVTT."""
    total_ms = int(round(seconds * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{ms:03}"


def export_vtt(transcript: Transcript, output_path: Path, unit: str = "segments") -> None:
    """Write WebVTT subtitles."""
    rows = _collect_segments(transcript, unit=unit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for row in rows:
            f.write(f"{_format_ts(row['start'])} --> {_format_ts(row['end'])}\n")
            # WebVTT spec allows HTML tags, so escape to prevent XSS
            speaker_safe = html.escape(row["speaker"]) if row.get("speaker") else ""
            prefix = f"{speaker_safe}: " if speaker_safe else ""
            text_safe = html.escape(row["text"])
            f.write(f"{prefix}{text_safe}\n\n")


def _speaker_color(speaker: str) -> str:
    """Generate a stable pastel color for a speaker label."""
    digest = hashlib.sha1(speaker.encode("utf-8")).hexdigest()
    hue = int(digest[:2], 16) / 255
    saturation = 0.55
    lightness = 0.82
    return f"hsl({int(hue * 360)}, {int(saturation * 100)}%, {int(lightness * 100)}%)"


def export_html(transcript: Transcript, output_path: Path, unit: str = "segments") -> None:
    """Write a simple annotated HTML transcript."""
    rows = _collect_segments(transcript, unit=unit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8" />',
        "<title>Transcript</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; color: #222; margin: 24px; }",
        ".turn { padding: 6px 10px; border-radius: 8px; margin-bottom: 8px; }",
        ".speaker { font-weight: 600; margin-right: 8px; }",
        ".time { color: #666; font-size: 0.9em; margin-right: 6px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h2>{html.escape(transcript.file_name)}</h2>",
    ]
    for row in rows:
        color = _speaker_color(row["speaker"])
        start = _format_ts(row["start"])
        end = _format_ts(row["end"])
        parts.append(
            f'<div class="turn" style="background:{color};">'
            f'<span class="speaker">{html.escape(str(row["speaker"]))}</span>'
            f'<span class="time">{start} - {end}</span>'
            f"<div>{html.escape(row['text'])}</div>"
            "</div>"
        )
    parts.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(parts), encoding="utf-8")


def export_textgrid(transcript: Transcript, output_path: Path, unit: str = "segments") -> None:
    """Write a simple single-tier TextGrid."""
    rows = _collect_segments(transcript, unit=unit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xmin = 0.0
    xmax = max((row["end"] for row in rows), default=0.0)

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {xmin:.6f}",
        f"xmax = {xmax:.6f}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "segments"',
        f"        xmin = {xmin:.6f}",
        f"        xmax = {xmax:.6f}",
        f"        intervals: size = {len(rows)}",
    ]

    for idx, row in enumerate(rows, start=1):
        # Escape quotes for TextGrid format (Praat uses '' for literal quotes)
        speaker_escaped = str(row["speaker"]).replace('"', "''")
        text_escaped = str(row["text"]).replace('"', "''")
        label = f"{speaker_escaped}: {text_escaped}".strip()
        lines.extend(
            [
                f"        intervals [{idx}]:",
                f"            xmin = {row['start']:.6f}",
                f"            xmax = {row['end']:.6f}",
                f'            text = "{label}"',
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_transcript(
    transcript: Transcript, fmt: ExportFormat, output_path: Path, unit: str = "segments"
) -> None:
    """Dispatch export to the requested format."""
    fmt = fmt.lower()
    if fmt not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {fmt}")

    if fmt == "csv":
        export_csv(transcript, output_path, unit=unit)
    elif fmt == "html":
        export_html(transcript, output_path, unit=unit)
    elif fmt == "vtt":
        export_vtt(transcript, output_path, unit=unit)
    elif fmt == "textgrid":
        export_textgrid(transcript, output_path, unit=unit)
