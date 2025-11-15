"""Generate indices of all transcripts for discovery and analysis."""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def generate_index(json_dir: Path, output_path: Path, format: str = "json") -> None:
    """
    Generate an index of all transcripts.

    Args:
        json_dir: Directory containing transcript JSON files
        output_path: Where to write the index
        format: "json" or "csv"
    """
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    # Collect transcript metadata
    transcripts = []
    json_files = sorted(json_dir.glob("*.json"))

    logger.info(f"Scanning {len(json_files)} transcript files...")

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract metadata
            meta = data.get("meta", {})
            enrichments = meta.get("enrichments", {})

            transcript_info = {
                "file_name": data.get("file_name", ""),
                "json_path": str(json_path.relative_to(json_path.parent.parent)),
                "duration_sec": meta.get("audio_duration_sec", 0.0),
                "language": data.get("language", ""),
                "segment_count": len(data.get("segments", [])),
                "has_tone": "tone_version" in enrichments,
                "has_speakers": "speaker_version" in enrichments,
                "transcribed_at": meta.get("generated_at", ""),
                "model_name": meta.get("model_name", ""),
            }

            transcripts.append(transcript_info)

        except Exception as e:
            logger.warning(f"Failed to read {json_path.name}: {e}")
            continue

    # Calculate totals
    total_duration = sum(t["duration_sec"] for t in transcripts)

    # Generate index
    if format == "json":
        _write_json_index(transcripts, total_duration, output_path)
    elif format == "csv":
        _write_csv_index(transcripts, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Generated {format.upper()} index: {output_path}")


def _write_json_index(transcripts: List[Dict[str, Any]], total_duration: float, output_path: Path):
    """Write index in JSON format."""
    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(transcripts),
        "total_duration_sec": total_duration,
        "total_duration_hours": total_duration / 3600,
        "transcripts": transcripts,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def _write_csv_index(transcripts: List[Dict[str, Any]], output_path: Path):
    """Write index in CSV format."""
    if not transcripts:
        logger.warning("No transcripts to write to CSV")
        return

    fieldnames = [
        "file_name",
        "json_path",
        "duration_sec",
        "duration_min",
        "language",
        "segment_count",
        "has_tone",
        "has_speakers",
        "transcribed_at",
        "model_name",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in transcripts:
            row = t.copy()
            row["duration_min"] = round(t["duration_sec"] / 60, 2)
            writer.writerow(row)
