"""Generate analytical reports from transcripts."""

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def generate_tone_report(json_dir: Path, output_path: Path) -> None:
    """
    Generate a tone analysis report from all enriched transcripts.

    Args:
        json_dir: Directory containing transcript JSON files
        output_path: Where to write the Markdown report
    """
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    # Collect tone statistics
    all_tones = []
    file_stats = []
    json_files = sorted(json_dir.glob("*.json"))

    logger.info(f"Analyzing tone in {len(json_files)} transcript files...")

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if tone enriched
            enrichments = data.get("meta", {}).get("enrichments", {})
            if "tone_version" not in enrichments:
                logger.debug(f"Skipping {json_path.name} - no tone enrichment")
                continue

            # Extract tones
            segments = data.get("segments", [])
            file_tones = [seg.get("tone") for seg in segments if seg.get("tone")]

            if not file_tones:
                continue

            all_tones.extend(file_tones)

            # Per-file stats
            tone_counts = Counter(file_tones)
            duration = data.get("meta", {}).get("audio_duration_sec", 0.0)

            file_stats.append({
                "file_name": data.get("file_name", ""),
                "duration_sec": duration,
                "segment_count": len(segments),
                "tone_counts": tone_counts,
            })

        except Exception as e:
            logger.warning(f"Failed to analyze {json_path.name}: {e}")
            continue

    if not all_tones:
        logger.warning("No tone-enriched transcripts found")
        return

    # Generate report
    _write_tone_report(all_tones, file_stats, output_path)
    logger.info(f"Generated tone report: {output_path}")


def _write_tone_report(all_tones: List[str], file_stats: List[Dict[str, Any]], output_path: Path):
    """Write tone analysis report in Markdown format."""
    total_segments = len(all_tones)
    tone_counts = Counter(all_tones)

    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write("# Tone Analysis Report\n\n")
        f.write(f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write(f"**Files analyzed**: {len(file_stats)}\n\n")
        f.write(f"**Total segments**: {total_segments:,}\n\n")

        # Overall tone distribution
        f.write("## Overall Tone Distribution\n\n")
        for tone, count in tone_counts.most_common():
            percentage = (count / total_segments) * 100
            f.write(f"- **{tone.capitalize()}**: {percentage:.1f}% ({count:,} segments)\n")
        f.write("\n")

        # Tone chart (simple text-based)
        f.write("### Distribution Chart\n\n")
        f.write("```\n")
        max_count = max(tone_counts.values())
        for tone, count in sorted(tone_counts.items(), key=lambda x: -x[1]):
            bar_length = int((count / max_count) * 50)
            bar = "█" * bar_length
            percentage = (count / total_segments) * 100
            f.write(f"{tone:12} {bar} {percentage:5.1f}% ({count:,})\n")
        f.write("```\n\n")

        # Per-file breakdown
        f.write("## Per-File Summary\n\n")
        for file_stat in file_stats:
            duration_min = file_stat["duration_sec"] / 60
            f.write(f"### {file_stat['file_name']}\n\n")
            f.write(f"- **Duration**: {duration_min:.1f} min\n")
            f.write(f"- **Segments**: {file_stat['segment_count']}\n")
            f.write(f"- **Tone breakdown**:\n")

            total_file_segments = sum(file_stat["tone_counts"].values())
            for tone, count in file_stat["tone_counts"].most_common():
                percentage = (count / total_file_segments) * 100
                f.write(f"  - {tone.capitalize()}: {percentage:.1f}% ({count})\n")
            f.write("\n")

        # Summary insights
        f.write("## Insights\n\n")

        # Dominant tone
        dominant_tone, dominant_count = tone_counts.most_common(1)[0]
        dominant_pct = (dominant_count / total_segments) * 100
        f.write(f"- **Most common tone**: {dominant_tone.capitalize()} ({dominant_pct:.1f}%)\n")

        # Emotional variance
        unique_tones = len(tone_counts)
        f.write(f"- **Unique tones detected**: {unique_tones}\n")

        # Average tones per file
        avg_segments_per_file = total_segments / len(file_stats) if file_stats else 0
        f.write(f"- **Average segments per file**: {avg_segments_per_file:.0f}\n")

        f.write("\n---\n\n")
        f.write("*Generated by slower-whisper tone enrichment pipeline*\n")


def generate_speaker_report(json_dir: Path, output_path: Path) -> None:
    """
    Generate a speaker diarization report from all enriched transcripts.

    Args:
        json_dir: Directory containing transcript JSON files
        output_path: Where to write the Markdown report
    """
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    file_stats = []
    json_files = sorted(json_dir.glob("*.json"))

    logger.info(f"Analyzing speakers in {len(json_files)} transcript files...")

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if speaker enriched
            enrichments = data.get("meta", {}).get("enrichments", {})
            if "speaker_version" not in enrichments:
                logger.debug(f"Skipping {json_path.name} - no speaker enrichment")
                continue

            # Extract speakers
            segments = data.get("segments", [])
            speakers = [seg.get("speaker") for seg in segments if seg.get("speaker")]

            if not speakers:
                continue

            speaker_counts = Counter(speakers)
            duration = data.get("meta", {}).get("audio_duration_sec", 0.0)

            # Calculate speaking time per speaker
            speaker_times = defaultdict(float)
            for seg in segments:
                speaker = seg.get("speaker")
                if speaker:
                    seg_duration = seg.get("end", 0) - seg.get("start", 0)
                    speaker_times[speaker] += seg_duration

            file_stats.append({
                "file_name": data.get("file_name", ""),
                "duration_sec": duration,
                "segment_count": len(segments),
                "speaker_counts": speaker_counts,
                "speaker_times": speaker_times,
            })

        except Exception as e:
            logger.warning(f"Failed to analyze {json_path.name}: {e}")
            continue

    if not file_stats:
        logger.warning("No speaker-enriched transcripts found")
        return

    # Generate report
    _write_speaker_report(file_stats, output_path)
    logger.info(f"Generated speaker report: {output_path}")


def _write_speaker_report(file_stats: List[Dict[str, Any]], output_path: Path):
    """Write speaker analysis report in Markdown format."""
    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write("# Speaker Diarization Report\n\n")
        f.write(f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write(f"**Files analyzed**: {len(file_stats)}\n\n")

        # Per-file breakdown
        f.write("## Per-File Summary\n\n")
        for file_stat in file_stats:
            duration_min = file_stat["duration_sec"] / 60
            f.write(f"### {file_stat['file_name']}\n\n")
            f.write(f"- **Duration**: {duration_min:.1f} min\n")
            f.write(f"- **Total segments**: {file_stat['segment_count']}\n")
            f.write(f"- **Unique speakers**: {len(file_stat['speaker_counts'])}\n\n")

            f.write("**Speaker breakdown**:\n\n")
            for speaker, count in file_stat["speaker_counts"].most_common():
                time_sec = file_stat["speaker_times"][speaker]
                time_min = time_sec / 60
                time_pct = (time_sec / file_stat["duration_sec"]) * 100 if file_stat["duration_sec"] > 0 else 0
                f.write(f"- **{speaker}**: {count} segments, {time_min:.1f} min ({time_pct:.1f}% of total)\n")
            f.write("\n")

        f.write("\n---\n\n")
        f.write("*Generated by slower-whisper speaker diarization pipeline*\n")
