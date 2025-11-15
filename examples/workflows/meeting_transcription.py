#!/usr/bin/env python3
"""
Meeting Transcription and Analysis Workflow

This workflow processes meeting recordings (Zoom, Teams, etc.) to create:
- Searchable transcripts with timestamps
- Emotional markers for engagement tracking
- Key moment detection (decisions, action items, heated discussions)
- Comprehensive analysis reports

Usage:
    python meeting_transcription.py --audio raw_audio/team_meeting.mp3
    python meeting_transcription.py --config configs/meeting_config.json --audio meeting.mp3
    python meeting_transcription.py --batch raw_audio/*.mp3
"""

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transcription.models import Transcript
from transcription.writers import load_transcript_from_json, write_txt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MeetingConfig:
    """Configuration for meeting transcription workflow."""

    # Transcription settings
    model: str = "large-v3"
    language: str = "en"
    device: str = "cuda"
    compute_type: str = "float16"
    vad_min_silence_ms: int = 500

    # Enrichment settings
    enable_prosody: bool = True
    enable_emotion_dimensional: bool = True
    enable_emotion_categorical: bool = True

    # Analysis settings
    detect_key_moments: bool = True
    engagement_tracking: bool = True
    detect_action_items: bool = True
    detect_decisions: bool = True

    # Output settings
    export_formats: list[str] = None

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["json", "txt", "csv", "annotated"]

    @classmethod
    def from_json(cls, config_path: Path) -> "MeetingConfig":
        """Load configuration from JSON file."""
        with open(config_path) as f:
            config_data = json.load(f)

        # Flatten nested structure if present
        if "transcription" in config_data:
            trans = config_data["transcription"]
            config_data.update(
                {
                    "model": trans.get("model", "large-v3"),
                    "language": trans.get("language", "en"),
                    "device": trans.get("device", "cuda"),
                    "compute_type": trans.get("compute_type", "float16"),
                    "vad_min_silence_ms": trans.get("vad_min_silence_ms", 500),
                }
            )

        if "enrichment" in config_data:
            enrich = config_data["enrichment"]
            config_data.update(
                {
                    "enable_prosody": enrich.get("enable_prosody", True),
                    "enable_emotion_dimensional": enrich.get("enable_emotion_dimensional", True),
                    "enable_emotion_categorical": enrich.get("enable_emotion_categorical", True),
                }
            )

        if "analysis" in config_data:
            analysis = config_data["analysis"]
            config_data.update(
                {
                    "detect_key_moments": analysis.get("detect_key_moments", True),
                    "engagement_tracking": analysis.get("engagement_tracking", True),
                    "detect_action_items": analysis.get("detect_action_items", True),
                    "detect_decisions": analysis.get("detect_decisions", True),
                    "export_formats": analysis.get(
                        "export_formats", ["json", "txt", "csv", "annotated"]
                    ),
                }
            )

        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in config_data.items() if k in valid_fields}

        return cls(**filtered_data)


class MeetingAnalyzer:
    """Analyze meeting transcripts for key moments and engagement."""

    def __init__(self, transcript: Transcript):
        self.transcript = transcript

    def detect_key_moments(self) -> list[dict[str, Any]]:
        """
        Detect key moments in the meeting based on emotional and prosodic features.

        Key moments include:
        - High engagement/excitement
        - Heated discussions
        - Unanimous agreement
        - Decisions being made
        - Action items being assigned
        """
        key_moments = []

        for segment in self.transcript.segments:
            if not segment.audio_state:
                continue

            moment_type = None
            confidence = 0.0
            indicators = []

            # Extract features
            emotion = segment.audio_state.get("emotion", {})
            prosody_pitch = segment.audio_state.get("pitch", {})
            prosody_energy = segment.audio_state.get("energy", {})
            prosody_rate = segment.audio_state.get("rate", {})

            # High excitement (high energy + high pitch + fast rate)
            if (
                prosody_energy.get("level") == "loud"
                and prosody_pitch.get("level") == "high"
                and prosody_rate.get("level") == "fast"
            ):
                moment_type = "high_excitement"
                confidence = 0.85
                indicators = ["loud voice", "high pitch", "fast speech"]

            # Heated discussion (high arousal + varied pitch)
            elif emotion.get("dimensional", {}).get("arousal", {}).get("score", 0) > 0.7:
                if prosody_pitch.get("contour") in ["rising", "varied"]:
                    moment_type = "heated_discussion"
                    confidence = 0.78
                    indicators = ["high emotional intensity", "pitch variation"]

            # Decision point (declarative + confident)
            elif prosody_pitch.get("contour") == "falling" and prosody_energy.get("level") in [
                "moderate",
                "loud",
            ]:
                # Check for decision keywords in text
                decision_keywords = [
                    "decide",
                    "decided",
                    "will do",
                    "going to",
                    "agreed",
                    "approve",
                    "let's",
                    "we'll",
                ]
                if any(kw in segment.text.lower() for kw in decision_keywords):
                    moment_type = "decision_point"
                    confidence = 0.82
                    indicators = ["declarative tone", "decision language"]

            # Action item (imperative + clear)
            elif prosody_pitch.get("contour") == "rising":
                action_keywords = [
                    "need to",
                    "should",
                    "will",
                    "can you",
                    "please",
                    "action",
                    "task",
                    "assign",
                ]
                if any(kw in segment.text.lower() for kw in action_keywords):
                    moment_type = "action_item"
                    confidence = 0.75
                    indicators = ["action language", "rising intonation"]

            # Agreement (positive emotion + low energy)
            elif emotion.get("dimensional", {}).get("valence", {}).get(
                "score", 0
            ) > 0.65 and prosody_energy.get("level") in ["quiet", "moderate"]:
                agreement_keywords = [
                    "agree",
                    "yes",
                    "exactly",
                    "right",
                    "sounds good",
                    "perfect",
                    "okay",
                ]
                if any(kw in segment.text.lower() for kw in agreement_keywords):
                    moment_type = "agreement"
                    confidence = 0.70
                    indicators = ["positive sentiment", "agreement language"]

            if moment_type:
                key_moments.append(
                    {
                        "segment_id": segment.id,
                        "timestamp": segment.start,
                        "duration": segment.end - segment.start,
                        "type": moment_type,
                        "confidence": confidence,
                        "text": segment.text,
                        "indicators": indicators,
                    }
                )

        return key_moments

    def calculate_engagement_timeline(self, window_size: int = 10) -> list[dict[str, Any]]:
        """
        Calculate engagement level over time using a sliding window.

        Args:
            window_size: Number of segments to average over

        Returns:
            List of engagement measurements over time
        """
        timeline = []

        for i in range(0, len(self.transcript.segments), window_size // 2):
            window = self.transcript.segments[i : i + window_size]

            if not window:
                continue

            # Calculate average features for window
            arousal_scores = []
            energy_levels = []
            rate_levels = []

            for seg in window:
                if seg.audio_state:
                    emotion = seg.audio_state.get("emotion", {})
                    arousal = emotion.get("dimensional", {}).get("arousal", {}).get("score")
                    if arousal is not None:
                        arousal_scores.append(arousal)

                    energy = seg.audio_state.get("energy", {})
                    if energy.get("db_rms"):
                        energy_levels.append(energy["db_rms"])

                    rate = seg.audio_state.get("rate", {})
                    if rate.get("syllables_per_sec"):
                        rate_levels.append(rate["syllables_per_sec"])

            # Calculate engagement score (0-1)
            engagement_score = 0.0
            if arousal_scores:
                engagement_score += (sum(arousal_scores) / len(arousal_scores)) * 0.5

            if energy_levels:
                # Normalize energy to 0-1 (assuming -40 to 0 dB range)
                normalized_energy = [(e + 40) / 40 for e in energy_levels]
                engagement_score += (sum(normalized_energy) / len(normalized_energy)) * 0.3

            if rate_levels:
                # Normalize rate to 0-1 (assuming 2-8 syl/sec range)
                normalized_rate = [(r - 2) / 6 for r in rate_levels]
                engagement_score += (sum(normalized_rate) / len(normalized_rate)) * 0.2

            engagement_level = "low"
            if engagement_score > 0.7:
                engagement_level = "high"
            elif engagement_score > 0.4:
                engagement_level = "medium"

            timeline.append(
                {
                    "timestamp": window[0].start,
                    "end_timestamp": window[-1].end,
                    "engagement_score": engagement_score,
                    "engagement_level": engagement_level,
                    "window_segments": len(window),
                }
            )

        return timeline

    def generate_summary(self, key_moments: list[dict], engagement_timeline: list[dict]) -> str:
        """Generate a text summary of the meeting analysis."""
        lines = []
        lines.append("MEETING ANALYSIS SUMMARY")
        lines.append("=" * 80)
        lines.append("")

        # Meeting metadata
        lines.append(f"Meeting Recording: {self.transcript.file_name}")
        lines.append(f"Language: {self.transcript.language}")
        lines.append(f"Duration: {self.transcript.segments[-1].end / 60:.1f} minutes")
        lines.append(f"Total Segments: {len(self.transcript.segments)}")
        lines.append("")

        # Engagement overview
        if engagement_timeline:
            avg_engagement = sum(t["engagement_score"] for t in engagement_timeline) / len(
                engagement_timeline
            )
            high_engagement = sum(1 for t in engagement_timeline if t["engagement_level"] == "high")

            lines.append("--- Engagement Overview ---")
            lines.append(
                f"Average Engagement: {avg_engagement:.2f} ({'High' if avg_engagement > 0.7 else 'Medium' if avg_engagement > 0.4 else 'Low'})"
            )
            lines.append(f"High Engagement Periods: {high_engagement} / {len(engagement_timeline)}")
            lines.append("")

        # Key moments summary
        if key_moments:
            from collections import Counter

            moment_types = Counter(m["type"] for m in key_moments)

            lines.append("--- Key Moments Detected ---")
            lines.append(f"Total Key Moments: {len(key_moments)}")
            lines.append("")

            for moment_type, count in moment_types.most_common():
                lines.append(f"  {moment_type.replace('_', ' ').title()}: {count}")

            lines.append("")
            lines.append("--- Key Moment Details ---")
            lines.append("")

            # Show key moments by type
            for moment_type in [
                "decision_point",
                "action_item",
                "heated_discussion",
                "high_excitement",
                "agreement",
            ]:
                type_moments = [m for m in key_moments if m["type"] == moment_type]
                if type_moments:
                    lines.append(f"{moment_type.replace('_', ' ').title()}s:")
                    for moment in type_moments[:5]:  # Show first 5 of each type
                        timestamp = moment["timestamp"]
                        minutes = int(timestamp // 60)
                        seconds = int(timestamp % 60)
                        text_preview = (
                            moment["text"][:70] + "..."
                            if len(moment["text"]) > 70
                            else moment["text"]
                        )
                        lines.append(f"  [{minutes:02d}:{seconds:02d}] {text_preview}")
                        lines.append(
                            f"    Confidence: {moment['confidence']:.0%} | Indicators: {', '.join(moment['indicators'])}"
                        )

                    if len(type_moments) > 5:
                        lines.append(f"  ... and {len(type_moments) - 5} more")
                    lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


def create_annotated_transcript(transcript: Transcript, output_path: Path):
    """Create a human-readable annotated transcript with emotional markers."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Meeting Transcript: {transcript.file_name}\n")
        f.write(f"# Language: {transcript.language}\n")
        f.write(f"# Duration: {transcript.segments[-1].end / 60:.1f} minutes\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        f.write("=" * 80 + "\n\n")

        for segment in transcript.segments:
            # Format timestamp
            start_min = int(segment.start // 60)
            start_sec = int(segment.start % 60)
            end_min = int(segment.end // 60)
            end_sec = int(segment.end % 60)

            f.write(f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]\n")
            f.write(f"{segment.text}\n")

            # Add audio state annotations
            if segment.audio_state:
                annotations = []

                # Prosody
                if "pitch" in segment.audio_state:
                    pitch = segment.audio_state["pitch"]
                    annotations.append(
                        f"Pitch: {pitch.get('level', 'unknown')} ({pitch.get('contour', 'flat')})"
                    )

                if "energy" in segment.audio_state:
                    energy = segment.audio_state["energy"]
                    annotations.append(f"Energy: {energy.get('level', 'unknown')}")

                if "rate" in segment.audio_state:
                    rate = segment.audio_state["rate"]
                    annotations.append(f"Speech: {rate.get('level', 'unknown')} pace")

                # Emotion
                if "emotion" in segment.audio_state:
                    emotion = segment.audio_state["emotion"]
                    if "categorical" in emotion:
                        cat = emotion["categorical"]
                        annotations.append(
                            f"Emotion: {cat.get('primary', 'neutral')} ({cat.get('confidence', 0):.0%})"
                        )
                    elif "dimensional" in emotion:
                        dim = emotion["dimensional"]
                        valence = dim.get("valence", {}).get("level", "neutral")
                        arousal = dim.get("arousal", {}).get("level", "moderate")
                        annotations.append(f"Sentiment: {valence}, Energy: {arousal}")

                # Pauses
                if "pauses" in segment.audio_state:
                    pause_count = segment.audio_state["pauses"].get("count", 0)
                    if pause_count > 0:
                        annotations.append(f"Pauses: {pause_count}")

                if annotations:
                    f.write("  >> " + " | ".join(annotations) + "\n")

            f.write("\n")


def export_to_csv(
    transcript: Transcript,
    key_moments: list[dict],
    engagement_timeline: list[dict],
    output_path: Path,
):
    """Export meeting data to CSV for spreadsheet analysis."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "segment_id",
            "timestamp",
            "timestamp_formatted",
            "duration",
            "text",
            "pitch_level",
            "pitch_contour",
            "energy_level",
            "speech_rate_level",
            "pause_count",
            "emotion_category",
            "emotion_confidence",
            "valence_score",
            "arousal_score",
            "engagement_level",
            "is_key_moment",
            "key_moment_type",
            "key_moment_confidence",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Create lookup for engagement levels and key moments
        engagement_map = {}
        for period in engagement_timeline:
            for seg in transcript.segments:
                if period["timestamp"] <= seg.start < period["end_timestamp"]:
                    engagement_map[seg.id] = period["engagement_level"]

        key_moment_map = {m["segment_id"]: m for m in key_moments}

        for segment in transcript.segments:
            minutes = int(segment.start // 60)
            seconds = int(segment.start % 60)

            row = {
                "segment_id": segment.id,
                "timestamp": f"{segment.start:.2f}",
                "timestamp_formatted": f"{minutes:02d}:{seconds:02d}",
                "duration": f"{segment.end - segment.start:.2f}",
                "text": segment.text,
            }

            # Extract audio features
            if segment.audio_state:
                pitch = segment.audio_state.get("pitch", {})
                row["pitch_level"] = pitch.get("level", "")
                row["pitch_contour"] = pitch.get("contour", "")

                energy = segment.audio_state.get("energy", {})
                row["energy_level"] = energy.get("level", "")

                rate = segment.audio_state.get("rate", {})
                row["speech_rate_level"] = rate.get("level", "")

                pauses = segment.audio_state.get("pauses", {})
                row["pause_count"] = pauses.get("count", 0)

                emotion = segment.audio_state.get("emotion", {})
                if "categorical" in emotion:
                    cat = emotion["categorical"]
                    row["emotion_category"] = cat.get("primary", "")
                    row["emotion_confidence"] = cat.get("confidence", "")

                if "dimensional" in emotion:
                    dim = emotion["dimensional"]
                    row["valence_score"] = dim.get("valence", {}).get("score", "")
                    row["arousal_score"] = dim.get("arousal", {}).get("score", "")

            # Add engagement level
            row["engagement_level"] = engagement_map.get(segment.id, "")

            # Add key moment info
            if segment.id in key_moment_map:
                moment = key_moment_map[segment.id]
                row["is_key_moment"] = "TRUE"
                row["key_moment_type"] = moment["type"]
                row["key_moment_confidence"] = moment["confidence"]
            else:
                row["is_key_moment"] = "FALSE"
                row["key_moment_type"] = ""
                row["key_moment_confidence"] = ""

            writer.writerow(row)


def process_meeting(audio_path: Path, output_dir: Path, config: MeetingConfig):
    """
    Process a single meeting recording through the complete workflow.

    Args:
        audio_path: Path to audio file
        output_dir: Directory for outputs
        config: Meeting configuration
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = audio_path.stem
    json_path = output_dir / f"{base_name}.json"

    logger.info(f"Processing meeting: {audio_path.name}")

    # Step 1: Check if transcript exists
    if json_path.exists():
        logger.info("Loading existing transcript...")
        transcript = load_transcript_from_json(json_path)
    else:
        logger.error("Transcript not found. Please run transcription pipeline first:")
        logger.error(f"  python transcribe_pipeline.py --audio {audio_path}")
        logger.error("Then run enrichment:")
        logger.error(f"  python audio_enrich.py --file {json_path}")
        return

    # Step 2: Analyze meeting
    logger.info("Analyzing meeting content...")
    analyzer = MeetingAnalyzer(transcript)

    key_moments = []
    if config.detect_key_moments:
        logger.info("Detecting key moments...")
        key_moments = analyzer.detect_key_moments()
        logger.info(f"Found {len(key_moments)} key moments")

    engagement_timeline = []
    if config.engagement_tracking:
        logger.info("Calculating engagement timeline...")
        engagement_timeline = analyzer.calculate_engagement_timeline()
        logger.info(f"Generated engagement timeline with {len(engagement_timeline)} periods")

    # Step 3: Generate outputs
    outputs_generated = []

    if "txt" in config.export_formats:
        txt_path = output_dir / f"{base_name}.txt"
        logger.info(f"Writing plain transcript: {txt_path.name}")
        write_txt(transcript, txt_path)
        outputs_generated.append(txt_path)

    if "annotated" in config.export_formats:
        annotated_path = output_dir / f"{base_name}_annotated.txt"
        logger.info(f"Creating annotated transcript: {annotated_path.name}")
        create_annotated_transcript(transcript, annotated_path)
        outputs_generated.append(annotated_path)

    if "csv" in config.export_formats:
        csv_path = output_dir / f"{base_name}.csv"
        logger.info(f"Exporting to CSV: {csv_path.name}")
        export_to_csv(transcript, key_moments, engagement_timeline, csv_path)
        outputs_generated.append(csv_path)

    # Analysis report
    analysis_path = output_dir / f"{base_name}_analysis.txt"
    logger.info(f"Generating analysis report: {analysis_path.name}")
    summary = analyzer.generate_summary(key_moments, engagement_timeline)
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(summary)
    outputs_generated.append(analysis_path)

    # Save key moments JSON
    if key_moments:
        moments_path = output_dir / f"{base_name}_key_moments.json"
        with open(moments_path, "w", encoding="utf-8") as f:
            json.dump(key_moments, f, indent=2)
        outputs_generated.append(moments_path)

    # Save engagement timeline JSON
    if engagement_timeline:
        timeline_path = output_dir / f"{base_name}_engagement.json"
        with open(timeline_path, "w", encoding="utf-8") as f:
            json.dump(engagement_timeline, f, indent=2)
        outputs_generated.append(timeline_path)

    logger.info(f"Meeting analysis complete! Generated {len(outputs_generated)} files:")
    for path in outputs_generated:
        logger.info(f"  - {path}")

    # Print summary to console
    print("\n" + summary + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Meeting Transcription and Analysis Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single meeting
  python meeting_transcription.py --audio raw_audio/team_meeting.mp3

  # Use configuration file
  python meeting_transcription.py --config configs/meeting_config.json --audio meeting.mp3

  # Specify output directory
  python meeting_transcription.py --audio meeting.mp3 --output outputs/meetings

Prerequisites:
  1. Run transcription: python transcribe_pipeline.py
  2. Run enrichment: python audio_enrich.py
  3. Run this workflow for analysis
        """,
    )

    parser.add_argument("--audio", type=Path, help="Audio file to process")
    parser.add_argument("--config", type=Path, help="Configuration JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/meetings"),
        help="Output directory (default: outputs/meetings)",
    )
    parser.add_argument("--batch", nargs="+", type=Path, help="Process multiple audio files")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = MeetingConfig.from_json(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = MeetingConfig()
        logger.info("Using default configuration")

    # Process files
    if args.batch:
        logger.info(f"Batch processing {len(args.batch)} files...")
        for audio_path in args.batch:
            if not audio_path.exists():
                logger.warning(f"File not found: {audio_path}")
                continue

            try:
                process_meeting(audio_path, args.output, config)
            except Exception as e:
                logger.error(f"Failed to process {audio_path.name}: {e}")
                continue

    elif args.audio:
        if not args.audio.exists():
            logger.error(f"Audio file not found: {args.audio}")
            sys.exit(1)

        process_meeting(args.audio, args.output, config)

    else:
        parser.print_help()
        sys.exit(1)

    logger.info("Workflow complete!")


if __name__ == "__main__":
    main()
