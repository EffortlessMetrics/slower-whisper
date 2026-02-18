#!/usr/bin/env python3
"""
Interview Analysis Workflow

Analyze research interviews, job interviews, or user testing sessions with:
- Emotional journey tracking
- Hesitation and uncertainty detection
- Confidence level analysis
- Key moment identification

Usage:
    python interview_analysis.py --audio interview_05.wav --output outputs/interview_05
    python interview_analysis.py --batch raw_audio/interviews/*.wav --aggregate
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from slower_whisper.pipeline.models import Segment, Transcript
from slower_whisper.pipeline.writers import load_transcript_from_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class InterviewConfig:
    """Configuration for interview analysis."""

    detect_emotional_shifts: bool = True
    track_confidence_levels: bool = True
    identify_key_moments: bool = True
    export_timeline: bool = True

    # Thresholds
    emotion_shift_threshold: float = 0.3
    confidence_threshold_high: float = 0.7
    confidence_threshold_low: float = 0.3


class InterviewAnalyzer:
    """Analyze interview transcripts for emotional patterns and insights."""

    def __init__(self, transcript: Transcript):
        self.transcript = transcript

    def detect_emotional_shifts(self, threshold: float = 0.3) -> list[dict[str, Any]]:
        """Detect significant emotional shifts during the interview."""
        shifts = []
        prev_valence = None
        prev_arousal = None

        for _i, segment in enumerate(self.transcript.segments):
            if not segment.audio_state or "emotion" not in segment.audio_state:
                continue

            emotion = segment.audio_state["emotion"]
            if "dimensional" not in emotion:
                continue

            valence = emotion["dimensional"].get("valence", {}).get("score")
            arousal = emotion["dimensional"].get("arousal", {}).get("score")

            if valence is None or arousal is None:
                continue

            if prev_valence is not None:
                valence_change = abs(valence - prev_valence)
                arousal_change = abs(arousal - prev_arousal)

                if valence_change >= threshold or arousal_change >= threshold:
                    # Significant emotional shift detected
                    from_emotion = self._classify_emotion(prev_valence, prev_arousal)
                    to_emotion = self._classify_emotion(valence, arousal)

                    confidence = min(valence_change, arousal_change) / threshold

                    shifts.append(
                        {
                            "timestamp": segment.start,
                            "from_emotion": from_emotion,
                            "to_emotion": to_emotion,
                            "valence_change": valence_change,
                            "arousal_change": arousal_change,
                            "confidence": min(confidence, 1.0),
                            "text": segment.text,
                        }
                    )

            prev_valence = valence
            prev_arousal = arousal

        return shifts

    def _classify_emotion(self, valence: float, arousal: float) -> str:
        """Classify emotion based on valence and arousal."""
        if arousal > 0.6:
            return "excited" if valence > 0.5 else "frustrated"
        elif arousal < 0.4:
            return "satisfied" if valence > 0.5 else "disappointed"
        else:
            return "positive" if valence > 0.5 else "negative"

    def analyze_confidence_levels(self) -> dict[str, Any]:
        """Analyze confidence indicators across segments."""
        high_confidence = []
        low_confidence = []

        for segment in self.transcript.segments:
            if not segment.audio_state:
                continue

            confidence_score = self._calculate_confidence_score(segment)

            segment_data = {
                "timestamp": segment.start,
                "text": segment.text,
                "confidence_score": confidence_score,
            }

            if confidence_score >= 0.7:
                high_confidence.append(segment_data)
            elif confidence_score <= 0.3:
                low_confidence.append(segment_data)

        return {
            "high_confidence_count": len(high_confidence),
            "low_confidence_count": len(low_confidence),
            "high_confidence_segments": high_confidence,
            "low_confidence_segments": low_confidence,
            "confidence_ratio": len(high_confidence) / max(len(self.transcript.segments), 1),
        }

    def _calculate_confidence_score(self, segment: Segment) -> float:
        """Calculate confidence score based on prosodic features."""
        score = 0.5  # Neutral baseline

        if not segment.audio_state:
            return score

        # Fast speech rate indicates confidence
        rate = segment.audio_state.get("rate", {})
        if rate.get("level") == "fast":
            score += 0.2
        elif rate.get("level") == "slow":
            score -= 0.2

        # Few pauses indicates confidence
        pauses = segment.audio_state.get("pauses", {})
        pause_count = pauses.get("count", 0)
        if pause_count == 0:
            score += 0.1
        elif pause_count >= 3:
            score -= 0.2

        # Rising pitch contour can indicate confidence
        pitch = segment.audio_state.get("pitch", {})
        if pitch.get("contour") == "rising":
            score += 0.1
        elif pitch.get("contour") == "flat":
            score -= 0.1

        # Moderate to loud energy indicates confidence
        energy = segment.audio_state.get("energy", {})
        if energy.get("level") in ["moderate", "loud"]:
            score += 0.1
        elif energy.get("level") == "quiet":
            score -= 0.1

        return max(0.0, min(1.0, score))

    def identify_key_moments(self) -> list[dict[str, Any]]:
        """Identify notable moments in the interview."""
        key_moments = []

        for segment in self.transcript.segments:
            if not segment.audio_state:
                continue

            moment_type = None
            confidence = 0.0

            # Peak excitement
            emotion = segment.audio_state.get("emotion", {})
            arousal = emotion.get("dimensional", {}).get("arousal", {}).get("score", 0)
            valence = emotion.get("dimensional", {}).get("valence", {}).get("score", 0.5)

            if arousal > 0.8 and valence > 0.7:
                moment_type = "peak_excitement"
                confidence = arousal * valence

            # Concern expressed
            elif arousal > 0.6 and valence < 0.4:
                moment_type = "concern_expressed"
                confidence = arousal * (1 - valence)

            # Insight/realization
            elif "categorical" in emotion and emotion["categorical"].get("primary") in [
                "surprised",
                "interested",
            ]:
                moment_type = "insight"
                confidence = emotion["categorical"].get("confidence", 0.5)

            # Uncertainty
            pauses = segment.audio_state.get("pauses", {})
            if pauses.get("count", 0) >= 3:
                moment_type = "uncertainty"
                confidence = min(pauses.get("count") / 5, 1.0)

            if moment_type:
                key_moments.append(
                    {
                        "timestamp": segment.start,
                        "type": moment_type,
                        "confidence": confidence,
                        "text": segment.text,
                    }
                )

        return key_moments

    def create_timeline(self) -> list[dict[str, Any]]:
        """Create emotional timeline for the interview."""
        timeline = []

        for segment in self.transcript.segments:
            entry = {
                "timestamp": segment.start,
                "text": segment.text,
            }

            if segment.audio_state:
                emotion = segment.audio_state.get("emotion", {})
                if "dimensional" in emotion:
                    entry["valence"] = emotion["dimensional"].get("valence", {}).get("score")
                    entry["arousal"] = emotion["dimensional"].get("arousal", {}).get("score")
                    entry["valence_level"] = emotion["dimensional"].get("valence", {}).get("level")
                    entry["arousal_level"] = emotion["dimensional"].get("arousal", {}).get("level")

                if "categorical" in emotion:
                    entry["emotion_category"] = emotion["categorical"].get("primary")

                entry["confidence_score"] = self._calculate_confidence_score(segment)

            timeline.append(entry)

        return timeline

    def generate_report(
        self, shifts: list[dict], confidence_data: dict, key_moments: list[dict]
    ) -> str:
        """Generate comprehensive analysis report."""
        lines = []
        lines.append("INTERVIEW ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Metadata
        lines.append(f"Interview: {self.transcript.file_name}")
        duration = self.transcript.segments[-1].end
        lines.append(f"Duration: {int(duration // 60)}:{int(duration % 60):02d}")
        lines.append(f"Segments: {len(self.transcript.segments)}")
        lines.append("")

        # Emotional journey
        if shifts:
            lines.append("--- Emotional Journey ---")
            lines.append("")
            lines.append(f"Emotional Shifts Detected: {len(shifts)}")
            lines.append("")

            for shift in shifts[:10]:  # Show first 10
                mins = int(shift["timestamp"] // 60)
                secs = int(shift["timestamp"] % 60)
                lines.append(
                    f"  [{mins:02d}:{secs:02d}] {shift['from_emotion'].title()} → {shift['to_emotion'].title()}"
                )
                lines.append(f"    Confidence: {shift['confidence']:.0%}")
                lines.append(f"    Context: {shift['text'][:60]}...")
                lines.append("")

        # Confidence analysis
        lines.append("--- Confidence Indicators ---")
        lines.append("")
        total = len(self.transcript.segments)
        high_pct = (confidence_data["high_confidence_count"] / total) * 100
        low_pct = (confidence_data["low_confidence_count"] / total) * 100

        lines.append(
            f"High Confidence Segments: {confidence_data['high_confidence_count']} ({high_pct:.1f}%)"
        )
        lines.append(
            f"Low Confidence/Uncertain Segments: {confidence_data['low_confidence_count']} ({low_pct:.1f}%)"
        )
        lines.append("")

        # Key moments
        if key_moments:
            lines.append("--- Key Moments ---")
            lines.append("")

            moment_types = Counter(m["type"] for m in key_moments)
            for moment_type, count in moment_types.most_common():
                lines.append(f"{moment_type.replace('_', ' ').title()}: {count}")

            lines.append("")

            for moment in key_moments[:5]:
                mins = int(moment["timestamp"] // 60)
                secs = int(moment["timestamp"] % 60)
                emoji = {
                    "peak_excitement": "🎯",
                    "concern_expressed": "⚠️",
                    "insight": "💡",
                    "uncertainty": "🤔",
                }.get(moment["type"], "•")

                lines.append(
                    f"{emoji} [{mins:02d}:{secs:02d}] {moment['type'].replace('_', ' ').title()}"
                )
                lines.append(f'  "{moment["text"][:70]}..."')
                lines.append(f"  Confidence: {moment['confidence']:.0%}")
                lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


def process_interview(audio_path: Path, output_dir: Path, config: InterviewConfig):
    """Process a single interview."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = audio_path.stem
    json_path = output_dir / f"{base_name}.json"

    if not json_path.exists():
        alt_path = Path("whisper_json") / f"{base_name}.json"
        if alt_path.exists():
            json_path = alt_path

    logger.info(f"Processing interview: {audio_path.name}")

    if not json_path.exists():
        logger.error("Transcript not found. Run transcription and enrichment first.")
        return

    transcript = load_transcript_from_json(json_path)
    analyzer = InterviewAnalyzer(transcript)

    # Analyze
    shifts = analyzer.detect_emotional_shifts() if config.detect_emotional_shifts else []
    confidence_data = analyzer.analyze_confidence_levels() if config.track_confidence_levels else {}
    key_moments = analyzer.identify_key_moments() if config.identify_key_moments else []
    timeline = analyzer.create_timeline() if config.export_timeline else []

    # Generate report
    report = analyzer.generate_report(shifts, confidence_data, key_moments)

    report_path = output_dir / f"{base_name}_analysis.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Saved report: {report_path}")

    # Save data files
    if shifts:
        shifts_path = output_dir / f"{base_name}_shifts.json"
        with open(shifts_path, "w") as f:
            json.dump(shifts, f, indent=2)

    if key_moments:
        moments_path = output_dir / f"{base_name}_moments.json"
        with open(moments_path, "w") as f:
            json.dump(key_moments, f, indent=2)

    if timeline:
        timeline_path = output_dir / f"{base_name}_timeline.csv"
        with open(timeline_path, "w", newline="", encoding="utf-8") as f:
            if timeline:
                writer = csv.DictWriter(f, fieldnames=timeline[0].keys())
                writer.writeheader()
                writer.writerows(timeline)

    print("\n" + report + "\n")
    logger.info("Interview analysis complete!")


def main():
    parser = argparse.ArgumentParser(description="Interview Analysis Workflow")
    parser.add_argument("--audio", type=Path, help="Audio file to process")
    parser.add_argument("--output", type=Path, default=Path("outputs/interviews"))
    parser.add_argument("--batch", nargs="+", type=Path, help="Process multiple files")
    parser.add_argument("--report", action="store_true", help="Generate full report")

    args = parser.parse_args()

    config = InterviewConfig()

    if args.batch:
        for audio_path in args.batch:
            if audio_path.exists():
                process_interview(audio_path, args.output, config)
    elif args.audio:
        if args.audio.exists():
            process_interview(args.audio, args.output, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
