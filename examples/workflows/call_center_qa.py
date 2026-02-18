#!/usr/bin/env python3
"""
Call Center QA Workflow

Analyze customer service calls for quality assurance:
- Frustration and satisfaction detection
- Call quality scoring
- Agent performance metrics
- Training opportunity identification

Usage:
    python call_center_qa.py --audio call_12345.wav --agent-id agent_042
    python call_center_qa.py --batch raw_audio/calls/*.wav --aggregate-by-agent
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
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
class QAConfig:
    """Configuration for call center QA."""

    frustration_arousal_threshold: float = 0.7
    frustration_valence_threshold: float = 0.3
    satisfaction_valence_threshold: float = 0.7
    escalation_risk_threshold: float = 0.8


@dataclass
class CallQAMetrics:
    """Quality assurance metrics for a call."""

    call_id: str
    agent_id: str
    duration: float
    overall_quality_score: float = 0.0

    frustration_count: int = 0
    satisfaction_count: int = 0
    escalation_risk_count: int = 0

    avg_customer_valence: float = 0.0
    avg_customer_arousal: float = 0.0

    agent_empathy_markers: int = 0
    resolution_achieved: bool = False
    final_sentiment: str = "neutral"

    training_opportunities: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)


class CallCenterAnalyzer:
    """Analyze customer service calls for QA."""

    def __init__(self, transcript: Transcript, call_id: str, agent_id: str = "unknown"):
        self.transcript = transcript
        self.call_id = call_id
        self.agent_id = agent_id

    def analyze_call(self, config: QAConfig) -> CallQAMetrics:
        """Perform comprehensive call analysis."""
        metrics = CallQAMetrics(
            call_id=self.call_id,
            agent_id=self.agent_id,
            duration=self.transcript.segments[-1].end if self.transcript.segments else 0.0,
        )

        valence_scores = []
        arousal_scores = []

        frustration_segments = []
        satisfaction_segments = []

        for segment in self.transcript.segments:
            if not segment.audio_state or "emotion" not in segment.audio_state:
                continue

            emotion = segment.audio_state["emotion"]
            if "dimensional" not in emotion:
                continue

            valence = emotion["dimensional"].get("valence", {}).get("score")
            arousal = emotion["dimensional"].get("arousal", {}).get("score")

            if valence is not None:
                valence_scores.append(valence)
            if arousal is not None:
                arousal_scores.append(arousal)

            # Detect frustration
            if (
                arousal
                and arousal > config.frustration_arousal_threshold
                and valence
                and valence < config.frustration_valence_threshold
            ):
                metrics.frustration_count += 1
                frustration_segments.append(segment)

            # Detect satisfaction
            if valence and valence > config.satisfaction_valence_threshold:
                metrics.satisfaction_count += 1
                satisfaction_segments.append(segment)

            # Escalation risk
            if arousal and arousal > config.escalation_risk_threshold:
                metrics.escalation_risk_count += 1

            # Empathy markers (calm, reassuring tone)
            if arousal and arousal < 0.4 and valence and valence > 0.5:
                empathy_words = ["understand", "help", "sorry", "apologize", "appreciate"]
                if any(word in segment.text.lower() for word in empathy_words):
                    metrics.agent_empathy_markers += 1

        # Calculate averages
        if valence_scores:
            metrics.avg_customer_valence = sum(valence_scores) / len(valence_scores)
        if arousal_scores:
            metrics.avg_customer_arousal = sum(arousal_scores) / len(arousal_scores)

        # Determine final sentiment
        if self.transcript.segments:
            last_segments = self.transcript.segments[-3:]  # Last 3 segments
            final_valence = []

            for seg in last_segments:
                if seg.audio_state and "emotion" in seg.audio_state:
                    v = (
                        seg.audio_state["emotion"]
                        .get("dimensional", {})
                        .get("valence", {})
                        .get("score")
                    )
                    if v is not None:
                        final_valence.append(v)

            if final_valence:
                avg_final = sum(final_valence) / len(final_valence)
                if avg_final > 0.6:
                    metrics.final_sentiment = "positive"
                    metrics.resolution_achieved = True
                elif avg_final < 0.4:
                    metrics.final_sentiment = "negative"
                else:
                    metrics.final_sentiment = "neutral"

        # Calculate quality score
        metrics.overall_quality_score = self._calculate_quality_score(metrics)

        # Identify training opportunities and strengths
        self._identify_training_and_strengths(metrics, frustration_segments)

        return metrics

    def _calculate_quality_score(self, metrics: CallQAMetrics) -> float:
        """Calculate overall quality score (0-10)."""
        score = 5.0  # Start at neutral

        # Resolution bonus
        if metrics.resolution_achieved:
            score += 2.0

        # Final sentiment impact
        if metrics.final_sentiment == "positive":
            score += 1.5
        elif metrics.final_sentiment == "negative":
            score -= 1.5

        # Frustration penalty
        if metrics.frustration_count > 3:
            score -= 1.0
        elif metrics.frustration_count == 0:
            score += 0.5

        # Empathy bonus
        if metrics.agent_empathy_markers >= 5:
            score += 1.0
        elif metrics.agent_empathy_markers >= 3:
            score += 0.5

        # Escalation penalty
        if metrics.escalation_risk_count > 2:
            score -= 1.5

        return max(0.0, min(10.0, score))

    def _identify_training_and_strengths(
        self, metrics: CallQAMetrics, frustration_segments: list[Segment]
    ):
        """Identify training opportunities and agent strengths."""
        # Strengths
        if metrics.agent_empathy_markers >= 5:
            metrics.strengths.append("Excellent empathy and acknowledgment")

        if metrics.resolution_achieved:
            metrics.strengths.append("Successful issue resolution")

        if metrics.frustration_count > 0 and metrics.final_sentiment == "positive":
            metrics.strengths.append("Effective de-escalation")

        # Training opportunities
        if metrics.escalation_risk_count > 2:
            metrics.training_opportunities.append("De-escalation techniques")

        if metrics.agent_empathy_markers < 2 and metrics.duration > 300:
            metrics.training_opportunities.append("Active listening and empathy")

        if not metrics.resolution_achieved and metrics.duration > 300:
            metrics.training_opportunities.append("Problem-solving and resolution strategies")

        # Check for repeated customer concerns (poor active listening)
        concern_words = ["said", "told", "mentioned", "again", "already"]
        repeat_count = sum(
            1
            for seg in self.transcript.segments
            if any(word in seg.text.lower() for word in concern_words)
        )

        if repeat_count >= 3:
            metrics.training_opportunities.append("Active listening and summarization")

    def generate_timeline(self) -> list[dict[str, Any]]:
        """Generate sentiment timeline for the call."""
        timeline = []

        for segment in self.transcript.segments:
            entry = {
                "timestamp": segment.start,
                "text": segment.text[:100],
            }

            if segment.audio_state and "emotion" in segment.audio_state:
                emotion = segment.audio_state["emotion"]

                if "dimensional" in emotion:
                    valence = emotion["dimensional"].get("valence", {})
                    arousal = emotion["dimensional"].get("arousal", {})

                    entry["valence_score"] = valence.get("score")
                    entry["valence_level"] = valence.get("level")
                    entry["arousal_score"] = arousal.get("score")
                    entry["arousal_level"] = arousal.get("level")

                if "categorical" in emotion:
                    entry["emotion"] = emotion["categorical"].get("primary")

            timeline.append(entry)

        return timeline

    def generate_qa_report(self, metrics: CallQAMetrics) -> str:
        """Generate comprehensive QA report."""
        lines = []
        lines.append("CALL CENTER QA REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Call metadata
        lines.append(f"Call ID: {metrics.call_id}")
        lines.append(f"Agent: {metrics.agent_id}")
        mins = int(metrics.duration // 60)
        secs = int(metrics.duration % 60)
        lines.append(f"Duration: {mins}:{secs:02d}")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("")

        # Quality metrics
        lines.append("--- Quality Metrics ---")
        lines.append("")
        lines.append(f"Overall Call Quality Score: {metrics.overall_quality_score:.1f}/10")
        lines.append("")

        lines.append("Customer Satisfaction Indicators:")
        lines.append(
            f"  {'✓' if metrics.resolution_achieved else '✗'} Issue resolved: {'Yes' if metrics.resolution_achieved else 'No'}"
        )
        lines.append(
            f"  {'✓' if metrics.final_sentiment == 'positive' else '⚠' if metrics.final_sentiment == 'neutral' else '✗'} Final sentiment: {metrics.final_sentiment.title()} ({metrics.avg_customer_valence:.2f})"
        )
        lines.append(
            f"  {'✓' if metrics.escalation_risk_count == 0 else '⚠'} Escalation: {'None' if metrics.escalation_risk_count == 0 else f'{metrics.escalation_risk_count} moments'}"
        )

        if metrics.frustration_count > 0:
            lines.append(f"  ⚠ Frustration detected: {metrics.frustration_count} instances")

        lines.append("")

        lines.append("Agent Performance:")
        lines.append(
            f"  {'✓' if metrics.agent_empathy_markers >= 3 else '⚠'} Empathy markers: {metrics.agent_empathy_markers} instances"
        )
        lines.append(
            f"  {'✓' if metrics.final_sentiment == 'positive' else '⚠'} Professional tone: {'Maintained' if metrics.final_sentiment != 'negative' else 'Needs improvement'}"
        )

        lines.append("")

        # Strengths
        if metrics.strengths:
            lines.append("--- Strengths ---")
            lines.append("")
            for strength in metrics.strengths:
                lines.append(f"  ✅ {strength}")
            lines.append("")

        # Training opportunities
        if metrics.training_opportunities:
            lines.append("--- Training Opportunities ---")
            lines.append("")
            for opportunity in metrics.training_opportunities:
                lines.append(f"  📚 {opportunity}")
            lines.append("")

        # Recommendations
        lines.append("--- Recommendations ---")
        lines.append("")

        if metrics.overall_quality_score >= 8.0:
            lines.append("1. ✅ Excellent call handling - consider for training examples")
        elif metrics.overall_quality_score >= 6.0:
            lines.append("1. ✅ Good performance - maintain current approach")
        else:
            lines.append("1. ⚠️ Performance below standard - immediate coaching recommended")

        if metrics.training_opportunities:
            lines.append("2. 📚 Schedule training on identified opportunities")

        if metrics.final_sentiment != "positive":
            lines.append("3. 🔄 Follow-up with customer recommended")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


def process_call(audio_path: Path, output_dir: Path, config: QAConfig, agent_id: str = "unknown"):
    """Process a single call."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = audio_path.stem
    call_id = base_name

    json_path = output_dir / f"{base_name}.json"
    if not json_path.exists():
        alt_path = Path("whisper_json") / f"{base_name}.json"
        if alt_path.exists():
            json_path = alt_path

    logger.info(f"Processing call: {call_id}")

    if not json_path.exists():
        logger.error("Transcript not found. Run transcription and enrichment first.")
        return None

    transcript = load_transcript_from_json(json_path)
    analyzer = CallCenterAnalyzer(transcript, call_id, agent_id)

    # Analyze call
    metrics = analyzer.analyze_call(config)
    timeline = analyzer.generate_timeline()

    # Generate report
    report = analyzer.generate_qa_report(metrics)

    # Save outputs
    report_path = output_dir / f"{base_name}_qa_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Save metrics as JSON
    metrics_dict = {
        "call_id": metrics.call_id,
        "agent_id": metrics.agent_id,
        "duration": metrics.duration,
        "overall_quality_score": metrics.overall_quality_score,
        "frustration_count": metrics.frustration_count,
        "satisfaction_count": metrics.satisfaction_count,
        "escalation_risk_count": metrics.escalation_risk_count,
        "avg_customer_valence": metrics.avg_customer_valence,
        "avg_customer_arousal": metrics.avg_customer_arousal,
        "agent_empathy_markers": metrics.agent_empathy_markers,
        "resolution_achieved": metrics.resolution_achieved,
        "final_sentiment": metrics.final_sentiment,
        "training_opportunities": metrics.training_opportunities,
        "strengths": metrics.strengths,
    }

    metrics_path = output_dir / f"{base_name}_qa_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Save timeline
    if timeline:
        timeline_path = output_dir / f"{base_name}_timeline.csv"
        with open(timeline_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=timeline[0].keys())
            writer.writeheader()
            writer.writerows(timeline)

    print("\n" + report + "\n")
    logger.info(f"QA analysis complete! Quality score: {metrics.overall_quality_score:.1f}/10")

    return metrics_dict


def aggregate_by_agent(metrics_list: list[dict], output_dir: Path):
    """Aggregate metrics by agent for performance analysis."""
    agent_stats = defaultdict(
        lambda: {
            "call_count": 0,
            "total_quality": 0.0,
            "total_empathy": 0,
            "resolutions": 0,
            "escalations": 0,
            "training_needs": Counter(),
        }
    )

    for metrics in metrics_list:
        agent_id = metrics["agent_id"]
        stats = agent_stats[agent_id]

        stats["call_count"] += 1
        stats["total_quality"] += metrics["overall_quality_score"]
        stats["total_empathy"] += metrics["agent_empathy_markers"]

        if metrics["resolution_achieved"]:
            stats["resolutions"] += 1

        stats["escalations"] += metrics["escalation_risk_count"]

        for opportunity in metrics["training_opportunities"]:
            stats["training_needs"][opportunity] += 1

    # Generate agent performance report
    report_lines = []
    report_lines.append("AGENT PERFORMANCE SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("")

    for agent_id in sorted(agent_stats.keys()):
        stats = agent_stats[agent_id]
        avg_quality = stats["total_quality"] / stats["call_count"]
        resolution_rate = (stats["resolutions"] / stats["call_count"]) * 100

        report_lines.append(f"Agent: {agent_id}")
        report_lines.append(f"  Calls Reviewed: {stats['call_count']}")
        report_lines.append(f"  Average Quality Score: {avg_quality:.1f}/10")
        report_lines.append(f"  Resolution Rate: {resolution_rate:.1f}%")
        report_lines.append(
            f"  Avg Empathy Markers: {stats['total_empathy'] / stats['call_count']:.1f}"
        )

        if stats["training_needs"]:
            report_lines.append("  Top Training Needs:")
            for need, count in stats["training_needs"].most_common(3):
                report_lines.append(f"    - {need} ({count} calls)")

        report_lines.append("")

    report = "\n".join(report_lines)

    # Save report
    summary_path = output_dir / "agent_performance_summary.txt"
    with open(summary_path, "w") as f:
        f.write(report)

    print("\n" + report)
    logger.info(f"Saved agent summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Call Center QA Workflow")
    parser.add_argument("--audio", type=Path, help="Audio file to process")
    parser.add_argument("--agent-id", type=str, default="unknown", help="Agent identifier")
    parser.add_argument("--output", type=Path, default=Path("outputs/qa"))
    parser.add_argument("--batch", nargs="+", type=Path, help="Process multiple files")
    parser.add_argument(
        "--aggregate-by-agent", action="store_true", help="Generate agent performance summary"
    )

    args = parser.parse_args()

    config = QAConfig()
    metrics_list = []

    if args.batch:
        logger.info(f"Batch processing {len(args.batch)} calls...")
        for audio_path in args.batch:
            if audio_path.exists():
                metrics = process_call(audio_path, args.output, config, args.agent_id)
                if metrics:
                    metrics_list.append(metrics)

        if args.aggregate_by_agent and metrics_list:
            aggregate_by_agent(metrics_list, args.output)

    elif args.audio:
        if args.audio.exists():
            process_call(args.audio, args.output, config, args.agent_id)
        else:
            logger.error(f"Audio file not found: {args.audio}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
