#!/usr/bin/env python3
"""
Podcast Processing Workflow

Process podcast episodes to create:
- Enhanced show notes with timestamps
- Highlight reel suggestions (exciting moments)
- Quote extraction for social media
- Chapter markers for navigation
- Emotional context for better content understanding

Usage:
    python podcast_processing.py --audio episode_042.mp3 --title "Episode 42: AI Future"
    python podcast_processing.py --config configs/podcast_config.json --audio episode.mp3
    python podcast_processing.py --highlights-only --audio episode.mp3
"""

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transcription.models import Segment, Transcript
from transcription.writers import load_transcript_from_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PodcastConfig:
    """Configuration for podcast processing workflow."""

    # Processing settings
    generate_show_notes: bool = True
    detect_highlights: bool = True
    extract_quotes: bool = True
    chapter_markers: bool = True

    # Thresholds
    engagement_threshold: float = 0.7
    min_highlight_duration: float = 10.0
    max_highlight_duration: float = 120.0
    quote_min_words: int = 10
    quote_max_words: int = 50

    @classmethod
    def from_json(cls, config_path: Path) -> "PodcastConfig":
        """Load configuration from JSON file."""
        with open(config_path) as f:
            config_data = json.load(f)

        if "podcast_specific" in config_data:
            config_data.update(config_data["podcast_specific"])

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in config_data.items() if k in valid_fields}

        return cls(**filtered_data)


class PodcastAnalyzer:
    """Analyze podcast transcripts for highlights and content."""

    def __init__(self, transcript: Transcript, title: str = ""):
        self.transcript = transcript
        self.title = title or transcript.file_name

    def detect_highlights(
        self,
        engagement_threshold: float = 0.7,
        min_duration: float = 10.0,
        max_duration: float = 120.0,
    ) -> list[dict[str, Any]]:
        """
        Detect highlight-worthy moments based on engagement and emotion.

        Highlights are characterized by:
        - High energy/arousal
        - Interesting emotional content
        - Sustained engagement over multiple segments
        - Appropriate duration for sharing
        """
        highlights = []
        current_highlight = None

        for i, segment in enumerate(self.transcript.segments):
            if not segment.audio_state:
                continue

            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(segment)

            if engagement_score >= engagement_threshold:
                if current_highlight is None:
                    # Start new highlight
                    current_highlight = {
                        "start_segment": i,
                        "start_time": segment.start,
                        "segments": [segment],
                        "max_engagement": engagement_score,
                        "emotions": [],
                    }
                else:
                    # Extend current highlight
                    current_highlight["segments"].append(segment)
                    current_highlight["max_engagement"] = max(
                        current_highlight["max_engagement"], engagement_score
                    )
            else:
                # End of high engagement period
                if current_highlight is not None:
                    # Check duration
                    duration = (
                        current_highlight["segments"][-1].end - current_highlight["start_time"]
                    )

                    if min_duration <= duration <= max_duration:
                        # This is a valid highlight
                        highlights.append(self._finalize_highlight(current_highlight))

                    current_highlight = None

        # Handle final highlight
        if current_highlight is not None:
            duration = current_highlight["segments"][-1].end - current_highlight["start_time"]
            if min_duration <= duration <= max_duration:
                highlights.append(self._finalize_highlight(current_highlight))

        return highlights

    def _calculate_engagement_score(self, segment: Segment) -> float:
        """Calculate engagement score for a segment (0-1)."""
        if not segment.audio_state:
            return 0.0

        score = 0.0
        weights = {"arousal": 0.4, "energy": 0.3, "rate": 0.2, "pitch_var": 0.1}

        # Arousal score
        emotion = segment.audio_state.get("emotion", {})
        arousal = emotion.get("dimensional", {}).get("arousal", {}).get("score")
        if arousal is not None:
            score += arousal * weights["arousal"]

        # Energy score (normalize -40 to 0 dB)
        energy = segment.audio_state.get("energy", {})
        db_rms = energy.get("db_rms")
        if db_rms is not None:
            energy_normalized = max(0, (db_rms + 40) / 40)
            score += energy_normalized * weights["energy"]

        # Speech rate score (normalize 2-8 syl/sec)
        rate = segment.audio_state.get("rate", {})
        syl_per_sec = rate.get("syllables_per_sec")
        if syl_per_sec is not None:
            rate_normalized = max(0, min(1, (syl_per_sec - 2) / 6))
            score += rate_normalized * weights["rate"]

        # Pitch variation (rising/falling contours are more engaging)
        pitch = segment.audio_state.get("pitch", {})
        contour = pitch.get("contour", "flat")
        if contour in ["rising", "falling", "varied"]:
            score += weights["pitch_var"]

        return min(1.0, score)

    def _finalize_highlight(self, highlight_data: dict) -> dict[str, Any]:
        """Convert raw highlight data into final format."""
        segments = highlight_data["segments"]
        start_time = highlight_data["start_time"]
        end_time = segments[-1].end
        duration = end_time - start_time

        # Combine text
        text = " ".join(seg.text for seg in segments)

        # Collect emotions
        emotions = []
        for seg in segments:
            if seg.audio_state and "emotion" in seg.audio_state:
                emotion = seg.audio_state["emotion"]
                if "categorical" in emotion:
                    emotions.append(emotion["categorical"].get("primary", "neutral"))

        primary_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "engaging"

        return {
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "text": text,
            "primary_emotion": primary_emotion,
            "max_engagement": highlight_data["max_engagement"],
            "segment_count": len(segments),
        }

    def extract_quotes(self, min_words: int = 10, max_words: int = 50) -> list[dict[str, Any]]:
        """
        Extract quotable segments suitable for social media.

        Good quotes are:
        - Complete thoughts (complete sentences)
        - Appropriate length
        - High emotional content or interesting prosody
        - Standalone comprehensible
        """
        quotes = []

        for segment in self.transcript.segments:
            text = segment.text.strip()

            # Word count check
            word_count = len(text.split())
            if not (min_words <= word_count <= max_words):
                continue

            # Check for complete sentence
            if text[-1] not in ".!?":
                continue

            # Calculate "quotability" score
            quotability = 0.0

            if segment.audio_state:
                # High valence (positive or negative extremes are quotable)
                emotion = segment.audio_state.get("emotion", {})
                valence = emotion.get("dimensional", {}).get("valence", {}).get("score")
                if valence is not None:
                    # Distance from neutral (0.5)
                    valence_magnitude = abs(valence - 0.5) * 2  # 0-1 scale
                    quotability += valence_magnitude * 0.4

                # Strong emotions are quotable
                if "categorical" in emotion:
                    strong_emotions = ["excited", "happy", "surprised", "inspired"]
                    if emotion["categorical"].get("primary") in strong_emotions:
                        quotability += 0.3

                # Clear, confident delivery
                pitch = segment.audio_state.get("pitch", {})
                if pitch.get("contour") in ["rising", "falling"]:  # Declarative or questioning
                    quotability += 0.2

                energy = segment.audio_state.get("energy", {})
                if energy.get("level") in ["moderate", "loud"]:  # Not too quiet
                    quotability += 0.1

            if quotability >= 0.5:  # Threshold for quotability
                quotes.append(
                    {
                        "timestamp": segment.start,
                        "text": text,
                        "quotability_score": quotability,
                        "word_count": word_count,
                        "emotion": segment.audio_state.get("emotion", {})
                        .get("categorical", {})
                        .get("primary", "neutral")
                        if segment.audio_state
                        else "neutral",
                    }
                )

        # Sort by quotability
        quotes.sort(key=lambda q: q["quotability_score"], reverse=True)

        return quotes

    def generate_chapter_markers(self, min_chapter_duration: float = 300.0) -> list[dict[str, Any]]:
        """
        Generate chapter markers based on topic shifts and pauses.

        Chapters are identified by:
        - Long pauses (topic transitions)
        - Emotional shifts
        - Minimum chapter duration
        """
        chapters = []
        last_chapter_time = 0.0

        for i, segment in enumerate(self.transcript.segments):
            # Check for chapter boundary indicators
            is_boundary = False

            # Long pause before this segment
            if i > 0:
                prev_segment = self.transcript.segments[i - 1]
                gap = segment.start - prev_segment.end
                if gap > 2.0:  # 2+ second gap
                    is_boundary = True

            # Minimum chapter duration check
            potential_duration = segment.start - last_chapter_time

            if is_boundary and potential_duration >= min_chapter_duration:
                # Create chapter for previous content
                if i > 0:
                    # Find a good chapter title from first few segments
                    chapter_segments = [
                        s
                        for s in self.transcript.segments
                        if last_chapter_time <= s.start < segment.start
                    ][:3]

                    chapter_text = " ".join(s.text for s in chapter_segments)
                    title = self._generate_chapter_title(chapter_text)

                    chapters.append(
                        {
                            "start_time": last_chapter_time,
                            "title": title,
                            "duration": potential_duration,
                        }
                    )

                last_chapter_time = segment.start

        # Final chapter
        if last_chapter_time < self.transcript.segments[-1].end:
            final_segments = [s for s in self.transcript.segments if s.start >= last_chapter_time][
                :3
            ]

            chapter_text = " ".join(s.text for s in final_segments)
            title = self._generate_chapter_title(chapter_text)

            chapters.append(
                {
                    "start_time": last_chapter_time,
                    "title": title,
                    "duration": self.transcript.segments[-1].end - last_chapter_time,
                }
            )

        return chapters

    def _generate_chapter_title(self, text: str, max_length: int = 60) -> str:
        """Generate a chapter title from the beginning text."""
        # Simple implementation: use first sentence or truncate
        sentences = text.split(".")
        if sentences:
            title = sentences[0].strip()
            if len(title) > max_length:
                title = title[:max_length].rsplit(" ", 1)[0] + "..."
            return title
        return "Chapter"

    def generate_show_notes(
        self, highlights: list[dict], quotes: list[dict], chapters: list[dict]
    ) -> str:
        """Generate formatted show notes in Markdown."""
        lines = []

        # Header
        lines.append(f"# {self.title}")
        lines.append("")

        # Duration
        total_duration = self.transcript.segments[-1].end
        minutes = int(total_duration // 60)
        seconds = int(total_duration % 60)
        lines.append(f"**Duration:** {minutes}:{seconds:02d}")
        lines.append("")

        # Highlights
        if highlights:
            lines.append("## Highlights")
            lines.append("")

            for _, highlight in enumerate(highlights[:5], 1):  # Top 5 highlights
                timestamp = self._format_timestamp(highlight["start_time"])
                emotion_emoji = self._get_emotion_emoji(highlight["primary_emotion"])

                lines.append(
                    f"### {emotion_emoji} [{timestamp}] - {highlight['primary_emotion'].title()} Moment"
                )
                lines.append(
                    f"*Duration: {highlight['duration']:.0f}s | Engagement: {highlight['max_engagement']:.0%}*"
                )
                lines.append("")

                # Preview text (first 100 chars)
                preview = (
                    highlight["text"][:100] + "..."
                    if len(highlight["text"]) > 100
                    else highlight["text"]
                )
                lines.append(preview)
                lines.append("")

        # Chapters
        if chapters:
            lines.append("## Chapters")
            lines.append("")

            for chapter in chapters:
                timestamp = self._format_timestamp(chapter["start_time"])
                lines.append(f"- [{timestamp}] {chapter['title']}")

            lines.append("")

        # Quotes
        if quotes:
            lines.append("## Memorable Quotes")
            lines.append("")

            for quote in quotes[:10]:  # Top 10 quotes
                timestamp = self._format_timestamp(quote["timestamp"])
                lines.append(f"> {quote['text']}")
                lines.append("> ")
                lines.append(f"> *— [{timestamp}]*")
                lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Show notes generated automatically from audio analysis*")
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d')}*")

        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def _get_emotion_emoji(self, emotion: str) -> str:
        """Get emoji for emotion."""
        emoji_map = {
            "excited": "🔥",
            "happy": "😊",
            "inspired": "💡",
            "surprised": "😲",
            "thoughtful": "🤔",
            "analytical": "📊",
            "engaging": "✨",
            "calm": "😌",
            "confident": "💪",
        }
        return emoji_map.get(emotion, "🎯")


def process_podcast(
    audio_path: Path,
    output_dir: Path,
    config: PodcastConfig,
    title: str = "",
    highlights_only: bool = False,
):
    """
    Process a podcast episode through the complete workflow.

    Args:
        audio_path: Path to audio file
        output_dir: Directory for outputs
        config: Podcast configuration
        title: Episode title
        highlights_only: Only generate highlights, skip full processing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = audio_path.stem
    json_path = output_dir / f"{base_name}.json"

    # Check if we need to look in whisper_json directory
    if not json_path.exists():
        alt_json_path = Path("whisper_json") / f"{base_name}.json"
        if alt_json_path.exists():
            json_path = alt_json_path

    logger.info(f"Processing podcast: {audio_path.name}")

    # Load transcript
    if json_path.exists():
        logger.info("Loading existing transcript...")
        transcript = load_transcript_from_json(json_path)
    else:
        logger.error("Transcript not found. Please run transcription and enrichment first:")
        logger.error("  1. python transcribe_pipeline.py")
        logger.error("  2. python audio_enrich.py")
        return

    # Analyze podcast
    analyzer = PodcastAnalyzer(transcript, title)

    highlights = []
    quotes = []
    chapters = []

    if config.detect_highlights or highlights_only:
        logger.info("Detecting highlights...")
        highlights = analyzer.detect_highlights(
            engagement_threshold=config.engagement_threshold,
            min_duration=config.min_highlight_duration,
            max_duration=config.max_highlight_duration,
        )
        logger.info(f"Found {len(highlights)} highlights")

        # Save highlights
        highlights_path = output_dir / f"{base_name}_highlights.txt"
        with open(highlights_path, "w", encoding="utf-8") as f:
            f.write(f"# Podcast Highlights: {title or base_name}\n\n")

            for i, highlight in enumerate(highlights, 1):
                timestamp = analyzer._format_timestamp(highlight["start_time"])
                f.write(f"{i}. [{timestamp}] - {highlight['primary_emotion'].title()}\n")
                f.write(
                    f"   Duration: {highlight['duration']:.0f}s | Engagement: {highlight['max_engagement']:.0%}\n"
                )
                f.write(f"   {highlight['text'][:150]}...\n\n")

        logger.info(f"Saved highlights to: {highlights_path}")

        if highlights_only:
            logger.info("Highlights-only mode complete!")
            return

    if config.extract_quotes:
        logger.info("Extracting quotable moments...")
        quotes = analyzer.extract_quotes(
            min_words=config.quote_min_words, max_words=config.quote_max_words
        )
        logger.info(f"Found {len(quotes)} potential quotes")

        # Save quotes
        quotes_path = output_dir / f"{base_name}_quotes.txt"
        with open(quotes_path, "w", encoding="utf-8") as f:
            f.write(f"# Quotable Moments: {title or base_name}\n\n")

            for i, quote in enumerate(quotes[:20], 1):  # Top 20
                timestamp = analyzer._format_timestamp(quote["timestamp"])
                f.write(f"{i}. [{timestamp}] ({quote['quotability_score']:.0%})\n")
                f.write(f'   "{quote["text"]}"\n\n')

        logger.info(f"Saved quotes to: {quotes_path}")

    if config.chapter_markers:
        logger.info("Generating chapter markers...")
        chapters = analyzer.generate_chapter_markers()
        logger.info(f"Created {len(chapters)} chapters")

        # Save chapters
        chapters_path = output_dir / f"{base_name}_chapters.json"
        with open(chapters_path, "w", encoding="utf-8") as f:
            json.dump(chapters, f, indent=2)

        logger.info(f"Saved chapters to: {chapters_path}")

    if config.generate_show_notes:
        logger.info("Generating show notes...")
        show_notes = analyzer.generate_show_notes(highlights, quotes, chapters)

        show_notes_path = output_dir / f"{base_name}_show_notes.md"
        with open(show_notes_path, "w", encoding="utf-8") as f:
            f.write(show_notes)

        logger.info(f"Saved show notes to: {show_notes_path}")

        # Print to console
        print("\n" + "=" * 80)
        print(show_notes)
        print("=" * 80 + "\n")

    logger.info("Podcast processing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Podcast Processing Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process podcast with title
  python podcast_processing.py --audio episode_042.mp3 --title "Episode 42: AI Future"

  # Use configuration file
  python podcast_processing.py --config configs/podcast_config.json --audio episode.mp3

  # Generate highlights only (quick)
  python podcast_processing.py --audio episode.mp3 --highlights-only

  # Specify output directory
  python podcast_processing.py --audio episode.mp3 --output outputs/podcasts
        """,
    )

    parser.add_argument("--audio", type=Path, help="Audio file to process")
    parser.add_argument("--config", type=Path, help="Configuration JSON file")
    parser.add_argument("--title", type=str, default="", help="Episode title")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/podcasts"),
        help="Output directory (default: outputs/podcasts)",
    )
    parser.add_argument(
        "--highlights-only", action="store_true", help="Only generate highlights (faster)"
    )
    parser.add_argument(
        "--min-duration", type=float, default=10.0, help="Minimum highlight duration in seconds"
    )

    args = parser.parse_args()

    if not args.audio:
        parser.print_help()
        sys.exit(1)

    if not args.audio.exists():
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    # Load configuration
    if args.config:
        config = PodcastConfig.from_json(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = PodcastConfig()
        if args.min_duration:
            config.min_highlight_duration = args.min_duration
        logger.info("Using default configuration")

    process_podcast(args.audio, args.output, config, args.title, args.highlights_only)


if __name__ == "__main__":
    main()
