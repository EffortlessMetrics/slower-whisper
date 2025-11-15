#!/usr/bin/env python3
"""
LLM Prompt Builder for Enriched Transcripts

This module provides utilities to format enriched transcripts for LLM consumption.
It creates LLM-ready prompts that include both transcribed speech and audio annotations,
enabling the LLM to understand not just what was said, but how it was said.

Key Features:
- Format transcripts with inline audio annotations
- Support speaker-aware formatting (when diarization is available)
- Include summary context about audio characteristics
- Generate different prompt styles for various analysis tasks

Usage:
    from prompt_builder import TranscriptPromptBuilder

    builder = TranscriptPromptBuilder("enriched_transcript.json")
    prompt = builder.build_analysis_prompt(
        task="Find moments of escalation",
        include_audio_summary=True
    )
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transcription.models import Segment
from transcription.writers import load_transcript_from_json


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""

    include_timestamps: bool = True
    include_audio_annotations: bool = True
    include_prosody_details: bool = False
    include_emotion_details: bool = False
    format_style: Literal["inline", "separate", "minimal"] = "inline"
    max_segments: int | None = None
    speaker_aware: bool = True


class TranscriptPromptBuilder:
    """
    Build LLM-ready prompts from enriched transcripts.

    This class converts enriched JSON transcripts into formatted prompts
    suitable for various LLM analysis tasks.
    """

    def __init__(self, transcript_path: Path):
        """
        Initialize the prompt builder.

        Args:
            transcript_path: Path to enriched JSON transcript file
        """
        self.transcript_path = Path(transcript_path)
        self.transcript = load_transcript_from_json(self.transcript_path)

    def build_basic_prompt(self, config: PromptConfig | None = None) -> str:
        """
        Build a basic formatted transcript prompt.

        Args:
            config: Optional configuration for prompt generation

        Returns:
            Formatted transcript string ready for LLM input
        """
        if config is None:
            config = PromptConfig()

        segments = self.transcript.segments
        if config.max_segments:
            segments = segments[: config.max_segments]

        lines = []

        # Add header
        lines.append("TRANSCRIPT")
        lines.append(f"File: {self.transcript.file_name}")
        lines.append(f"Language: {self.transcript.language}")
        lines.append("")

        # Format segments
        for segment in segments:
            line = self._format_segment(segment, config)
            lines.append(line)
            lines.append("")

        return "\n".join(lines)

    def build_analysis_prompt(
        self,
        task: str,
        include_audio_summary: bool = True,
        include_instructions: bool = True,
        config: PromptConfig | None = None,
    ) -> str:
        """
        Build a prompt for a specific analysis task.

        Args:
            task: Description of the analysis task (e.g., "Find moments of escalation")
            include_audio_summary: Whether to include overall audio statistics
            include_instructions: Whether to include task instructions
            config: Optional configuration for prompt generation

        Returns:
            Complete prompt including task, transcript, and optional context
        """
        if config is None:
            config = PromptConfig()

        lines = []

        # Task description
        if include_instructions:
            lines.append("TASK")
            lines.append(task)
            lines.append("")

        # Audio summary
        if include_audio_summary:
            summary = self._generate_audio_summary()
            if summary:
                lines.append("AUDIO CHARACTERISTICS SUMMARY")
                lines.extend(summary)
                lines.append("")

        # Transcript
        lines.append(self.build_basic_prompt(config))

        return "\n".join(lines)

    def build_comparison_prompt(
        self, segment_ids: list[int], context: str = "Compare these segments"
    ) -> str:
        """
        Build a prompt comparing specific segments.

        Args:
            segment_ids: List of segment IDs to compare
            context: Context or question for the comparison

        Returns:
            Prompt with selected segments and comparison context
        """
        lines = []
        lines.append("COMPARISON TASK")
        lines.append(context)
        lines.append("")

        config = PromptConfig(include_prosody_details=True, include_emotion_details=True)

        for seg_id in segment_ids:
            segment = self._get_segment_by_id(seg_id)
            if segment:
                lines.append(f"SEGMENT {seg_id}:")
                lines.append(self._format_segment(segment, config))

                # Add detailed features
                if segment.audio_state:
                    lines.append(self._format_detailed_features(segment.audio_state))
                lines.append("")

        return "\n".join(lines)

    def build_speaker_aware_prompt(self, config: PromptConfig | None = None) -> str:
        """
        Build a prompt with speaker-aware formatting.

        Groups consecutive segments by speaker for dialogue-style formatting.

        Args:
            config: Optional configuration for prompt generation

        Returns:
            Speaker-formatted transcript
        """
        if config is None:
            config = PromptConfig(speaker_aware=True)

        lines = []
        lines.append("TRANSCRIPT (SPEAKER-AWARE)")
        lines.append(f"File: {self.transcript.file_name}")
        lines.append("")

        current_speaker = None
        speaker_lines = []

        for segment in self.transcript.segments:
            speaker = segment.speaker or "Speaker"

            if speaker != current_speaker:
                # Flush previous speaker's text
                if speaker_lines:
                    lines.append(f"{current_speaker}:")
                    lines.extend(speaker_lines)
                    lines.append("")

                current_speaker = speaker
                speaker_lines = []

            # Format segment
            text = segment.text

            if config.include_audio_annotations and segment.audio_state:
                rendering = segment.audio_state.get("rendering", "")
                if rendering and rendering != "[audio: neutral]":
                    text = f"{text} {rendering}"

            if config.include_timestamps:
                text = f"[{segment.start:.1f}s] {text}"

            speaker_lines.append(f"  {text}")

        # Flush final speaker
        if speaker_lines:
            lines.append(f"{current_speaker}:")
            lines.extend(speaker_lines)

        return "\n".join(lines)

    def build_context_injection_prompt(
        self, user_query: str, relevant_segment_ids: list[int] | None = None
    ) -> str:
        """
        Build a prompt that injects transcript context into a user query.

        Useful for RAG-style applications where transcript segments are
        retrieved based on semantic similarity or audio features.

        Args:
            user_query: The user's question or query
            relevant_segment_ids: Optional list of most relevant segment IDs

        Returns:
            Prompt with context and query
        """
        lines = []
        lines.append("CONTEXT FROM TRANSCRIPT")
        lines.append(f"Source: {self.transcript.file_name}")
        lines.append("")

        segments = self.transcript.segments
        if relevant_segment_ids:
            segments = [self._get_segment_by_id(sid) for sid in relevant_segment_ids]
            segments = [s for s in segments if s is not None]

        config = PromptConfig(include_timestamps=True, include_audio_annotations=True)

        for segment in segments[:10]:  # Limit context length
            lines.append(self._format_segment(segment, config))

        lines.append("")
        lines.append("QUERY")
        lines.append(user_query)

        return "\n".join(lines)

    def _format_segment(self, segment: Segment, config: PromptConfig) -> str:
        """Format a single segment according to configuration."""
        parts = []

        # Timestamp
        if config.include_timestamps:
            parts.append(f"[{segment.start:.1f}s - {segment.end:.1f}s]")

        # Speaker (if available and enabled)
        if config.speaker_aware and segment.speaker:
            parts.append(f"{segment.speaker}:")

        # Text
        text = segment.text
        parts.append(text)

        # Audio annotations
        if config.include_audio_annotations and segment.audio_state:
            rendering = segment.audio_state.get("rendering", "")

            if config.format_style == "inline" and rendering:
                if rendering != "[audio: neutral]":
                    parts.append(rendering)

            elif config.format_style == "separate" and rendering:
                # Return with annotation on separate line
                base = " ".join(parts)
                return f"{base}\n    {rendering}"

        return " ".join(parts)

    def _format_detailed_features(self, audio_state: dict[str, Any]) -> str:
        """Format detailed audio features for comparison tasks."""
        lines = []
        lines.append("  Audio Features:")

        # Prosody
        prosody = audio_state.get("prosody", {})
        if prosody:
            pitch = prosody.get("pitch", {})
            energy = prosody.get("energy", {})
            rate = prosody.get("rate", {})

            if pitch:
                lines.append(
                    f"    Pitch: {pitch.get('level', 'N/A')} ({pitch.get('mean_hz', 0):.1f} Hz)"
                )
            if energy:
                lines.append(
                    f"    Energy: {energy.get('level', 'N/A')} ({energy.get('db_rms', 0):.1f} dB)"
                )
            if rate:
                lines.append(
                    f"    Rate: {rate.get('level', 'N/A')} ({rate.get('syllables_per_sec', 0):.2f} syl/sec)"
                )

        # Emotion
        emotion = audio_state.get("emotion", {})
        if emotion:
            valence = emotion.get("valence", {})
            arousal = emotion.get("arousal", {})
            categorical = emotion.get("categorical", {})

            if categorical:
                lines.append(
                    f"    Emotion: {categorical.get('primary', 'N/A')} (confidence: {categorical.get('confidence', 0):.2f})"
                )
            elif valence or arousal:
                val_str = f"valence={valence.get('score', 0):.2f}" if valence else ""
                aro_str = f"arousal={arousal.get('score', 0):.2f}" if arousal else ""
                lines.append(f"    Emotion: {val_str}, {aro_str}")

        return "\n".join(lines)

    def _generate_audio_summary(self) -> list[str]:
        """Generate summary statistics about audio characteristics."""
        lines = []

        total_segments = len(self.transcript.segments)
        enriched_segments = sum(1 for s in self.transcript.segments if s.audio_state)

        if enriched_segments == 0:
            return []

        lines.append(f"Total segments: {total_segments}")
        lines.append(f"Segments with audio features: {enriched_segments}")

        # Collect statistics
        pitch_levels = []
        energy_levels = []
        emotions = []

        for segment in self.transcript.segments:
            if not segment.audio_state:
                continue

            prosody = segment.audio_state.get("prosody", {})
            if prosody:
                if "pitch" in prosody:
                    pitch_levels.append(prosody["pitch"].get("level", "unknown"))
                if "energy" in prosody:
                    energy_levels.append(prosody["energy"].get("level", "unknown"))

            emotion = segment.audio_state.get("emotion", {})
            if emotion and "categorical" in emotion:
                emotions.append(emotion["categorical"].get("primary", "unknown"))

        # Summarize distributions
        if pitch_levels:
            lines.append(f"Pitch distribution: {self._distribution_summary(pitch_levels)}")
        if energy_levels:
            lines.append(f"Energy distribution: {self._distribution_summary(energy_levels)}")
        if emotions:
            lines.append(f"Emotion distribution: {self._distribution_summary(emotions)}")

        return lines

    def _distribution_summary(self, items: list[str]) -> str:
        """Create a summary of item distribution."""
        from collections import Counter

        counts = Counter(items)
        total = len(items)
        top_3 = counts.most_common(3)

        parts = [f"{item} ({count}/{total})" for item, count in top_3]
        return ", ".join(parts)

    def _get_segment_by_id(self, segment_id: int) -> Segment | None:
        """Get a segment by its ID."""
        for segment in self.transcript.segments:
            if segment.id == segment_id:
                return segment
        return None


# Example usage
def demonstrate_prompt_builder():
    """Demonstrate the prompt builder with example output."""
    print("=" * 80)
    print("ENRICHED TRANSCRIPT PROMPT BUILDER DEMO")
    print("=" * 80)
    print()

    # Create sample transcript data
    print("This example demonstrates various prompt formatting styles.")
    print("In practice, you would use:")
    print()
    print("    builder = TranscriptPromptBuilder('path/to/enriched.json')")
    print("    prompt = builder.build_analysis_prompt('Find moments of escalation')")
    print()
    print("=" * 80)
    print()

    # Example 1: Basic inline format
    print("EXAMPLE 1: Basic Inline Format")
    print("-" * 80)
    example_inline = """TRANSCRIPT
File: meeting.wav
Language: en

[0.0s - 2.5s] Hello everyone, thanks for joining today. [audio: high pitch, loud volume]

[2.5s - 5.8s] Let's start with the quarterly results. [audio: neutral]

[5.8s - 9.2s] I'm really excited about our progress! [audio: high pitch, loud volume, fast speech, excited tone]

[9.2s - 12.1s] However, we need to address some concerns. [audio: low pitch, slow speech, concerned tone]"""

    print(example_inline)
    print()

    # Example 2: Speaker-aware format
    print("EXAMPLE 2: Speaker-Aware Format")
    print("-" * 80)
    example_speaker = """TRANSCRIPT (SPEAKER-AWARE)
File: meeting.wav

Alice:
  [0.0s] Hello everyone, thanks for joining today. [audio: high pitch, loud volume]
  [2.5s] Let's start with the quarterly results. [audio: neutral]

Bob:
  [5.8s] I'm really excited about our progress! [audio: high pitch, loud volume, fast speech, excited tone]
  [9.2s] However, we need to address some concerns. [audio: low pitch, slow speech, concerned tone]"""

    print(example_speaker)
    print()

    # Example 3: Analysis prompt
    print("EXAMPLE 3: Analysis Task Prompt")
    print("-" * 80)
    example_analysis = """TASK
Identify moments where the speaker's emotional state shifts significantly.
Look for changes in pitch, energy, speech rate, or emotional tone.

AUDIO CHARACTERISTICS SUMMARY
Total segments: 145
Segments with audio features: 145
Pitch distribution: medium (68/145), high (42/145), low (35/145)
Energy distribution: medium (62/145), high (51/145), low (32/145)
Emotion distribution: neutral (85/145), joy (24/145), concern (14/145)

TRANSCRIPT
File: meeting.wav
Language: en

[0.0s - 2.5s] Hello everyone, thanks for joining today. [audio: neutral]

[2.5s - 5.8s] Let's start with the quarterly results. [audio: neutral]

[5.8s - 9.2s] I'm really excited about our progress! [audio: high pitch, loud volume, fast speech, excited tone]

[9.2s - 12.1s] However, we need to address some concerns. [audio: low pitch, slow speech, concerned tone]"""

    print(example_analysis)
    print()

    # Example 4: Comparison prompt
    print("EXAMPLE 4: Segment Comparison Prompt")
    print("-" * 80)
    example_comparison = """COMPARISON TASK
Compare the emotional tone and delivery style of these two segments.
What differences do you notice?

SEGMENT 23:
[45.2s - 48.5s] I'm really excited about this direction! [audio: high pitch, loud volume, fast speech, excited tone]
  Audio Features:
    Pitch: high (245.3 Hz)
    Energy: loud (-12.1 dB)
    Rate: fast (5.2 syl/sec)
    Emotion: joy (confidence: 0.85)

SEGMENT 67:
[156.8s - 160.2s] I'm not sure this is the right approach. [audio: low pitch, quiet volume, slow speech, uncertain tone]
  Audio Features:
    Pitch: low (125.7 Hz)
    Energy: quiet (-28.5 dB)
    Rate: slow (2.8 syl/sec)
    Emotion: concern (confidence: 0.72)"""

    print(example_comparison)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build LLM prompts from enriched transcripts")
    parser.add_argument("transcript", nargs="?", type=Path, help="Path to enriched JSON transcript")
    parser.add_argument("--task", help="Analysis task description")
    parser.add_argument(
        "--style",
        choices=["basic", "speaker", "analysis"],
        default="basic",
        help="Prompt style to generate",
    )
    parser.add_argument("--demo", action="store_true", help="Show demonstration examples")

    args = parser.parse_args()

    if args.demo or not args.transcript:
        demonstrate_prompt_builder()
    else:
        builder = TranscriptPromptBuilder(args.transcript)

        if args.style == "basic":
            print(builder.build_basic_prompt())
        elif args.style == "speaker":
            print(builder.build_speaker_aware_prompt())
        elif args.style == "analysis":
            task = args.task or "Analyze this transcript"
            print(builder.build_analysis_prompt(task))
