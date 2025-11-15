#!/usr/bin/env python3
"""
LLM Analysis Query Templates

This module provides pre-built query templates for common analysis tasks
on enriched transcripts. These queries leverage audio features to enable
sophisticated analysis beyond simple text processing.

Query Categories:
- Emotional Analysis: Find emotional shifts, peaks, intensity
- Communication Patterns: Detect hesitation, certainty, sarcasm
- Interaction Analysis: Identify escalation, agreement, conflict
- Content Analysis: Summarize with emotional context, find key moments

Usage:
    from analysis_queries import QueryTemplates

    templates = QueryTemplates("enriched_transcript.json")
    prompt = templates.find_escalation_moments()
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompt_builder import PromptConfig, TranscriptPromptBuilder


class QueryTemplates:
    """
    Pre-built query templates for LLM analysis of enriched transcripts.

    This class provides ready-to-use prompts for common analysis patterns,
    making it easy to extract insights from audio-enriched transcripts.
    """

    def __init__(self, transcript_path: Path):
        """
        Initialize query templates.

        Args:
            transcript_path: Path to enriched JSON transcript
        """
        self.builder = TranscriptPromptBuilder(transcript_path)

    def find_escalation_moments(self) -> str:
        """
        Generate prompt to find moments of escalation.

        Escalation is characterized by:
        - Increasing pitch
        - Increasing volume/energy
        - Faster speech rate
        - Rising emotional intensity
        """
        task = """Identify moments where the speaker's intensity escalates.

Look for patterns indicating escalation:
- Rising pitch (low → medium → high)
- Increasing volume/energy (quiet → loud)
- Faster speech rate
- Emotional shift toward higher arousal (calm → agitated, neutral → excited)

For each escalation moment, provide:
1. Time range (start - end)
2. The escalation trigger (what caused the shift)
3. Audio indicators (pitch, energy, rate changes)
4. The content being discussed

Format your response as a numbered list of escalation moments."""

        return self.builder.build_analysis_prompt(task=task, include_audio_summary=True)

    def find_uncertain_statements(self) -> str:
        """
        Generate prompt to identify uncertain or hesitant statements.

        Uncertainty is characterized by:
        - Frequent pauses
        - Slower speech rate
        - Hesitation markers in speech
        - Lower confidence in emotional classification
        """
        task = """Identify statements where the speaker sounds uncertain or hesitant.

Look for uncertainty indicators:
- Frequent pauses or long pauses (suggests thinking/uncertainty)
- Slower than normal speech rate
- Lower pitch or dropping pitch
- Hesitation words (um, uh, well, etc.)
- Audio annotations indicating "uncertain tone"

For each uncertain statement, provide:
1. Timestamp and full text
2. Uncertainty indicators (pauses, rate, etc.)
3. What topic they're uncertain about
4. Confidence level (very uncertain, somewhat uncertain, slightly uncertain)

Group similar uncertainties together if they relate to the same topic."""

        return self.builder.build_analysis_prompt(task=task, include_audio_summary=True)

    def detect_sarcasm_contradiction(self) -> str:
        """
        Generate prompt to detect sarcasm or contradictions between words and tone.

        Sarcasm detection looks for:
        - Mismatch between positive words and negative/neutral tone
        - Exaggerated prosody patterns
        - Specific emotional tones (sarcastic, ironic)
        """
        task = """Identify potential sarcasm or contradictions between what is said and how it's said.

Look for mismatches:
- Positive words with neutral/negative emotional tone
- Enthusiastic words with low energy/flat delivery
- Agreement words with agitated/concerned tone
- Compliments with unusual prosody (exaggerated pitch, etc.)

Also look for explicit sarcasm indicators:
- Audio annotations mentioning "sarcastic" tone
- Exaggerated prosodic features (very high pitch, etc.)

For each potential sarcasm/contradiction:
1. Timestamp and text
2. The contradiction (what's said vs. how it's said)
3. Audio evidence (tone, pitch, energy)
4. Confidence in sarcasm detection (high/medium/low)
5. Probable intended meaning

Be conservative - only flag clear contradictions."""

        return self.builder.build_analysis_prompt(task=task, include_audio_summary=True)

    def summarize_emotional_arc(self) -> str:
        """
        Generate prompt to summarize the emotional journey through the transcript.

        Analyzes the overall emotional progression and key shifts.
        """
        task = """Summarize the emotional arc of this transcript.

Analyze the emotional journey:
1. Opening emotional state (first ~10% of transcript)
2. Key emotional shifts (when and why emotions changed)
3. Emotional peaks (highest arousal/intensity moments)
4. Emotional valleys (lowest energy/most subdued moments)
5. Closing emotional state (final ~10% of transcript)

For the overall arc, describe:
- Dominant emotions throughout
- How emotions evolved over time
- Critical moments that shifted the emotional tone
- Overall emotional trajectory (positive to negative? stable? volatile?)

Use the audio features (pitch, energy, rate, emotional tone) as evidence.
Structure as a narrative summary, not a list."""

        return self.builder.build_analysis_prompt(task=task, include_audio_summary=True)

    def find_high_confidence_statements(self) -> str:
        """
        Generate prompt to find moments of high confidence or certainty.

        Confidence is characterized by:
        - Strong, consistent energy
        - Clear, steady prosody
        - Minimal pauses
        - Faster speech rate
        - Assertive emotional tones
        """
        task = """Identify statements where the speaker sounds highly confident or certain.

Look for confidence indicators:
- High energy/loud volume (assertive delivery)
- Steady pitch without excessive variation
- Fast or normal speech rate (no hesitation)
- Minimal pauses between words
- Emotional tones: confident, assertive, determined
- Clear, declarative language

For each confident statement:
1. Timestamp and full text
2. Confidence indicators from audio
3. What claim/assertion they're making
4. Whether confidence is justified by context

Separate into:
- Factual assertions (claiming something is true)
- Opinions/judgments (evaluating or recommending)
- Commitments (promising to do something)"""

        return self.builder.build_analysis_prompt(task=task, include_audio_summary=True)

    def identify_questions_by_tone(self) -> str:
        """
        Generate prompt to identify questions, including those without question marks.

        Uses rising intonation and other prosodic cues to detect questions.
        """
        task = """Identify all questions in this transcript, including those not marked with "?".

Look for question indicators:
1. Explicit questions (ending with "?")
2. Rising pitch contour (characteristic of questions)
3. Higher pitch at end compared to beginning
4. Question words (what, where, when, why, how, who)

Classify each question:
- Type: yes/no question, information seeking, rhetorical
- Tone: genuine curiosity, skeptical, challenging, uncertain
- Audio evidence: rising pitch, high pitch, specific tone

List all questions with:
1. Timestamp and text
2. Question type and tone
3. Audio indicators (especially pitch contour)
4. Whether it received an answer (if this is a dialogue)"""

        return self.builder.build_analysis_prompt(task=task, include_audio_summary=True)

    def detect_agreement_disagreement(self) -> str:
        """
        Generate prompt to detect agreement and disagreement in dialogues.

        Uses both words and tone to identify true agreement vs. reluctant compliance.
        """
        task = """Analyze this transcript for moments of agreement and disagreement.

For each speaker interaction, identify:

AGREEMENTS:
- Strong agreement: positive words + positive tone + high energy
- Weak agreement: agreement words + neutral/low tone (possible reluctance)
- Enthusiastic agreement: "yes" + excited tone + fast speech

DISAGREEMENTS:
- Direct disagreement: "no" or disagreement words + any tone
- Soft disagreement: hedging language + uncertain/concerned tone
- Strong disagreement: disagreement words + loud volume + agitated tone

For each agreement/disagreement:
1. Timestamp and text
2. Type (strong/weak agreement, direct/soft disagreement)
3. Audio evidence (tone, energy, pitch)
4. What they're agreeing/disagreeing about
5. Whether the tone matches the words

Focus on identifying:
- False agreements (saying yes but tone suggests reluctance)
- Hidden disagreements (polite language but negative tone)"""

        config = PromptConfig(speaker_aware=True)
        return self.builder.build_analysis_prompt(
            task=task, include_audio_summary=True, config=config
        )

    def find_key_moments(self) -> str:
        """
        Generate prompt to identify the most important/impactful moments.

        Uses audio features to find moments of highest significance.
        """
        task = """Identify the 5-10 most important or impactful moments in this transcript.

Key moments are characterized by:
- High emotional intensity (strong tone, high arousal)
- Significant topic shifts
- Peak energy/volume (drawing attention)
- Unusual prosody (very high/low pitch, very fast/slow)
- Strong reactions or emotional shifts

For each key moment:
1. Timestamp and text (include surrounding context if needed)
2. Why this moment is significant
3. Audio indicators (what makes it stand out)
4. Impact on the overall conversation/narrative
5. Key insights or takeaways

Rank by importance, with #1 being most significant.
Explain your ranking criteria."""

        return self.builder.build_analysis_prompt(task=task, include_audio_summary=True)

    def analyze_speaker_states(self) -> str:
        """
        Generate prompt to analyze the emotional/mental state of each speaker.

        Useful for understanding speaker dynamics in conversations.
        """
        task = """Analyze the emotional and mental state of each speaker throughout this transcript.

For each speaker, determine:

EMOTIONAL PATTERNS:
- Baseline emotion (most common emotional tone)
- Emotional range (how much variation do they show?)
- Emotional triggers (what causes shifts in their emotion?)

COMMUNICATION STYLE:
- Speech rate pattern (consistently fast/slow or variable?)
- Energy level pattern (high energy vs. low energy speaker?)
- Pitch patterns (wide range vs. monotone?)

MENTAL STATE INDICATORS:
- Confidence level (based on prosody and tone)
- Stress/pressure indicators (fast speech, pauses, pitch changes)
- Engagement level (energy, variation in prosody)

INTERACTION PATTERNS:
- How they respond to other speakers
- When they become most animated/intense
- When they become most subdued

Provide a character profile for each speaker based on audio evidence."""

        config = PromptConfig(speaker_aware=True)
        return self.builder.build_analysis_prompt(
            task=task, include_audio_summary=True, config=config
        )

    def custom_query(self, task_description: str, include_summary: bool = True) -> str:
        """
        Generate a custom analysis query.

        Args:
            task_description: Custom task description
            include_summary: Whether to include audio summary

        Returns:
            Custom prompt
        """
        return self.builder.build_analysis_prompt(
            task=task_description, include_audio_summary=include_summary
        )


# Demonstration and examples
def demonstrate_query_templates():
    """Show example queries and their expected outputs."""
    print("=" * 80)
    print("LLM ANALYSIS QUERY TEMPLATES - EXAMPLES")
    print("=" * 80)
    print()

    examples = [
        (
            "Find Escalation Moments",
            """
Expected Output Format:

ESCALATION MOMENTS IDENTIFIED:

1. Escalation at 45.2s - 52.8s
   Trigger: Discussion of budget constraints
   Audio Indicators:
   - Pitch increased from medium (165 Hz) to high (245 Hz)
   - Volume increased from normal to loud
   - Speech rate accelerated from 3.5 to 5.2 syl/sec
   - Tone shifted from neutral to agitated

   Content: Speaker became increasingly animated when discussing
   the budget limitations, emphasizing the negative impact on the
   timeline.

2. Escalation at 123.5s - 128.1s
   ...
""",
        ),
        (
            "Detect Uncertain Statements",
            """
Expected Output Format:

UNCERTAIN STATEMENTS:

Topic: Project Timeline
- [45.2s] "Well, I think we might be able to finish by... um...
  maybe end of Q3?"
  Indicators: 3 pauses, slow speech (2.8 syl/sec), falling pitch,
  uncertain tone
  Confidence: Very uncertain

- [67.8s] "The, uh, the database migration could take... anywhere
  from two to six weeks."
  Indicators: 2 pauses, hedge words ("could", "anywhere from"),
  uncertain tone
  Confidence: Somewhat uncertain

Topic: Resource Allocation
...
""",
        ),
        (
            "Detect Sarcasm/Contradiction",
            """
Expected Output Format:

POTENTIAL SARCASM/CONTRADICTIONS:

1. [89.5s] "Oh great, another meeting. Just what we needed."
   Contradiction: Positive words ("great", "just what we needed")
   with sarcastic tone and low energy
   Audio Evidence: Flat pitch contour, low energy (-25 dB),
   sarcastic tone annotation
   Confidence: High
   Intended Meaning: Expressing frustration with excessive meetings

2. [145.2s] "Yeah, that's a brilliant idea."
   Contradiction: Praise words with neutral/negative delivery
   Audio Evidence: Very low pitch, quiet volume, concerned tone
   Confidence: Medium
   Intended Meaning: Likely disagreement or skepticism expressed
   politely

...
""",
        ),
        (
            "Summarize Emotional Arc",
            """
Expected Output Format:

EMOTIONAL ARC SUMMARY:

The conversation begins with a neutral, professional tone as
participants exchange greetings and settle in. The opening shows
medium pitch and moderate energy, suggesting a calm, focused start.

Around the 2-minute mark, there's a noticeable shift as the team
discusses Q3 results. Energy and pitch both increase, with several
segments showing "excited tone" and fast speech, indicating
enthusiasm about positive outcomes.

The emotional peak occurs at 5:30-6:15, when discussing the new
product launch. This segment features the highest pitch (250 Hz),
loudest volume, and fastest speech rate of the entire transcript,
with multiple speakers expressing joy and excitement.

A significant valley follows at 8:45-10:30 during the budget
discussion. Energy drops, speech slows, and concerned tones emerge.
Multiple speakers show uncertainty indicators (pauses, slower rate),
reflecting anxiety about resource constraints.

The transcript closes with a cautiously optimistic tone. While not
as energetic as the peak, the final segments show rising pitch
contours and moderate positive emotion, suggesting the team reached
a constructive resolution.

Overall trajectory: Starts neutral → rises to excited peak →
drops to concerned valley → recovers to cautious optimism. The
arc reflects a problem-solving journey from enthusiasm through
challenge to resolution.
""",
        ),
    ]

    for title, example in examples:
        print(f"\nQUERY: {title}")
        print("-" * 80)
        print(example)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM analysis query templates")
    parser.add_argument("transcript", nargs="?", type=Path, help="Path to enriched JSON transcript")
    parser.add_argument(
        "--query",
        choices=[
            "escalation",
            "uncertain",
            "sarcasm",
            "emotional_arc",
            "confident",
            "questions",
            "agreement",
            "key_moments",
            "speaker_states",
        ],
        help="Query template to generate",
    )
    parser.add_argument("--custom", help="Custom query description")
    parser.add_argument("--demo", action="store_true", help="Show example queries and outputs")

    args = parser.parse_args()

    if args.demo or not args.transcript:
        demonstrate_query_templates()
    else:
        templates = QueryTemplates(args.transcript)

        if args.custom:
            print(templates.custom_query(args.custom))
        elif args.query == "escalation":
            print(templates.find_escalation_moments())
        elif args.query == "uncertain":
            print(templates.find_uncertain_statements())
        elif args.query == "sarcasm":
            print(templates.detect_sarcasm_contradiction())
        elif args.query == "emotional_arc":
            print(templates.summarize_emotional_arc())
        elif args.query == "confident":
            print(templates.find_high_confidence_statements())
        elif args.query == "questions":
            print(templates.identify_questions_by_tone())
        elif args.query == "agreement":
            print(templates.detect_agreement_disagreement())
        elif args.query == "key_moments":
            print(templates.find_key_moments())
        elif args.query == "speaker_states":
            print(templates.analyze_speaker_states())
        else:
            print("Please specify a query type or use --custom")
