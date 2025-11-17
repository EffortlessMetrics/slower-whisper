#!/usr/bin/env python3
"""
Comparison Demo: With vs Without Audio Enrichment

This script demonstrates the difference in LLM analysis quality when using
enriched transcripts (with audio features) versus plain transcripts (text only).

The comparison shows:
1. What insights are possible with audio enrichment
2. What is missed with text-only transcripts
3. Concrete examples of improved analysis

Usage:
    python comparison_demo.py enriched_transcript.json
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prompt_builder import TranscriptPromptBuilder

from transcription.writers import load_transcript_from_json


class ComparisonDemo:
    """
    Compare LLM prompts with and without audio enrichment.

    This class generates side-by-side comparisons showing how audio
    features enhance LLM understanding.
    """

    def __init__(self, transcript_path: Path):
        """
        Initialize comparison demo.

        Args:
            transcript_path: Path to enriched JSON transcript
        """
        self.transcript_path = Path(transcript_path)
        self.transcript = load_transcript_from_json(self.transcript_path)

    def generate_text_only_prompt(self, task: str) -> str:
        """
        Generate a text-only prompt (no audio features).

        Args:
            task: Analysis task description

        Returns:
            Prompt with only transcribed text
        """
        lines = []
        lines.append("TASK")
        lines.append(task)
        lines.append("")
        lines.append("TRANSCRIPT")
        lines.append(f"File: {self.transcript.file_name}")
        lines.append("")

        for segment in self.transcript.segments:
            lines.append(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")

        return "\n".join(lines)

    def generate_enriched_prompt(self, task: str) -> str:
        """
        Generate an enriched prompt (with audio features).

        Args:
            task: Analysis task description

        Returns:
            Prompt with text and audio annotations
        """
        builder = TranscriptPromptBuilder(self.transcript_path)
        return builder.build_analysis_prompt(task, include_audio_summary=True)

    def compare_prompts(self, task: str) -> tuple[str, str]:
        """
        Generate both versions for comparison.

        Args:
            task: Analysis task description

        Returns:
            Tuple of (text_only_prompt, enriched_prompt)
        """
        return (self.generate_text_only_prompt(task), self.generate_enriched_prompt(task))

    def show_example_scenarios(self):
        """Show concrete examples of how enrichment improves analysis."""
        scenarios = [
            self._scenario_sarcasm_detection(),
            self._scenario_emotion_tracking(),
            self._scenario_speaker_state(),
            self._scenario_question_detection(),
        ]

        for i, scenario in enumerate(scenarios, 1):
            print(f"\nSCENARIO {i}: {scenario['title']}")
            print("=" * 80)
            print()
            print(scenario["description"])
            print()
            print("-" * 80)
            print("WITHOUT AUDIO ENRICHMENT:")
            print("-" * 80)
            print(scenario["text_only"])
            print()
            print("-" * 80)
            print("WITH AUDIO ENRICHMENT:")
            print("-" * 80)
            print(scenario["enriched"])
            print()
            print("-" * 80)
            print("ANALYSIS IMPROVEMENT:")
            print("-" * 80)
            print(scenario["improvement"])
            print()
            print("=" * 80)

    def _scenario_sarcasm_detection(self) -> dict[str, str]:
        """Example: Detecting sarcasm."""
        return {
            "title": "Sarcasm Detection",
            "description": "Detecting sarcasm requires understanding the mismatch between words and tone.",
            "text_only": """[45.2s - 47.8s] Oh great, another meeting. Just what we needed.

TEXT-ONLY ANALYSIS:
An LLM might interpret this literally as positive, or might detect
sarcasm from words like "oh great" but with low confidence since
these words can be used genuinely or sarcastically.

Confidence: LOW (30-50%)
Evidence: Word choice only, ambiguous context""",
            "enriched": """[45.2s - 47.8s] Oh great, another meeting. Just what we needed. [audio: low pitch, quiet volume, slow speech, sarcastic tone]

ENRICHED ANALYSIS:
The audio clearly indicates sarcasm:
- Low pitch (contrasts with genuine excitement which would be high)
- Quiet volume (contrasts with enthusiastic delivery)
- Slow speech rate (contrasts with energetic speech)
- Explicit "sarcastic tone" annotation from audio analysis

Confidence: HIGH (85-95%)
Evidence: Clear contradiction between words and delivery""",
            "improvement": """The enriched version provides:
1. DEFINITIVE tone classification ("sarcastic tone")
2. Multiple corroborating audio signals
3. Clear contrast with expected prosody for genuine enthusiasm
4. Much higher confidence in sarcasm detection

Result: Reduces false positives and enables reliable sarcasm detection.""",
        }

    def _scenario_emotion_tracking(self) -> dict[str, str]:
        """Example: Tracking emotional shifts."""
        return {
            "title": "Emotional Shift Detection",
            "description": "Tracking how emotions change over time during a conversation.",
            "text_only": """[12.5s - 15.2s] I think we can make this work.
[15.2s - 18.7s] Actually, looking at the numbers again...
[18.7s - 22.1s] This might be more challenging than I thought.

TEXT-ONLY ANALYSIS:
Words suggest a shift from optimism to uncertainty, but the
intensity and timing of the shift are unclear. The emotional
trajectory relies heavily on interpreting hedging language.

Detected shift: Optimism → Uncertainty (based on words)
Confidence: MEDIUM""",
            "enriched": """[12.5s - 15.2s] I think we can make this work. [audio: high pitch, loud volume, fast speech, confident tone]
[15.2s - 18.7s] Actually, looking at the numbers again... [audio: medium pitch, normal volume, slow speech, concerned tone]
[18.7s - 22.1s] This might be more challenging than I thought. [audio: low pitch, quiet volume, slow speech, uncertain tone]

ENRICHED ANALYSIS:
Clear emotional descent captured in audio:
- Pitch drops from high → medium → low (confidence diminishing)
- Volume decreases from loud → normal → quiet (energy draining)
- Speech slows from fast → normal → slow (processing difficulty)
- Tone shifts: confident → concerned → uncertain

Detected shift: Strong confidence → Growing concern → Uncertainty
Timing: Rapid shift over 10 seconds
Confidence: HIGH""",
            "improvement": """The enriched version provides:
1. QUANTIFIABLE emotional trajectory (pitch/volume/rate changes)
2. Precise TIMING of the emotional shift
3. INTENSITY measurement (how dramatic the shift is)
4. Multiple CORROBORATING signals (not just word choice)

Result: Enables tracking of emotional dynamics with precision and confidence.""",
        }

    def _scenario_speaker_state(self) -> dict[str, str]:
        """Example: Assessing speaker mental state."""
        return {
            "title": "Speaker Mental State Assessment",
            "description": "Understanding a speaker's confidence, stress, or cognitive load.",
            "text_only": """[89.3s - 95.6s] Well, I mean, the timeline is... uh... we're looking at,
you know, probably sometime in Q3, maybe Q4, depending on... um... various factors.

TEXT-ONLY ANALYSIS:
The text shows verbal hesitation markers (uh, um, well, I mean)
and vague language ("various factors", "probably", "maybe").
Suggests uncertainty but can't measure intensity.

Assessment: Uncertain, possibly stressed
Confidence: MEDIUM
Evidence: Filler words and hedging language""",
            "enriched": """[89.3s - 95.6s] Well, I mean, the timeline is... uh... we're looking at,
you know, probably sometime in Q3, maybe Q4, depending on... um... various factors.
[audio: low pitch, quiet volume, very slow speech (2.1 syl/sec), 5 pauses, uncertain tone]

ENRICHED ANALYSIS:
Multiple stress/cognitive load indicators:
- Very slow speech rate (2.1 syl/sec vs. baseline 3.8 syl/sec)
- 5 distinct pauses in 6.3 seconds = high pause density
- Low pitch + quiet volume = low confidence
- Uncertain tone explicitly detected

Assessment: High uncertainty, significant cognitive load, possible stress
Confidence: HIGH
Evidence: Converging audio signals + verbal markers""",
            "improvement": """The enriched version provides:
1. QUANTITATIVE measures (speech rate, pause count)
2. COMPARISON to speaker baseline (shows deviation)
3. MULTIPLE independent indicators all pointing same direction
4. INTENSITY measurement (how uncertain/stressed)

Result: Enables nuanced assessment of speaker mental state, distinguishing
between minor uncertainty and significant stress/cognitive load.""",
        }

    def _scenario_question_detection(self) -> dict[str, str]:
        """Example: Detecting questions without question marks."""
        return {
            "title": "Question Detection (No Question Mark)",
            "description": "Identifying questions that lack punctuation, using rising intonation.",
            "text_only": """[34.2s - 36.8s] So we're launching this next month.

TEXT-ONLY ANALYSIS:
Appears to be a statement (no question mark). Without context
or punctuation, impossible to determine if this is a question
or a declarative statement.

Classification: Statement
Confidence: MEDIUM (relies on punctuation)""",
            "enriched": """[34.2s - 36.8s] So we're launching this next month. [audio: rising pitch contour, high final pitch (245 Hz)]

ENRICHED ANALYSIS:
The audio reveals this is actually a question:
- Rising pitch contour (characteristic of questions)
- High final pitch (245 Hz, up from 165 Hz baseline)
- Questioning tone

Despite lack of "?" punctuation, this is clearly seeking
confirmation or expressing surprise.

Classification: Question (confirmation-seeking)
Confidence: HIGH (based on prosody)""",
            "improvement": """The enriched version provides:
1. PROSODIC cues that override punctuation
2. Detection of IMPLIED questions not marked textually
3. DIFFERENTIATION between statement types (declarative vs. interrogative)
4. Understanding of PRAGMATIC intent (seeking confirmation)

Result: More accurate dialogue understanding, catches questions that
text processing would miss.""",
        }


def demonstrate_comparison():
    """Show demonstration of comparison scenarios."""
    print("=" * 80)
    print("AUDIO ENRICHMENT COMPARISON DEMO")
    print("=" * 80)
    print()
    print("This demo shows concrete examples of how audio enrichment improves")
    print("LLM analysis compared to text-only transcripts.")
    print()
    print("For each scenario, we compare:")
    print("  - What an LLM can infer from TEXT ONLY")
    print("  - What an LLM can infer with AUDIO ENRICHMENT")
    print("  - The specific improvements gained")
    print()

    # Create a mock demo instance for examples
    demo = ComparisonDemo.__new__(ComparisonDemo)
    demo.show_example_scenarios()

    print()
    print("=" * 80)
    print("SUMMARY OF BENEFITS")
    print("=" * 80)
    print()
    print("Audio enrichment enables:")
    print()
    print("1. SARCASM & TONE DETECTION")
    print("   - Detect contradictions between words and delivery")
    print("   - Identify irony, sarcasm, genuine vs. fake enthusiasm")
    print("   - Higher confidence in tone classification")
    print()
    print("2. EMOTIONAL ANALYSIS")
    print("   - Track emotional shifts with precision")
    print("   - Measure intensity of emotions")
    print("   - Identify subtle emotional nuances")
    print()
    print("3. SPEAKER STATE ASSESSMENT")
    print("   - Detect stress, cognitive load, fatigue")
    print("   - Measure confidence vs. uncertainty")
    print("   - Quantify hesitation and processing difficulty")
    print()
    print("4. PRAGMATIC UNDERSTANDING")
    print("   - Detect questions without punctuation (using intonation)")
    print("   - Identify emphasis and focus")
    print("   - Understand speaker intent beyond words")
    print()
    print("5. INTERACTION DYNAMICS")
    print("   - Detect escalation/de-escalation")
    print("   - Identify genuine vs. reluctant agreement")
    print("   - Track turn-taking and interruption patterns")
    print()
    print("=" * 80)


def compare_prompt_sizes(transcript_path: Path):
    """Compare prompt sizes with and without enrichment."""
    demo = ComparisonDemo(transcript_path)

    task = "Analyze the emotional tone of this conversation."

    text_only, enriched = demo.compare_prompts(task)

    print("=" * 80)
    print("PROMPT SIZE COMPARISON")
    print("=" * 80)
    print()
    print(f"Text-only prompt: {len(text_only)} characters")
    print(f"Enriched prompt: {len(enriched)} characters")
    print(
        f"Additional size: {len(enriched) - len(text_only)} characters ({(len(enriched) / len(text_only) - 1) * 100:.1f}% increase)"
    )
    print()
    print("The enriched prompt is larger, but the added audio annotations")
    print("provide context that would otherwise require multiple LLM queries")
    print("or remain undetectable.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare analysis with vs without audio enrichment"
    )
    parser.add_argument("transcript", nargs="?", type=Path, help="Path to enriched JSON transcript")
    parser.add_argument("--demo", action="store_true", help="Show demonstration scenarios")
    parser.add_argument("--task", help="Custom analysis task for comparison")

    args = parser.parse_args()

    if args.demo or not args.transcript:
        demonstrate_comparison()
    else:
        if args.task:
            demo = ComparisonDemo(args.transcript)
            text_only, enriched = demo.compare_prompts(args.task)

            print("=" * 80)
            print("COMPARISON: TEXT-ONLY vs ENRICHED")
            print("=" * 80)
            print()
            print("TEXT-ONLY PROMPT:")
            print("-" * 80)
            print(text_only)
            print()
            print("=" * 80)
            print()
            print("ENRICHED PROMPT:")
            print("-" * 80)
            print(enriched)
            print()
            print("=" * 80)
        else:
            compare_prompt_sizes(args.transcript)
