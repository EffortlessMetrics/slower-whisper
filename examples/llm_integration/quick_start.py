#!/usr/bin/env python3
"""
Quick Start Guide - LLM Integration

This script provides a hands-on walkthrough of the LLM integration workflow.
Run this to see working examples with sample data.

Usage:
    python quick_start.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    print_section("LLM INTEGRATION QUICK START")

    print("This guide demonstrates how to use enriched transcripts with LLMs.")
    print("We'll walk through the complete workflow step by step.")
    print()

    # Step 1: Understanding the Input
    print_section("STEP 1: Understanding Enriched Transcripts")

    print("Enriched transcripts contain three types of information:")
    print()
    print("1. TRANSCRIBED TEXT")
    print('   "I\'m really excited about this project!"')
    print()
    print("2. AUDIO ANNOTATIONS (human-readable)")
    print("   [audio: high pitch, loud volume, fast speech, excited tone]")
    print()
    print("3. DETAILED FEATURES (machine-readable)")
    print("   {")
    print('     "pitch": {"level": "high", "mean_hz": 245.3},')
    print('     "emotion": {"categorical": {"primary": "joy", "confidence": 0.85}}')
    print("   }")
    print()

    input("Press Enter to continue...")

    # Step 2: Building Prompts
    print_section("STEP 2: Building LLM Prompts")

    print("The prompt_builder.py module converts JSON to LLM-friendly prompts.")
    print()
    print("Example code:")
    print("-" * 80)
    print("""
from prompt_builder import TranscriptPromptBuilder

# Load transcript
builder = TranscriptPromptBuilder("enriched_transcript.json")

# Generate basic prompt
prompt = builder.build_basic_prompt()

# Generate analysis prompt with task
prompt = builder.build_analysis_prompt(
    task="Find moments of escalation",
    include_audio_summary=True
)
""")
    print("-" * 80)
    print()
    print("Output example:")
    print("-" * 80)
    print("""
TASK
Find moments of escalation

AUDIO CHARACTERISTICS SUMMARY
Total segments: 145
Pitch distribution: medium (68/145), high (42/145), low (35/145)
Emotion distribution: neutral (85/145), joy (24/145), concern (14/145)

TRANSCRIPT
File: meeting.wav

[0.0s - 2.5s] Let's discuss the timeline. [audio: neutral]
[2.5s - 5.8s] This is completely unacceptable! [audio: high pitch, loud volume, agitated tone]
""")
    print("-" * 80)
    print()

    input("Press Enter to continue...")

    # Step 3: Pre-Built Queries
    print_section("STEP 3: Using Pre-Built Query Templates")

    print("The analysis_queries.py module provides ready-to-use prompts.")
    print()
    print("Available queries:")
    print("  • find_escalation_moments() - Rising intensity/conflict")
    print("  • find_uncertain_statements() - Hesitation and doubt")
    print("  • detect_sarcasm_contradiction() - Tone vs. words mismatch")
    print("  • summarize_emotional_arc() - Emotional journey")
    print("  • find_high_confidence_statements() - Assertive moments")
    print("  • identify_questions_by_tone() - Questions via intonation")
    print("  • detect_agreement_disagreement() - Consensus analysis")
    print("  • find_key_moments() - Most important moments")
    print("  • analyze_speaker_states() - Speaker profiling")
    print()
    print("Example code:")
    print("-" * 80)
    print("""
from analysis_queries import QueryTemplates

templates = QueryTemplates("enriched_transcript.json")

# Generate query for finding escalation
prompt = templates.find_escalation_moments()

# Generate query for detecting sarcasm
prompt = templates.detect_sarcasm_contradiction()

# Generate emotional arc summary
prompt = templates.summarize_emotional_arc()
""")
    print("-" * 80)
    print()

    input("Press Enter to continue...")

    # Step 4: LLM Integration
    print_section("STEP 4: Integrating with LLM APIs")

    print("The llm_integration_demo.py module handles API calls.")
    print()
    print("Supported providers:")
    print("  • OpenAI (GPT-4, GPT-3.5)")
    print("  • Anthropic (Claude)")
    print("  • Local (OpenAI-compatible APIs)")
    print()
    print("Example code:")
    print("-" * 80)
    print("""
import os
from llm_integration_demo import LLMIntegration, LLMConfig
from prompt_builder import TranscriptPromptBuilder

# Configure LLM
config = LLMConfig(
    provider="openai",
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")
)

llm = LLMIntegration(config)

# Build prompt
builder = TranscriptPromptBuilder("enriched_transcript.json")
prompt = builder.build_analysis_prompt("Find moments of escalation")

# Query LLM
result = llm.query(prompt)

print(result["response"])
print(f"Tokens used: {result['usage']['total_tokens']}")
""")
    print("-" * 80)
    print()

    input("Press Enter to continue...")

    # Step 5: Comparison
    print_section("STEP 5: Benefits of Audio Enrichment")

    print("The comparison_demo.py shows how audio enrichment improves analysis.")
    print()
    print("WITHOUT audio enrichment:")
    print("-" * 80)
    print("[45.2s] Oh great, another meeting. Just what we needed.")
    print()
    print("LLM Analysis: Possibly sarcastic based on word choice")
    print("Confidence: LOW (30-50%)")
    print("-" * 80)
    print()
    print("WITH audio enrichment:")
    print("-" * 80)
    print("[45.2s] Oh great, another meeting. Just what we needed.")
    print("[audio: low pitch, quiet volume, slow speech, sarcastic tone]")
    print()
    print("LLM Analysis: Clearly sarcastic - audio contradicts positive words")
    print("Confidence: HIGH (85-95%)")
    print("-" * 80)
    print()
    print("Key improvements:")
    print("  ✓ Higher confidence in tone detection")
    print("  ✓ Multiple corroborating signals (pitch, volume, rate)")
    print("  ✓ Distinction between genuine and sarcastic expressions")
    print("  ✓ Quantitative metrics for emotional intensity")
    print()

    input("Press Enter to continue...")

    # Step 6: Common Use Cases
    print_section("STEP 6: Common Use Cases")

    print("1. CUSTOMER SERVICE ANALYSIS")
    print("   • Detect frustrated customers before they complain")
    print("   • Identify when agents need support")
    print("   • Measure de-escalation effectiveness")
    print()
    print("2. MEETING INTELLIGENCE")
    print("   • Distinguish genuine vs. reluctant agreement")
    print("   • Find unresolved concerns (tone mismatch)")
    print("   • Identify key decision moments")
    print()
    print("3. INTERVIEW ANALYSIS")
    print("   • Measure candidate confidence objectively")
    print("   • Detect areas of uncertainty vs. expertise")
    print("   • Reduce interviewer bias")
    print()
    print("4. CONTENT ANALYSIS")
    print("   • Find engaging moments for clips")
    print("   • Identify emotional peaks")
    print("   • Track audience engagement")
    print()
    print("5. THERAPY/COUNSELING")
    print("   • Track patient emotional state")
    print("   • Identify breakthrough moments")
    print("   • Monitor treatment progress")
    print()

    input("Press Enter to continue...")

    # Step 7: Next Steps
    print_section("STEP 7: Next Steps")

    print("Now you're ready to use the LLM integration tools!")
    print()
    print("Try these commands:")
    print()
    print("1. View demonstrations:")
    print("   $ python prompt_builder.py --demo")
    print("   $ python analysis_queries.py --demo")
    print("   $ python comparison_demo.py --demo")
    print()
    print("2. Process your own transcript:")
    print("   # First, enrich your transcript")
    print("   $ python audio_enrich.py your_audio.wav")
    print()
    print("   # Then build prompts")
    print("   $ python prompt_builder.py output/your_audio.json")
    print()
    print("3. Generate specific queries:")
    print("   $ python analysis_queries.py output/your_audio.json --query escalation")
    print("   $ python analysis_queries.py output/your_audio.json --query sarcasm")
    print()
    print("4. Integrate with LLM (set API key first):")
    print("   $ export OPENAI_API_KEY='your-key-here'")
    print("   $ python llm_integration_demo.py output/your_audio.json --provider openai")
    print()
    print("5. Read the full documentation:")
    print("   $ cat README.md")
    print()

    print_section("QUICK START COMPLETE")
    print("You're ready to integrate enriched transcripts with LLMs!")
    print()


if __name__ == "__main__":
    main()
