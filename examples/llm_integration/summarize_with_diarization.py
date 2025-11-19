#!/usr/bin/env python3
"""
Example: Summarize a conversation using Claude with speaker diarization.

This script demonstrates the complete slower-whisper → LLM workflow:
1. Load a transcript JSON file (produced by `slower-whisper transcribe`)
2. Render it for LLM consumption using the built-in API
3. Send to Claude for analysis with speaker-aware prompting

Prerequisites:
- A transcript JSON file with diarization and optionally audio enrichment
- ANTHROPIC_API_KEY environment variable set
- Anthropic SDK installed: `uv pip install anthropic`

Usage:
    python examples/llm_integration/summarize_with_diarization.py \
        whisper_json/support_call_001.json

Output:
    Conversation summary with quality scores and coaching feedback
"""

import argparse
import os
import sys

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed.", file=sys.stderr)
    print("Install with: uv pip install anthropic", file=sys.stderr)
    sys.exit(1)

from transcription import load_transcript, render_conversation_for_llm


def analyze_conversation(json_path: str, use_audio_cues: bool = True) -> str:
    """
    Load a transcript and analyze it with Claude.

    Args:
        json_path: Path to the transcript JSON file
        use_audio_cues: Whether to include prosody/emotion cues in the rendering

    Returns:
        Claude's analysis text
    """
    # 1. Load transcript from JSON
    print(f"Loading transcript: {json_path}")
    transcript = load_transcript(json_path)

    # 2. Optionally infer speaker roles from talk time
    speaker_labels = None
    if transcript.speakers and len(transcript.speakers) == 2:
        # Simple heuristic: assume longest speaker is the agent
        # In production, you'd use external metadata or user input
        print("Inferring speaker roles from talk time...")
        speaker_labels = {"spk_0": "Agent", "spk_1": "Customer"}

    # 3. Render for LLM consumption
    print("Rendering conversation for LLM...")
    context = render_conversation_for_llm(
        transcript,
        mode="turns",  # Use turn structure (cleaner for multi-speaker)
        include_audio_cues=use_audio_cues,  # Include prosody/emotion if available
        include_timestamps=True,  # Add [HH:MM:SS] timestamps for reference
        include_metadata=True,  # Include conversation header
        speaker_labels=speaker_labels,  # Map spk_0/spk_1 → Agent/Customer
    )

    print(f"Rendered context: {len(context)} chars\n")

    # 4. Send to Claude for analysis
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    print("Sending to Claude for analysis...")
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""{context}

Please analyze this conversation and provide:

1. **Summary** (2-3 sentences)
   - What did the customer want?
   - Was the issue resolved?

2. **Quality Score** (1-10)
   - Overall call quality rating
   - Brief justification

3. **Coaching Feedback** (2-3 actionable improvements)
   - Specific suggestions for the agent
   - Reference timestamps where relevant

4. **Customer Sentiment Trajectory**
   - How did the customer's tone/emotion change over time?
   - Note any audio cues that indicate frustration, relief, etc.
"""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def main():
    parser = argparse.ArgumentParser(description="Analyze a slower-whisper transcript with Claude")
    parser.add_argument(
        "json_path", help="Path to transcript JSON file (e.g., whisper_json/call_001.json)"
    )
    parser.add_argument(
        "--no-audio-cues",
        action="store_true",
        help="Exclude prosody/emotion cues from the rendering",
    )

    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: File not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)

    analysis = analyze_conversation(args.json_path, use_audio_cues=not args.no_audio_cues)

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(analysis)
    print("=" * 80)


if __name__ == "__main__":
    main()
