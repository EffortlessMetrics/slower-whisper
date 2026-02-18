"""Render a transcript for LLM use alongside semantic annotations.

Usage:
    uv run python examples/llm_integration/semantic_summary.py path/to/transcript.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slower_whisper.pipeline import load_transcript, render_conversation_for_llm
from slower_whisper.pipeline.semantic import KeywordSemanticAnnotator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render transcript + semantic annotations for LLM consumption."
    )
    parser.add_argument(
        "transcript",
        type=Path,
        help="Path to a transcript JSON file (schema v2).",
    )
    parser.add_argument(
        "--mode",
        choices=["turns", "segments"],
        default="turns",
        help="Rendering mode for LLM context (default: turns).",
    )
    args = parser.parse_args()

    transcript = load_transcript(args.transcript)
    annotator = KeywordSemanticAnnotator()
    annotated = annotator.annotate(transcript)

    semantic = (annotated.annotations or {}).get("semantic", {})
    print("Semantic signals")
    print("  keywords :", semantic.get("keywords", []))
    print("  risk_tags:", semantic.get("risk_tags", []))
    print("  actions  :", semantic.get("actions", []))

    conversation = render_conversation_for_llm(
        annotated,
        mode=args.mode,
        include_audio_cues=True,
    )
    print("\nConversation (for LLM input):\n")
    print(conversation)


if __name__ == "__main__":
    main()
