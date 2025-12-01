"""
Minimal example: build a speaker-aware prompt that includes analytics.

Run with an enriched transcript (turn metadata + speaker_stats):
    uv run python examples/llm_integration/speaker_aware_summary.py whisper_json/meeting.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transcription import load_transcript
from transcription.llm_utils import to_speaker_summary, to_turn_view


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render speaker-aware prompt from a transcript")
    parser.add_argument(
        "transcript",
        nargs="?",
        default="whisper_json/meeting.json",
        help="Path to enriched transcript JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    transcript_path = Path(args.transcript)

    transcript = load_transcript(transcript_path)

    # Map canonical speaker IDs to human labels (customize for your data)
    speaker_labels = {"spk_0": "Agent", "spk_1": "Customer"}

    speaker_block = to_speaker_summary(transcript, speaker_labels=speaker_labels)
    turns_block = to_turn_view(
        transcript,
        include_audio_state=True,
        include_timestamps=True,
        speaker_labels=speaker_labels,
    )

    prompt = f"""{speaker_block}

Conversation:
{turns_block}
"""

    print(prompt)


if __name__ == "__main__":
    main()
