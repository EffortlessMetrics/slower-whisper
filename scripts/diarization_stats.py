#!/usr/bin/env python3
"""
Generate human-readable diarization statistics from a transcript JSON.

Usage:
    uv run python scripts/diarization_stats.py whisper_json/audio.json
"""

import sys
from pathlib import Path

# Add parent directory to path to import transcription module
sys.path.insert(0, str(Path(__file__).parent.parent))

from transcription.dogfood_utils import (  # noqa: E402
    compute_diarization_stats,
    print_diarization_stats,
)


def main():
    """Print diarization stats using unified utilities."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/diarization_stats.py <transcript.json>")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    try:
        stats = compute_diarization_stats(json_path)
        print_diarization_stats(stats)
    except Exception as e:
        print(f"Error analyzing transcript: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
