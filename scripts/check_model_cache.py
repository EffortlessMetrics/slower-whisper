#!/usr/bin/env python3
"""
Check what models are already cached and what will be downloaded.

Usage:
    uv run python scripts/check_model_cache.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import transcription module
sys.path.insert(0, str(Path(__file__).parent.parent))

from transcription.dogfood_utils import get_model_cache_status, print_cache_status  # noqa: E402


def main():
    """Print model cache status using unified cache utilities."""
    print("Checking model cache status...\n")

    status = get_model_cache_status()
    print_cache_status(status)

    print("\nNotes:")
    print("- Models are cached permanently after first download")
    print("- Cache location: $SLOWER_WHISPER_CACHE_ROOT or ~/.cache/slower-whisper/")
    print("- Use 'slower-whisper cache --show' for detailed cache info")
    print("- Use 'slower-whisper cache --clear <target>' to remove specific caches")


if __name__ == "__main__":
    main()
