#!/usr/bin/env python3
"""Demonstrate word-level timestamp extraction.

Usage:
    python examples/word_timestamps_demo.py path/to/audio.wav

This example shows how to:
1. Transcribe audio with word-level timestamps enabled
2. Access individual word timings
3. Format output with precise word boundaries

Word-level timestamps are useful for:
- Karaoke-style subtitle highlighting
- Precise audio editing and clipping
- Word-level search in audio archives
- Quality assurance (low-confidence word detection)
- Speaking rate analysis

Requirements:
- Audio file in any ffmpeg-supported format
- GPU recommended for faster transcription (use --device cpu for CPU inference)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS.mmm for display."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def main() -> int:
    """Demonstrate word-level timestamp extraction."""
    parser = argparse.ArgumentParser(
        description="Demonstrate word-level timestamp extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to audio file (any format supported by ffmpeg)",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model to use (default: base)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device for inference (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for transcript files (default: creates temp dir)",
    )

    args = parser.parse_args()

    audio_path: Path = args.audio_file
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        return 1

    # Import after argument parsing to fail fast on bad args
    from transcription import TranscriptionConfig, transcribe_file

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        import tempfile

        output_dir = Path(tempfile.mkdtemp(prefix="word_timestamps_"))

    print("=" * 70)
    print("WORD-LEVEL TIMESTAMPS DEMO")
    print("=" * 70)
    print(f"\nAudio file: {audio_path}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")

    # Step 1: Transcribe with word_timestamps enabled
    print("\n[1] Transcribing with word-level timestamps enabled...")

    config = TranscriptionConfig(
        model=args.model,
        device=args.device,
        word_timestamps=True,  # Key setting for word-level timing
    )

    transcript = transcribe_file(audio_path, output_dir, config)

    print(f"    Language detected: {transcript.language}")
    print(f"    Total segments: {len(transcript.segments)}")
    print(f"    Total duration: {transcript.duration:.2f}s")

    # Step 2: Access individual word timings
    print("\n[2] Accessing word-level timing data...")

    total_words = 0
    for segment in transcript.segments:
        if segment.words:
            total_words += len(segment.words)

    print(f"    Total words with timestamps: {total_words}")

    # Step 3: Format output with precise word boundaries
    print("\n[3] Word-level output (first 3 segments):")
    print("-" * 70)

    for _i, segment in enumerate(transcript.segments[:3]):
        print(
            f"\nSegment {segment.id} [{format_time(segment.start)} - {format_time(segment.end)}]:"
        )
        print(f"  Text: {segment.text.strip()}")

        if segment.words:
            print("  Words:")
            for word in segment.words:
                # Show confidence as a visual bar
                conf_bar = "#" * int(word.probability * 10)
                conf_bar = conf_bar.ljust(10, ".")

                print(
                    f"    {format_time(word.start)} - {format_time(word.end)} "
                    f"[{conf_bar}] {word.word}"
                )
        else:
            print("  Words: (not available)")

    if len(transcript.segments) > 3:
        print(f"\n... ({len(transcript.segments) - 3} more segments)")

    # Bonus: Show some useful analytics
    print("\n" + "=" * 70)
    print("ANALYTICS")
    print("=" * 70)

    if total_words > 0:
        # Calculate speaking rate
        wpm = (total_words / transcript.duration) * 60 if transcript.duration > 0 else 0
        print(f"\nSpeaking rate: {wpm:.0f} words per minute")

        # Find low-confidence words
        low_conf_words = []
        for segment in transcript.segments:
            if segment.words:
                for word in segment.words:
                    if word.probability < 0.7:
                        low_conf_words.append((word, segment.id))

        if low_conf_words:
            print(f"\nLow-confidence words (< 70%): {len(low_conf_words)}")
            for word, seg_id in low_conf_words[:5]:
                print(
                    f"  [{format_time(word.start)}] '{word.word}' "
                    f"(confidence: {word.probability:.0%}, segment {seg_id})"
                )
            if len(low_conf_words) > 5:
                print(f"  ... and {len(low_conf_words) - 5} more")
        else:
            print("\nAll words have confidence >= 70%")

    # Show programmatic usage
    print("\n" + "=" * 70)
    print("PROGRAMMATIC USAGE")
    print("=" * 70)
    print(
        """
To use word timestamps in your own code:

    from transcription import TranscriptionConfig, transcribe_file

    # Enable word_timestamps in config
    config = TranscriptionConfig(
        model="base",
        word_timestamps=True,
    )

    # Transcribe
    transcript = transcribe_file("audio.wav", "./output", config)

    # Access word-level data
    for segment in transcript.segments:
        if segment.words:
            for word in segment.words:
                print(f"{word.word}: {word.start:.2f}s - {word.end:.2f}s")
                print(f"  Confidence: {word.probability:.0%}")

    # Use the full_text property for quick access
    print(transcript.full_text)

    # Find segment at a specific time
    segment = transcript.get_segment_at_time(10.5)
    if segment:
        print(f"At 10.5s: {segment.text}")
"""
    )

    print(f"\nOutput files saved to: {output_dir}")
    print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
