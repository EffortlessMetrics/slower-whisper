#!/usr/bin/env python3
"""
Word-level timestamps example.

Demonstrates how to extract word-level timing information from transcripts,
useful for subtitle generation, forced alignment, and word-level search.

Word timestamps are available since v1.8.0 via the `word_timestamps` config option.
When enabled, each segment includes a list of Word objects with precise timing
for every word, plus confidence scores from the ASR model.

Use cases:
- Subtitle generation with word-by-word highlighting (karaoke-style)
- Forced alignment for dubbing/voice-over work
- Word-level search and navigation in long recordings
- Confidence-based quality filtering
- Speaker attribution at word level (with diarization)

Usage:
    python examples/word_timestamps_example.py <audio_file> [output_dir]

Example:
    python examples/word_timestamps_example.py interview.wav ./output
    python examples/word_timestamps_example.py podcast.mp3 ./output --device cpu
"""

import sys
from pathlib import Path

from transcription import TranscriptionConfig, Word, transcribe_file


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def demo_basic_word_access(segments: list) -> None:
    """
    Demo 1: Basic word-level data access.

    Shows how to iterate through segments and access word timing information.
    """
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Word Access")
    print("=" * 80)

    for i, segment in enumerate(segments[:3]):  # Show first 3 segments
        print(f"\nSegment {segment.id}: {segment.text.strip()}")
        print(f"  Time: {segment.start:.2f}s - {segment.end:.2f}s")

        if segment.words:
            print(f"  Words ({len(segment.words)}):")
            for word in segment.words:
                conf_bar = "*" * int(word.probability * 10)
                print(
                    f"    '{word.word}': "
                    f"{word.start:.2f}s - {word.end:.2f}s "
                    f"(conf: {word.probability:.2f} {conf_bar})"
                )
        else:
            print("  Words: None (word_timestamps may not be enabled)")

        if i >= 2:
            print("\n  ... (showing first 3 segments only)")
            break


def demo_word_level_srt(segments: list, output_path: Path | None = None) -> str:
    """
    Demo 2: Generate word-level SRT subtitles.

    Creates subtitles where each word appears individually, useful for
    karaoke-style highlighting or precise subtitle timing.
    """
    print("\n" + "=" * 80)
    print("DEMO 2: Word-Level SRT Generation")
    print("=" * 80)

    srt_lines = []
    subtitle_index = 1

    for segment in segments:
        if not segment.words:
            continue

        for word in segment.words:
            # Skip very short words (likely artifacts)
            if word.end - word.start < 0.05:
                continue

            start_ts = format_srt_timestamp(word.start)
            end_ts = format_srt_timestamp(word.end)

            srt_lines.append(str(subtitle_index))
            srt_lines.append(f"{start_ts} --> {end_ts}")
            srt_lines.append(word.word.strip())
            srt_lines.append("")

            subtitle_index += 1

    srt_content = "\n".join(srt_lines)

    # Show preview
    preview_lines = srt_content.split("\n")[:16]  # First 4 subtitles
    print("\nWord-level SRT preview:")
    print("-" * 40)
    print("\n".join(preview_lines))
    if len(srt_lines) > 16:
        print("...")
    print("-" * 40)
    print(f"Total word subtitles: {subtitle_index - 1}")

    if output_path:
        output_path.write_text(srt_content, encoding="utf-8")
        print(f"Saved to: {output_path}")

    return srt_content


def demo_confidence_filtering(segments: list, threshold: float = 0.8) -> None:
    """
    Demo 3: Filter words by confidence score.

    Identifies low-confidence words that may need review or correction.
    Useful for quality assurance workflows.
    """
    print("\n" + "=" * 80)
    print(f"DEMO 3: Confidence Filtering (threshold: {threshold})")
    print("=" * 80)

    low_confidence_words: list[tuple[Word, int]] = []
    total_words = 0

    for segment in segments:
        if not segment.words:
            continue

        for word in segment.words:
            total_words += 1
            if word.probability < threshold:
                low_confidence_words.append((word, segment.id))

    print(f"\nTotal words: {total_words}")
    print(f"Low confidence words (< {threshold}): {len(low_confidence_words)}")

    if low_confidence_words:
        print("\nWords that may need review:")
        for word, seg_id in low_confidence_words[:10]:  # Show first 10
            print(
                f"  [{format_timestamp(word.start)}] '{word.word}' "
                f"(conf: {word.probability:.2f}, segment {seg_id})"
            )
        if len(low_confidence_words) > 10:
            print(f"  ... and {len(low_confidence_words) - 10} more")


def demo_word_search(segments: list, search_term: str) -> None:
    """
    Demo 4: Search for specific words and get their timestamps.

    Enables navigation to specific words in long recordings.
    """
    print("\n" + "=" * 80)
    print(f"DEMO 4: Word Search ('{search_term}')")
    print("=" * 80)

    search_lower = search_term.lower()
    matches: list[tuple[Word, int]] = []

    for segment in segments:
        if not segment.words:
            continue

        for word in segment.words:
            if search_lower in word.word.lower():
                matches.append((word, segment.id))

    print(f"\nFound {len(matches)} occurrence(s) of '{search_term}':")
    for word, seg_id in matches[:10]:
        print(
            f"  [{format_timestamp(word.start)}] '{word.word}' "
            f"(segment {seg_id}, conf: {word.probability:.2f})"
        )
    if len(matches) > 10:
        print(f"  ... and {len(matches) - 10} more")


def demo_timing_statistics(segments: list) -> None:
    """
    Demo 5: Word timing statistics.

    Analyzes speaking rate and word duration patterns.
    """
    print("\n" + "=" * 80)
    print("DEMO 5: Timing Statistics")
    print("=" * 80)

    durations: list[float] = []
    gaps: list[float] = []
    prev_end: float | None = None

    for segment in segments:
        if not segment.words:
            continue

        for word in segment.words:
            duration = word.end - word.start
            if duration > 0:
                durations.append(duration)

            if prev_end is not None:
                gap = word.start - prev_end
                if 0 <= gap < 2.0:  # Reasonable gap threshold
                    gaps.append(gap)

            prev_end = word.end

    if durations:
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        # Calculate words per minute (rough estimate)
        if segments and segments[-1].end > 0:
            total_duration_min = segments[-1].end / 60
            wpm = len(durations) / total_duration_min if total_duration_min > 0 else 0
        else:
            wpm = 0

        print("\nWord duration statistics:")
        print(f"  Total words: {len(durations)}")
        print(f"  Average duration: {avg_duration * 1000:.0f}ms")
        print(f"  Min duration: {min_duration * 1000:.0f}ms")
        print(f"  Max duration: {max_duration * 1000:.0f}ms")
        print(f"  Estimated speaking rate: {wpm:.0f} words/minute")

    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        print("\nInter-word gap statistics:")
        print(f"  Average gap: {avg_gap * 1000:.0f}ms")


def main():
    """
    Main entry point demonstrating word-level timestamp features.
    """
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python word_timestamps_example.py <audio_file> [output_dir] [options]")
        print("\nOptions:")
        print("  --model <name>       Whisper model (default: base)")
        print("  --device <cuda|cpu>  Device to use (default: cuda)")
        print("  --language <code>    Language code (default: auto-detect)")
        print("  --search <term>      Word to search for in demo 4")
        print("\nExamples:")
        print("  python word_timestamps_example.py interview.wav")
        print("  python word_timestamps_example.py podcast.mp3 ./output --device cpu")
        print("  python word_timestamps_example.py speech.wav ./out --search hello")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    output_dir = (
        Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    )

    # Parse optional arguments
    model = "base"  # Use smaller model for faster demo
    device = "cuda"
    language = None
    search_term = "the"  # Default search term

    i = 2 if output_dir is None else 3
    while i < len(sys.argv):
        if sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--device" and i + 1 < len(sys.argv):
            device = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--language" and i + 1 < len(sys.argv):
            language = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--search" and i + 1 < len(sys.argv):
            search_term = sys.argv[i + 1]
            i += 2
        elif not sys.argv[i].startswith("--"):
            # This might be output_dir if not already set
            if output_dir is None:
                output_dir = Path(sys.argv[i])
            i += 1
        else:
            print(f"Unknown option: {sys.argv[i]}")
            sys.exit(1)

    # Validate audio file
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path("./word_timestamps_output")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("WORD-LEVEL TIMESTAMPS EXAMPLE")
    print("=" * 80)
    print(f"\nAudio file: {audio_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Language: {language or 'auto-detect'}")

    # Create configuration with word_timestamps enabled
    # This is the key setting that enables word-level timing
    config = TranscriptionConfig(
        model=model,
        device=device,
        language=language,
        word_timestamps=True,  # Enable word-level timestamps
    )

    print("\nConfiguration:")
    print(f"  word_timestamps: {config.word_timestamps}")

    # Transcribe the file
    print("\nTranscribing with word-level timestamps...")
    try:
        transcript = transcribe_file(audio_path, output_dir, config)
    except Exception as e:
        print(f"\nError during transcription: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Display basic results
    print("\n" + "=" * 80)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 80)
    print(f"\nFile: {transcript.file_name}")
    print(f"Language: {transcript.language}")
    print(f"Segments: {len(transcript.segments)}")

    if transcript.segments:
        total_duration = transcript.segments[-1].end
        print(f"Duration: {format_timestamp(total_duration)}")

        # Count total words
        total_words = sum(len(seg.words) if seg.words else 0 for seg in transcript.segments)
        print(f"Total words with timestamps: {total_words}")

    # Run demos
    if transcript.segments:
        # Demo 1: Basic word access
        demo_basic_word_access(transcript.segments)

        # Demo 2: Word-level SRT generation
        srt_path = output_dir / f"{audio_path.stem}_word_level.srt"
        demo_word_level_srt(transcript.segments, srt_path)

        # Demo 3: Confidence filtering
        demo_confidence_filtering(transcript.segments, threshold=0.8)

        # Demo 4: Word search
        demo_word_search(transcript.segments, search_term)

        # Demo 5: Timing statistics
        demo_timing_statistics(transcript.segments)

    # Summary
    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    stem = audio_path.stem
    print(f"Standard JSON: {output_dir / 'whisper_json' / f'{stem}.json'}")
    print(f"Standard TXT:  {output_dir / 'transcripts' / f'{stem}.txt'}")
    print(f"Standard SRT:  {output_dir / 'transcripts' / f'{stem}.srt'}")
    print(f"Word-level SRT: {output_dir / f'{stem}_word_level.srt'}")

    print("\n" + "=" * 80)
    print("PROGRAMMATIC USAGE")
    print("=" * 80)
    print("""
To use word timestamps in your own code:

    from transcription import TranscriptionConfig, transcribe_file

    # Enable word_timestamps in config
    config = TranscriptionConfig(
        model="base",
        word_timestamps=True,  # Key setting
    )

    # Transcribe
    transcript = transcribe_file("audio.wav", "./output", config)

    # Access word-level data
    for segment in transcript.segments:
        if segment.words:
            for word in segment.words:
                print(f"{word.word}: {word.start:.2f}s - {word.end:.2f}s")
                print(f"  Confidence: {word.probability:.2f}")
                if word.speaker:
                    print(f"  Speaker: {word.speaker}")

The Word dataclass has these fields:
- word: str          - The transcribed word text
- start: float       - Start time in seconds
- end: float         - End time in seconds
- probability: float - ASR confidence score (0.0-1.0)
- speaker: str|None  - Optional speaker ID (with diarization)
""")

    print("\nDone!")


if __name__ == "__main__":
    main()
