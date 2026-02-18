#!/usr/bin/env python3
"""
Single File Transcription Example

This script demonstrates how to transcribe a single audio file using the
slower-whisper API. This is useful when you don't have a batch of files
or want to process files one at a time.

The script will:
1. Take a single audio file as input
2. Create the necessary directory structure
3. Transcribe the file
4. Display the transcript with timing information
5. Save outputs in JSON, TXT, and SRT formats

Usage:
    python single_file.py <audio_file> <output_dir> [options]

Example:
    python single_file.py interview.mp3 ./output --language en
    python single_file.py podcast.wav ./transcripts --model medium --device cpu
"""

import sys
from pathlib import Path

from slower_whisper.pipeline import TranscriptionConfig, transcribe_file


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def main():
    """
    Transcribe a single audio file and display the results.
    """
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python single_file.py <audio_file> <output_dir> [options]")
        print("\nOptions:")
        print("  --model <name>       Whisper model (default: large-v3)")
        print("  --device <cuda|cpu>  Device to use (default: cuda)")
        print("  --language <code>    Language code (default: auto-detect)")
        print("  --task <transcribe|translate>  Task type (default: transcribe)")
        print("  --show-segments      Show all segments with timestamps")
        print("\nExamples:")
        print("  python single_file.py interview.mp3 ./output")
        print("  python single_file.py podcast.wav ./output --language en --show-segments")
        print("  python single_file.py speech.m4a ./output --task translate --device cpu")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    # Parse optional arguments
    model = "large-v3"
    device = "cuda"
    language = None
    task = "transcribe"
    show_segments = False

    i = 3
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
        elif sys.argv[i] == "--task" and i + 1 < len(sys.argv):
            task = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--show-segments":
            show_segments = True
            i += 1
        else:
            print(f"Unknown option: {sys.argv[i]}")
            sys.exit(1)

    # Validate inputs
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    if not audio_path.is_file():
        print(f"Error: Path is not a file: {audio_path}")
        sys.exit(1)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SINGLE FILE TRANSCRIPTION")
    print("=" * 80)
    print(f"\nAudio file: {audio_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Language: {language or 'auto-detect'}")
    print(f"Task: {task}")
    print()

    # Create configuration
    config = TranscriptionConfig(
        model=model,
        device=device,
        language=language,
        task=task,
    )

    # Transcribe the file
    print("Transcribing...")
    try:
        transcript = transcribe_file(audio_path, output_dir, config)
    except Exception as e:
        print(f"\nError during transcription: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Display results
    print("\n" + "=" * 80)
    print("TRANSCRIPTION RESULTS")
    print("=" * 80)

    print(f"\nFile: {transcript.file_name}")
    print(f"Language: {transcript.language}")
    print(f"Segments: {len(transcript.segments)}")

    if transcript.segments:
        # Calculate statistics
        total_duration = transcript.segments[-1].end
        total_words = sum(len(seg.text.split()) for seg in transcript.segments)
        avg_segment_duration = total_duration / len(transcript.segments)

        print(f"Duration: {format_timestamp(total_duration)} ({total_duration:.1f}s)")
        print(f"Total words: {total_words}")
        print(f"Average segment length: {avg_segment_duration:.2f}s")

        # Display full transcript or segment-by-segment
        if show_segments:
            print("\n" + "-" * 80)
            print("SEGMENTS")
            print("-" * 80)
            for segment in transcript.segments:
                timestamp = (
                    f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}]"
                )
                print(f"\n{timestamp}")
                print(segment.text.strip())
        else:
            # Show full text
            print("\n" + "-" * 80)
            print("FULL TRANSCRIPT")
            print("-" * 80)
            full_text = " ".join(seg.text.strip() for seg in transcript.segments)
            print(f"\n{full_text}\n")

            # Show first and last segments as preview
            print("-" * 80)
            print("FIRST SEGMENT:")
            print(
                f"[{format_timestamp(transcript.segments[0].start)}] {transcript.segments[0].text.strip()}"
            )
            print("\nLAST SEGMENT:")
            print(
                f"[{format_timestamp(transcript.segments[-1].start)}] {transcript.segments[-1].text.strip()}"
            )

    # Show output file locations
    stem = audio_path.stem
    print("\n" + "-" * 80)
    print("OUTPUT FILES")
    print("-" * 80)
    print(f"JSON: {output_dir / 'whisper_json' / f'{stem}.json'}")
    print(f"TXT:  {output_dir / 'transcripts' / f'{stem}.txt'}")
    print(f"SRT:  {output_dir / 'transcripts' / f'{stem}.srt'}")

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
