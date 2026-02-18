#!/usr/bin/env python3
"""
Basic Transcription Example

This script demonstrates the simplest way to transcribe all audio files in a
directory using the slower-whisper API. It processes all files in raw_audio/
and outputs transcripts in multiple formats (JSON, TXT, SRT).

Expected directory structure:
    project_root/
        raw_audio/          # Place your audio files here (MP3, WAV, M4A, etc.)
        input_audio/        # Auto-created: normalized WAV files
        whisper_json/       # Auto-created: JSON transcripts
        transcripts/        # Auto-created: TXT and SRT outputs

Usage:
    python basic_transcription.py /path/to/project/root

Requirements:
    - CUDA-enabled GPU (or use --device cpu for CPU inference)
    - Audio files in project_root/raw_audio/
"""

import sys
from pathlib import Path

# Import the public API
from slower_whisper.pipeline import TranscriptionConfig, transcribe_directory


def main():
    """
    Transcribe all audio files in a project directory.
    """
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python basic_transcription.py <project_root> [options]")
        print("\nOptions:")
        print("  --device <cuda|cpu>     Device to use (default: cuda)")
        print("  --model <model_name>    Whisper model (default: large-v3)")
        print("  --language <code>       Language code (default: auto-detect)")
        print("\nExample:")
        print("  python basic_transcription.py /data/interviews --language en")
        sys.exit(1)

    project_root = Path(sys.argv[1])

    # Parse optional arguments
    device = "cuda"
    model = "large-v3"
    language = None

    for i in range(2, len(sys.argv), 2):
        if i + 1 < len(sys.argv):
            if sys.argv[i] == "--device":
                device = sys.argv[i + 1]
            elif sys.argv[i] == "--model":
                model = sys.argv[i + 1]
            elif sys.argv[i] == "--language":
                language = sys.argv[i + 1]

    # Validate project root exists
    if not project_root.exists():
        print(f"Error: Project root does not exist: {project_root}")
        sys.exit(1)

    # Check for raw_audio directory
    raw_audio_dir = project_root / "raw_audio"
    if not raw_audio_dir.exists():
        print(f"Error: raw_audio directory not found: {raw_audio_dir}")
        print("\nPlease create the directory and add audio files:")
        print(f"  mkdir -p {raw_audio_dir}")
        sys.exit(1)

    # Count audio files
    audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
    audio_files = [f for f in raw_audio_dir.iterdir() if f.suffix.lower() in audio_extensions]

    if not audio_files:
        print(f"Error: No audio files found in {raw_audio_dir}")
        print(f"\nSupported formats: {', '.join(audio_extensions)}")
        sys.exit(1)

    print("=" * 80)
    print("BASIC TRANSCRIPTION")
    print("=" * 80)
    print(f"\nProject root: {project_root}")
    print(f"Audio files found: {len(audio_files)}")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Language: {language or 'auto-detect'}")
    print()

    # Create configuration
    try:
        config = TranscriptionConfig(
            model=model,
            device=device,
            language=language,
            skip_existing_json=True,  # Skip files already transcribed
        )
    except Exception as e:
        print(f"Error creating configuration: {e}")
        sys.exit(1)

    # Run transcription
    print("Starting transcription...\n")
    try:
        transcripts = transcribe_directory(project_root, config)
    except Exception as e:
        print(f"\nError during transcription: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Display results
    print("\n" + "=" * 80)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 80)
    print(f"\nSuccessfully transcribed {len(transcripts)} file(s)")

    if transcripts:
        print("\nTranscripts:")
        for transcript in transcripts:
            print(f"\n  File: {transcript.file_name}")
            print(f"  Language: {transcript.language}")
            print(f"  Segments: {len(transcript.segments)}")

            # Calculate total duration
            if transcript.segments:
                total_duration = transcript.segments[-1].end
                print(f"  Duration: {total_duration:.1f}s")

                # Show first few words
                first_text = transcript.segments[0].text.strip()
                preview = first_text[:60] + "..." if len(first_text) > 60 else first_text
                print(f"  Preview: {preview}")

        # Show output locations
        print("\nOutputs saved to:")
        print(f"  JSON: {project_root / 'whisper_json'}")
        print(f"  TXT:  {project_root / 'transcripts'}")
        print(f"  SRT:  {project_root / 'transcripts'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
