#!/usr/bin/env python3
"""
Custom Configuration Example

This script demonstrates advanced configuration options for the transcription
API, including:
- Different Whisper model sizes
- CPU vs GPU device selection
- Language hints and auto-detection
- Translation mode (transcribe non-English to English)
- Advanced VAD (Voice Activity Detection) settings
- Beam search parameters
- Skipping already-transcribed files

This is useful when you need fine-grained control over the transcription
process for specific use cases like:
- Low-resource environments (CPU-only, smaller models)
- Multilingual content with known languages
- Real-time vs batch processing trade-offs
- Quality vs speed optimization

Usage:
    python custom_config.py <project_root> --preset <preset_name>
    python custom_config.py <project_root> --custom

Available presets:
    - fast_draft: Fastest transcription, lower quality (tiny model, GPU)
    - balanced: Good balance of speed and quality (medium model, GPU)
    - high_quality: Best quality, slower (large-v3 model, GPU)
    - cpu_fallback: For systems without GPU (medium model, CPU)
    - multilingual: Auto-detect language, translate to English
"""

import sys
from pathlib import Path

from transcription import TranscriptionConfig, transcribe_directory

# Predefined configuration presets
PRESETS = {
    "fast_draft": {
        "model": "tiny",
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 1,  # Faster, less accurate
        "vad_min_silence_ms": 1000,  # Longer silence threshold
        "language": None,
        "description": "Fastest transcription for quick drafts (tiny model, minimal beam search)",
    },
    "balanced": {
        "model": "medium",
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 5,
        "vad_min_silence_ms": 500,
        "language": None,
        "description": "Balanced speed and quality (medium model, standard settings)",
    },
    "high_quality": {
        "model": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 10,  # More thorough search
        "vad_min_silence_ms": 300,  # Detect shorter pauses
        "language": None,
        "description": "Best quality, slower processing (large-v3 model, thorough beam search)",
    },
    "cpu_fallback": {
        "model": "medium",
        "device": "cpu",
        "compute_type": "int8",  # CPU-optimized quantization
        "beam_size": 5,
        "vad_min_silence_ms": 500,
        "language": None,
        "description": "CPU-only processing with optimized quantization",
    },
    "multilingual": {
        "model": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 5,
        "vad_min_silence_ms": 500,
        "language": None,  # Auto-detect
        "task": "translate",  # Translate to English
        "description": "Auto-detect language and translate to English",
    },
    "english_only": {
        "model": "medium.en",  # English-only model (faster)
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 5,
        "vad_min_silence_ms": 500,
        "language": "en",
        "description": "Optimized for English-only content (faster English model)",
    },
}


def print_presets():
    """Display available configuration presets."""
    print("\nAvailable Presets:")
    print("=" * 80)
    for name, config in PRESETS.items():
        print(f"\n{name}:")
        print(f"  {config['description']}")
        print(f"  Model: {config['model']}")
        print(f"  Device: {config['device']}")
        print(f"  Beam size: {config['beam_size']}")
        print(f"  VAD silence: {config['vad_min_silence_ms']}ms")


def interactive_config() -> dict:
    """Interactively build a custom configuration."""
    print("\n" + "=" * 80)
    print("CUSTOM CONFIGURATION BUILDER")
    print("=" * 80)

    config = {}

    # Model selection
    print("\nModel size (affects quality and speed):")
    print("  1. tiny     - Fastest, lowest quality")
    print("  2. base     - Fast, basic quality")
    print("  3. small    - Good balance")
    print("  4. medium   - High quality, slower")
    print("  5. large-v3 - Best quality, slowest (recommended)")
    choice = input("Select model [1-5] (default: 5): ").strip() or "5"
    models = ["tiny", "base", "small", "medium", "large-v3"]
    config["model"] = (
        models[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= 5 else "large-v3"
    )

    # Device selection
    print("\nDevice:")
    print("  1. cuda - GPU acceleration (recommended if available)")
    print("  2. cpu  - CPU only (slower)")
    choice = input("Select device [1-2] (default: 1): ").strip() or "1"
    config["device"] = "cuda" if choice == "1" else "cpu"

    # Compute type (based on device)
    if config["device"] == "cuda":
        config["compute_type"] = "float16"
    else:
        print("\nCPU Compute type:")
        print("  1. int8    - Faster, lower memory")
        print("  2. float32 - Slower, higher quality")
        choice = input("Select compute type [1-2] (default: 1): ").strip() or "1"
        config["compute_type"] = "int8" if choice == "1" else "float32"

    # Language
    print("\nLanguage (leave empty for auto-detection):")
    print("  Common codes: en (English), es (Spanish), fr (French), de (German), zh (Chinese)")
    language = input("Enter language code (default: auto-detect): ").strip()
    config["language"] = language if language else None

    # Task
    if config["language"] and config["language"] != "en":
        print("\nTask:")
        print("  1. transcribe - Keep original language")
        print("  2. translate  - Translate to English")
        choice = input("Select task [1-2] (default: 1): ").strip() or "1"
        config["task"] = "translate" if choice == "2" else "transcribe"

    # Advanced options
    print("\nAdvanced options (press Enter for defaults):")

    beam_size = input("Beam size (1-10, higher=better quality, default: 5): ").strip()
    config["beam_size"] = int(beam_size) if beam_size.isdigit() else 5

    vad_silence = input("VAD min silence in ms (default: 500): ").strip()
    config["vad_min_silence_ms"] = int(vad_silence) if vad_silence.isdigit() else 500

    return config


def main():
    """
    Main entry point for custom configuration example.
    """
    if len(sys.argv) < 2:
        print("Usage: python custom_config.py <project_root> [options]")
        print("\nOptions:")
        print("  --preset <name>    Use a predefined preset")
        print("  --custom           Interactive configuration builder")
        print("  --list-presets     Show available presets")
        print("\nExamples:")
        print("  python custom_config.py /data/audio --preset balanced")
        print("  python custom_config.py /data/audio --custom")
        print_presets()
        sys.exit(1)

    project_root = Path(sys.argv[1])

    # Parse mode
    mode = "balanced"  # default preset
    if len(sys.argv) > 2:
        if sys.argv[2] == "--list-presets":
            print_presets()
            sys.exit(0)
        elif sys.argv[2] == "--preset" and len(sys.argv) > 3:
            mode = sys.argv[3]
            if mode not in PRESETS:
                print(f"Error: Unknown preset '{mode}'")
                print_presets()
                sys.exit(1)
        elif sys.argv[2] == "--custom":
            mode = "custom"
        else:
            print(f"Error: Unknown option '{sys.argv[2]}'")
            sys.exit(1)

    # Validate project root
    if not project_root.exists():
        print(f"Error: Project root does not exist: {project_root}")
        sys.exit(1)

    raw_audio_dir = project_root / "raw_audio"
    if not raw_audio_dir.exists() or not any(raw_audio_dir.iterdir()):
        print(f"Error: No audio files found in {raw_audio_dir}")
        sys.exit(1)

    # Get configuration
    if mode == "custom":
        config_dict = interactive_config()
    else:
        config_dict = PRESETS[mode].copy()
        print(f"\nUsing preset: {mode}")
        print(f"Description: {config_dict.pop('description')}")

    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    for key, value in config_dict.items():
        print(f"  {key}: {value}")

    # Create TranscriptionConfig
    try:
        config = TranscriptionConfig(
            model=config_dict.get("model", "large-v3"),
            device=config_dict.get("device", "cuda"),
            compute_type=config_dict.get("compute_type", "float16"),
            language=config_dict.get("language"),
            task=config_dict.get("task", "transcribe"),
            beam_size=config_dict.get("beam_size", 5),
            vad_min_silence_ms=config_dict.get("vad_min_silence_ms", 500),
            skip_existing_json=True,
        )
    except Exception as e:
        print(f"\nError creating configuration: {e}")
        sys.exit(1)

    # Run transcription
    print("\n" + "=" * 80)
    print("STARTING TRANSCRIPTION")
    print("=" * 80)
    print()

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
    print(f"\nSuccessfully processed {len(transcripts)} file(s)")

    if transcripts:
        print("\nResults:")
        total_duration = 0
        total_segments = 0

        for transcript in transcripts:
            segments = len(transcript.segments)
            duration = transcript.segments[-1].end if transcript.segments else 0
            total_segments += segments
            total_duration += duration

            print(f"\n  {transcript.file_name}")
            print(f"    Language: {transcript.language}")
            print(f"    Duration: {duration:.1f}s")
            print(f"    Segments: {segments}")

        print("\nTotals:")
        print(f"  Duration: {total_duration:.1f}s ({total_duration / 60:.1f} min)")
        print(f"  Segments: {total_segments}")

    print("\nDone!")


if __name__ == "__main__":
    main()
