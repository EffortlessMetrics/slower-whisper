#!/usr/bin/env python3
"""
Two-Stage Enrichment Workflow Example

This script demonstrates the complete two-stage pipeline:
1. Stage 1: Transcribe audio files to text with Whisper
2. Stage 2: Enrich transcripts with audio-derived features (prosody & emotion)

The enrichment stage adds:
- Prosody features (pitch, energy, speech rate, pauses)
- Dimensional emotion (valence, arousal)
- Categorical emotion (happy, sad, angry, neutral, etc.)

This two-stage approach allows you to:
- Transcribe first, then enrich later (useful for large batches)
- Re-enrich existing transcripts with different settings
- Skip enrichment for files where you only need text

Usage:
    # Run both stages:
    python enrichment_workflow.py /path/to/project --transcribe --enrich

    # Only transcribe (Stage 1):
    python enrichment_workflow.py /path/to/project --transcribe

    # Only enrich existing transcripts (Stage 2):
    python enrichment_workflow.py /path/to/project --enrich

    # Full workflow with custom options:
    python enrichment_workflow.py /path/to/project --transcribe --enrich \\
        --model large-v3 --language en --enable-categorical
"""

import sys
from pathlib import Path

from slower_whisper.pipeline import (
    EnrichmentConfig,
    TranscriptionConfig,
    enrich_directory,
    transcribe_directory,
)


def print_stage_banner(stage_name: str):
    """Print a formatted stage banner."""
    print("\n" + "=" * 80)
    print(f"STAGE: {stage_name}")
    print("=" * 80)


def print_transcript_summary(transcripts: list, stage: str):
    """Print a summary of transcription results."""
    print(f"\n{stage} Summary:")
    print("-" * 80)

    if not transcripts:
        print("No transcripts processed.")
        return

    total_segments = 0
    total_duration = 0.0
    languages = set()

    for transcript in transcripts:
        segments = len(transcript.segments)
        duration = transcript.segments[-1].end if transcript.segments else 0
        total_segments += segments
        total_duration += duration
        languages.add(transcript.language)

    print(f"Files processed: {len(transcripts)}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration / 60:.1f} min)")
    print(f"Total segments: {total_segments}")
    print(f"Average segments/file: {total_segments / len(transcripts):.1f}")
    print(f"Languages detected: {', '.join(sorted(languages))}")


def print_enrichment_summary(transcripts: list):
    """Print a summary of enrichment results."""
    print("\nEnrichment Summary:")
    print("-" * 80)

    if not transcripts:
        print("No transcripts enriched.")
        return

    enriched_count = 0
    feature_counts = {
        "prosody": 0,
        "emotion": 0,
        "categorical_emotion": 0,
    }

    for transcript in transcripts:
        has_enrichment = False
        for segment in transcript.segments:
            if segment.audio_state:
                has_enrichment = True

                # Count feature types
                if "pitch" in segment.audio_state or "energy" in segment.audio_state:
                    feature_counts["prosody"] += 1

                if "emotion" in segment.audio_state:
                    feature_counts["emotion"] += 1
                    if "categorical" in segment.audio_state["emotion"]:
                        feature_counts["categorical_emotion"] += 1

        if has_enrichment:
            enriched_count += 1

    print(f"Files enriched: {enriched_count}/{len(transcripts)}")
    print(f"Segments with prosody: {feature_counts['prosody']}")
    print(f"Segments with emotion: {feature_counts['emotion']}")
    if feature_counts["categorical_emotion"] > 0:
        print(f"Segments with categorical emotion: {feature_counts['categorical_emotion']}")


def show_sample_enrichment(transcripts: list):
    """Display a sample of enriched segments."""
    print("\nSample Enriched Segments:")
    print("-" * 80)

    shown = 0
    max_samples = 3

    for transcript in transcripts:
        for segment in transcript.segments:
            if segment.audio_state and shown < max_samples:
                print(f"\n[{segment.start:.1f}s] {segment.text[:60]}...")

                # Show prosody features
                if "pitch" in segment.audio_state:
                    pitch = segment.audio_state["pitch"]
                    print(
                        f"  Pitch: {pitch.get('level', 'unknown')} ({pitch.get('mean_hz', 0):.1f} Hz)"
                    )

                if "energy" in segment.audio_state:
                    energy = segment.audio_state["energy"]
                    print(
                        f"  Energy: {energy.get('level', 'unknown')} ({energy.get('db_rms', 0):.1f} dB)"
                    )

                if "rate" in segment.audio_state:
                    rate = segment.audio_state["rate"]
                    print(
                        f"  Speech Rate: {rate.get('level', 'unknown')} ({rate.get('syllables_per_sec', 0):.2f} syl/sec)"
                    )

                # Show emotion
                if "emotion" in segment.audio_state:
                    emotion = segment.audio_state["emotion"]
                    if "categorical" in emotion:
                        cat = emotion["categorical"]
                        print(
                            f"  Emotion: {cat.get('primary', 'unknown')} ({cat.get('confidence', 0):.1%})"
                        )
                    elif "dimensional" in emotion:
                        dim = emotion["dimensional"]
                        valence = dim.get("valence", {}).get("score", 0.5)
                        arousal = dim.get("arousal", {}).get("score", 0.5)
                        print(f"  Emotion: valence={valence:.2f}, arousal={arousal:.2f}")

                shown += 1

        if shown >= max_samples:
            break


def main():
    """
    Main entry point for two-stage enrichment workflow.
    """
    if len(sys.argv) < 2:
        print("Usage: python enrichment_workflow.py <project_root> [options]")
        print("\nStage Selection (at least one required):")
        print("  --transcribe              Run Stage 1 (transcription)")
        print("  --enrich                  Run Stage 2 (enrichment)")
        print("\nTranscription Options (Stage 1):")
        print("  --model <name>            Whisper model (default: large-v3)")
        print("  --device <cuda|cpu>       Device for transcription (default: cuda)")
        print("  --language <code>         Language hint (default: auto-detect)")
        print("\nEnrichment Options (Stage 2):")
        print("  --enrich-device <cuda|cpu>  Device for enrichment (default: cpu)")
        print("  --enable-categorical      Enable categorical emotion detection")
        print("  --skip-prosody            Disable prosody features")
        print("  --skip-emotion            Disable emotion features")
        print("\nExamples:")
        print("  # Full workflow:")
        print("  python enrichment_workflow.py /data/audio --transcribe --enrich")
        print("\n  # Transcribe only:")
        print("  python enrichment_workflow.py /data/audio --transcribe --language en")
        print("\n  # Enrich existing transcripts:")
        print("  python enrichment_workflow.py /data/audio --enrich --enable-categorical")
        sys.exit(1)

    project_root = Path(sys.argv[1])

    # Parse options
    run_transcribe = "--transcribe" in sys.argv
    run_enrich = "--enrich" in sys.argv

    if not run_transcribe and not run_enrich:
        print("Error: Must specify at least one stage (--transcribe or --enrich)")
        sys.exit(1)

    # Transcription options
    model = "large-v3"
    device = "cuda"
    language = None

    # Enrichment options
    enrich_device = "cpu"
    enable_categorical = "--enable-categorical" in sys.argv
    enable_prosody = "--skip-prosody" not in sys.argv
    enable_emotion = "--skip-emotion" not in sys.argv

    # Parse command-line arguments
    i = 2
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
        elif sys.argv[i] == "--enrich-device" and i + 1 < len(sys.argv):
            enrich_device = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    # Validate project root
    if not project_root.exists():
        print(f"Error: Project root does not exist: {project_root}")
        sys.exit(1)

    print("=" * 80)
    print("TWO-STAGE ENRICHMENT WORKFLOW")
    print("=" * 80)
    print(f"\nProject root: {project_root}")
    print("Stages to run: ", end="")
    stages = []
    if run_transcribe:
        stages.append("Transcription")
    if run_enrich:
        stages.append("Enrichment")
    print(" + ".join(stages))

    # Stage 1: Transcription
    transcripts = []
    if run_transcribe:
        print_stage_banner("TRANSCRIPTION (Stage 1)")

        # Check for raw audio
        raw_audio_dir = project_root / "raw_audio"
        if not raw_audio_dir.exists():
            print(f"Error: raw_audio directory not found: {raw_audio_dir}")
            sys.exit(1)

        print("\nConfiguration:")
        print(f"  Model: {model}")
        print(f"  Device: {device}")
        print(f"  Language: {language or 'auto-detect'}")

        try:
            config = TranscriptionConfig(
                model=model,
                device=device,
                language=language,
                skip_existing_json=True,
            )

            print("\nTranscribing...")
            transcripts = transcribe_directory(project_root, config)

            print_transcript_summary(transcripts, "Transcription")

        except Exception as e:
            print(f"\nError during transcription: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # Stage 2: Enrichment
    if run_enrich:
        print_stage_banner("ENRICHMENT (Stage 2)")

        # Check for existing transcripts
        json_dir = project_root / "whisper_json"
        if not json_dir.exists():
            print("Error: No transcripts found. Run --transcribe first.")
            print(f"Expected directory: {json_dir}")
            sys.exit(1)

        json_files = list(json_dir.glob("*.json"))
        if not json_files:
            print(f"Error: No JSON transcripts found in {json_dir}")
            sys.exit(1)

        print("\nConfiguration:")
        print(f"  Device: {enrich_device}")
        print(f"  Prosody features: {enable_prosody}")
        print(f"  Emotion features: {enable_emotion}")
        print(f"  Categorical emotion: {enable_categorical}")

        try:
            config = EnrichmentConfig(
                device=enrich_device,
                enable_prosody=enable_prosody,
                enable_emotion=enable_emotion,
                enable_categorical_emotion=enable_categorical,
                skip_existing=True,
            )

            print(f"\nEnriching {len(json_files)} transcript(s)...")
            enriched = enrich_directory(project_root, config)

            print_enrichment_summary(enriched)
            show_sample_enrichment(enriched)

        except Exception as e:
            print(f"\nError during enrichment: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # Final summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)

    print("\nOutput locations:")
    print(f"  JSON transcripts: {project_root / 'whisper_json'}")
    print(f"  Text transcripts: {project_root / 'transcripts'}")

    if run_enrich:
        print("\nEnriched transcripts contain:")
        if enable_prosody:
            print("  - Prosody: pitch, energy, speech rate, pauses")
        if enable_emotion:
            print("  - Emotion: dimensional (valence, arousal)")
        if enable_categorical:
            print("  - Categorical emotion: happy, sad, angry, etc.")

    print("\nNext steps:")
    print("  - Load enriched transcripts with: load_transcript('path/to/file.json')")
    print("  - Query features with the analysis tools in examples/")
    print("  - Export to CSV/TXT for further analysis")

    print("\nDone!")


if __name__ == "__main__":
    main()
