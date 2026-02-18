"""Dogfood workflow CLI - test real-world use cases before release.

This module provides a complete end-to-end testing workflow:
1. Prepare sample audio (generate synthetic or use cached datasets)
2. Run transcription with diarization
3. Analyze quality metrics
4. Optionally test LLM integration

Usage:
    slower-whisper-dogfood --sample synthetic
    slower-whisper-dogfood --sample mini-diarization
    slower-whisper-dogfood --file raw_audio/custom.wav
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .api import transcribe_directory
from .config import TranscriptionConfig
from .dogfood_utils import (
    compute_diarization_stats,
    get_model_cache_status,
    print_cache_status,
    print_diarization_stats,
    save_dogfood_results,
)
from .samples import copy_sample_to_project, generate_synthetic_2speaker, get_sample_test_files


def main(argv: Sequence[str] | None = None) -> int:
    """Main dogfood CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="slower-whisper-dogfood",
        description="Test slower-whisper with real-world samples before release",
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--sample",
        choices=["synthetic", "mini-diarization"],
        help="Use built-in sample dataset",
    )
    mode_group.add_argument(
        "--file",
        type=Path,
        help="Use custom audio file",
    )

    # Options
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Skip transcription, use existing JSON",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM integration test (even if API key present)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--save-results",
        type=Path,
        help="Save structured results to JSON file (e.g., dogfood_results/run.json)",
    )

    args = parser.parse_args(argv)

    # Resolve audio file
    if args.sample:
        if args.sample == "synthetic":
            print("Preparing synthetic 2-speaker sample...")
            output_dir = args.root / "raw_audio"
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_file = output_dir / "synthetic_2speaker.wav"
            if not audio_file.exists():
                generate_synthetic_2speaker(audio_file)
            else:
                print(f"Using existing: {audio_file}")
        elif args.sample == "mini-diarization":
            print("Preparing mini-diarization sample...")
            project_dir = args.root / "raw_audio"
            try:
                copy_sample_to_project("mini_diarization", project_dir)
                test_files = get_sample_test_files("mini_diarization")
                audio_file = project_dir / test_files[0].name
            except (ValueError, FileNotFoundError) as e:
                print(f"Error: {e}", file=sys.stderr)
                print("\nTo download mini-diarization dataset:")
                print("  slower-whisper samples list")
                print("  # Follow instructions to download manually")
                return 1
    else:
        audio_file = args.file
        if not audio_file.exists():
            print(f"Error: File not found: {audio_file}", file=sys.stderr)
            return 1

    basename = audio_file.stem
    json_file = args.root / "whisper_json" / f"{basename}.json"

    print(f"\n=== Dogfood Workflow: {basename} ===\n")

    # Step 1: Check cache
    print("Step 1: Checking model cache...\n")
    cache_status = get_model_cache_status()
    print_cache_status(cache_status)
    print()

    # Check for required environment variables
    if args.sample and args.sample != "synthetic":
        if not os.getenv("HF_TOKEN"):
            print("Warning: HF_TOKEN not set (required for diarization models)")
            print("Set with: export HF_TOKEN=hf_...")
            print("Get token from: https://huggingface.co/settings/tokens")
            print()
            print("Also ensure you've accepted pyannote model terms:")
            print("  https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("  https://huggingface.co/pyannote/segmentation-3.0")
            print()

    # Step 2: Transcribe (unless skipped)
    if args.skip_transcribe:
        print("Step 2: Skipping transcription (using existing JSON)\n")
        if not json_file.exists():
            print(f"Error: JSON file not found: {json_file}", file=sys.stderr)
            return 1
    else:
        print("Step 2: Transcribing with diarization...\n")
        config = TranscriptionConfig(
            enable_diarization=True,
            min_speakers=2,
            max_speakers=2,
        )
        try:
            transcribe_directory(root=args.root, config=config)
            print("\n✓ Transcription complete\n")
        except Exception as e:
            print(f"Error during transcription: {e}", file=sys.stderr)
            return 1

    # Step 3: Diarization stats
    print("Step 3: Diarization statistics\n")
    try:
        stats = compute_diarization_stats(json_file)
        print_diarization_stats(stats)
    except Exception as e:
        print(f"Error computing stats: {e}", file=sys.stderr)
        return 1

    # Step 4: LLM integration (optional)
    llm_output = None
    if not args.skip_llm and os.getenv("ANTHROPIC_API_KEY"):
        print("\nStep 4: Testing LLM integration...\n")
        llm_example = Path("examples/llm_integration/summarize_with_diarization.py")
        if llm_example.exists():
            # Try to run the example script
            import subprocess

            try:
                result = subprocess.run(
                    ["python", str(llm_example), "--", str(json_file)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                llm_output = result.stdout
                print(llm_output)
                print("\n✓ LLM integration test complete\n")
            except subprocess.CalledProcessError as e:
                print(f"Warning: LLM integration failed: {e}")
                print(e.stderr)
        else:
            print(f"Skipping: LLM example script not found ({llm_example})\n")
    else:
        print("\nStep 4: Skipping LLM integration")
        if args.skip_llm:
            print("  (--skip-llm specified)\n")
        else:
            print("  (ANTHROPIC_API_KEY not set)\n")

    # Save results if requested
    if args.save_results:
        sample_name = args.sample or "custom"
        save_dogfood_results(
            output_path=args.save_results,
            sample_name=sample_name,
            stats=stats,
            cache_status=cache_status,
            llm_output=llm_output,
        )

    # Summary
    print("\n=== Dogfood Complete ===\n")
    print("Results:")
    print(f"  JSON:   {json_file}")
    print(f"  Audio:  {audio_file}")
    if args.save_results:
        print(f"  Report: {args.save_results}")
    print()
    print("Next steps:")
    print("  1. Review stats output above")
    print(f"  2. Inspect JSON: jq . {json_file} | less")
    print("  3. Record findings in docs/DOGFOOD_NOTES.md")
    print()

    return 0
