#!/usr/bin/env python3
"""Evaluate ASR and diarization quality using local reference annotations.

This script evaluates slower-whisper's transcription and speaker diarization
against reference annotations from benchmark datasets. It computes:
- WER (Word Error Rate): Accuracy of transcribed text vs reference
- DER (Diarization Error Rate): Accuracy of speaker assignments vs reference

This is a PURE LOCAL evaluation - no LLM APIs involved.

Supported Datasets:
- AMI Meeting Corpus: Multi-speaker meetings with diarization ground truth
- LibriSpeech: Clean single-speaker read speech for WER-only evaluation

Workflow:
1. Load dataset samples with reference transcripts (and speaker segments for AMI)
2. Run slower-whisper transcription (with diarization for AMI)
3. Compute WER using jiwer (local)
4. Compute DER using pyannote.metrics (local, AMI only)
5. Save metrics JSON for later analysis

Usage:
    # Evaluate AMI dataset (WER + DER)
    uv run python benchmarks/eval_asr_diarization.py --dataset ami --n 2

    # Evaluate LibriSpeech dev-clean (WER only, no diarization)
    uv run python benchmarks/eval_asr_diarization.py --dataset librispeech --n 10

    # Evaluate LibriSpeech test-clean split
    uv run python benchmarks/eval_asr_diarization.py --dataset librispeech \\
        --split test-clean --n 50

    # Use specific model and device
    uv run python benchmarks/eval_asr_diarization.py --dataset librispeech \\
        --model base --device cpu --n 5

Requirements:
    - Base: uv sync --extra full
    - For AMI diarization: uv sync --extra diarization
    - HF_TOKEN environment variable (for pyannote.audio models, AMI only)
    - jiwer package: uv pip install jiwer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from slower_whisper.pipeline import TranscriptionConfig, transcribe_file
from slower_whisper.pipeline.benchmarks import EvalSample, iter_ami_meetings, iter_librispeech

try:
    from pyannote.core import Annotation
except ImportError:
    Annotation = None  # type: ignore


def check_dependencies() -> None:
    """Check that required packages are installed."""
    missing = []

    try:
        import jiwer  # noqa: F401
    except ImportError:
        missing.append("jiwer")

    try:
        from pyannote.core import Annotation  # noqa: F401
        from pyannote.metrics.diarization import DiarizationErrorRate  # noqa: F401
    except ImportError:
        missing.append("pyannote.audio (and dependencies)")

    if missing:
        print(
            f"Error: Missing required packages: {', '.join(missing)}",
            file=sys.stderr,
        )
        print("\nInstall with:")
        print("  uv sync --extra full --extra diarization")
        print("  uv pip install jiwer")
        sys.exit(1)


def check_hf_token() -> None:
    """Check that HF_TOKEN is set for pyannote.audio models."""
    if not os.getenv("HF_TOKEN"):
        print(
            "Warning: HF_TOKEN environment variable not set.",
            file=sys.stderr,
        )
        print(
            "Diarization may fail without it. Get a token from:",
            file=sys.stderr,
        )
        print("  https://huggingface.co/settings/tokens", file=sys.stderr)
        print("Then set it: export HF_TOKEN=hf_...", file=sys.stderr)
        print()


def build_reference_annotation(speakers: list[dict[str, Any]]) -> Annotation:
    """Build pyannote Annotation from reference speaker segments.

    Args:
        speakers: List of speaker dicts with "id" and "segments" fields
            Example: [{"id": "SPEAKER_00", "segments": [{"start": 0.0, "end": 3.0}]}]

    Returns:
        pyannote.core.Annotation object
    """
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for speaker in speakers:
        speaker_id = speaker["id"]
        for seg in speaker["segments"]:
            ann[Segment(seg["start"], seg["end"])] = speaker_id
    return ann


def build_hypothesis_annotation(transcript: Any) -> Annotation:
    """Build pyannote Annotation from slower-whisper transcript.

    Args:
        transcript: Transcript object from slower-whisper

    Returns:
        pyannote.core.Annotation object
    """
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for seg in transcript.segments:
        if seg.speaker is None:
            # Unlabeled segment - assign to generic "UNKNOWN" speaker
            speaker_id = "UNKNOWN"
        else:
            speaker_id = seg.speaker["id"]
        ann[Segment(seg.start, seg.end)] = speaker_id
    return ann


def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    """Compute Word Error Rate between reference and hypothesis.

    Applies standard ASR normalization:
    - Convert to uppercase
    - Remove punctuation
    - Collapse whitespace

    This ensures fair comparison between different transcript formats
    (e.g., LibriSpeech all-caps vs Whisper normal-case).

    Args:
        reference_text: Ground truth transcript
        hypothesis_text: Predicted transcript from ASR

    Returns:
        WER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    import jiwer

    # Standard ASR normalization pipeline
    transformation = jiwer.Compose(
        [
            jiwer.ToUpperCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )

    ref = transformation(reference_text)
    hyp = transformation(hypothesis_text)

    if not ref:
        # Empty reference - if hypothesis is also empty, WER=0, else WER=1
        return 0.0 if not hyp else 1.0

    return jiwer.wer(ref, hyp)


def compute_der(reference_annotation: Any, hypothesis_annotation: Any) -> float:
    """Compute Diarization Error Rate.

    Args:
        reference_annotation: pyannote Annotation with ground truth speakers
        hypothesis_annotation: pyannote Annotation with predicted speakers

    Returns:
        DER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate()
    der = metric(reference_annotation, hypothesis_annotation)
    return float(der)


def run_transcription_on_sample(
    sample: EvalSample,
    config: TranscriptionConfig,
) -> Any:
    """Run slower-whisper transcription on a single sample.

    Args:
        sample: EvalSample with audio_path
        config: TranscriptionConfig with diarization enabled

    Returns:
        Transcript object
    """
    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Run transcription
        transcript = transcribe_file(
            audio_path=sample.audio_path,
            root=root,
            config=config,
        )

    return transcript


def evaluate_sample(
    sample: EvalSample,
    config: TranscriptionConfig,
) -> dict[str, Any]:
    """Evaluate a single sample: run ASR+diarization and compute metrics.

    Args:
        sample: EvalSample with reference data
        config: TranscriptionConfig

    Returns:
        Dict with WER, DER, and metadata
    """
    print(f"  Evaluating {sample.id}...")

    # Initialize scores (will be computed after transcription if reference data exists)
    wer_score = None
    der_score = None

    # Run transcription
    try:
        transcript = run_transcription_on_sample(sample, config)
    except Exception as e:
        print(f"    Error during transcription: {e}")
        return {
            "id": sample.id,
            "error": str(e),
            "WER": None,
            "DER": None,
        }

    # Compute WER
    if sample.reference_transcript:
        hyp_text = " ".join(s.text for s in transcript.segments).strip()
        ref_text = sample.reference_transcript.strip()
        wer_score = compute_wer(ref_text, hyp_text)
        print(f"    WER: {wer_score:.3f}")

    # Compute DER
    if sample.reference_speakers:
        try:
            ref_ann = build_reference_annotation(sample.reference_speakers)
            hyp_ann = build_hypothesis_annotation(transcript)
            der_score = compute_der(ref_ann, hyp_ann)
            print(f"    DER: {der_score:.3f}")
        except Exception as e:
            print(f"    Error computing DER: {e}")
            der_score = None

    return {
        "id": sample.id,
        "WER": float(wer_score) if wer_score is not None else None,
        "DER": float(der_score) if der_score is not None else None,
        "num_segments": len(transcript.segments),
        "num_speakers_detected": len(
            {s.speaker["id"] for s in transcript.segments if s.speaker is not None}
        )
        if any(s.speaker for s in transcript.segments)
        else 0,
    }


def evaluate_dataset(
    dataset: str,
    n: int | None,
    config: TranscriptionConfig,
    split: str | None = None,
) -> dict[str, Any]:
    """Evaluate multiple samples from a dataset.

    Args:
        dataset: Dataset name (e.g., "ami", "librispeech")
        n: Number of samples to evaluate (None = all)
        config: TranscriptionConfig
        split: Dataset split (used for librispeech: "dev-clean", "test-clean", etc.)

    Returns:
        Dict with aggregate metrics and per-sample results
    """
    print(f"Loading {dataset} samples (limit={n})...")

    if dataset == "ami":
        samples = list(iter_ami_meetings(limit=n))
    elif dataset == "librispeech":
        split = split or "dev-clean"
        print(f"  Using LibriSpeech split: {split}")
        samples = list(iter_librispeech(split=split, limit=n))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not samples:
        print(f"No samples found for dataset '{dataset}'")
        return {
            "dataset": dataset,
            "num_samples": 0,
            "samples": [],
            "aggregate": {},
        }

    print(f"Evaluating {len(samples)} samples...")
    results = []

    for sample in samples:
        result = evaluate_sample(sample, config)
        results.append(result)

    # Compute aggregates
    valid_wer = [r["WER"] for r in results if r.get("WER") is not None]
    valid_der = [r["DER"] for r in results if r.get("DER") is not None]

    aggregate = {
        "num_samples": len(results),
        "num_valid_wer": len(valid_wer),
        "num_valid_der": len(valid_der),
        "avg_WER": sum(valid_wer) / len(valid_wer) if valid_wer else None,
        "avg_DER": sum(valid_der) / len(valid_der) if valid_der else None,
        "min_WER": min(valid_wer) if valid_wer else None,
        "max_WER": max(valid_wer) if valid_wer else None,
        "min_DER": min(valid_der) if valid_der else None,
        "max_DER": max(valid_der) if valid_der else None,
    }

    result_dict = {
        "dataset": dataset,
        "config": {
            "model": config.model,
            "device": config.device,
            "enable_diarization": config.enable_diarization,
            "min_speakers": config.min_speakers,
            "max_speakers": config.max_speakers,
        },
        "timestamp": datetime.now().isoformat(),
        "samples": results,
        "aggregate": aggregate,
    }

    # Add split info if provided
    if split:
        result_dict["split"] = split

    return result_dict


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate ASR and diarization quality (local-only, no LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="ami",
        choices=["ami", "librispeech"],
        help="Dataset to evaluate (default: ami)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split (for librispeech: dev-clean, test-clean, dev-other, test-other)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: benchmarks/results/asr_diar_{dataset}.json)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model to use (default: large-v3)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers (diarization hint)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers (diarization hint)",
    )

    args = parser.parse_args(argv)

    # Set default output path based on dataset
    if args.output is None:
        if args.dataset == "librispeech":
            split_suffix = f"_{args.split}" if args.split else "_dev-clean"
            args.output = Path(f"benchmarks/results/asr_librispeech{split_suffix}.json")
        else:
            args.output = Path(f"benchmarks/results/asr_diar_{args.dataset}.json")

    # Check dependencies
    check_dependencies()

    # Only check HF token if diarization is needed (not required for LibriSpeech WER-only eval)
    if args.dataset == "ami" or args.min_speakers or args.max_speakers:
        check_hf_token()

    # Create config
    # Auto-select compute_type based on device
    if args.device == "cpu":
        compute_type = "int8"  # CPU doesn't support float16
    else:
        compute_type = "float16"

    # Enable diarization only if dataset requires it (AMI) or explicitly requested
    enable_diarization = (
        args.dataset == "ami" or args.min_speakers is not None or args.max_speakers is not None
    )

    config = TranscriptionConfig(
        model=args.model,
        device=args.device,
        compute_type=compute_type,
        enable_diarization=enable_diarization,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    print("Configuration:")
    print(f"  Model: {config.model}")
    print(f"  Device: {config.device}")
    print(f"  Compute type: {config.compute_type}")
    print(f"  Diarization: {config.enable_diarization}")
    if config.min_speakers:
        print(f"  Min speakers: {config.min_speakers}")
    if config.max_speakers:
        print(f"  Max speakers: {config.max_speakers}")
    print()

    # Run evaluation
    results = evaluate_dataset(
        dataset=args.dataset,
        n=args.n,
        config=config,
        split=args.split,
    )

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Dataset: {results['dataset']}")
    print(f"Samples evaluated: {results['aggregate']['num_samples']}")
    if results["aggregate"].get("avg_WER") is not None:
        print(f"Average WER: {results['aggregate']['avg_WER']:.3f}")
        print(
            f"WER range: [{results['aggregate']['min_WER']:.3f}, {results['aggregate']['max_WER']:.3f}]"
        )
    if results["aggregate"].get("avg_DER") is not None:
        print(f"Average DER: {results['aggregate']['avg_DER']:.3f}")
        print(
            f"DER range: [{results['aggregate']['min_DER']:.3f}, {results['aggregate']['max_DER']:.3f}]"
        )
    print(f"\nResults written to: {args.output}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
