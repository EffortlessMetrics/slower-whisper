#!/usr/bin/env python3
"""Evaluate emotion recognition accuracy using IEMOCAP dataset.

This script evaluates slower-whisper's emotion extraction (categorical and dimensional)
against IEMOCAP ground truth annotations.

Workflow:
1. Load IEMOCAP clips with reference emotion labels
2. Extract emotion features using slower-whisper's emotion models
3. Compare predictions to ground truth using:
   - Categorical: Accuracy, F1-score, confusion matrix
   - Dimensional: MAE, RMSE, correlation (valence/arousal/dominance)
4. Save detailed results for analysis

Usage:
    # Evaluate categorical emotions on Session 5 (test set)
    python benchmarks/eval_emotion.py --session Session5 --mode categorical

    # Evaluate dimensional emotions on all sessions
    python benchmarks/eval_emotion.py --mode dimensional

    # Full evaluation (both categorical and dimensional)
    python benchmarks/eval_emotion.py --mode both

    # Quick sanity check (10 samples)
    python benchmarks/eval_emotion.py --limit 10 --mode categorical

Requirements:
    - IEMOCAP dataset set up (see docs/IEMOCAP_SETUP.md)
    - uv sync --extra emotion (for emotion models)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transcription.benchmarks import iter_iemocap_clips
from transcription.emotion import (
    EMOTION_AVAILABLE,
    extract_emotion_categorical,
    extract_emotion_dimensional,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# IEMOCAP emotion codes to full labels
IEMOCAP_EMOTION_MAP = {
    "ang": "angry",
    "hap": "happy",
    "sad": "sad",
    "neu": "neutral",
    "exc": "excited",
    "fru": "frustrated",
    "fea": "fearful",
    "sur": "surprised",
    "dis": "disgusted",
    "oth": "other",
}

# IEMOCAP to model categorical mapping (for models with limited emotion set)
IEMOCAP_TO_MODEL_CATEGORICAL = {
    "ang": "angry",
    "hap": "happy",
    "sad": "sad",
    "neu": "neutral",
    "exc": "happy",  # Excitement → happy (high arousal positive)
    "fru": "angry",  # Frustration → angry (high arousal negative)
    "fea": "fear",
    "sur": "surprise",
    "dis": "disgust",
    "oth": None,  # Skip "other" category
}


def check_emotion_available() -> None:
    """Check if emotion dependencies are installed."""
    if not EMOTION_AVAILABLE:
        logger.error("Emotion recognition dependencies not available")
        print("Error: Emotion recognition requires torch and transformers", file=sys.stderr)
        print("Install with: uv sync --extra emotion", file=sys.stderr)
        sys.exit(1)


def parse_iemocap_dimensional(annotation_line: str) -> dict[str, float] | None:
    """Parse dimensional emotion ratings from IEMOCAP annotation line.

    IEMOCAP format: [6.2901 - 8.2357]	Ses01F_impro01_F000	ang	[2.5000, 2.5000, 2.5000]

    Args:
        annotation_line: Raw annotation line from EmoEvaluation file

    Returns:
        Dict with valence, arousal, dominance (0-1 scale) or None if parsing fails
    """
    try:
        # Extract dimensional ratings [val, act, dom] from line
        parts = annotation_line.strip().split("\t")
        if len(parts) < 4:
            return None

        # Parse bracketed values: [2.5, 2.5, 2.5]
        dim_str = parts[3].strip("[]")
        val, act, dom = map(float, dim_str.split(","))

        # Convert from 1-5 scale to 0-1 scale
        return {
            "valence": (val - 1.0) / 4.0,
            "arousal": (act - 1.0) / 4.0,
            "dominance": (dom - 1.0) / 4.0,
        }
    except Exception as e:
        logger.warning(f"Failed to parse dimensional ratings: {e}")
        return None


def load_audio_segment(audio_path: Path) -> tuple[np.ndarray, int] | None:
    """Load audio segment for emotion extraction.

    Args:
        audio_path: Path to WAV file

    Returns:
        Tuple of (audio_samples, sample_rate) or None if loading fails
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(str(audio_path))
        return audio, sr
    except ImportError:
        logger.error("soundfile package required for audio loading")
        print("Error: Install soundfile: uv sync --extra full", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.warning(f"Failed to load audio {audio_path}: {e}")
        return None


def evaluate_categorical(
    session: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate categorical emotion recognition accuracy.

    Args:
        session: Specific IEMOCAP session to evaluate (or None for all)
        limit: Maximum number of clips to evaluate

    Returns:
        Dict with evaluation results:
        {
            "accuracy": float,
            "f1_weighted": float,
            "f1_macro": float,
            "confusion_matrix": dict,
            "per_class_scores": dict,
            "predictions": list of prediction dicts
        }
    """
    check_emotion_available()

    predictions = []
    count = 0

    logger.info(f"Starting categorical emotion evaluation (session={session}, limit={limit})")

    for sample in iter_iemocap_clips(session=session, limit=limit):
        if not sample.reference_emotions or not sample.reference_emotions[0]:
            logger.debug(f"Skipping {sample.id}: no reference emotion")
            continue

        # Get ground truth
        ref_emotion_code = sample.reference_emotions[0]
        ref_emotion = IEMOCAP_EMOTION_MAP.get(ref_emotion_code, ref_emotion_code)

        # Map to model's emotion set
        mapped_emotion = IEMOCAP_TO_MODEL_CATEGORICAL.get(ref_emotion_code)
        if mapped_emotion is None:
            logger.debug(f"Skipping {sample.id}: emotion '{ref_emotion}' not in model set")
            continue

        # Load audio
        audio_data = load_audio_segment(sample.audio_path)
        if audio_data is None:
            continue

        audio, sr = audio_data

        # Extract emotion
        try:
            result = extract_emotion_categorical(audio, sr)
            predicted_emotion = result["categorical"]["primary"]
            confidence = result["categorical"]["confidence"]
            all_scores = result["categorical"]["all_scores"]

            predictions.append(
                {
                    "clip_id": sample.id,
                    "reference": mapped_emotion,
                    "reference_original": ref_emotion,
                    "predicted": predicted_emotion,
                    "confidence": confidence,
                    "all_scores": all_scores,
                    "correct": predicted_emotion == mapped_emotion,
                }
            )

            count += 1
            if count % 100 == 0:
                logger.info(f"Processed {count} clips...")

        except Exception as e:
            logger.warning(f"Emotion extraction failed for {sample.id}: {e}")
            continue

    if not predictions:
        logger.error("No predictions collected - check IEMOCAP setup and audio files")
        return {}

    # Compute metrics
    correct = sum(1 for p in predictions if p["correct"])
    accuracy = correct / len(predictions)

    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for p in predictions:
        confusion[p["reference"]][p["predicted"]] += 1

    # Per-class metrics
    per_class = {}
    all_emotions = {p["reference"] for p in predictions} | {p["predicted"] for p in predictions}

    for emotion in all_emotions:
        tp = confusion[emotion][emotion]
        fp = sum(confusion[ref][emotion] for ref in confusion if ref != emotion)
        fn = sum(confusion[emotion][pred] for pred in confusion[emotion] if pred != emotion)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[emotion] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(1 for p in predictions if p["reference"] == emotion),
        }

    # Weighted and macro F1
    f1_weighted = sum(per_class[e]["f1"] * per_class[e]["support"] for e in per_class) / len(
        predictions
    )
    f1_macro = sum(per_class[e]["f1"] for e in per_class) / len(per_class)

    logger.info(
        f"Categorical evaluation complete: {len(predictions)} clips, {accuracy:.3f} accuracy"
    )

    return {
        "accuracy": round(accuracy, 3),
        "f1_weighted": round(f1_weighted, 3),
        "f1_macro": round(f1_macro, 3),
        "confusion_matrix": dict(confusion),
        "per_class_scores": per_class,
        "predictions": predictions,
        "n_samples": len(predictions),
    }


def evaluate_dimensional(
    session: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate dimensional emotion recognition (valence, arousal, dominance).

    Args:
        session: Specific IEMOCAP session to evaluate (or None for all)
        limit: Maximum number of clips to evaluate

    Returns:
        Dict with evaluation results:
        {
            "valence": {"mae": float, "rmse": float, "correlation": float},
            "arousal": {"mae": float, "rmse": float, "correlation": float},
            "dominance": {"mae": float, "rmse": float, "correlation": float},
            "predictions": list of prediction dicts
        }
    """
    check_emotion_available()

    predictions = []
    count = 0

    logger.info(f"Starting dimensional emotion evaluation (session={session}, limit={limit})")

    for sample in iter_iemocap_clips(session=session, limit=limit):
        # IEMOCAP dimensional ratings are in metadata, need to parse from annotation file
        # For now, we'll extract from the annotation file directly
        if not sample.metadata or "dialog_id" not in sample.metadata:
            continue

        # Construct annotation file path
        session_name = sample.metadata.get("session", "Session1")
        dialog_id = sample.metadata["dialog_id"]

        from transcription.benchmarks import get_benchmarks_root

        eval_file = (
            get_benchmarks_root()
            / "iemocap"
            / session_name
            / "dialog"
            / "EmoEvaluation"
            / f"{dialog_id}.txt"
        )

        if not eval_file.exists():
            logger.debug(f"Annotation file not found: {eval_file}")
            continue

        # Parse dimensional ratings for this clip
        ref_dimensional = None
        try:
            with open(eval_file) as f:
                for line in f:
                    if sample.id in line:
                        ref_dimensional = parse_iemocap_dimensional(line)
                        break
        except Exception as e:
            logger.warning(f"Failed to read annotation file {eval_file}: {e}")
            continue

        if ref_dimensional is None:
            logger.debug(f"No dimensional ratings found for {sample.id}")
            continue

        # Load audio
        audio_data = load_audio_segment(sample.audio_path)
        if audio_data is None:
            continue

        audio, sr = audio_data

        # Extract emotion
        try:
            result = extract_emotion_dimensional(audio, sr)

            predictions.append(
                {
                    "clip_id": sample.id,
                    "reference": ref_dimensional,
                    "predicted": {
                        "valence": result["valence"]["score"],
                        "arousal": result["arousal"]["score"],
                        "dominance": result["dominance"]["score"],
                    },
                    "predicted_levels": {
                        "valence": result["valence"]["level"],
                        "arousal": result["arousal"]["level"],
                        "dominance": result["dominance"]["level"],
                    },
                }
            )

            count += 1
            if count % 100 == 0:
                logger.info(f"Processed {count} clips...")

        except Exception as e:
            logger.warning(f"Emotion extraction failed for {sample.id}: {e}")
            continue

    if not predictions:
        logger.error("No predictions collected - check IEMOCAP setup and annotation files")
        return {}

    # Compute metrics for each dimension
    dimensions = ["valence", "arousal", "dominance"]
    metrics = {}

    for dim in dimensions:
        ref_values = np.array([p["reference"][dim] for p in predictions])
        pred_values = np.array([p["predicted"][dim] for p in predictions])

        # Mean Absolute Error
        mae = np.mean(np.abs(ref_values - pred_values))

        # Root Mean Square Error
        rmse = np.sqrt(np.mean((ref_values - pred_values) ** 2))

        # Pearson correlation
        correlation = np.corrcoef(ref_values, pred_values)[0, 1]

        metrics[dim] = {
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "correlation": round(float(correlation), 3),
        }

        logger.info(f"{dim.capitalize()}: MAE={mae:.3f}, RMSE={rmse:.3f}, r={correlation:.3f}")

    logger.info(f"Dimensional evaluation complete: {len(predictions)} clips")

    return {
        "valence": metrics["valence"],
        "arousal": metrics["arousal"],
        "dominance": metrics["dominance"],
        "predictions": predictions,
        "n_samples": len(predictions),
    }


def print_categorical_report(results: dict[str, Any]) -> None:
    """Print categorical evaluation report to console."""
    print()
    print("=" * 70)
    print("CATEGORICAL EMOTION EVALUATION RESULTS")
    print("=" * 70)
    print(f"Samples evaluated: {results['n_samples']}")
    print(f"Overall accuracy: {results['accuracy']:.1%}")
    print(f"F1-score (weighted): {results['f1_weighted']:.3f}")
    print(f"F1-score (macro): {results['f1_macro']:.3f}")
    print()

    print("Per-class performance:")
    print(f"{'Emotion':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 70)
    for emotion, scores in sorted(results["per_class_scores"].items()):
        print(
            f"{emotion:<15} "
            f"{scores['precision']:<12.3f} "
            f"{scores['recall']:<12.3f} "
            f"{scores['f1']:<12.3f} "
            f"{scores['support']:<10}"
        )
    print()

    # Show most common confusions
    print("Most common confusions (top 5):")
    confusions = []
    for ref, pred_dict in results["confusion_matrix"].items():
        for pred, count in pred_dict.items():
            if ref != pred and count > 0:
                confusions.append((count, ref, pred))

    for count, ref, pred in sorted(confusions, reverse=True)[:5]:
        print(f"  {ref} → {pred}: {count} times")
    print()


def print_dimensional_report(results: dict[str, Any]) -> None:
    """Print dimensional evaluation report to console."""
    print()
    print("=" * 70)
    print("DIMENSIONAL EMOTION EVALUATION RESULTS")
    print("=" * 70)
    print(f"Samples evaluated: {results['n_samples']}")
    print()

    print(f"{'Dimension':<15} {'MAE':<12} {'RMSE':<12} {'Correlation':<15}")
    print("-" * 70)
    for dim in ["valence", "arousal", "dominance"]:
        metrics = results[dim]
        print(
            f"{dim.capitalize():<15} "
            f"{metrics['mae']:<12.3f} "
            f"{metrics['rmse']:<12.3f} "
            f"{metrics['correlation']:<15.3f}"
        )
    print()

    print("Interpretation:")
    print("  MAE/RMSE: Lower is better (0.0 = perfect, range 0-1)")
    print("  Correlation: Higher is better (-1 to 1, 1 = perfect)")
    print()


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Evaluate emotion recognition using IEMOCAP dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate categorical emotions on Session 5
  python benchmarks/eval_emotion.py --session Session5 --mode categorical

  # Evaluate dimensional emotions on all sessions
  python benchmarks/eval_emotion.py --mode dimensional

  # Quick sanity check (10 samples)
  python benchmarks/eval_emotion.py --limit 10 --mode categorical

  # Full evaluation (both modes)
  python benchmarks/eval_emotion.py --mode both
""",
    )

    parser.add_argument(
        "--session",
        choices=["Session1", "Session2", "Session3", "Session4", "Session5"],
        default=None,
        help="Specific session to evaluate (default: all sessions)",
    )
    parser.add_argument(
        "--mode",
        choices=["categorical", "dimensional", "both"],
        default="categorical",
        help="Evaluation mode (default: categorical)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of clips to evaluate (for testing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for results JSON (default: benchmarks/results/iemocap_emotion_<date>.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        results_dir = Path("benchmarks/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"iemocap_emotion_{timestamp}.json"

    # Run evaluation based on mode
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": "iemocap",
            "session": args.session,
            "mode": args.mode,
            "limit": args.limit,
        }
    }

    if args.mode in ["categorical", "both"]:
        logger.info("Running categorical evaluation...")
        categorical_results = evaluate_categorical(session=args.session, limit=args.limit)
        if categorical_results:
            results["categorical"] = categorical_results
            print_categorical_report(categorical_results)

    if args.mode in ["dimensional", "both"]:
        logger.info("Running dimensional evaluation...")
        dimensional_results = evaluate_dimensional(session=args.session, limit=args.limit)
        if dimensional_results:
            results["dimensional"] = dimensional_results
            print_dimensional_report(dimensional_results)

    # Save results
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Review results: jq . {output_path} | less")
    print("  2. Compare with baseline models")
    print("  3. Analyze error patterns for model improvements")

    return 0


if __name__ == "__main__":
    sys.exit(main())
