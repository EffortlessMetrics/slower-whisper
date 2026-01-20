"""
Benchmark CLI subcommand for slower-whisper.

Provides infrastructure for running standardized quality evaluations across
different tracks (ASR, diarization, streaming, semantic).

This module implements the v2.0.0 benchmark CLI feature, integrating with
the existing benchmark infrastructure in transcription/benchmarks.py and
benchmarks/*.py.

Example usage:
    slower-whisper benchmark --track asr --dataset librispeech --output report.json
    slower-whisper benchmark --track diarization --dataset ami --split test
    slower-whisper benchmark list  # List available datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmarks import (
    EvalSample,
    get_benchmarks_root,
    iter_ami_meetings,
    iter_iemocap_clips,
    iter_librispeech,
    list_available_benchmarks,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark Track Definitions
# =============================================================================


@dataclass
class TrackConfig:
    """Configuration for a benchmark track."""

    name: str
    description: str
    supported_datasets: list[str]
    metrics: list[str]
    default_dataset: str | None = None


# Available benchmark tracks
BENCHMARK_TRACKS: dict[str, TrackConfig] = {
    "asr": TrackConfig(
        name="ASR (Automatic Speech Recognition)",
        description="Evaluate transcription accuracy using WER/CER metrics",
        supported_datasets=["librispeech"],
        metrics=["wer", "cer", "rtf"],
        default_dataset="librispeech",
    ),
    "diarization": TrackConfig(
        name="Speaker Diarization",
        description="Evaluate speaker segmentation using DER metrics",
        supported_datasets=["ami", "libricss"],
        metrics=["der", "jer", "speaker_count_accuracy"],
        default_dataset="ami",
    ),
    "streaming": TrackConfig(
        name="Streaming Performance",
        description="Evaluate latency and throughput for streaming transcription",
        supported_datasets=["librispeech", "ami"],
        metrics=["latency_p50", "latency_p99", "throughput", "rtf"],
        default_dataset="librispeech",
    ),
    "semantic": TrackConfig(
        name="Semantic Annotation",
        description="Evaluate semantic enrichment quality (summaries, actions, etc.)",
        supported_datasets=["ami"],
        metrics=["faithfulness", "coverage", "clarity"],
        default_dataset="ami",
    ),
    "emotion": TrackConfig(
        name="Emotion Recognition",
        description="Evaluate emotion classification accuracy",
        supported_datasets=["iemocap"],
        metrics=["accuracy", "f1_weighted", "confusion_matrix"],
        default_dataset="iemocap",
    ),
}


# =============================================================================
# Benchmark Results
# =============================================================================


@dataclass
class BenchmarkMetric:
    """A single evaluation metric."""

    name: str
    value: float
    unit: str = ""
    description: str = ""


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    track: str
    dataset: str
    split: str
    samples_evaluated: int
    samples_failed: int
    metrics: list[BenchmarkMetric]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    system_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "track": self.track,
            "dataset": self.dataset,
            "split": self.split,
            "samples_evaluated": self.samples_evaluated,
            "samples_failed": self.samples_failed,
            "metrics": [asdict(m) for m in self.metrics],
            "timestamp": self.timestamp,
            "config": self.config,
            "errors": self.errors,
            "system_info": self.system_info,
        }


# =============================================================================
# Benchmark Runners (Scaffolding)
# =============================================================================


class BenchmarkRunner:
    """Base class for benchmark runners."""

    def __init__(self, track: str, dataset: str, split: str = "test"):
        self.track = track
        self.dataset = dataset
        self.split = split

    def get_samples(self, limit: int | None = None) -> list[EvalSample]:
        """Load evaluation samples for this benchmark."""
        raise NotImplementedError

    def evaluate_sample(self, sample: EvalSample) -> dict[str, Any]:
        """Evaluate a single sample and return metrics."""
        raise NotImplementedError

    def aggregate_metrics(self, sample_results: list[dict[str, Any]]) -> list[BenchmarkMetric]:
        """Aggregate per-sample metrics into summary metrics."""
        raise NotImplementedError

    def run(self, limit: int | None = None, verbose: bool = False) -> BenchmarkResult:
        """Run the benchmark and return results."""
        samples = self.get_samples(limit=limit)
        sample_results = []
        errors = []

        for i, sample in enumerate(samples):
            if verbose:
                logger.info(f"[{i + 1}/{len(samples)}] Processing {sample.id}...")

            try:
                result = self.evaluate_sample(sample)
                sample_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to evaluate {sample.id}: {e}")
                errors.append(f"{sample.id}: {str(e)}")

        metrics = self.aggregate_metrics(sample_results)

        return BenchmarkResult(
            track=self.track,
            dataset=self.dataset,
            split=self.split,
            samples_evaluated=len(sample_results),
            samples_failed=len(errors),
            metrics=metrics,
            errors=errors,
            system_info=self._get_system_info(),
        )

    def _get_system_info(self) -> dict[str, Any]:
        """Collect system information for reproducibility."""
        import platform

        info: dict[str, Any] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }

        # Try to get GPU info
        try:
            import torch

            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_device"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["cuda_available"] = False

        return info


class ASRBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for ASR (WER/CER) evaluation.

    This is scaffolding - full implementation would:
    1. Transcribe each sample using slower-whisper
    2. Compare with reference transcript
    3. Calculate WER/CER using jiwer or similar
    """

    def get_samples(self, limit: int | None = None) -> list[EvalSample]:
        if self.dataset == "librispeech":
            return list(iter_librispeech(split=self.split, limit=limit))
        raise ValueError(f"Dataset {self.dataset} not supported for ASR track")

    def evaluate_sample(self, sample: EvalSample) -> dict[str, Any]:
        # TODO: Implement actual ASR evaluation
        # 1. Transcribe sample.audio_path
        # 2. Compare with sample.reference_transcript
        # 3. Return WER/CER metrics
        logger.debug(f"ASR evaluation for {sample.id} (not implemented)")
        return {
            "id": sample.id,
            "wer": 0.0,  # Placeholder
            "cer": 0.0,  # Placeholder
            "reference_length": len(sample.reference_transcript or ""),
        }

    def aggregate_metrics(self, sample_results: list[dict[str, Any]]) -> list[BenchmarkMetric]:
        if not sample_results:
            return []

        avg_wer = sum(r["wer"] for r in sample_results) / len(sample_results)
        avg_cer = sum(r["cer"] for r in sample_results) / len(sample_results)

        return [
            BenchmarkMetric(
                name="wer",
                value=avg_wer,
                unit="%",
                description="Word Error Rate (lower is better)",
            ),
            BenchmarkMetric(
                name="cer",
                value=avg_cer,
                unit="%",
                description="Character Error Rate (lower is better)",
            ),
        ]


class DiarizationBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for speaker diarization (DER) evaluation.

    This is scaffolding - full implementation would:
    1. Run diarization on each sample
    2. Compare with reference speaker annotations
    3. Calculate DER using pyannote.metrics
    """

    def get_samples(self, limit: int | None = None) -> list[EvalSample]:
        if self.dataset == "ami":
            return list(iter_ami_meetings(split=self.split, limit=limit))
        raise ValueError(f"Dataset {self.dataset} not supported for diarization track")

    def evaluate_sample(self, sample: EvalSample) -> dict[str, Any]:
        # TODO: Implement actual diarization evaluation
        # 1. Run diarization on sample.audio_path
        # 2. Compare with sample.reference_speakers
        # 3. Return DER metrics
        logger.debug(f"Diarization evaluation for {sample.id} (not implemented)")
        return {
            "id": sample.id,
            "der": 0.0,  # Placeholder
            "jer": 0.0,  # Placeholder (Jaccard Error Rate)
            "speaker_count_ref": len(
                {s.get("speaker_id") for s in (sample.reference_speakers or [])}
            ),
            "speaker_count_hyp": 0,
        }

    def aggregate_metrics(self, sample_results: list[dict[str, Any]]) -> list[BenchmarkMetric]:
        if not sample_results:
            return []

        avg_der = sum(r["der"] for r in sample_results) / len(sample_results)
        avg_jer = sum(r["jer"] for r in sample_results) / len(sample_results)

        return [
            BenchmarkMetric(
                name="der",
                value=avg_der,
                unit="%",
                description="Diarization Error Rate (lower is better)",
            ),
            BenchmarkMetric(
                name="jer",
                value=avg_jer,
                unit="%",
                description="Jaccard Error Rate (lower is better)",
            ),
        ]


class StreamingBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for streaming performance evaluation.

    This is scaffolding - full implementation would:
    1. Run streaming transcription
    2. Measure latency (time to first token, time to completion)
    3. Calculate throughput and RTF
    """

    def get_samples(self, limit: int | None = None) -> list[EvalSample]:
        if self.dataset == "librispeech":
            return list(iter_librispeech(split=self.split, limit=limit))
        elif self.dataset == "ami":
            return list(iter_ami_meetings(split=self.split, limit=limit))
        raise ValueError(f"Dataset {self.dataset} not supported for streaming track")

    def evaluate_sample(self, sample: EvalSample) -> dict[str, Any]:
        # TODO: Implement actual streaming evaluation
        # 1. Run streaming transcription
        # 2. Measure latencies
        # 3. Return timing metrics
        logger.debug(f"Streaming evaluation for {sample.id} (not implemented)")
        return {
            "id": sample.id,
            "latency_first_token_ms": 0.0,
            "latency_total_ms": 0.0,
            "audio_duration_s": 0.0,
            "rtf": 0.0,
        }

    def aggregate_metrics(self, sample_results: list[dict[str, Any]]) -> list[BenchmarkMetric]:
        if not sample_results:
            return []

        latencies = [
            r["latency_first_token_ms"] for r in sample_results if r["latency_first_token_ms"] > 0
        ]
        rtfs = [r["rtf"] for r in sample_results if r["rtf"] > 0]

        metrics = []
        if latencies:
            import statistics

            metrics.append(
                BenchmarkMetric(
                    name="latency_p50",
                    value=statistics.median(latencies),
                    unit="ms",
                    description="Median latency to first token",
                )
            )
            if len(latencies) >= 10:
                sorted_latencies = sorted(latencies)
                p99_idx = int(len(sorted_latencies) * 0.99)
                metrics.append(
                    BenchmarkMetric(
                        name="latency_p99",
                        value=sorted_latencies[p99_idx],
                        unit="ms",
                        description="99th percentile latency",
                    )
                )

        if rtfs:
            metrics.append(
                BenchmarkMetric(
                    name="rtf",
                    value=sum(rtfs) / len(rtfs),
                    unit="x",
                    description="Real-time factor (lower is faster)",
                )
            )

        return metrics


class SemanticBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for semantic annotation evaluation.

    This is scaffolding - full implementation would:
    1. Generate summaries/annotations using LLM
    2. Compare with reference using Claude-as-judge
    3. Return quality scores
    """

    def get_samples(self, limit: int | None = None) -> list[EvalSample]:
        if self.dataset == "ami":
            return list(iter_ami_meetings(split=self.split, limit=limit, require_summary=True))
        raise ValueError(f"Dataset {self.dataset} not supported for semantic track")

    def evaluate_sample(self, sample: EvalSample) -> dict[str, Any]:
        # TODO: Implement actual semantic evaluation
        # 1. Generate summary from transcript
        # 2. Compare with sample.reference_summary
        # 3. Return quality scores
        logger.debug(f"Semantic evaluation for {sample.id} (not implemented)")
        return {
            "id": sample.id,
            "faithfulness": 0.0,
            "coverage": 0.0,
            "clarity": 0.0,
        }

    def aggregate_metrics(self, sample_results: list[dict[str, Any]]) -> list[BenchmarkMetric]:
        if not sample_results:
            return []

        avg_faith = sum(r["faithfulness"] for r in sample_results) / len(sample_results)
        avg_cov = sum(r["coverage"] for r in sample_results) / len(sample_results)
        avg_clar = sum(r["clarity"] for r in sample_results) / len(sample_results)

        return [
            BenchmarkMetric(
                name="faithfulness",
                value=avg_faith,
                unit="/10",
                description="Factual accuracy of generated content",
            ),
            BenchmarkMetric(
                name="coverage",
                value=avg_cov,
                unit="/10",
                description="Completeness of key information",
            ),
            BenchmarkMetric(
                name="clarity",
                value=avg_clar,
                unit="/10",
                description="Readability and coherence",
            ),
        ]


class EmotionBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for emotion recognition evaluation.

    This is scaffolding - full implementation would:
    1. Run emotion recognition on each sample
    2. Compare with reference emotion labels
    3. Calculate accuracy/F1 metrics
    """

    def get_samples(self, limit: int | None = None) -> list[EvalSample]:
        if self.dataset == "iemocap":
            return list(iter_iemocap_clips(limit=limit))
        raise ValueError(f"Dataset {self.dataset} not supported for emotion track")

    def evaluate_sample(self, sample: EvalSample) -> dict[str, Any]:
        # TODO: Implement actual emotion evaluation
        # 1. Run emotion recognition on sample.audio_path
        # 2. Compare with sample.reference_emotions
        # 3. Return accuracy metrics
        logger.debug(f"Emotion evaluation for {sample.id} (not implemented)")
        return {
            "id": sample.id,
            "predicted": None,
            "reference": sample.reference_emotions[0] if sample.reference_emotions else None,
            "correct": False,
        }

    def aggregate_metrics(self, sample_results: list[dict[str, Any]]) -> list[BenchmarkMetric]:
        if not sample_results:
            return []

        correct = sum(1 for r in sample_results if r["correct"])
        total = len(sample_results)
        accuracy = correct / total if total > 0 else 0.0

        return [
            BenchmarkMetric(
                name="accuracy",
                value=accuracy * 100,
                unit="%",
                description="Classification accuracy",
            ),
        ]


# =============================================================================
# Runner Factory
# =============================================================================


def get_benchmark_runner(track: str, dataset: str, split: str = "test") -> BenchmarkRunner:
    """Get the appropriate benchmark runner for a track."""
    runners = {
        "asr": ASRBenchmarkRunner,
        "diarization": DiarizationBenchmarkRunner,
        "streaming": StreamingBenchmarkRunner,
        "semantic": SemanticBenchmarkRunner,
        "emotion": EmotionBenchmarkRunner,
    }

    if track not in runners:
        raise ValueError(f"Unknown track: {track}. Available: {list(runners.keys())}")

    return runners[track](track=track, dataset=dataset, split=split)


# =============================================================================
# CLI Handler Functions
# =============================================================================


def handle_benchmark_list() -> int:
    """Handle 'benchmark list' command - show available datasets and tracks."""
    print("Available Benchmark Tracks:")
    print("-" * 60)

    for track_id, config in BENCHMARK_TRACKS.items():
        print(f"\n  {track_id}: {config.name}")
        print(f"     {config.description}")
        print(f"     Datasets: {', '.join(config.supported_datasets)}")
        print(f"     Metrics: {', '.join(config.metrics)}")

    print("\n")
    print("Available Datasets:")
    print("-" * 60)

    datasets = list_available_benchmarks()
    for name, info in datasets.items():
        status = "[available]" if info["available"] else "[not staged]"
        print(f"\n  {name}: {status}")
        print(f"     {info['description']}")
        print(f"     Path: {info['path']}")
        print(f"     Tasks: {', '.join(info.get('tasks', []))}")
        if not info["available"]:
            print(f"     Setup: See {info['setup_doc']}")

    print("\n")
    print(f"Benchmarks root: {get_benchmarks_root()}")

    return 0


def handle_benchmark_run(
    track: str,
    dataset: str | None,
    split: str,
    limit: int | None,
    output: Path | None,
    verbose: bool,
) -> int:
    """Handle 'benchmark run' command - run a specific benchmark."""
    # Validate track
    if track not in BENCHMARK_TRACKS:
        print(f"Error: Unknown track '{track}'.", file=sys.stderr)
        print(f"Available tracks: {', '.join(BENCHMARK_TRACKS.keys())}", file=sys.stderr)
        return 1

    track_config = BENCHMARK_TRACKS[track]

    # Use default dataset if not specified
    if dataset is None:
        dataset = track_config.default_dataset
        if dataset is None:
            print(
                f"Error: No dataset specified and no default for track '{track}'.", file=sys.stderr
            )
            return 1

    # Validate dataset for track
    if dataset not in track_config.supported_datasets:
        print(f"Error: Dataset '{dataset}' not supported for track '{track}'.", file=sys.stderr)
        print(f"Supported datasets: {', '.join(track_config.supported_datasets)}", file=sys.stderr)
        return 1

    # Check dataset availability
    available = list_available_benchmarks()
    if dataset in available and not available[dataset]["available"]:
        print(f"Warning: Dataset '{dataset}' is not staged.", file=sys.stderr)
        print(f"See {available[dataset]['setup_doc']} for setup instructions.", file=sys.stderr)
        # Continue anyway - runner will fail with helpful error

    print(f"Running benchmark: {track}")
    print(f"  Dataset: {dataset}")
    print(f"  Split: {split}")
    if limit:
        print(f"  Limit: {limit} samples")
    print()

    try:
        runner = get_benchmark_runner(track, dataset, split)
        result = runner.run(limit=limit, verbose=verbose)

        # Display results
        print("=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"\nTrack: {result.track}")
        print(f"Dataset: {result.dataset} ({result.split})")
        print(f"Samples: {result.samples_evaluated} evaluated, {result.samples_failed} failed")
        print(f"Timestamp: {result.timestamp}")

        print("\nMetrics:")
        for metric in result.metrics:
            print(f"  {metric.name}: {metric.value:.4f}{metric.unit}")
            if metric.description:
                print(f"    ({metric.description})")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors[:5]:
                print(f"  - {error}")
            if len(result.errors) > 5:
                print(f"  ... and {len(result.errors) - 5} more")

        # Save results if output specified
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to: {output}")

        return 0 if result.samples_failed == 0 else 1

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def handle_benchmark_status() -> int:
    """Handle 'benchmark status' command - show current benchmark setup status."""
    print("Benchmark Infrastructure Status")
    print("=" * 60)

    root = get_benchmarks_root()
    print(f"\nBenchmarks root: {root}")
    print(f"  Exists: {root.exists()}")

    datasets = list_available_benchmarks()
    available_count = sum(1 for d in datasets.values() if d["available"])
    print(f"\nDatasets: {available_count}/{len(datasets)} available")

    for name, info in datasets.items():
        status = "[OK]" if info["available"] else "[MISSING]"
        print(f"  {status} {name}")

    print("\nTo stage a dataset, see the setup documentation:")
    for name, info in datasets.items():
        if not info["available"]:
            print(f"  {name}: {info['setup_doc']}")

    return 0


# =============================================================================
# Argument Parser Builder
# =============================================================================


def build_benchmark_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Build the benchmark subcommand parser.

    Args:
        subparsers: The subparsers object from the main CLI parser.

    Returns:
        The benchmark subcommand parser.
    """
    p_benchmark: argparse.ArgumentParser = subparsers.add_parser(
        "benchmark",
        help="Run quality benchmarks against standard datasets.",
        description=(
            "Evaluate slower-whisper quality on standard benchmark datasets. "
            "Supports multiple evaluation tracks (ASR, diarization, streaming, semantic) "
            "with configurable datasets and output formats."
        ),
    )

    # Create nested subparsers for benchmark actions
    benchmark_subparsers = p_benchmark.add_subparsers(
        dest="benchmark_action",
        help="Benchmark action to perform",
    )

    # benchmark list
    benchmark_subparsers.add_parser(
        "list",
        help="List available benchmark tracks and datasets.",
    )

    # benchmark status
    benchmark_subparsers.add_parser(
        "status",
        help="Show benchmark infrastructure status.",
    )

    # benchmark run
    p_run = benchmark_subparsers.add_parser(
        "run",
        help="Run a benchmark evaluation.",
    )

    p_run.add_argument(
        "--track",
        "-t",
        choices=list(BENCHMARK_TRACKS.keys()),
        required=True,
        help="Evaluation track to run.",
    )

    p_run.add_argument(
        "--dataset",
        "-d",
        help="Dataset to evaluate on (default: track-specific).",
    )

    p_run.add_argument(
        "--split",
        "-s",
        default="test",
        help="Dataset split to use (default: test).",
    )

    p_run.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (for quick testing).",
    )

    p_run.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for benchmark results JSON.",
    )

    p_run.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress.",
    )

    # Default action if no subcommand given - show help
    p_benchmark.set_defaults(benchmark_action=None)

    return p_benchmark


def handle_benchmark_command(args: argparse.Namespace) -> int:
    """Handle the benchmark command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    if args.benchmark_action is None or args.benchmark_action == "list":
        return handle_benchmark_list()

    if args.benchmark_action == "status":
        return handle_benchmark_status()

    if args.benchmark_action == "run":
        return handle_benchmark_run(
            track=args.track,
            dataset=args.dataset,
            split=args.split,
            limit=args.limit,
            output=args.output,
            verbose=args.verbose,
        )

    # Unknown action
    print(f"Unknown benchmark action: {args.benchmark_action}", file=sys.stderr)
    return 1
