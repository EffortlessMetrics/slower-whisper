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
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from types import ModuleType

    from .asr_engine import TranscriptionEngine as _TranscriptionEngine
    from .diarization import Diarizer
    from .models import Transcript

from .benchmark.semantic_metrics import (
    compute_action_metrics,
    compute_risk_metrics,
    compute_topic_f1,
)
from .benchmarks import (
    EvalSample,
    get_benchmarks_root,
    iter_ami_meetings,
    iter_callhome,
    iter_commonvoice,
    iter_iemocap_clips,
    iter_librispeech,
    iter_smoke_asr,
    iter_smoke_diarization,
    list_available_benchmarks,
)
from .semantic import KeywordSemanticAnnotator

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
        supported_datasets=["smoke", "librispeech", "commonvoice_en_smoke"],
        metrics=["wer", "cer", "rtf"],
        default_dataset="smoke",
    ),
    "diarization": TrackConfig(
        name="Speaker Diarization",
        description="Evaluate speaker segmentation using DER metrics",
        supported_datasets=["smoke", "ami", "callhome", "libricss"],
        metrics=["der", "jer", "speaker_count_accuracy"],
        default_dataset="smoke",
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
    value: float | None
    unit: str = ""
    description: str = ""
    measured_count: int | None = None
    total_count: int | None = None
    reason: str | None = None  # Why metric is None (e.g., "missing_api_key")


@dataclass
class BenchmarkReceipt:
    """Provenance information for a benchmark run.

    Provides traceability for benchmark results including tool version,
    config hash, and git commit for reproducibility.
    """

    tool_version: str
    config_hash: str = ""
    git_commit: str | None = None
    mode: str = ""  # For semantic track: "tags", "summary", "both"


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
    receipt: BenchmarkReceipt | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
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
        if self.receipt is not None:
            result["receipt"] = asdict(self.receipt)
        return result


# =============================================================================
# Baseline File Format (#137)
# =============================================================================

BASELINE_SCHEMA_VERSION = 1
DEFAULT_REGRESSION_THRESHOLD = 0.10  # 10% regression allowed


@dataclass
class BaselineMetric:
    """A metric in a baseline file with optional threshold."""

    value: float
    unit: str = ""
    threshold: float | None = None  # Regression threshold (e.g., 0.10 = 10%)


@dataclass
class BaselineReceipt:
    """Provenance information for a baseline."""

    tool_version: str
    model: str = ""
    device: str = ""
    compute_type: str = ""
    git_commit: str | None = None


@dataclass
class BaselineFile:
    """A baseline file for comparing benchmark results.

    Location: benchmarks/baselines/<track>/<dataset>.json

    Example:
        {
            "schema_version": 1,
            "track": "asr",
            "dataset": "librispeech-test-clean",
            "created_at": "2026-01-07T12:00:00Z",
            "metrics": {
                "wer": {"value": 0.045, "unit": "ratio", "threshold": 0.05}
            },
            "receipt": {
                "tool_version": "1.9.2",
                "model": "large-v3"
            }
        }
    """

    schema_version: int
    track: str
    dataset: str
    created_at: str
    metrics: dict[str, BaselineMetric]
    receipt: BaselineReceipt

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "track": self.track,
            "dataset": self.dataset,
            "created_at": self.created_at,
            "metrics": {
                name: {
                    "value": m.value,
                    "unit": m.unit,
                    **({"threshold": m.threshold} if m.threshold is not None else {}),
                }
                for name, m in self.metrics.items()
            },
            "receipt": asdict(self.receipt),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineFile:
        """Create from dictionary (JSON deserialization)."""
        metrics = {}
        for name, m in data.get("metrics", {}).items():
            metrics[name] = BaselineMetric(
                value=m["value"],
                unit=m.get("unit", ""),
                threshold=m.get("threshold"),
            )

        receipt_data = data.get("receipt", {})
        receipt = BaselineReceipt(
            tool_version=receipt_data.get("tool_version", ""),
            model=receipt_data.get("model", ""),
            device=receipt_data.get("device", ""),
            compute_type=receipt_data.get("compute_type", ""),
            git_commit=receipt_data.get("git_commit"),
        )

        return cls(
            schema_version=data.get("schema_version", 1),
            track=data["track"],
            dataset=data["dataset"],
            created_at=data.get("created_at", ""),
            metrics=metrics,
            receipt=receipt,
        )


@dataclass
class ComparisonResult:
    """Result of comparing a metric against baseline."""

    metric_name: str
    current_value: float
    baseline_value: float
    regression: float  # (current - baseline) / baseline
    threshold: float | None
    passed: bool  # True if regression <= threshold (or no threshold)
    unit: str = ""


@dataclass
class BaselineComparison:
    """Full comparison of benchmark results against baseline."""

    track: str
    dataset: str
    baseline_created_at: str
    comparison_timestamp: str
    results: list[ComparisonResult]
    overall_passed: bool  # All metrics passed
    mode: Literal["report", "gate"]  # report = never fail, gate = fail on regression

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "track": self.track,
            "dataset": self.dataset,
            "baseline_created_at": self.baseline_created_at,
            "comparison_timestamp": self.comparison_timestamp,
            "results": [
                {
                    "metric": r.metric_name,
                    "current": r.current_value,
                    "baseline": r.baseline_value,
                    "regression": r.regression,
                    "threshold": r.threshold,
                    "passed": r.passed,
                    "unit": r.unit,
                }
                for r in self.results
            ],
            "overall_passed": self.overall_passed,
            "mode": self.mode,
        }


def get_baselines_dir() -> Path:
    """Get the baselines directory path.

    Returns the baselines directory from the project's benchmarks folder.
    Baselines are version-controlled artifacts, not cache data.

    Location priority:
    1. SLOWER_WHISPER_BASELINES environment variable (if set)
    2. Project root's benchmarks/baselines directory

    Returns:
        Path to baselines directory
    """
    # Check for explicit override
    if baselines_override := os.environ.get("SLOWER_WHISPER_BASELINES"):
        return Path(baselines_override).expanduser()

    # Use project root's benchmarks/baselines directory
    # This is the package's parent directory (transcription -> project root)
    project_root = Path(__file__).parent.parent
    return project_root / "benchmarks" / "baselines"


def get_baseline_path(track: str, dataset: str) -> Path:
    """Get the path to a baseline file."""
    return get_baselines_dir() / track / f"{dataset}.json"


def load_baseline(track: str, dataset: str) -> BaselineFile | None:
    """Load a baseline file if it exists.

    Args:
        track: Benchmark track (asr, diarization, etc.)
        dataset: Dataset name (librispeech, ami, etc.)

    Returns:
        BaselineFile if exists, None otherwise
    """
    path = get_baseline_path(track, dataset)
    if not path.exists():
        logger.debug(f"No baseline file at {path}")
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        return BaselineFile.from_dict(data)
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.warning(f"Failed to load baseline from {path}: {e}")
        return None


def save_baseline(baseline: BaselineFile) -> Path:
    """Save a baseline file.

    Args:
        baseline: The baseline to save

    Returns:
        Path where baseline was saved
    """
    path = get_baseline_path(baseline.track, baseline.dataset)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(baseline.to_dict(), f, indent=2)

    return path


def create_baseline_from_result(
    result: BenchmarkResult,
    thresholds: dict[str, float] | None = None,
) -> BaselineFile:
    """Create a baseline file from benchmark results.

    Args:
        result: Benchmark results to use as baseline
        thresholds: Optional dict of metric_name -> threshold (e.g., {"wer": 0.10})

    Returns:
        BaselineFile ready to save
    """
    thresholds = thresholds or {}

    # Get version info
    try:
        from . import __version__
    except ImportError:
        __version__ = "unknown"

    # Get git commit if available
    git_commit = None
    try:
        import subprocess

        git_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if git_result.returncode == 0:
            git_commit = git_result.stdout.strip()
    except Exception:
        pass

    # Build metrics dict
    metrics = {}
    for m in result.metrics:
        if m.value is not None:
            metrics[m.name] = BaselineMetric(
                value=m.value,
                unit=m.unit,
                threshold=thresholds.get(m.name),
            )

    # Build receipt from system info
    receipt = BaselineReceipt(
        tool_version=__version__,
        model=result.config.get("model", ""),
        device=result.system_info.get("cuda_device", "cpu"),
        compute_type=result.config.get("compute_type", ""),
        git_commit=git_commit,
    )

    return BaselineFile(
        schema_version=BASELINE_SCHEMA_VERSION,
        track=result.track,
        dataset=result.dataset,
        created_at=datetime.now().isoformat(),
        metrics=metrics,
        receipt=receipt,
    )


def compare_with_baseline(
    result: BenchmarkResult,
    baseline: BaselineFile,
    mode: Literal["report", "gate"] = "report",
    default_threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> BaselineComparison:
    """Compare benchmark results against a baseline.

    Args:
        result: Current benchmark results
        baseline: Baseline to compare against
        mode: "report" (never fail) or "gate" (fail on regression)
        default_threshold: Default regression threshold if metric has none

    Returns:
        BaselineComparison with per-metric results
    """
    comparison_results = []

    for metric in result.metrics:
        if metric.value is None:
            continue

        baseline_metric = baseline.metrics.get(metric.name)
        if baseline_metric is None:
            logger.debug(f"Metric {metric.name} not in baseline, skipping comparison")
            continue

        # Calculate regression
        if baseline_metric.value == 0:
            regression = 0.0 if metric.value == 0 else float("inf")
        else:
            regression = (metric.value - baseline_metric.value) / baseline_metric.value

        # Determine threshold
        threshold = baseline_metric.threshold
        if threshold is None:
            threshold = default_threshold

        # Check if passed (for metrics where lower is better, regression > 0 is bad)
        passed = regression <= threshold

        comparison_results.append(
            ComparisonResult(
                metric_name=metric.name,
                current_value=metric.value,
                baseline_value=baseline_metric.value,
                regression=regression,
                threshold=threshold,
                passed=passed,
                unit=metric.unit,
            )
        )

    overall_passed = all(r.passed for r in comparison_results)

    return BaselineComparison(
        track=result.track,
        dataset=result.dataset,
        baseline_created_at=baseline.created_at,
        comparison_timestamp=datetime.now().isoformat(),
        results=comparison_results,
        overall_passed=overall_passed,
        mode=mode,
    )


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
        receipt = self._generate_receipt()

        return BenchmarkResult(
            track=self.track,
            dataset=self.dataset,
            split=self.split,
            samples_evaluated=len(sample_results),
            samples_failed=len(errors),
            metrics=metrics,
            errors=errors,
            system_info=self._get_system_info(),
            receipt=receipt,
        )

    def _generate_receipt(self) -> BenchmarkReceipt:
        """Generate provenance receipt for the benchmark run.

        Returns:
            BenchmarkReceipt with tool version, config hash, and git commit
        """
        import hashlib

        # Get tool version
        try:
            from . import __version__
        except ImportError:
            __version__ = "unknown"

        # Generate config hash from track, dataset, split
        config_str = f"{self.track}:{self.dataset}:{self.split}"
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

        # Get git commit if available
        git_commit = self._get_git_commit()

        return BenchmarkReceipt(
            tool_version=__version__,
            config_hash=config_hash,
            git_commit=git_commit,
        )

    def _get_git_commit(self) -> str | None:
        """Get the current git commit hash if available."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

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

    Implements full ASR evaluation:
    1. Transcribe each sample using slower-whisper (TranscriptionEngine)
    2. Compare with reference transcript
    3. Calculate WER/CER using jiwer

    Note:
        Currently hardcodes language="en" since the supported dataset (LibriSpeech)
        is English-only. Future datasets may require language configuration.
    """

    engine: _TranscriptionEngine | None
    jiwer: ModuleType | None

    def __init__(self, track: str, dataset: str, split: str = "test"):
        super().__init__(track, dataset, split)
        self.engine = None
        self.jiwer = None

        # Initialize jiwer first (fast, no side effects)
        try:
            import jiwer

            self.jiwer = jiwer
        except ImportError:
            logger.warning("jiwer not installed. WER/CER calculation will fail.")

        # Initialize ASR engine
        try:
            from .asr_engine import TranscriptionEngine
            from .config import AsrConfig, TranscriptionConfig

            # Load config from env (respects SLOWER_WHISPER_MODEL, etc.)
            t_cfg = TranscriptionConfig.from_env()
            config = AsrConfig(
                model_name=t_cfg.model,
                device=t_cfg.device,
                compute_type=t_cfg.compute_type,
                beam_size=t_cfg.beam_size,
                task="transcribe",
                language="en",  # LibriSpeech is English-only
            )
            self.engine = TranscriptionEngine(config)
        except ImportError as e:
            logger.error(f"ASR dependencies not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize ASR engine: {e}")

    def get_samples(self, limit: int | None = None) -> list[EvalSample]:
        if self.dataset == "librispeech":
            return list(iter_librispeech(split=self.split, limit=limit))
        elif self.dataset == "commonvoice_en_smoke":
            return list(iter_commonvoice(subset="en_smoke", limit=limit))
        elif self.dataset == "smoke":
            return list(iter_smoke_asr(limit=limit))
        raise ValueError(f"Dataset {self.dataset} not supported for ASR track")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for ASR evaluation (lowercase, remove punctuation)."""
        if not text:
            return ""
        # Lowercase
        text = text.lower()
        # Remove punctuation (keep alphanumeric and spaces)
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse whitespace
        return " ".join(text.split())

    def evaluate_sample(self, sample: EvalSample) -> dict[str, Any]:
        if not self.engine:
            raise RuntimeError("ASR engine not initialized")

        if not self.jiwer:
            raise ImportError("jiwer package is required for ASR evaluation")

        # 1. Transcribe
        logger.debug(f"Transcribing {sample.id}...")
        try:
            transcript = self.engine.transcribe_file(sample.audio_path)
        except Exception as e:
            logger.error(f"Transcription failed for {sample.id}: {e}")
            raise

        # Join segments to get full hypothesis text
        hypothesis = " ".join(seg.text for seg in transcript.segments)

        # 2. Prepare texts
        reference = sample.reference_transcript or ""

        # Normalize both reference and hypothesis
        ref_norm = self._normalize_text(reference)
        hyp_norm = self._normalize_text(hypothesis)

        # 3. Calculate metrics
        wer = 0.0
        cer = 0.0

        if ref_norm:
            wer = self.jiwer.wer(ref_norm, hyp_norm)
            cer = self.jiwer.cer(ref_norm, hyp_norm)
        elif hyp_norm:
            # If reference is empty but hypothesis is not, error is 100% (or infinite?)
            # Usually treat as insertion error = length of hypothesis
            wer = 1.0
            cer = 1.0

        return {
            "id": sample.id,
            "wer": wer * 100,  # Convert to percentage
            "cer": cer * 100,  # Convert to percentage
            "reference_length": len(ref_norm),
            "hypothesis_length": len(hyp_norm),
            "reference": ref_norm,
            "hypothesis": hyp_norm,
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

    Evaluates speaker diarization quality by:
    1. Running diarization on each sample
    2. Comparing with reference speaker annotations
    3. Calculating DER/JER using pyannote.metrics

    Requirements:
        - pyannote.metrics (install with: uv sync --extra dev)
        - pyannote.audio for actual diarization (install with: uv sync --extra diarization)
    """

    def __init__(self, track: str, dataset: str, split: str = "test"):
        super().__init__(track, dataset, split)
        self._diarizer: Diarizer | None = None

    @property
    def diarizer(self):
        if self._diarizer is None:
            from transcription.diarization import Diarizer

            self._diarizer = Diarizer()
        return self._diarizer

    def get_samples(self, limit: int | None = None) -> list[EvalSample]:
        if self.dataset == "smoke":
            return list(iter_smoke_diarization(limit=limit))
        elif self.dataset == "ami":
            return list(iter_ami_meetings(split=self.split, limit=limit))
        elif self.dataset == "callhome":
            return list(iter_callhome(split=self.split, limit=limit))
        raise ValueError(f"Dataset {self.dataset} not supported for diarization track")

    def evaluate_sample(self, sample: EvalSample) -> dict[str, Any]:
        try:
            from pyannote.core import Annotation, Segment
            from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
        except ImportError as e:
            raise ImportError(
                "pyannote.metrics is required for diarization benchmark. "
                "Install it with: uv pip install pyannote.metrics"
            ) from e

        if not sample.reference_speakers:
            # Cannot evaluate without reference - return None metrics
            return {
                "id": sample.id,
                "der": None,
                "jer": None,
                "speaker_count_ref": 0,
                "speaker_count_hyp": 0,
                "reason": "no_reference_speakers",
            }

        # 1. Run diarization
        turns = self.diarizer.run(sample.audio_path)

        # 2. Prepare reference annotation
        ref = Annotation()
        ref_speakers = set()
        for s in sample.reference_speakers:
            ref[Segment(s["start"], s["end"])] = s["speaker_id"]
            ref_speakers.add(s["speaker_id"])

        # 3. Prepare hypothesis annotation
        hyp = Annotation()
        hyp_speakers = set()
        for turn in turns:
            hyp[Segment(turn.start, turn.end)] = turn.speaker_id
            hyp_speakers.add(turn.speaker_id)

        # 4. Calculate metrics
        der_metric = DiarizationErrorRate()
        jer_metric = JaccardErrorRate()

        # pyannote.metrics handles speaker permutation automatically
        try:
            der = der_metric(ref, hyp)
            jer = jer_metric(ref, hyp)
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            logger.error(f"Ref: {ref}")
            logger.error(f"Hyp: {hyp}")
            raise

        return {
            "id": sample.id,
            "der": der * 100.0,  # Convert to percentage
            "jer": jer * 100.0,  # Convert to percentage
            "speaker_count_ref": len(ref_speakers),
            "speaker_count_hyp": len(hyp_speakers),
        }

    def aggregate_metrics(self, sample_results: list[dict[str, Any]]) -> list[BenchmarkMetric]:
        if not sample_results:
            return []

        total_count = len(sample_results)

        # Filter out samples with None metrics
        der_values = [r["der"] for r in sample_results if r.get("der") is not None]
        jer_values = [r["jer"] for r in sample_results if r.get("jer") is not None]

        metrics: list[BenchmarkMetric] = []

        if der_values:
            avg_der = sum(der_values) / len(der_values)
            metrics.append(
                BenchmarkMetric(
                    name="der",
                    value=avg_der,
                    unit="%",
                    description="Diarization Error Rate (lower is better)",
                    measured_count=len(der_values),
                    total_count=total_count,
                )
            )
        else:
            # Find reason from samples
            reasons = [r.get("reason") for r in sample_results if r.get("der") is None]
            reason = reasons[0] if reasons else "not_measured"
            metrics.append(
                BenchmarkMetric(
                    name="der",
                    value=None,
                    unit="%",
                    description="Diarization Error Rate (lower is better)",
                    measured_count=0,
                    total_count=total_count,
                    reason=reason,
                )
            )

        if jer_values:
            avg_jer = sum(jer_values) / len(jer_values)
            metrics.append(
                BenchmarkMetric(
                    name="jer",
                    value=avg_jer,
                    unit="%",
                    description="Jaccard Error Rate (lower is better)",
                    measured_count=len(jer_values),
                    total_count=total_count,
                )
            )
        else:
            reasons = [r.get("reason") for r in sample_results if r.get("jer") is None]
            reason = reasons[0] if reasons else "not_measured"
            metrics.append(
                BenchmarkMetric(
                    name="jer",
                    value=None,
                    unit="%",
                    description="Jaccard Error Rate (lower is better)",
                    measured_count=0,
                    total_count=total_count,
                    reason=reason,
                )
            )

        return metrics


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
        import time

        from .asr_engine import AsrConfig, TranscriptionEngine
        from .streaming import StreamChunk
        from .streaming_enrich import StreamingEnrichmentConfig, StreamingEnrichmentSession

        logger.debug(f"Streaming evaluation for {sample.id}")

        # Initialize ASR engine
        # Note: In a real scenario we'd reuse the engine across samples,
        # but the interface currently evaluates one sample at a time.
        # Ideally, we should initialize the engine in __init__ or run().
        if not hasattr(self, "_engine"):
            # Use default config
            asr_config = AsrConfig()
            self._engine = TranscriptionEngine(asr_config)

        # Initialize streaming enrichment session
        # We disable enrichment features to focus on core streaming mechanics,
        # but using the session ensures we exercise the streaming pipeline.
        enrich_config = StreamingEnrichmentConfig(
            enable_prosody=False,
            enable_emotion=False,
        )
        session = StreamingEnrichmentSession(sample.audio_path, enrich_config)

        # Measure timing
        start_time = time.perf_counter()
        first_token_time: float | None = None

        try:
            # Use internal _transcribe_with_model to get the segment generator
            # This simulates receiving segments from a streaming ASR engine
            raw_segments, info = self._engine._transcribe_with_model(sample.audio_path)

            for segment in raw_segments:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                # Feed to streaming session
                chunk = StreamChunk(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker_id=None,
                )
                session.ingest_chunk(chunk)

            # Finalize stream
            session.end_of_stream()
            end_time = time.perf_counter()

            # Calculate metrics
            total_latency_ms = (end_time - start_time) * 1000
            latency_first_token_ms = (
                (first_token_time - start_time) * 1000 if first_token_time else total_latency_ms
            )

            # Audio duration
            # Try to get from session extractor, fallback to info.duration
            try:
                audio_duration_s = session._extractor.duration_seconds
            except Exception:
                audio_duration_s = getattr(info, "duration", 0.0)

            rtf = (total_latency_ms / 1000) / audio_duration_s if audio_duration_s > 0 else 0.0

            return {
                "id": sample.id,
                "latency_first_token_ms": latency_first_token_ms,
                "latency_total_ms": total_latency_ms,
                "audio_duration_s": audio_duration_s,
                "rtf": rtf,
            }

        except Exception as e:
            logger.error(f"Streaming evaluation failed for {sample.id}: {e}")
            raise

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


# Type alias for semantic evaluation mode
SemanticMode = Literal["tags", "summary", "both"]


class SemanticBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for semantic annotation evaluation.

    Evaluates semantic tagging (deterministic, CI-friendly) and/or
    LLM-based summary quality (requires ANTHROPIC_API_KEY).

    Args:
        track: Benchmark track name
        dataset: Dataset to evaluate on
        split: Dataset split (train/dev/test)
        mode: Evaluation mode - "tags" (deterministic), "summary" (LLM), or "both"
    """

    def __init__(
        self,
        track: str,
        dataset: str,
        split: str = "test",
        mode: SemanticMode = "tags",
    ):
        super().__init__(track, dataset, split)
        self.mode = mode
        self._annotator = KeywordSemanticAnnotator()
        # Gold labels are in project's benchmarks/gold/semantic directory
        self._gold_dir = self._get_gold_dir()
        # Directory containing sample transcript JSON files with proper segments
        self._samples_dir = self._get_samples_dir()

    def _get_gold_dir(self) -> Path:
        """Get the directory containing gold semantic labels.

        Returns the gold directory from benchmarks/gold/semantic/ in the project
        (version-controlled test fixtures, not cache data).
        """
        project_root = Path(__file__).parent.parent
        return project_root / "benchmarks" / "gold" / "semantic"

    def _get_samples_dir(self) -> Path:
        """Get the directory containing sample transcript JSON files.

        Returns the samples directory from benchmarks/data/speaker_analytics_samples/
        which contains properly segmented transcript JSON files for evaluation.
        """
        # Project root's benchmarks/data/speaker_analytics_samples directory
        project_root = Path(__file__).parent.parent
        return project_root / "benchmarks" / "data" / "speaker_analytics_samples"

    def get_samples(self, limit: int | None = None) -> list[EvalSample]:
        if self.dataset == "ami":
            return list(iter_ami_meetings(split=self.split, limit=limit, require_summary=True))
        raise ValueError(f"Dataset {self.dataset} not supported for semantic track")

    def _generate_receipt(self) -> BenchmarkReceipt:
        """Generate provenance receipt with semantic-specific mode info."""
        import hashlib

        # Get tool version
        try:
            from . import __version__
        except ImportError:
            __version__ = "unknown"

        # Generate config hash including mode for semantic track
        config_str = f"{self.track}:{self.dataset}:{self.split}:{self.mode}"
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

        # Get git commit if available
        git_commit = self._get_git_commit()

        return BenchmarkReceipt(
            tool_version=__version__,
            config_hash=config_hash,
            git_commit=git_commit,
            mode=self.mode,
        )

    def _load_gold_labels(self, meeting_id: str) -> dict[str, Any] | None:
        """Load gold labels from benchmarks/gold/semantic/{meeting_id}.json if exists."""
        gold_path = self._gold_dir / f"{meeting_id}.json"
        if gold_path.exists():
            try:
                with open(gold_path) as f:
                    data: dict[str, Any] = json.load(f)
                    return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load gold labels from {gold_path}: {e}")
        return None

    def _load_transcript_json(self, meeting_id: str) -> Transcript | None:
        """Load transcript from JSON file if available.

        Attempts to load from benchmarks/data/speaker_analytics_samples/{meeting_id}.json.
        Falls back to None if not available, in which case _evaluate_tags will
        create a synthetic single-segment transcript.

        Args:
            meeting_id: The meeting/sample identifier

        Returns:
            Transcript object if JSON file exists and is valid, None otherwise
        """
        from .writers import load_transcript_from_json

        transcript_path = self._samples_dir / f"{meeting_id}.json"
        if transcript_path.exists():
            try:
                return load_transcript_from_json(transcript_path)
            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.warning(f"Failed to load transcript from {transcript_path}: {e}")
        return None

    def _not_measured_tags(self, reason: str) -> dict[str, Any]:
        """Return a tags result dict with all metrics set to None."""
        return {
            "topic_precision": None,
            "topic_recall": None,
            "topic_f1": None,
            "risk_precision": None,
            "risk_recall": None,
            "risk_f1": None,
            "action_precision": None,
            "action_recall": None,
            "action_f1": None,
            "tags_reason": reason,
        }

    def _evaluate_tags(self, sample: EvalSample, gold: dict[str, Any] | None) -> dict[str, Any]:
        """Evaluate semantic tagging against gold labels.

        Returns metrics dict with topic_precision, topic_recall, topic_f1,
        risk_precision, risk_recall, risk_f1, action_precision, action_recall, action_f1.
        All values are None if no gold labels exist.

        Uses semantic_metrics module for deterministic scoring against gold
        schema fields (topics, risks, actions).
        """
        if gold is None:
            return self._not_measured_tags("no_gold_labels")

        # Load transcript and run annotator
        from .models import Segment, Transcript

        # Try to load actual transcript JSON with proper segments first
        transcript = self._load_transcript_json(sample.id)
        synthetic_segments = False

        if transcript is None:
            # Fall back to creating a minimal transcript from the sample's reference transcript
            transcript_text = sample.reference_transcript or ""
            if not transcript_text:
                return self._not_measured_tags("no_transcript")

            # Build a simple transcript with one segment
            # NOTE: This is a synthetic single-segment transcript. Gold labels with
            # segment_id > 0 will not match predicted segment IDs.
            segments = [Segment(id=0, start=0.0, end=1.0, text=transcript_text)]
            transcript = Transcript(
                file_name=f"{sample.id}.json",
                language="en",
                segments=segments,
            )
            synthetic_segments = True  # Flag for measurement integrity note

        # Run annotator
        annotated = self._annotator.annotate(transcript)
        semantic = (annotated.annotations or {}).get("semantic", {})

        # --- TOPICS ---
        # Gold schema uses topics[].label; annotator produces keywords + risk_tags
        gold_topics = [t.get("label") for t in (gold.get("topics") or []) if t.get("label")]

        # Best available predicted topic proxy: risk_tags are closer to coarse topics
        # than raw keywords, include both for broader vocabulary overlap
        pred_topics = list(
            dict.fromkeys((semantic.get("risk_tags") or []) + (semantic.get("keywords") or []))
        )

        topic_result = compute_topic_f1(pred_topics, gold_topics)
        topic_precision = topic_result["precision"] if topic_result else None
        topic_recall = topic_result["recall"] if topic_result else None
        topic_f1_val = topic_result["f1"] if topic_result else None

        topic_note = None
        if gold_topics and pred_topics and set(gold_topics).isdisjoint(set(pred_topics)):
            topic_note = "vocabulary_mismatch"

        # --- RISKS ---
        # Gold schema uses risks[].{type, severity, segment_id}
        # Annotator produces matches[].{risk, keyword, segment_id} - no severity
        gold_risks = gold.get("risks") or []

        # Map risk type names (annotator uses "churn_risk", gold may use "churn")
        risk_type_map = {"churn_risk": "churn", "escalation": "escalation", "pricing": "pricing"}

        pred_risks = []
        for m in semantic.get("matches") or []:
            r = m.get("risk")
            if not r:
                continue
            pred_risks.append(
                {
                    "type": risk_type_map.get(r, r),
                    "segment_id": m.get("segment_id"),
                    # No severity produced by KeywordSemanticAnnotator
                }
            )

        risk_result = compute_risk_metrics(pred_risks, gold_risks)
        risk_precision = risk_result["overall"]["precision"] if risk_result else None
        risk_recall = risk_result["overall"]["recall"] if risk_result else None
        risk_f1_val = risk_result["overall"]["f1"] if risk_result else None

        risk_note = None
        # If gold requires severity matching, note that we can't measure it
        if any(r.get("severity") for r in gold_risks):
            risk_note = "severity_not_measured"

        # --- ACTIONS ---
        # Gold schema uses actions[].{text, speaker_id?, segment_ids?}
        # Annotator uses same format: actions[].{text, speaker_id, segment_ids}
        gold_actions = gold.get("actions") or []
        pred_actions = semantic.get("actions") or []

        action_result = compute_action_metrics(pred_actions, gold_actions)
        action_precision = action_result["precision"] if action_result else None
        action_recall = action_result["recall"] if action_result else None
        # compute_action_metrics uses accuracy as harmonic mean (same as F1)
        action_f1_val = action_result["accuracy"] if action_result else None
        action_matched = action_result.get("matched_count") if action_result else None
        action_gold = action_result.get("gold_count") if action_result else None

        out: dict[str, Any] = {
            "topic_precision": topic_precision,
            "topic_recall": topic_recall,
            "topic_f1": topic_f1_val,
            "risk_precision": risk_precision,
            "risk_recall": risk_recall,
            "risk_f1": risk_f1_val,
            "action_precision": action_precision,
            "action_recall": action_recall,
            "action_f1": action_f1_val,
            "tags_reason": None,
        }

        # Add optional notes for known measurement gaps
        if synthetic_segments:
            out["tags_note"] = "synthetic_transcript_segments"
        if topic_note:
            out["topic_note"] = topic_note
        if risk_note:
            out["risk_note"] = risk_note
        if action_matched is not None:
            out["action_matched"] = action_matched
            out["action_gold"] = action_gold

        return out

    def _evaluate_summary(self, sample: EvalSample) -> dict[str, Any]:
        """Evaluate summary generation using Claude-as-judge.

        Returns metrics dict with faithfulness, coverage, clarity (0-10 scale).
        All values are None if API key is missing or evaluation fails.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {
                "faithfulness": None,
                "coverage": None,
                "clarity": None,
                "summary_reason": "missing_api_key",
            }

        if not sample.reference_summary:
            return {
                "faithfulness": None,
                "coverage": None,
                "clarity": None,
                "summary_reason": "no_reference_summary",
            }

        try:
            from anthropic import Anthropic
        except ImportError:
            return {
                "faithfulness": None,
                "coverage": None,
                "clarity": None,
                "summary_reason": "anthropic_not_installed",
            }

        # Get transcript context
        transcript_text = sample.reference_transcript or ""
        if not transcript_text:
            return {
                "faithfulness": None,
                "coverage": None,
                "clarity": None,
                "summary_reason": "no_transcript",
            }

        try:
            client = Anthropic(api_key=api_key)

            # Generate summary
            gen_prompt = f"""You are analyzing a meeting transcript. Generate a concise summary that captures:
- Main topics discussed
- Key decisions made
- Action items (if any)
- Important outcomes

Keep the summary focused and factual. Use 3-5 bullet points.

Transcript:
{transcript_text[:8000]}

Summary:"""

            gen_response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": gen_prompt}],
            )
            candidate_summary = gen_response.content[0].text

            # Judge summary
            judge_prompt = f"""You are evaluating a meeting summary against a reference summary.

Reference summary (ground truth):
---
{sample.reference_summary}
---

Candidate summary (to evaluate):
---
{candidate_summary}
---

Score the candidate on three dimensions (0-10 scale):

1. **Faithfulness** (0-10): Does the candidate avoid hallucinations? Does it only include information that's actually in the meeting?
   - 10 = completely faithful, no false information
   - 5 = some minor inaccuracies
   - 0 = major hallucinations or fabrications

2. **Coverage** (0-10): Does the candidate capture the main points from the reference?
   - 10 = covers all important points
   - 5 = misses about half the key information
   - 0 = misses most/all key information

3. **Clarity** (0-10): Is the candidate well-structured and easy to understand?
   - 10 = excellent organization and readability
   - 5 = acceptable but could be clearer
   - 0 = confusing or poorly structured

Respond in JSON format:
{{"faithfulness": <score 0-10>, "coverage": <score 0-10>, "clarity": <score 0-10>, "comments": "<1-2 sentence analysis>"}}"""

            judge_response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": judge_prompt}],
            )

            result = json.loads(judge_response.content[0].text)
            return {
                "faithfulness": result.get("faithfulness"),
                "coverage": result.get("coverage"),
                "clarity": result.get("clarity"),
                "summary_comments": result.get("comments", ""),
                "candidate_summary": candidate_summary,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse judge response: {e}")
            return {
                "faithfulness": None,
                "coverage": None,
                "clarity": None,
                "summary_reason": "json_parse_error",
            }
        except Exception as e:
            logger.warning(f"Summary evaluation failed: {e}")
            return {
                "faithfulness": None,
                "coverage": None,
                "clarity": None,
                "summary_reason": f"error: {str(e)[:100]}",
            }

    def evaluate_sample(self, sample: EvalSample) -> dict[str, Any]:
        """Evaluate a single sample based on the configured mode."""
        result: dict[str, Any] = {"id": sample.id}

        if self.mode in ("tags", "both"):
            gold = self._load_gold_labels(sample.id)
            tags_metrics = self._evaluate_tags(sample, gold)
            result.update(tags_metrics)

        if self.mode in ("summary", "both"):
            summary_metrics = self._evaluate_summary(sample)
            result.update(summary_metrics)

        return result

    def aggregate_metrics(self, sample_results: list[dict[str, Any]]) -> list[BenchmarkMetric]:
        """Aggregate per-sample metrics, handling None values properly."""
        if not sample_results:
            return []

        metrics: list[BenchmarkMetric] = []
        total_count = len(sample_results)

        def aggregate_metric(key: str, unit: str, description: str) -> BenchmarkMetric | None:
            """Compute average for a metric, excluding None values."""
            values = [r[key] for r in sample_results if r.get(key) is not None]
            measured_count = len(values)

            if measured_count == 0:
                # Find reason if available
                reasons = [
                    r.get(f"{key.split('_')[0]}_reason")
                    or r.get("tags_reason")
                    or r.get("summary_reason")
                    for r in sample_results
                    if r.get(key) is None
                ]
                reason = reasons[0] if reasons else "not_measured"
                return BenchmarkMetric(
                    name=key,
                    value=None,
                    unit=unit,
                    description=description,
                    measured_count=0,
                    total_count=total_count,
                    reason=reason,
                )

            avg_value = sum(values) / measured_count
            return BenchmarkMetric(
                name=key,
                value=avg_value,
                unit=unit,
                description=description,
                measured_count=measured_count,
                total_count=total_count,
            )

        # Tags metrics (if mode includes tags)
        if self.mode in ("tags", "both"):
            tag_metrics = [
                ("topic_precision", "", "Precision of keyword/topic detection"),
                ("topic_recall", "", "Recall of keyword/topic detection"),
                ("topic_f1", "", "F1 score for keyword/topic detection"),
                ("risk_precision", "", "Precision of risk tag detection"),
                ("risk_recall", "", "Recall of risk tag detection"),
                ("risk_f1", "", "F1 score for risk tag detection"),
                ("action_precision", "", "Precision of action item detection"),
                ("action_recall", "", "Recall of action item detection"),
                ("action_f1", "", "F1 score for action item detection"),
            ]
            for key, unit, desc in tag_metrics:
                m = aggregate_metric(key, unit, desc)
                if m is not None:
                    metrics.append(m)

        # Summary metrics (if mode includes summary)
        if self.mode in ("summary", "both"):
            summary_metrics_defs = [
                ("faithfulness", "/10", "Factual accuracy of generated summary"),
                ("coverage", "/10", "Completeness of key information in summary"),
                ("clarity", "/10", "Readability and coherence of summary"),
            ]
            for key, unit, desc in summary_metrics_defs:
                m = aggregate_metric(key, unit, desc)
                if m is not None:
                    metrics.append(m)

        return metrics


class EmotionBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner for emotion recognition evaluation.

    Evaluates categorical emotion classification on IEMOCAP dataset:
    1. Runs emotion recognition using wav2vec2 model
    2. Compares predicted labels with IEMOCAP ground truth
    3. Calculates accuracy and weighted F1 metrics

    Note:
        The wav2vec2 categorical model uses different label vocabulary than IEMOCAP.
        Labels are normalized for comparison (e.g., model's "fear" → IEMOCAP's "fearful").
        Some IEMOCAP labels (excited, frustrated, other) have no direct model equivalent.
    """

    # Map model output labels to IEMOCAP vocabulary
    # Model labels: angry, disgust, fear, happy, neutral, sad, surprised, calm
    # IEMOCAP labels: angry, happy, sad, neutral, excited, frustrated, fearful, surprised, disgusted, other
    LABEL_MAP: dict[str, str] = {
        "disgust": "disgusted",
        "fear": "fearful",
        "calm": "neutral",  # Best approximation
    }

    def get_samples(self, limit: int | None = None) -> list[EvalSample]:
        if self.dataset == "iemocap":
            return list(iter_iemocap_clips(limit=limit))
        raise ValueError(f"Dataset {self.dataset} not supported for emotion track")

    def evaluate_sample(self, sample: EvalSample) -> dict[str, Any]:
        import librosa

        from .emotion import get_emotion_recognizer

        # Load audio at 16kHz (wav2vec2 expected sample rate)
        try:
            audio, _ = librosa.load(sample.audio_path, sr=16000)
        except Exception as e:
            logger.error(f"Failed to load audio {sample.audio_path}: {e}")
            raise

        # Get recognizer and predict
        recognizer = get_emotion_recognizer()
        result = recognizer.extract_emotion_categorical(audio, sr=16000)

        predicted_raw = result["categorical"]["primary"]

        # Reference is a list of strings (IEMOCAP may have multiple annotators)
        reference_list = sample.reference_emotions or []
        reference = reference_list[0] if reference_list else None

        is_correct = False
        predicted_normalized = None

        if predicted_raw:
            # Normalize predicted label to IEMOCAP vocabulary
            p = predicted_raw.lower()
            predicted_normalized = self.LABEL_MAP.get(p, p)

            if reference:
                r = reference.lower()
                is_correct = predicted_normalized == r

        return {
            "id": sample.id,
            "predicted": predicted_normalized,
            "predicted_raw": predicted_raw,
            "reference": reference,
            "correct": is_correct,
            "confidence": result["categorical"]["confidence"],
            "all_scores": result["categorical"]["all_scores"],
        }

    def aggregate_metrics(self, sample_results: list[dict[str, Any]]) -> list[BenchmarkMetric]:
        if not sample_results:
            return []

        correct = sum(1 for r in sample_results if r["correct"])
        total = len(sample_results)
        accuracy = correct / total if total > 0 else 0.0

        metrics = [
            BenchmarkMetric(
                name="accuracy",
                value=accuracy * 100,
                unit="%",
                description="Classification accuracy",
            ),
        ]

        # Compute weighted F1 if sklearn is available
        try:
            from sklearn.metrics import f1_score

            # Extract predictions and references (skip samples with missing reference)
            y_true = [r["reference"] for r in sample_results if r["reference"]]
            y_pred = [r["predicted"] for r in sample_results if r["reference"]]

            if y_true and y_pred:
                # Use weighted average to account for class imbalance
                f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                metrics.append(
                    BenchmarkMetric(
                        name="f1_weighted",
                        value=f1_weighted * 100,
                        unit="%",
                        description="Weighted F1 score (accounts for class imbalance)",
                    )
                )
        except ImportError:
            logger.debug("sklearn not available; skipping F1 computation")

        return metrics


# =============================================================================
# Runner Factory
# =============================================================================


def get_benchmark_runner(
    track: str,
    dataset: str,
    split: str = "test",
    mode: SemanticMode | None = None,
) -> BenchmarkRunner:
    """Get the appropriate benchmark runner for a track.

    Args:
        track: Benchmark track name (asr, diarization, streaming, semantic, emotion)
        dataset: Dataset to evaluate on
        split: Dataset split (train/dev/test)
        mode: Semantic evaluation mode (only used for semantic track)

    Returns:
        Configured benchmark runner instance
    """
    if track not in BENCHMARK_TRACKS:
        raise ValueError(f"Unknown track: {track}. Available: {list(BENCHMARK_TRACKS.keys())}")

    if track == "asr":
        return ASRBenchmarkRunner(track=track, dataset=dataset, split=split)
    elif track == "diarization":
        return DiarizationBenchmarkRunner(track=track, dataset=dataset, split=split)
    elif track == "streaming":
        return StreamingBenchmarkRunner(track=track, dataset=dataset, split=split)
    elif track == "semantic":
        return SemanticBenchmarkRunner(
            track=track, dataset=dataset, split=split, mode=mode or "tags"
        )
    elif track == "emotion":
        return EmotionBenchmarkRunner(track=track, dataset=dataset, split=split)
    else:
        raise ValueError(f"Unknown track: {track}")


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


def _perform_dry_run_validation(track: str, dataset: str) -> int:
    """Perform dry-run validation for a benchmark dataset.

    Validates manifest and configuration without running actual benchmarks.
    Reports validation status, sample counts, and staging status.

    Output format:
        Dry-run validation for {track}/{dataset}:
        [checkmark] Manifest found: {manifest_path}
        [checkmark] Schema version: {version}
        [checkmark] Samples defined: {count}
        [warning] Audio files staged: {staged_count}/{total_count}

        To stage audio: {staging_command}

    Args:
        track: Benchmark track name
        dataset: Dataset name

    Returns:
        Exit code (0 = manifest valid, 1 = manifest invalid/missing)
    """
    import csv

    # Unicode symbols for output
    CHECK = "\u2713"  # checkmark
    WARN = "\u26a0"  # warning sign

    project_root = Path(__file__).parent.parent

    print(f"Dry-run validation for {track}/{dataset}:")

    # Find manifest path based on track/dataset
    if dataset == "commonvoice_en_smoke":
        manifest_dir = project_root / "benchmarks" / "datasets" / "asr" / "commonvoice_en_smoke"
        manifest_path = manifest_dir / "manifest.json"
    elif dataset == "librispeech":
        # LibriSpeech doesn't use a manifest file in the same way
        benchmarks_root = get_benchmarks_root()
        librispeech_path = benchmarks_root / "librispeech" / "LibriSpeech"
        if librispeech_path.exists():
            print(f"{CHECK} Manifest found: {librispeech_path} (directory-based)")
            print(f"{CHECK} Schema version: N/A (file-based dataset)")
            # Count test-clean samples
            test_clean = librispeech_path / "test-clean"
            if test_clean.exists():
                flac_files = list(test_clean.rglob("*.flac"))
                print(f"{CHECK} Samples defined: {len(flac_files)}")
                print(f"{CHECK} Audio files staged: {len(flac_files)}/{len(flac_files)}")
            else:
                print(f"{WARN} Samples defined: unknown (test-clean not found)")
                print(f"{WARN} Audio files staged: 0/unknown")
            print()
            print(
                f"To stage audio: Download LibriSpeech test-clean to {benchmarks_root}/librispeech/"
            )
            return 0
        else:
            print(f"{WARN} Manifest found: No (expected directory at {librispeech_path})")
            print()
            print(
                f"To stage audio: Download LibriSpeech test-clean to {benchmarks_root}/librispeech/"
            )
            return 1
    elif dataset == "ami":
        benchmarks_root = get_benchmarks_root()
        ami_path = benchmarks_root / "ami"
        if ami_path.exists():
            print(f"{CHECK} Manifest found: {ami_path} (directory-based)")
            print(f"{CHECK} Schema version: N/A (file-based dataset)")
            # Count audio files
            audio_files = list(ami_path.rglob("*.wav")) + list(ami_path.rglob("*.mp3"))
            print(f"{CHECK} Samples defined: {len(audio_files)}")
            print(f"{CHECK} Audio files staged: {len(audio_files)}/{len(audio_files)}")
            print()
            print("To stage audio: See docs/AMI_SETUP.md")
            return 0
        else:
            print(f"{WARN} Manifest found: No (expected directory at {ami_path})")
            print()
            print("To stage audio: See docs/AMI_SETUP.md")
            return 1
    elif dataset == "iemocap":
        benchmarks_root = get_benchmarks_root()
        iemocap_path = benchmarks_root / "iemocap"
        if iemocap_path.exists():
            print(f"{CHECK} Manifest found: {iemocap_path} (directory-based)")
            print(f"{CHECK} Schema version: N/A (file-based dataset)")
            audio_files = list(iemocap_path.rglob("*.wav"))
            print(f"{CHECK} Samples defined: {len(audio_files)}")
            print(f"{CHECK} Audio files staged: {len(audio_files)}/{len(audio_files)}")
            print()
            print("To stage audio: See docs/IEMOCAP_SETUP.md")
            return 0
        else:
            print(f"{WARN} Manifest found: No (expected directory at {iemocap_path})")
            print()
            print("To stage audio: See docs/IEMOCAP_SETUP.md")
            return 1
    else:
        print(f"{WARN} Manifest found: No (unsupported dataset: {dataset})")
        return 1

    # For manifest-based datasets (commonvoice_en_smoke)
    if not manifest_path.exists():
        print(f"{WARN} Manifest found: No (expected at {manifest_path})")
        return 1

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        print(f"{CHECK} Manifest found: {manifest_path}")
        print(f"{WARN} Schema version: ERROR (invalid JSON: {e})")
        return 1

    print(f"{CHECK} Manifest found: {manifest_path}")

    schema_version = manifest.get("schema_version", "unknown")
    print(f"{CHECK} Schema version: {schema_version}")

    # Determine sample count
    samples_in_manifest = manifest.get("samples", [])
    sample_source = manifest.get("sample_source", {})
    selection_file = sample_source.get("file")

    total_count = 0
    staged_count = 0

    if samples_in_manifest:
        # Samples are directly in the manifest
        total_count = len(samples_in_manifest)
        for sample in samples_in_manifest:
            audio_rel = sample.get("audio")
            if audio_rel:
                audio_path = manifest_dir / audio_rel
                if audio_path.exists():
                    staged_count += 1
        print(f"{CHECK} Samples defined: {total_count}")
    elif selection_file:
        # Samples are in a selection file (like commonvoice_en_smoke)
        selection_path = manifest_dir / selection_file
        if selection_path.exists():
            try:
                with open(selection_path, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    total_count = len(rows)

                    # Check for staged audio files
                    audio_dir = manifest_dir / "audio"
                    for row in rows:
                        clip_id = row.get("clip_id", "")
                        # Check common audio extensions
                        for ext in [".mp3", ".wav", ".flac"]:
                            audio_path = audio_dir / f"{clip_id}{ext}"
                            if audio_path.exists():
                                staged_count += 1
                                break
            except Exception as e:
                print(f"{WARN} Samples defined: ERROR (failed to read {selection_file}: {e})")
                return 1
            print(f"{CHECK} Samples defined: {total_count} (from {selection_file})")
        else:
            print(f"{WARN} Samples defined: ERROR (selection file not found: {selection_path})")
            return 1
    else:
        # Try to get count from meta
        meta = manifest.get("meta", {})
        total_count = meta.get("sample_count", 0)
        print(f"{CHECK} Samples defined: {total_count} (from meta.sample_count)")

    # Report staging status
    if staged_count == total_count and total_count > 0:
        print(f"{CHECK} Audio files staged: {staged_count}/{total_count}")
    else:
        print(f"{WARN} Audio files staged: {staged_count}/{total_count}")

    print()

    # Provide staging command
    if dataset == "commonvoice_en_smoke":
        staging_script = project_root / "benchmarks" / "scripts" / "stage_commonvoice.py"
        print(f"To stage audio: python {staging_script}")
    else:
        print(f"To stage audio: See documentation for {dataset}")

    return 0


def _print_gate_report(comparison: BaselineComparison) -> None:
    """Print a detailed gate mode regression report.

    Outputs a clear, CI-friendly report showing:
    - Which metrics passed/failed
    - Regression percentages vs thresholds
    - Overall gate status

    Args:
        comparison: The baseline comparison results
    """
    print("=" * 70)
    print("REGRESSION GATE CHECK")
    print("=" * 70)
    print(f"\nTrack: {comparison.track}")
    print(f"Dataset: {comparison.dataset}")
    print(f"Baseline: {comparison.baseline_created_at}")

    # Separate failed metrics for summary reporting
    failed_metrics = [r for r in comparison.results if not r.passed]

    print("\n" + "-" * 70)
    print(
        f"{'Metric':<20} {'Current':<12} {'Baseline':<12} {'Change':<12} "
        f"{'Threshold':<12} {'Status'}"
    )
    print("-" * 70)

    for r in comparison.results:
        # Format regression as percentage with sign
        if r.regression == float("inf"):
            change_str = "+inf%"
        elif r.regression == float("-inf"):
            change_str = "-inf%"
        else:
            change_str = f"{r.regression:+.1%}"

        # Format threshold
        threshold_str = f"{r.threshold:.0%}" if r.threshold is not None else "N/A"

        # Status indicator
        if r.passed:
            status = "\u2713 PASS"  # checkmark
        else:
            status = "\u2717 FAIL"  # cross mark

        print(
            f"{r.metric_name:<20} {r.current_value:<12.4f} {r.baseline_value:<12.4f} "
            f"{change_str:<12} {threshold_str:<12} {status}"
        )

    print("-" * 70)

    # Summary
    if comparison.overall_passed:
        print("\n\u2713 GATE PASSED: All metrics within thresholds.")
    else:
        print(f"\n\u2717 GATE FAILED: {len(failed_metrics)} metric(s) exceeded threshold.")
        print("\nFailed metrics:")
        for r in failed_metrics:
            if r.regression == float("inf"):
                change_str = "+inf%"
            else:
                change_str = f"{r.regression:+.1%}"
            threshold_str = f"{r.threshold:.0%}" if r.threshold is not None else "N/A"
            print(
                f"  - {r.metric_name}: {r.current_value:.4f} vs baseline {r.baseline_value:.4f} "
                f"({change_str} change, threshold {threshold_str})"
            )


def handle_benchmark_run(
    track: str,
    dataset: str | None,
    split: str,
    limit: int | None,
    output: Path | None,
    verbose: bool,
    mode: SemanticMode | None = None,
    dry_run: bool = False,
    gate: bool = False,
    threshold_overrides: dict[str, float] | None = None,
) -> int:
    """Handle 'benchmark run' command - run a specific benchmark.

    Args:
        track: Benchmark track name (asr, diarization, streaming, semantic, emotion)
        dataset: Dataset name (or None for default)
        split: Dataset split (train/dev/test)
        limit: Sample limit
        output: Optional output path for results JSON
        verbose: Show detailed progress
        mode: Semantic evaluation mode (only for semantic track)
        dry_run: If True, validate manifest and configuration without running benchmarks
        gate: If True, compare against baseline and fail on regression
        threshold_overrides: Optional per-metric threshold overrides (e.g., {"wer": 0.05})

    Returns:
        Exit code (0 = success, 1 = error/regression in gate mode)
    """
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

    # Handle dry-run mode - validate manifest and exit early
    if dry_run:
        return _perform_dry_run_validation(track, dataset)

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
    if track == "semantic" and mode:
        print(f"  Mode: {mode}")
    print()

    try:
        runner = get_benchmark_runner(track, dataset, split, mode=mode)
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
            if metric.value is not None:
                coverage_info = ""
                if metric.measured_count is not None and metric.total_count is not None:
                    coverage_info = f" [{metric.measured_count}/{metric.total_count} measured]"
                print(f"  {metric.name}: {metric.value:.4f}{metric.unit}{coverage_info}")
            else:
                reason_info = f" (reason: {metric.reason})" if metric.reason else ""
                print(f"  {metric.name}: None{reason_info}")
            if metric.description:
                print(f"    ({metric.description})")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors[:5]:
                print(f"  - {error}")
            if len(result.errors) > 5:
                print(f"  ... and {len(result.errors) - 5} more")

        # Display receipt for provenance tracking
        if result.receipt:
            print("\nReceipt:")
            print(f"  tool_version: {result.receipt.tool_version}")
            print(f"  config_hash: {result.receipt.config_hash}")
            if result.receipt.git_commit:
                print(f"  git_commit: {result.receipt.git_commit}")
            if result.receipt.mode:
                print(f"  mode: {result.receipt.mode}")

        # Save results if output specified
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to: {output}")

        # Gate mode: compare against baseline and fail on regression
        if gate:
            baseline = load_baseline(track, dataset)
            if baseline is None:
                baseline_path = get_baseline_path(track, dataset)
                print(f"\nWarning: No baseline found at {baseline_path}", file=sys.stderr)
                print("Gate mode requires a baseline. Run 'benchmark save-baseline' first.")
                print("Continuing without gate check...")
                return 0 if result.samples_failed == 0 else 1

            # Apply threshold overrides to baseline metrics
            if threshold_overrides:
                for metric_name, threshold in threshold_overrides.items():
                    if metric_name in baseline.metrics:
                        baseline.metrics[metric_name].threshold = threshold

            comparison = compare_with_baseline(result, baseline, mode="gate")

            print("\n")
            _print_gate_report(comparison)

            if not comparison.overall_passed:
                return 1

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

    # Show baseline status
    baselines_dir = get_baselines_dir()
    print(f"\nBaselines directory: {baselines_dir}")
    if baselines_dir.exists():
        baseline_files = list(baselines_dir.glob("*/*.json"))
        print(f"  Baseline files: {len(baseline_files)}")
        for bf in sorted(baseline_files):
            track = bf.parent.name
            dataset = bf.stem
            print(f"    [{track}] {dataset}")
    else:
        print("  No baselines directory found")

    print("\nTo stage a dataset, see the setup documentation:")
    for name, info in datasets.items():
        if not info["available"]:
            print(f"  {name}: {info['setup_doc']}")

    return 0


def handle_baseline_save(
    track: str,
    dataset: str | None,
    split: str,
    limit: int | None,
    output: Path | None,
    verbose: bool,
    thresholds: dict[str, float] | None = None,
    mode: SemanticMode | None = None,
) -> int:
    """Handle 'benchmark save-baseline' command - run benchmark and save as baseline.

    Args:
        track: Benchmark track (asr, diarization, etc.)
        dataset: Dataset name (or None for default)
        split: Dataset split (train/dev/test)
        limit: Sample limit
        output: Optional output path for results JSON
        verbose: Show detailed progress
        thresholds: Optional dict of metric_name -> threshold
        mode: Semantic evaluation mode

    Returns:
        Exit code (0 = success, 1 = error)
    """
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

    print(f"Running benchmark to create baseline: {track}")
    print(f"  Dataset: {dataset}")
    print(f"  Split: {split}")
    if limit:
        print(f"  Limit: {limit} samples")
    print()

    try:
        runner = get_benchmark_runner(track, dataset, split, mode=mode)
        result = runner.run(limit=limit, verbose=verbose)

        if result.samples_evaluated == 0:
            print("Error: No samples evaluated. Cannot create baseline.", file=sys.stderr)
            return 1

        # Create and save baseline
        baseline = create_baseline_from_result(result, thresholds=thresholds)
        baseline_path = save_baseline(baseline)

        print("=" * 60)
        print("BASELINE SAVED")
        print("=" * 60)
        print(f"\nTrack: {baseline.track}")
        print(f"Dataset: {baseline.dataset}")
        print(f"Samples evaluated: {result.samples_evaluated}")
        print(f"File: {baseline_path}")

        print("\nBaseline metrics:")
        for name, m in baseline.metrics.items():
            threshold_str = f" (threshold: {m.threshold:.1%})" if m.threshold else ""
            print(f"  {name}: {m.value:.4f}{m.unit}{threshold_str}")

        print("\nReceipt:")
        print(f"  tool_version: {baseline.receipt.tool_version}")
        if baseline.receipt.model:
            print(f"  model: {baseline.receipt.model}")
        if baseline.receipt.device:
            print(f"  device: {baseline.receipt.device}")
        if baseline.receipt.git_commit:
            print(f"  git_commit: {baseline.receipt.git_commit}")

        # Optionally save full results
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nFull results saved to: {output}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error creating baseline: {e}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def handle_baseline_compare(
    track: str,
    dataset: str | None,
    split: str,
    limit: int | None,
    output: Path | None,
    verbose: bool,
    gate_mode: bool = False,
    mode: SemanticMode | None = None,
    threshold_overrides: dict[str, float] | None = None,
) -> int:
    """Handle 'benchmark compare' command - run benchmark and compare with baseline.

    Args:
        track: Benchmark track (asr, diarization, etc.)
        dataset: Dataset name (or None for default)
        split: Dataset split (train/dev/test)
        limit: Sample limit
        output: Optional output path for comparison JSON
        verbose: Show detailed progress
        gate_mode: If True, exit with error on regression
        mode: Semantic evaluation mode
        threshold_overrides: Optional per-metric threshold overrides (e.g., {"wer": 0.05})

    Returns:
        Exit code (0 = success/no regression, 1 = error/regression in gate mode)
    """
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

    # Load baseline
    baseline = load_baseline(track, dataset)
    if baseline is None:
        baseline_path = get_baseline_path(track, dataset)
        print(f"Error: No baseline found at {baseline_path}", file=sys.stderr)
        print("Run 'benchmark save-baseline' first to create a baseline.", file=sys.stderr)
        return 1

    # Apply threshold overrides to baseline metrics
    if threshold_overrides:
        for metric_name, threshold in threshold_overrides.items():
            if metric_name in baseline.metrics:
                baseline.metrics[metric_name].threshold = threshold

    print(f"Comparing against baseline: {track}/{dataset}")
    print(f"  Baseline created: {baseline.created_at}")
    print(f"  Split: {split}")
    if limit:
        print(f"  Limit: {limit} samples")
    print(f"  Mode: {'gate' if gate_mode else 'report'}")
    print()

    try:
        runner = get_benchmark_runner(track, dataset, split, mode=mode)
        result = runner.run(limit=limit, verbose=verbose)

        if result.samples_evaluated == 0:
            print("Error: No samples evaluated.", file=sys.stderr)
            return 1

        # Compare with baseline
        comparison_mode: Literal["report", "gate"] = "gate" if gate_mode else "report"
        comparison = compare_with_baseline(result, baseline, mode=comparison_mode)

        print("=" * 60)
        print("BASELINE COMPARISON")
        print("=" * 60)
        print(f"\nTrack: {comparison.track}")
        print(f"Dataset: {comparison.dataset}")
        print(f"Baseline: {comparison.baseline_created_at}")
        print(f"Mode: {comparison.mode}")

        print("\nMetric Comparison:")
        print("-" * 60)
        print(f"{'Metric':<20} {'Current':<12} {'Baseline':<12} {'Regression':<12} {'Status'}")
        print("-" * 60)

        for r in comparison.results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            regression_str = f"{r.regression:+.1%}"
            threshold_str = f"(≤{r.threshold:.0%})" if r.threshold else ""
            print(
                f"{r.metric_name:<20} {r.current_value:<12.4f} "
                f"{r.baseline_value:<12.4f} {regression_str:<12} {status} {threshold_str}"
            )

        print("-" * 60)

        overall_status = "PASSED" if comparison.overall_passed else "FAILED"
        print(f"\nOverall: {overall_status}")

        if not comparison.overall_passed:
            failed_metrics = [r.metric_name for r in comparison.results if not r.passed]
            print(f"Failed metrics: {', '.join(failed_metrics)}")

        # Save comparison if output specified
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(comparison.to_dict(), f, indent=2)
            print(f"\nComparison saved to: {output}")

        # Return appropriate exit code
        if gate_mode and not comparison.overall_passed:
            print("\nGate mode: Failing due to regression.")
            return 1

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error running comparison: {e}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


def handle_baseline_list() -> int:
    """Handle 'benchmark baselines' command - list all baselines."""
    baselines_dir = get_baselines_dir()
    print("Stored Baselines")
    print("=" * 60)
    print(f"\nBaselines directory: {baselines_dir}")

    if not baselines_dir.exists():
        print("\nNo baselines directory found.")
        print("Run 'benchmark save-baseline' to create baselines.")
        return 0

    baseline_files = list(baselines_dir.glob("*/*.json"))
    if not baseline_files:
        print("\nNo baseline files found.")
        print("Run 'benchmark save-baseline' to create baselines.")
        return 0

    print(f"\nFound {len(baseline_files)} baseline(s):\n")

    for bf in sorted(baseline_files):
        baseline = load_baseline(bf.parent.name, bf.stem)
        if baseline is None:
            print(f"  [{bf.parent.name}] {bf.stem}: <error loading>")
            continue

        print(f"  [{baseline.track}] {baseline.dataset}")
        print(f"    Created: {baseline.created_at}")
        print(f"    Version: {baseline.receipt.tool_version}")
        if baseline.metrics:
            print(f"    Metrics: {', '.join(baseline.metrics.keys())}")
        print()

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

    p_run.add_argument(
        "--mode",
        "-m",
        choices=["tags", "summary", "both"],
        default="tags",
        help=(
            "Semantic evaluation mode (only for --track semantic). "
            "'tags' = deterministic keyword/risk/action detection (CI-friendly), "
            "'summary' = LLM-based summary evaluation (requires ANTHROPIC_API_KEY), "
            "'both' = run both evaluations. Default: tags"
        ),
    )

    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest and configuration without running benchmarks.",
    )

    p_run.add_argument(
        "--gate",
        action="store_true",
        help=(
            "Gate mode: compare results against baseline and exit with error "
            "if any metric exceeds its regression threshold. Use for CI gates."
        ),
    )

    p_run.add_argument(
        "--threshold",
        type=str,
        action="append",
        metavar="METRIC=VALUE",
        help=(
            "Override regression threshold for a metric (e.g., --threshold wer=0.05). "
            "Can be specified multiple times. Value is a decimal (0.05 = 5%%). "
            "Only used with --gate."
        ),
    )

    # benchmark baselines - list stored baselines
    benchmark_subparsers.add_parser(
        "baselines",
        help="List stored baselines.",
    )

    # benchmark save-baseline - run benchmark and save as baseline
    p_save = benchmark_subparsers.add_parser(
        "save-baseline",
        help="Run benchmark and save results as baseline for future comparisons.",
    )

    p_save.add_argument(
        "--track",
        "-t",
        choices=list(BENCHMARK_TRACKS.keys()),
        required=True,
        help="Evaluation track to run.",
    )

    p_save.add_argument(
        "--dataset",
        "-d",
        help="Dataset to evaluate on (default: track-specific).",
    )

    p_save.add_argument(
        "--split",
        "-s",
        default="test",
        help="Dataset split to use (default: test).",
    )

    p_save.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (for quick testing).",
    )

    p_save.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for full benchmark results JSON (baseline saved separately).",
    )

    p_save.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress.",
    )

    p_save.add_argument(
        "--threshold",
        type=str,
        action="append",
        metavar="METRIC=VALUE",
        help=(
            "Set regression threshold for a metric (e.g., --threshold wer=0.10). "
            "Can be specified multiple times. Value is a decimal (0.10 = 10%%)."
        ),
    )

    p_save.add_argument(
        "--mode",
        "-m",
        choices=["tags", "summary", "both"],
        default="tags",
        help="Semantic evaluation mode (only for --track semantic).",
    )

    # benchmark compare - run benchmark and compare with baseline
    p_compare = benchmark_subparsers.add_parser(
        "compare",
        help="Run benchmark and compare results against stored baseline.",
    )

    p_compare.add_argument(
        "--track",
        "-t",
        choices=list(BENCHMARK_TRACKS.keys()),
        required=True,
        help="Evaluation track to run.",
    )

    p_compare.add_argument(
        "--dataset",
        "-d",
        help="Dataset to evaluate on (default: track-specific).",
    )

    p_compare.add_argument(
        "--split",
        "-s",
        default="test",
        help="Dataset split to use (default: test).",
    )

    p_compare.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (for quick testing).",
    )

    p_compare.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for comparison results JSON.",
    )

    p_compare.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress.",
    )

    p_compare.add_argument(
        "--gate",
        action="store_true",
        help="Gate mode: exit with error if regression exceeds threshold.",
    )

    p_compare.add_argument(
        "--threshold",
        type=str,
        action="append",
        metavar="METRIC=VALUE",
        help=(
            "Override regression threshold for a metric (e.g., --threshold wer=0.05). "
            "Can be specified multiple times. Value is a decimal (0.05 = 5%%)."
        ),
    )

    p_compare.add_argument(
        "--mode",
        "-m",
        choices=["tags", "summary", "both"],
        default="tags",
        help="Semantic evaluation mode (only for --track semantic).",
    )

    # Default action if no subcommand given - show help
    p_benchmark.set_defaults(benchmark_action=None)

    return p_benchmark


def _parse_thresholds(threshold_args: list[str] | None) -> dict[str, float]:
    """Parse threshold arguments (--threshold metric=value) into a dict."""
    if not threshold_args:
        return {}

    thresholds = {}
    for arg in threshold_args:
        if "=" not in arg:
            logger.warning(f"Invalid threshold format: {arg} (expected METRIC=VALUE)")
            continue
        metric, value_str = arg.split("=", 1)
        try:
            thresholds[metric.strip()] = float(value_str.strip())
        except ValueError:
            logger.warning(f"Invalid threshold value: {value_str} (expected float)")

    return thresholds


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

    if args.benchmark_action == "baselines":
        return handle_baseline_list()

    if args.benchmark_action == "run":
        threshold_overrides = _parse_thresholds(getattr(args, "threshold", None))
        return handle_benchmark_run(
            track=args.track,
            dataset=args.dataset,
            split=args.split,
            limit=args.limit,
            output=args.output,
            verbose=args.verbose,
            mode=getattr(args, "mode", None),
            dry_run=getattr(args, "dry_run", False),
            gate=getattr(args, "gate", False),
            threshold_overrides=threshold_overrides if threshold_overrides else None,
        )

    if args.benchmark_action == "save-baseline":
        thresholds = _parse_thresholds(getattr(args, "threshold", None))
        return handle_baseline_save(
            track=args.track,
            dataset=args.dataset,
            split=args.split,
            limit=args.limit,
            output=args.output,
            verbose=args.verbose,
            thresholds=thresholds if thresholds else None,
            mode=getattr(args, "mode", None),
        )

    if args.benchmark_action == "compare":
        threshold_overrides = _parse_thresholds(getattr(args, "threshold", None))
        return handle_baseline_compare(
            track=args.track,
            dataset=args.dataset,
            split=args.split,
            limit=args.limit,
            output=args.output,
            verbose=args.verbose,
            gate_mode=getattr(args, "gate", False),
            mode=getattr(args, "mode", None),
            threshold_overrides=threshold_overrides if threshold_overrides else None,
        )

    # Unknown action
    print(f"Unknown benchmark action: {args.benchmark_action}", file=sys.stderr)
    return 1
