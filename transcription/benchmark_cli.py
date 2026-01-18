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
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .benchmark.semantic_metrics import (
    compute_action_metrics,
    compute_risk_metrics,
    compute_topic_f1,
)
from .benchmarks import (
    EvalSample,
    get_benchmarks_root,
    iter_ami_meetings,
    iter_iemocap_clips,
    iter_librispeech,
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
    value: float | None
    unit: str = ""
    description: str = ""
    measured_count: int | None = None
    total_count: int | None = None
    reason: str | None = None  # Why metric is None (e.g., "missing_api_key")


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
        self._gold_dir = get_benchmarks_root() / "gold" / "semantic"

    def get_samples(self, limit: int | None = None) -> list[EvalSample]:
        if self.dataset == "ami":
            return list(iter_ami_meetings(split=self.split, limit=limit, require_summary=True))
        raise ValueError(f"Dataset {self.dataset} not supported for semantic track")

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

        # Create a minimal transcript from the sample's reference transcript
        # In a full implementation, we'd load the actual JSON transcript with real segments
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
        from .emotion import get_emotion_recognizer
        import librosa

        # Load audio
        try:
            audio, _ = librosa.load(sample.audio_path, sr=16000)
        except Exception as e:
            logger.error(f"Failed to load audio {sample.audio_path}: {e}")
            raise

        # Get recognizer and predict
        recognizer = get_emotion_recognizer()
        result = recognizer.extract_emotion_categorical(audio, sr=16000)

        predicted = result["categorical"]["primary"]

        # Reference is a list of strings
        reference_list = sample.reference_emotions or []
        reference = reference_list[0] if reference_list else None

        is_correct = False

        if reference and predicted:
            # Normalize labels for comparison
            p = predicted.lower()
            r = reference.lower()

            # Map model labels to IEMOCAP labels
            if p == "disgust":
                p = "disgusted"

            is_correct = p == r

        return {
            "id": sample.id,
            "predicted": predicted,
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


def handle_benchmark_run(
    track: str,
    dataset: str | None,
    split: str,
    limit: int | None,
    output: Path | None,
    verbose: bool,
    mode: SemanticMode | None = None,
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
            mode=getattr(args, "mode", None),
        )

    # Unknown action
    print(f"Unknown benchmark action: {args.benchmark_action}", file=sys.stderr)
    return 1
