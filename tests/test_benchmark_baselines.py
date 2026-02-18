"""
Tests for benchmark baseline functionality (#137).

Tests cover:
- BaselineFile data structure serialization/deserialization
- Baseline creation from benchmark results
- Baseline comparison logic with threshold checking
- Baseline I/O (load/save)
- Threshold parsing helper
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from slower_whisper.pipeline.benchmark_cli import (
    BASELINE_SCHEMA_VERSION,
    DEFAULT_REGRESSION_THRESHOLD,
    BaselineFile,
    BaselineMetric,
    BaselineReceipt,
    BenchmarkMetric,
    BenchmarkResult,
    _parse_thresholds,
    compare_with_baseline,
    create_baseline_from_result,
    get_baseline_path,
    load_baseline,
    save_baseline,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_baseline() -> BaselineFile:
    """Create a sample baseline for testing."""
    return BaselineFile(
        schema_version=BASELINE_SCHEMA_VERSION,
        track="asr",
        dataset="librispeech",
        created_at="2026-01-07T12:00:00",
        metrics={
            "wer": BaselineMetric(value=4.5, unit="%", threshold=0.10),
            "cer": BaselineMetric(value=1.2, unit="%"),
        },
        receipt=BaselineReceipt(
            tool_version="1.9.2",
            model="large-v3",
            device="cuda",
            compute_type="float16",
            git_commit="abc1234",
        ),
    )


@pytest.fixture
def sample_benchmark_result() -> BenchmarkResult:
    """Create a sample benchmark result for testing."""
    return BenchmarkResult(
        track="asr",
        dataset="librispeech",
        split="test",
        samples_evaluated=100,
        samples_failed=2,
        metrics=[
            BenchmarkMetric(name="wer", value=4.8, unit="%", description="Word Error Rate"),
            BenchmarkMetric(name="cer", value=1.1, unit="%", description="Character Error Rate"),
        ],
        timestamp="2026-01-15T14:30:00",
        system_info={"cuda_device": "NVIDIA RTX 4090"},
        config={"model": "large-v3", "compute_type": "float16"},
    )


# =============================================================================
# BaselineFile Tests
# =============================================================================


class TestBaselineFile:
    """Tests for BaselineFile dataclass."""

    def test_to_dict(self, sample_baseline: BaselineFile) -> None:
        """BaselineFile.to_dict produces correct JSON-serializable dict."""
        data = sample_baseline.to_dict()

        assert data["schema_version"] == BASELINE_SCHEMA_VERSION
        assert data["track"] == "asr"
        assert data["dataset"] == "librispeech"
        assert data["created_at"] == "2026-01-07T12:00:00"

        # Check metrics
        assert "wer" in data["metrics"]
        assert data["metrics"]["wer"]["value"] == 4.5
        assert data["metrics"]["wer"]["unit"] == "%"
        assert data["metrics"]["wer"]["threshold"] == 0.10

        # cer has no threshold
        assert "cer" in data["metrics"]
        assert "threshold" not in data["metrics"]["cer"]

        # Check receipt
        assert data["receipt"]["tool_version"] == "1.9.2"
        assert data["receipt"]["model"] == "large-v3"

    def test_from_dict(self) -> None:
        """BaselineFile.from_dict correctly deserializes JSON data."""
        data: dict[str, Any] = {
            "schema_version": 1,
            "track": "diarization",
            "dataset": "ami",
            "created_at": "2026-01-08T10:00:00",
            "metrics": {
                "der": {"value": 15.2, "unit": "%", "threshold": 0.15},
                "jer": {"value": 22.1, "unit": "%"},
            },
            "receipt": {
                "tool_version": "2.0.0",
                "model": "",
                "device": "cpu",
            },
        }

        baseline = BaselineFile.from_dict(data)

        assert baseline.schema_version == 1
        assert baseline.track == "diarization"
        assert baseline.dataset == "ami"
        assert baseline.metrics["der"].value == 15.2
        assert baseline.metrics["der"].threshold == 0.15
        assert baseline.metrics["jer"].value == 22.1
        assert baseline.metrics["jer"].threshold is None
        assert baseline.receipt.tool_version == "2.0.0"
        assert baseline.receipt.device == "cpu"

    def test_roundtrip(self, sample_baseline: BaselineFile) -> None:
        """to_dict -> from_dict preserves all data."""
        data = sample_baseline.to_dict()
        json_str = json.dumps(data)
        loaded_data = json.loads(json_str)
        restored = BaselineFile.from_dict(loaded_data)

        assert restored.schema_version == sample_baseline.schema_version
        assert restored.track == sample_baseline.track
        assert restored.dataset == sample_baseline.dataset
        assert restored.metrics["wer"].value == sample_baseline.metrics["wer"].value
        assert restored.metrics["wer"].threshold == sample_baseline.metrics["wer"].threshold
        assert restored.receipt.tool_version == sample_baseline.receipt.tool_version


# =============================================================================
# Baseline Creation Tests
# =============================================================================


class TestCreateBaselineFromResult:
    """Tests for create_baseline_from_result function."""

    def test_creates_baseline_from_result(self, sample_benchmark_result: BenchmarkResult) -> None:
        """create_baseline_from_result produces correct BaselineFile."""
        baseline = create_baseline_from_result(sample_benchmark_result)

        assert baseline.schema_version == BASELINE_SCHEMA_VERSION
        assert baseline.track == "asr"
        assert baseline.dataset == "librispeech"
        assert baseline.metrics["wer"].value == 4.8
        assert baseline.metrics["wer"].unit == "%"
        assert baseline.metrics["cer"].value == 1.1

    def test_applies_thresholds(self, sample_benchmark_result: BenchmarkResult) -> None:
        """create_baseline_from_result applies provided thresholds."""
        thresholds = {"wer": 0.05, "cer": 0.15}
        baseline = create_baseline_from_result(sample_benchmark_result, thresholds=thresholds)

        assert baseline.metrics["wer"].threshold == 0.05
        assert baseline.metrics["cer"].threshold == 0.15

    def test_skips_none_metrics(self) -> None:
        """create_baseline_from_result skips metrics with None values."""
        result = BenchmarkResult(
            track="semantic",
            dataset="ami",
            split="test",
            samples_evaluated=10,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="topic_f1", value=0.85, unit=""),
                BenchmarkMetric(name="faithfulness", value=None, reason="missing_api_key"),
            ],
        )

        baseline = create_baseline_from_result(result)

        assert "topic_f1" in baseline.metrics
        assert "faithfulness" not in baseline.metrics

    def test_extracts_system_info(self, sample_benchmark_result: BenchmarkResult) -> None:
        """create_baseline_from_result extracts device from system_info."""
        baseline = create_baseline_from_result(sample_benchmark_result)

        assert baseline.receipt.device == "NVIDIA RTX 4090"


# =============================================================================
# Baseline Comparison Tests
# =============================================================================


class TestCompareWithBaseline:
    """Tests for compare_with_baseline function."""

    def test_calculates_regression(
        self, sample_baseline: BaselineFile, sample_benchmark_result: BenchmarkResult
    ) -> None:
        """compare_with_baseline calculates correct regression percentages."""
        comparison = compare_with_baseline(sample_benchmark_result, sample_baseline)

        # WER: 4.8 vs baseline 4.5 = (4.8-4.5)/4.5 = 0.0667 (6.67% regression)
        wer_result = next(r for r in comparison.results if r.metric_name == "wer")
        assert wer_result.current_value == 4.8
        assert wer_result.baseline_value == 4.5
        assert abs(wer_result.regression - 0.0667) < 0.001

        # CER: 1.1 vs baseline 1.2 = (1.1-1.2)/1.2 = -0.0833 (improvement)
        cer_result = next(r for r in comparison.results if r.metric_name == "cer")
        assert cer_result.current_value == 1.1
        assert cer_result.baseline_value == 1.2
        assert abs(cer_result.regression - (-0.0833)) < 0.001

    def test_passes_within_threshold(
        self, sample_baseline: BaselineFile, sample_benchmark_result: BenchmarkResult
    ) -> None:
        """Metrics within threshold pass."""
        comparison = compare_with_baseline(sample_benchmark_result, sample_baseline)

        # WER has 10% threshold, regression is 6.67% -> pass
        wer_result = next(r for r in comparison.results if r.metric_name == "wer")
        assert wer_result.passed is True
        assert wer_result.threshold == 0.10

    def test_fails_over_threshold(self, sample_baseline: BaselineFile) -> None:
        """Metrics over threshold fail."""
        # Create result with significant regression
        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=100,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="wer", value=6.0, unit="%"),  # 33% regression
            ],
        )

        comparison = compare_with_baseline(result, sample_baseline)

        wer_result = next(r for r in comparison.results if r.metric_name == "wer")
        assert wer_result.passed is False
        assert abs(wer_result.regression - 0.333) < 0.01  # 33% regression

    def test_overall_passed_all_pass(
        self, sample_baseline: BaselineFile, sample_benchmark_result: BenchmarkResult
    ) -> None:
        """overall_passed is True when all metrics pass."""
        comparison = compare_with_baseline(sample_benchmark_result, sample_baseline)
        assert comparison.overall_passed is True

    def test_overall_passed_any_fail(self, sample_baseline: BaselineFile) -> None:
        """overall_passed is False when any metric fails."""
        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=100,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="wer", value=10.0, unit="%"),  # 122% regression
            ],
        )

        comparison = compare_with_baseline(result, sample_baseline)
        assert comparison.overall_passed is False

    def test_uses_default_threshold(self, sample_baseline: BaselineFile) -> None:
        """Uses DEFAULT_REGRESSION_THRESHOLD when metric has no threshold."""
        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=100,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="cer", value=1.3, unit="%"),  # 8.3% regression
            ],
        )

        comparison = compare_with_baseline(result, sample_baseline)

        cer_result = next(r for r in comparison.results if r.metric_name == "cer")
        assert cer_result.threshold == DEFAULT_REGRESSION_THRESHOLD

    def test_report_mode_never_fails_exit(self, sample_baseline: BaselineFile) -> None:
        """Report mode sets correct mode but doesn't affect passed status."""
        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=100,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="wer", value=10.0, unit="%"),
            ],
        )

        comparison = compare_with_baseline(result, sample_baseline, mode="report")

        assert comparison.mode == "report"
        # Still reports pass/fail per metric
        assert comparison.overall_passed is False

    def test_gate_mode(self, sample_baseline: BaselineFile) -> None:
        """Gate mode sets correct mode."""
        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=100,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="wer", value=4.8, unit="%"),
            ],
        )

        comparison = compare_with_baseline(result, sample_baseline, mode="gate")

        assert comparison.mode == "gate"

    def test_handles_zero_baseline(self) -> None:
        """Handles zero baseline value gracefully."""
        baseline = BaselineFile(
            schema_version=1,
            track="test",
            dataset="test",
            created_at="2026-01-01",
            metrics={"metric": BaselineMetric(value=0.0, unit="")},
            receipt=BaselineReceipt(tool_version="1.0"),
        )

        # Zero current value -> regression = 0
        result1 = BenchmarkResult(
            track="test",
            dataset="test",
            split="test",
            samples_evaluated=1,
            samples_failed=0,
            metrics=[BenchmarkMetric(name="metric", value=0.0)],
        )
        comparison1 = compare_with_baseline(result1, baseline)
        metric_result = comparison1.results[0]
        assert metric_result.regression == 0.0

        # Non-zero current value -> regression = inf
        result2 = BenchmarkResult(
            track="test",
            dataset="test",
            split="test",
            samples_evaluated=1,
            samples_failed=0,
            metrics=[BenchmarkMetric(name="metric", value=1.0)],
        )
        comparison2 = compare_with_baseline(result2, baseline)
        metric_result = comparison2.results[0]
        assert metric_result.regression == float("inf")

    def test_skips_missing_baseline_metrics(self, sample_baseline: BaselineFile) -> None:
        """Skips metrics not present in baseline."""
        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=100,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="wer", value=4.8, unit="%"),
                BenchmarkMetric(name="new_metric", value=99.0, unit=""),  # Not in baseline
            ],
        )

        comparison = compare_with_baseline(result, sample_baseline)

        metric_names = [r.metric_name for r in comparison.results]
        assert "wer" in metric_names
        assert "new_metric" not in metric_names


class TestComparisonResult:
    """Tests for ComparisonResult and BaselineComparison dataclasses."""

    def test_to_dict(self, sample_baseline: BaselineFile) -> None:
        """BaselineComparison.to_dict produces correct structure."""
        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=100,
            samples_failed=0,
            metrics=[BenchmarkMetric(name="wer", value=4.8, unit="%")],
        )

        comparison = compare_with_baseline(result, sample_baseline)
        data = comparison.to_dict()

        assert data["track"] == "asr"
        assert data["dataset"] == "librispeech"
        assert isinstance(data["results"], list)
        assert data["results"][0]["metric"] == "wer"
        assert "regression" in data["results"][0]
        assert "overall_passed" in data


# =============================================================================
# Baseline I/O Tests
# =============================================================================


class TestBaselineIO:
    """Tests for baseline load/save functions."""

    def test_get_baseline_path(self) -> None:
        """get_baseline_path returns correct path."""
        path = get_baseline_path("asr", "librispeech")
        assert path.name == "librispeech.json"
        assert path.parent.name == "asr"
        assert "baselines" in str(path)

    def test_save_and_load_baseline(self, sample_baseline: BaselineFile, tmp_path: Path) -> None:
        """save_baseline and load_baseline roundtrip works."""
        with patch(
            "slower_whisper.pipeline.benchmark_cli.get_baselines_dir", return_value=tmp_path
        ):
            # Save
            saved_path = save_baseline(sample_baseline)
            assert saved_path.exists()

            # Load
            loaded = load_baseline(sample_baseline.track, sample_baseline.dataset)
            assert loaded is not None
            assert loaded.track == sample_baseline.track
            assert loaded.dataset == sample_baseline.dataset
            assert loaded.metrics["wer"].value == sample_baseline.metrics["wer"].value

    def test_load_nonexistent_baseline(self, tmp_path: Path) -> None:
        """load_baseline returns None for nonexistent baseline."""
        with patch(
            "slower_whisper.pipeline.benchmark_cli.get_baselines_dir", return_value=tmp_path
        ):
            result = load_baseline("asr", "nonexistent")
            assert result is None

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        """load_baseline returns None for invalid JSON."""
        with patch(
            "slower_whisper.pipeline.benchmark_cli.get_baselines_dir", return_value=tmp_path
        ):
            # Create invalid JSON file
            baseline_dir = tmp_path / "asr"
            baseline_dir.mkdir()
            (baseline_dir / "broken.json").write_text("{ invalid json }")

            result = load_baseline("asr", "broken")
            assert result is None


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestParseThresholds:
    """Tests for _parse_thresholds helper function."""

    def test_parses_valid_thresholds(self) -> None:
        """_parse_thresholds correctly parses valid threshold args."""
        args = ["wer=0.10", "cer=0.05", "der=0.15"]
        result = _parse_thresholds(args)

        assert result == {"wer": 0.10, "cer": 0.05, "der": 0.15}

    def test_handles_empty_input(self) -> None:
        """_parse_thresholds returns empty dict for empty input."""
        assert _parse_thresholds(None) == {}
        assert _parse_thresholds([]) == {}

    def test_skips_invalid_format(self) -> None:
        """_parse_thresholds skips invalid format entries."""
        args = ["wer=0.10", "invalid", "cer=0.05"]
        result = _parse_thresholds(args)

        assert "wer" in result
        assert "cer" in result
        assert len(result) == 2

    def test_skips_invalid_value(self) -> None:
        """_parse_thresholds skips non-numeric values."""
        args = ["wer=0.10", "cer=notanumber"]
        result = _parse_thresholds(args)

        assert result == {"wer": 0.10}

    def test_handles_whitespace(self) -> None:
        """_parse_thresholds handles whitespace around =."""
        args = ["wer = 0.10", " cer=0.05 "]
        result = _parse_thresholds(args)

        assert result == {"wer": 0.10, "cer": 0.05}


# =============================================================================
# Gate Mode Tests
# =============================================================================


class TestGateMode:
    """Tests for --gate flag on benchmark run command."""

    def test_gate_mode_passes_when_within_threshold(
        self, sample_baseline: BaselineFile, tmp_path: Path
    ) -> None:
        """Gate mode returns 0 when metrics are within thresholds."""
        # Mock the runner to return good results
        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=10,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="wer", value=4.6, unit="%"),  # Small regression
                BenchmarkMetric(name="cer", value=1.1, unit="%"),  # Improvement
            ],
        )

        comparison = compare_with_baseline(result, sample_baseline, mode="gate")
        assert comparison.overall_passed is True

    def test_gate_mode_fails_when_exceeds_threshold(self, sample_baseline: BaselineFile) -> None:
        """Gate mode returns 1 when any metric exceeds threshold."""
        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=10,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="wer", value=6.0, unit="%"),  # 33% regression
            ],
        )

        comparison = compare_with_baseline(result, sample_baseline, mode="gate")
        assert comparison.overall_passed is False

    def test_threshold_override_can_make_test_pass(self, sample_baseline: BaselineFile) -> None:
        """Threshold override can change pass/fail outcome."""
        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=10,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="wer", value=5.0, unit="%"),  # 11% regression
            ],
        )

        # With default threshold (10%), should fail
        comparison1 = compare_with_baseline(result, sample_baseline, mode="gate")
        wer_result1 = next(r for r in comparison1.results if r.metric_name == "wer")
        assert wer_result1.passed is False  # 11% > 10%

        # Override threshold to 15%, should pass now
        sample_baseline.metrics["wer"].threshold = 0.15
        comparison2 = compare_with_baseline(result, sample_baseline, mode="gate")
        wer_result2 = next(r for r in comparison2.results if r.metric_name == "wer")
        assert wer_result2.passed is True  # 11% < 15%

    def test_threshold_override_can_make_test_stricter(self, sample_baseline: BaselineFile) -> None:
        """Threshold override can make test stricter."""
        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=10,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="wer", value=4.7, unit="%"),  # 4.4% regression
            ],
        )

        # With default threshold (10%), should pass
        comparison1 = compare_with_baseline(result, sample_baseline, mode="gate")
        wer_result1 = next(r for r in comparison1.results if r.metric_name == "wer")
        assert wer_result1.passed is True  # 4.4% < 10%

        # Override threshold to 3%, should fail now
        sample_baseline.metrics["wer"].threshold = 0.03
        comparison2 = compare_with_baseline(result, sample_baseline, mode="gate")
        wer_result2 = next(r for r in comparison2.results if r.metric_name == "wer")
        assert wer_result2.passed is False  # 4.4% > 3%


class TestPrintGateReport:
    """Tests for _print_gate_report function output."""

    def test_gate_report_shows_all_metrics(
        self, sample_baseline: BaselineFile, capsys: pytest.CaptureFixture
    ) -> None:
        """Gate report shows all metrics in comparison."""
        from slower_whisper.pipeline.benchmark_cli import _print_gate_report

        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=10,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="wer", value=4.8, unit="%"),
                BenchmarkMetric(name="cer", value=1.1, unit="%"),
            ],
        )

        comparison = compare_with_baseline(result, sample_baseline, mode="gate")
        _print_gate_report(comparison)

        output = capsys.readouterr().out
        assert "wer" in output
        assert "cer" in output
        assert "REGRESSION GATE CHECK" in output

    def test_gate_report_shows_pass_status(
        self, sample_baseline: BaselineFile, capsys: pytest.CaptureFixture
    ) -> None:
        """Gate report shows GATE PASSED when all metrics pass."""
        from slower_whisper.pipeline.benchmark_cli import _print_gate_report

        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=10,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="wer", value=4.6, unit="%"),
            ],
        )

        comparison = compare_with_baseline(result, sample_baseline, mode="gate")
        _print_gate_report(comparison)

        output = capsys.readouterr().out
        assert "GATE PASSED" in output

    def test_gate_report_shows_fail_details(
        self, sample_baseline: BaselineFile, capsys: pytest.CaptureFixture
    ) -> None:
        """Gate report shows detailed failure information."""
        from slower_whisper.pipeline.benchmark_cli import _print_gate_report

        result = BenchmarkResult(
            track="asr",
            dataset="librispeech",
            split="test",
            samples_evaluated=10,
            samples_failed=0,
            metrics=[
                BenchmarkMetric(name="wer", value=6.0, unit="%"),  # Big regression
            ],
        )

        comparison = compare_with_baseline(result, sample_baseline, mode="gate")
        _print_gate_report(comparison)

        output = capsys.readouterr().out
        assert "GATE FAILED" in output
        assert "Failed metrics:" in output
        assert "wer" in output
