"""
Tests for StreamingBenchmarkRunner.evaluate_sample (streaming track).

Tests cover:
- Basic streaming evaluation flow (ASR -> chunks -> enrichment session)
- Timing metrics calculation (latency_first_token_ms, latency_total_ms, rtf)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcription.benchmark_cli import StreamingBenchmarkRunner
from transcription.benchmarks import EvalSample


@pytest.fixture
def mock_asr_engine() -> MagicMock:
    """Mock TranscriptionEngine returning two segments."""
    with patch("transcription.asr_engine.TranscriptionEngine") as mock:
        engine_instance = mock.return_value
        segment1 = MagicMock(start=0.0, end=1.0, text="hello")
        segment2 = MagicMock(start=1.0, end=2.0, text="world")
        info = MagicMock(duration=2.0)
        engine_instance._transcribe_with_model.return_value = ([segment1, segment2], info)
        yield engine_instance


@pytest.fixture
def mock_streaming_session() -> MagicMock:
    """Mock StreamingEnrichmentSession with 2s audio duration."""
    with patch("transcription.streaming_enrich.StreamingEnrichmentSession") as mock:
        session_instance = mock.return_value
        session_instance._extractor.duration_seconds = 2.0
        yield session_instance


def test_streaming_benchmark_runner_evaluate_sample(
    mock_asr_engine: MagicMock, mock_streaming_session: MagicMock
) -> None:
    """StreamingBenchmarkRunner.evaluate_sample produces timing metrics."""
    runner = StreamingBenchmarkRunner(track="streaming", dataset="test_ds")
    sample = EvalSample(
        dataset="test_ds",
        id="test_001",
        audio_path=Path("dummy.wav"),
        reference_transcript="hello world",
    )

    # Run
    result = runner.evaluate_sample(sample)

    # Verify ASR engine called
    mock_asr_engine._transcribe_with_model.assert_called_once_with(Path("dummy.wav"))

    # Verify chunks ingested
    assert mock_streaming_session.ingest_chunk.call_count == 2

    # Check first chunk argument
    call_args_list = mock_streaming_session.ingest_chunk.call_args_list
    chunk1 = call_args_list[0][0][0]  # First arg of first call
    assert chunk1["text"] == "hello"
    assert chunk1["start"] == 0.0

    # Verify end_of_stream called
    mock_streaming_session.end_of_stream.assert_called_once()

    # Verify result structure
    assert result["id"] == "test_001"
    assert "latency_first_token_ms" in result
    assert "latency_total_ms" in result
    assert result["audio_duration_s"] == 2.0
    assert "rtf" in result


# =============================================================================
# Dry-run validation tests
# =============================================================================


class TestDryRunValidation:
    """Tests for --dry-run flag and _perform_dry_run_validation function."""

    def test_dry_run_commonvoice_en_smoke_returns_zero_for_valid_manifest(self) -> None:
        """Dry-run returns exit code 0 for valid manifest (even if audio not staged)."""
        from transcription.benchmark_cli import handle_benchmark_run

        exit_code = handle_benchmark_run(
            track="asr",
            dataset="commonvoice_en_smoke",
            split="test",
            limit=None,
            output=None,
            verbose=False,
            mode=None,
            dry_run=True,
        )

        assert exit_code == 0

    def test_dry_run_invalid_track_returns_one(self) -> None:
        """Dry-run returns exit code 1 for invalid track."""
        from transcription.benchmark_cli import handle_benchmark_run

        exit_code = handle_benchmark_run(
            track="invalid_track",
            dataset="commonvoice_en_smoke",
            split="test",
            limit=None,
            output=None,
            verbose=False,
            mode=None,
            dry_run=True,
        )

        assert exit_code == 1

    def test_dry_run_invalid_dataset_for_track_returns_one(self) -> None:
        """Dry-run returns exit code 1 for dataset not supported by track."""
        from transcription.benchmark_cli import handle_benchmark_run

        exit_code = handle_benchmark_run(
            track="asr",
            dataset="ami",  # ami is not an ASR dataset
            split="test",
            limit=None,
            output=None,
            verbose=False,
            mode=None,
            dry_run=True,
        )

        assert exit_code == 1

    def test_dry_run_output_contains_expected_fields(self, capsys: pytest.CaptureFixture) -> None:
        """Dry-run output contains manifest path, schema version, samples, and staging info."""
        from transcription.benchmark_cli import handle_benchmark_run

        handle_benchmark_run(
            track="asr",
            dataset="commonvoice_en_smoke",
            split="test",
            limit=None,
            output=None,
            verbose=False,
            mode=None,
            dry_run=True,
        )

        captured = capsys.readouterr()
        output = captured.out

        # Check required output fields
        assert "Dry-run validation for asr/commonvoice_en_smoke:" in output
        assert "Manifest found:" in output
        assert "Schema version:" in output
        assert "Samples defined:" in output
        assert "Audio files staged:" in output
        assert "To stage audio:" in output

    def test_dry_run_uses_unicode_symbols(self, capsys: pytest.CaptureFixture) -> None:
        """Dry-run output uses Unicode check mark and warning symbols."""
        from transcription.benchmark_cli import handle_benchmark_run

        handle_benchmark_run(
            track="asr",
            dataset="commonvoice_en_smoke",
            split="test",
            limit=None,
            output=None,
            verbose=False,
            mode=None,
            dry_run=True,
        )

        captured = capsys.readouterr()
        output = captured.out

        # Should contain Unicode check mark for valid manifest fields
        assert "\u2713" in output  # CHECK mark

        # Should contain warning symbol for unstaged audio (commonvoice_en_smoke has 0 staged)
        assert "\u26a0" in output  # WARNING symbol

    def test_dry_run_does_not_run_benchmarks(self) -> None:
        """Dry-run mode exits early without running actual benchmarks."""
        from transcription.benchmark_cli import handle_benchmark_run

        # If benchmarks were run, this would try to load ASR model and fail
        # or take a long time. Dry-run should complete nearly instantly.
        with patch("transcription.benchmark_cli.get_benchmark_runner") as mock_runner:
            handle_benchmark_run(
                track="asr",
                dataset="commonvoice_en_smoke",
                split="test",
                limit=None,
                output=None,
                verbose=False,
                mode=None,
                dry_run=True,
            )

            # get_benchmark_runner should NOT be called in dry-run mode
            mock_runner.assert_not_called()

    def test_dry_run_respects_selection_csv_for_sample_count(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """Dry-run reads sample count from selection.csv for commonvoice_en_smoke."""
        from transcription.benchmark_cli import handle_benchmark_run

        handle_benchmark_run(
            track="asr",
            dataset="commonvoice_en_smoke",
            split="test",
            limit=None,
            output=None,
            verbose=False,
            mode=None,
            dry_run=True,
        )

        captured = capsys.readouterr()
        output = captured.out

        # selection.csv has 15 entries (from the test data)
        assert "Samples defined: 15" in output
        assert "from selection.csv" in output


def test_benchmark_run_passes_llm_delay_to_runner() -> None:
    """LLM delay is forwarded to the benchmark runner factory."""
    from transcription.benchmark_cli import BenchmarkMetric, BenchmarkResult, handle_benchmark_run

    dummy_result = BenchmarkResult(
        track="semantic",
        dataset="ami",
        split="test",
        samples_evaluated=1,
        samples_failed=0,
        metrics=[BenchmarkMetric(name="faithfulness", value=1.0, unit="/10")],
    )

    mock_runner = MagicMock()
    mock_runner.run.return_value = dummy_result

    with patch(
        "transcription.benchmark_cli.get_benchmark_runner", return_value=mock_runner
    ) as mock_get:
        exit_code = handle_benchmark_run(
            track="semantic",
            dataset="ami",
            split="test",
            limit=1,
            output=None,
            verbose=False,
            mode="summary",
            dry_run=False,
            gate=False,
            threshold_overrides=None,
            llm_delay_s=1.5,
        )

    assert exit_code == 0
    mock_get.assert_called_once_with("semantic", "ami", "test", mode="summary", llm_delay_s=1.5)
