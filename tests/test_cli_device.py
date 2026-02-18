"""Tests for CLI device resolution and preflight banner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from slower_whisper.pipeline.cli import build_parser


class TestDeviceCliFlag:
    """Tests for --device CLI flag."""

    def test_transcribe_device_accepts_auto(self) -> None:
        """--device auto is accepted by transcribe command."""
        parser = build_parser()
        args = parser.parse_args(["transcribe", "--device", "auto"])
        assert args.device == "auto"

    def test_transcribe_device_accepts_cuda(self) -> None:
        """--device cuda is accepted by transcribe command."""
        parser = build_parser()
        args = parser.parse_args(["transcribe", "--device", "cuda"])
        assert args.device == "cuda"

    def test_transcribe_device_accepts_cpu(self) -> None:
        """--device cpu is accepted by transcribe command."""
        parser = build_parser()
        args = parser.parse_args(["transcribe", "--device", "cpu"])
        assert args.device == "cpu"

    def test_transcribe_device_default_is_none(self) -> None:
        """--device defaults to None (resolved to auto at runtime)."""
        parser = build_parser()
        args = parser.parse_args(["transcribe"])
        assert args.device is None

    def test_enrich_device_accepts_auto(self) -> None:
        """--device auto is accepted by enrich command."""
        parser = build_parser()
        args = parser.parse_args(["enrich", "--device", "auto"])
        assert args.device == "auto"


class TestPreflightBanner:
    """Tests for preflight banner output."""

    @patch("slower_whisper.pipeline.cli.resolve_device")
    @patch("slower_whisper.pipeline.pipeline.run_pipeline")
    def test_banner_printed_on_transcribe(
        self, mock_run_pipeline: MagicMock, mock_resolve: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        """Preflight banner is printed when running transcribe command."""
        from slower_whisper.pipeline.device import ResolvedDevice

        # Mock device resolution
        mock_resolve.return_value = ResolvedDevice(
            device="cpu",
            compute_type="int8",
            requested_device="auto",
            fallback_reason="No CUDA devices found by CTranslate2",
            cuda_available=False,
            cuda_device_count=0,
        )

        # Mock the pipeline to avoid actual transcription
        from slower_whisper.pipeline.models import BatchProcessingResult

        mock_result = MagicMock(spec=BatchProcessingResult)
        mock_result.total_files = 0
        mock_result.processed = 0
        mock_result.skipped = 0
        mock_result.diarized_only = 0
        mock_result.failed = 0
        mock_result.total_audio_seconds = 0
        mock_result.total_time_seconds = 0
        mock_result.file_results = []
        mock_run_pipeline.return_value = mock_result

        from slower_whisper.pipeline.cli import main

        main(["transcribe", "--device", "auto"])

        # Check banner was printed to stderr (keeps stdout clean for structured output)
        captured = capsys.readouterr()
        assert "[Device] CPU (fallback from auto)" in captured.err
        assert "No CUDA devices found by CTranslate2" in captured.err

    @patch("slower_whisper.pipeline.cli.resolve_device")
    @patch("slower_whisper.pipeline.pipeline.run_pipeline")
    def test_banner_shows_cuda_when_available(
        self, mock_run_pipeline: MagicMock, mock_resolve: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        """Preflight banner shows CUDA when available."""
        from slower_whisper.pipeline.device import ResolvedDevice

        # Mock device resolution with CUDA available
        mock_resolve.return_value = ResolvedDevice(
            device="cuda",
            compute_type="float16",
            requested_device="auto",
            fallback_reason=None,
            cuda_available=True,
            cuda_device_count=1,
        )

        # Mock the pipeline
        from slower_whisper.pipeline.models import BatchProcessingResult

        mock_result = MagicMock(spec=BatchProcessingResult)
        mock_result.total_files = 0
        mock_result.processed = 0
        mock_result.skipped = 0
        mock_result.diarized_only = 0
        mock_result.failed = 0
        mock_result.total_audio_seconds = 0
        mock_result.total_time_seconds = 0
        mock_result.file_results = []
        mock_run_pipeline.return_value = mock_result

        from slower_whisper.pipeline.cli import main

        main(["transcribe", "--device", "auto"])

        # Check banner was printed to stderr with CUDA
        captured = capsys.readouterr()
        assert "[Device] CUDA" in captured.err
        assert "compute_type=float16" in captured.err
        assert "fallback" not in captured.err.lower()


class TestComputeTypeFallback:
    """Tests for compute_type coherence on device fallback."""

    @patch("slower_whisper.pipeline.cli.resolve_device")
    @patch("slower_whisper.pipeline.pipeline.run_pipeline")
    def test_cpu_fallback_sets_int8_compute_type(
        self,
        mock_run_pipeline: MagicMock,
        mock_resolve: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """When falling back to CPU, compute_type must be set to int8 (not float16).

        This is a critical regression test: GPU compute types (float16) are
        incompatible with CPU inference and cause runtime failures.
        """
        from slower_whisper.pipeline.device import ResolvedDevice

        # Mock device resolution: requested auto but falling back to CPU
        mock_resolve.return_value = ResolvedDevice(
            device="cpu",
            compute_type="int8",
            requested_device="auto",
            fallback_reason="No CUDA devices found by CTranslate2",
            cuda_available=False,
            cuda_device_count=0,
        )

        # Mock the pipeline
        from slower_whisper.pipeline.models import BatchProcessingResult

        mock_result = MagicMock(spec=BatchProcessingResult)
        mock_result.total_files = 0
        mock_result.processed = 0
        mock_result.skipped = 0
        mock_result.diarized_only = 0
        mock_result.failed = 0
        mock_result.total_audio_seconds = 0
        mock_result.total_time_seconds = 0
        mock_result.file_results = []
        mock_run_pipeline.return_value = mock_result

        from slower_whisper.pipeline.cli import main

        main(["transcribe", "--device", "auto"])

        # CRITICAL: Verify that run_pipeline was called with int8 compute_type
        mock_run_pipeline.assert_called_once()
        call_args = mock_run_pipeline.call_args
        app_config = call_args[0][0]  # First positional arg is AppConfig
        assert app_config.asr.compute_type == "int8", (
            f"Expected compute_type='int8' for CPU fallback, got '{app_config.asr.compute_type}'"
        )
        assert app_config.asr.device == "cpu"

        # Also verify banner shows correct info
        captured = capsys.readouterr()
        assert "compute_type=int8" in captured.err

    @patch("slower_whisper.pipeline.cli.resolve_device")
    @patch("slower_whisper.pipeline.pipeline.run_pipeline")
    def test_explicit_compute_type_preserved_on_fallback(
        self,
        mock_run_pipeline: MagicMock,
        mock_resolve: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """User's explicit --compute-type is preserved even on device fallback.

        If user explicitly passes --compute-type float32, we respect that
        even when falling back to CPU (float32 works on CPU).
        """
        from slower_whisper.pipeline.device import ResolvedDevice

        # Mock device resolution: requested cuda but falling back to CPU
        mock_resolve.return_value = ResolvedDevice(
            device="cpu",
            compute_type="int8",  # This would be the auto-derived value
            requested_device="cuda",
            fallback_reason="No CUDA devices found by CTranslate2",
            cuda_available=False,
            cuda_device_count=0,
        )

        # Mock the pipeline
        from slower_whisper.pipeline.models import BatchProcessingResult

        mock_result = MagicMock(spec=BatchProcessingResult)
        mock_result.total_files = 0
        mock_result.processed = 0
        mock_result.skipped = 0
        mock_result.diarized_only = 0
        mock_result.failed = 0
        mock_result.total_audio_seconds = 0
        mock_result.total_time_seconds = 0
        mock_result.file_results = []
        mock_run_pipeline.return_value = mock_result

        from slower_whisper.pipeline.cli import main

        # User explicitly requests float32 compute type
        main(["transcribe", "--device", "cuda", "--compute-type", "float32"])

        # Verify that run_pipeline was called with user's explicit float32
        mock_run_pipeline.assert_called_once()
        call_args = mock_run_pipeline.call_args
        app_config = call_args[0][0]
        assert app_config.asr.compute_type == "float32", (
            f"Expected explicit compute_type='float32' to be preserved, "
            f"got '{app_config.asr.compute_type}'"
        )

        _ = capsys.readouterr()  # consume output
