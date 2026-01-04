"""Tests for CLI device resolution and preflight banner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from transcription.cli import build_parser


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

    @patch("transcription.cli.resolve_device")
    @patch("transcription.pipeline.run_pipeline")
    def test_banner_printed_on_transcribe(
        self, mock_run_pipeline: MagicMock, mock_resolve: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        """Preflight banner is printed when running transcribe command."""
        from transcription.device import ResolvedDevice

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
        from transcription.models import BatchProcessingResult

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

        from transcription.cli import main

        main(["transcribe", "--device", "auto"])

        # Check banner was printed
        captured = capsys.readouterr()
        assert "[Device] CPU (fallback from auto)" in captured.out
        assert "No CUDA devices found by CTranslate2" in captured.out

    @patch("transcription.cli.resolve_device")
    @patch("transcription.pipeline.run_pipeline")
    def test_banner_shows_cuda_when_available(
        self, mock_run_pipeline: MagicMock, mock_resolve: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        """Preflight banner shows CUDA when available."""
        from transcription.device import ResolvedDevice

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
        from transcription.models import BatchProcessingResult

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

        from transcription.cli import main

        main(["transcribe", "--device", "auto"])

        # Check banner was printed with CUDA
        captured = capsys.readouterr()
        assert "[Device] CUDA" in captured.out
        assert "compute_type=float16" in captured.out
        assert "fallback" not in captured.out.lower()
