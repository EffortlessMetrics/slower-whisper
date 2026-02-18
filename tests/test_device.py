"""Tests for device detection and resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from slower_whisper.pipeline.device import (
    ResolvedDevice,
    format_preflight_banner,
    get_device_summary,
    resolve_compute_type,
    resolve_device,
)


class TestResolveDevice:
    """Tests for resolve_device()."""

    def test_cpu_requested_returns_cpu(self) -> None:
        """When CPU is explicitly requested, return CPU regardless of CUDA."""
        result = resolve_device("cpu")
        assert result.device == "cpu"
        assert result.requested_device == "cpu"
        assert result.fallback_reason is None
        assert not result.is_fallback

    @patch("slower_whisper.pipeline.device._detect_ctranslate2_cuda")
    def test_auto_with_cuda_available_returns_cuda(self, mock_detect: MagicMock) -> None:
        """When auto requested and CUDA available, return CUDA."""
        mock_detect.return_value = (True, 1, None)

        result = resolve_device("auto")

        assert result.device == "cuda"
        assert result.requested_device == "auto"
        assert result.fallback_reason is None
        assert result.cuda_available is True
        assert result.cuda_device_count == 1
        assert not result.is_fallback

    @patch("slower_whisper.pipeline.device._detect_ctranslate2_cuda")
    def test_auto_without_cuda_falls_back_to_cpu(self, mock_detect: MagicMock) -> None:
        """When auto requested and CUDA unavailable, fall back to CPU."""
        mock_detect.return_value = (False, 0, "No CUDA devices found by CTranslate2")

        result = resolve_device("auto")

        assert result.device == "cpu"
        assert result.requested_device == "auto"
        assert result.fallback_reason == "No CUDA devices found by CTranslate2"
        assert result.cuda_available is False
        assert result.is_fallback

    @patch("slower_whisper.pipeline.device._detect_ctranslate2_cuda")
    def test_cuda_requested_with_cuda_available_returns_cuda(self, mock_detect: MagicMock) -> None:
        """When CUDA explicitly requested and available, return CUDA."""
        mock_detect.return_value = (True, 2, None)

        result = resolve_device("cuda")

        assert result.device == "cuda"
        assert result.requested_device == "cuda"
        assert result.fallback_reason is None
        assert result.cuda_device_count == 2
        assert not result.is_fallback

    @patch("slower_whisper.pipeline.device._detect_ctranslate2_cuda")
    def test_cuda_requested_without_cuda_falls_back_to_cpu(self, mock_detect: MagicMock) -> None:
        """When CUDA requested but unavailable, fall back to CPU with reason."""
        mock_detect.return_value = (False, 0, "CTranslate2 not compiled with CUDA")

        result = resolve_device("cuda", allow_fallback=True)

        assert result.device == "cpu"
        assert result.requested_device == "cuda"
        assert result.fallback_reason == "CTranslate2 not compiled with CUDA"
        assert result.is_fallback

    @patch("slower_whisper.pipeline.device._detect_ctranslate2_cuda")
    def test_cuda_requested_without_cuda_raises_when_no_fallback(
        self, mock_detect: MagicMock
    ) -> None:
        """When CUDA requested, unavailable, and fallback disabled, raise error."""
        mock_detect.return_value = (False, 0, "No CUDA devices")

        with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
            resolve_device("cuda", allow_fallback=False)

    @patch("slower_whisper.pipeline.device._detect_torch_cuda")
    def test_torch_backend_uses_torch_detection(self, mock_detect: MagicMock) -> None:
        """When torch backend specified, use torch CUDA detection."""
        mock_detect.return_value = (True, 1, None)

        result = resolve_device("auto", backend="torch")

        assert result.device == "cuda"
        assert result.cuda_device_count == 1
        mock_detect.assert_called_once()


class TestResolveComputeType:
    """Tests for resolve_compute_type()."""

    def test_auto_on_cpu_returns_int8(self) -> None:
        """Auto compute type on CPU defaults to int8 for speed."""
        assert resolve_compute_type("auto", "cpu") == "int8"
        assert resolve_compute_type(None, "cpu") == "int8"

    def test_auto_on_cuda_returns_float16(self) -> None:
        """Auto compute type on CUDA defaults to float16 for quality."""
        assert resolve_compute_type("auto", "cuda") == "float16"
        assert resolve_compute_type(None, "cuda") == "float16"

    def test_explicit_compute_type_passed_through(self) -> None:
        """Explicit compute type is returned unchanged."""
        assert resolve_compute_type("float32", "cuda") == "float32"
        assert resolve_compute_type("int8_float16", "cuda") == "int8_float16"


class TestFormatPreflightBanner:
    """Tests for format_preflight_banner()."""

    def test_basic_cuda_banner(self) -> None:
        """Basic CUDA banner shows device and model."""
        resolved = ResolvedDevice(
            device="cuda",
            compute_type="float16",
            requested_device="auto",
            cuda_available=True,
            cuda_device_count=1,
        )

        banner = format_preflight_banner(resolved, "large-v3")

        assert "[Device] CUDA" in banner
        assert "compute_type=float16" in banner
        assert "model=large-v3" in banner
        assert "fallback" not in banner.lower()

    def test_fallback_banner_shows_reason(self) -> None:
        """Fallback banner shows reason for fallback."""
        resolved = ResolvedDevice(
            device="cpu",
            compute_type="int8",
            requested_device="cuda",
            fallback_reason="No CUDA devices found",
            cuda_available=False,
            cuda_device_count=0,
        )

        banner = format_preflight_banner(resolved, "base")

        assert "[Device] CPU (fallback from cuda)" in banner
        assert "Reason: No CUDA devices found" in banner

    def test_verbose_banner_shows_details(self) -> None:
        """Verbose banner shows additional details."""
        resolved = ResolvedDevice(
            device="cuda",
            compute_type="float16",
            requested_device="auto",
            cuda_available=True,
            cuda_device_count=2,
        )

        banner = format_preflight_banner(resolved, "large-v3", cache_dir="/cache", verbose=True)

        assert "CUDA devices: 2" in banner
        assert "Cache: /cache" in banner


class TestGetDeviceSummary:
    """Tests for get_device_summary()."""

    def test_summary_contains_all_fields(self) -> None:
        """Summary dict contains all relevant fields."""
        resolved = ResolvedDevice(
            device="cuda",
            compute_type="float16",
            requested_device="auto",
            fallback_reason=None,
            cuda_available=True,
            cuda_device_count=1,
        )

        summary = get_device_summary(resolved)

        assert summary["device"] == "cuda"
        assert summary["compute_type"] == "float16"
        assert summary["requested_device"] == "auto"
        assert summary["fallback_reason"] is None
        assert summary["cuda_available"] is True
        assert summary["cuda_device_count"] == 1

    def test_summary_with_fallback(self) -> None:
        """Summary includes fallback reason when present."""
        resolved = ResolvedDevice(
            device="cpu",
            compute_type="int8",
            requested_device="cuda",
            fallback_reason="CUDA unavailable",
            cuda_available=False,
            cuda_device_count=0,
        )

        summary = get_device_summary(resolved)

        assert summary["fallback_reason"] == "CUDA unavailable"


class TestDetectionFunctions:
    """Tests for internal detection functions."""

    def test_ctranslate2_detection_handles_import_error(self) -> None:
        """CTranslate2 detection handles import errors gracefully."""
        from slower_whisper.pipeline.device import _detect_ctranslate2_cuda

        # This should not raise even if ctranslate2 is missing/broken
        available, count, error = _detect_ctranslate2_cuda()
        # Result depends on environment, but should not crash
        assert isinstance(available, bool)
        assert isinstance(count, int)

    def test_torch_detection_handles_import_error(self) -> None:
        """Torch detection handles import errors gracefully."""
        from slower_whisper.pipeline.device import _detect_torch_cuda

        # This should not raise even if torch is missing
        available, count, error = _detect_torch_cuda()
        assert isinstance(available, bool)
        assert isinstance(count, int)
