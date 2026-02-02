"""Operational telemetry system for slower-whisper.

This module provides timing, metrics collection, and system diagnostics
for monitoring pipeline performance and debugging issues.

Key components:
- Timer: Context manager for measuring stage durations
- TelemetryCollector: Aggregates timing and counter metrics
- DoctorCheck: Individual diagnostic check result
- run_doctor: Comprehensive system diagnostics

Example usage:
    from transcription.telemetry import Timer, TelemetryCollector

    collector = TelemetryCollector()
    with Timer("asr") as t:
        # perform ASR
        pass
    collector.record_timing("asr_ms", t.elapsed_ms)
    collector.increment("segments_count", 10)
    print(collector.to_dict())
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Timer Context Manager
# =============================================================================


class Timer:
    """Context manager for measuring elapsed time.

    Measures wall-clock time in milliseconds with high precision.

    Example:
        with Timer("asr") as t:
            # do work
            pass
        print(f"Elapsed: {t.elapsed_ms:.2f} ms")
    """

    def __init__(self, name: str) -> None:
        """Initialize timer with a name for logging/identification.

        Args:
            name: Identifier for this timing measurement
        """
        self.name = name
        self._start: float | None = None
        self._end: float | None = None

    def __enter__(self) -> Timer:
        """Start timing when entering context."""
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Stop timing when exiting context."""
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds. If timer is still running,
            returns time since start. If never started, returns 0.0.
        """
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return (end - self._start) * 1000.0

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds.

        Returns:
            Elapsed time in seconds.
        """
        return self.elapsed_ms / 1000.0


# =============================================================================
# Telemetry Collector
# =============================================================================


@dataclass
class TelemetryCollector:
    """Collects and aggregates pipeline telemetry metrics.

    Tracks timing for pipeline stages, counts for processed items,
    and streaming-specific metrics.

    Attributes:
        timings: Stage timing measurements in milliseconds
        counters: Count-based metrics (segments, words, etc.)
        streaming: Streaming-specific metrics
        resource_usage: Resource consumption metrics
    """

    # Stage timings (ms)
    timings: dict[str, float] = field(default_factory=dict)

    # Counters
    counters: dict[str, int] = field(default_factory=dict)

    # Streaming metrics
    streaming: dict[str, int] = field(default_factory=dict)

    # Resource usage
    resource_usage: dict[str, float] = field(default_factory=dict)

    def record_timing(self, stage: str, ms: float) -> None:
        """Record a timing measurement for a pipeline stage.

        Args:
            stage: Stage identifier (e.g., "asr_ms", "diarization_ms")
            ms: Duration in milliseconds
        """
        self.timings[stage] = ms

    def increment(self, counter: str, value: int = 1) -> None:
        """Increment a counter by the given value.

        Args:
            counter: Counter identifier (e.g., "segments_count")
            value: Amount to increment (default: 1)
        """
        self.counters[counter] = self.counters.get(counter, 0) + value

    def record_streaming_metric(self, metric: str, value: int) -> None:
        """Record a streaming-specific metric.

        Args:
            metric: Metric identifier (e.g., "chunks_received")
            value: Metric value
        """
        self.streaming[metric] = value

    def increment_streaming(self, metric: str, value: int = 1) -> None:
        """Increment a streaming metric.

        Args:
            metric: Metric identifier
            value: Amount to increment (default: 1)
        """
        self.streaming[metric] = self.streaming.get(metric, 0) + value

    def record_resource(self, resource: str, value: float) -> None:
        """Record a resource usage metric.

        Args:
            resource: Resource identifier (e.g., "peak_memory_mb")
            value: Resource value
        """
        self.resource_usage[resource] = value

    def capture_memory_usage(self) -> None:
        """Capture current memory usage statistics.

        Records peak_memory_mb and optionally gpu_memory_mb if available.
        """
        try:
            import resource

            rusage = resource.getrusage(resource.RUSAGE_SELF)
            # maxrss is in KB on Linux, bytes on macOS
            if sys.platform == "darwin":
                peak_mb = rusage.ru_maxrss / (1024 * 1024)
            else:
                peak_mb = rusage.ru_maxrss / 1024
            self.record_resource("peak_memory_mb", peak_mb)
        except Exception:
            pass

        # Try to get GPU memory if torch is available
        try:
            import torch

            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                self.record_resource("gpu_memory_mb", gpu_memory_mb)
        except Exception:
            pass

    def to_dict(self) -> dict[str, Any]:
        """Convert telemetry to dictionary for JSON serialization.

        Returns:
            Dictionary with all telemetry metrics organized by category.
        """
        result: dict[str, Any] = {}

        if self.timings:
            result["timings"] = dict(self.timings)

        if self.counters:
            result["counters"] = dict(self.counters)

        if self.streaming:
            result["streaming"] = dict(self.streaming)

        if self.resource_usage:
            result["resource_usage"] = dict(self.resource_usage)

        # Add computed total if we have stage timings
        if self.timings:
            total_ms = sum(
                v for k, v in self.timings.items() if k.endswith("_ms") and k != "total_ms"
            )
            if total_ms > 0:
                result["timings"]["computed_total_ms"] = total_ms

        return result

    def reset(self) -> None:
        """Reset all telemetry data."""
        self.timings.clear()
        self.counters.clear()
        self.streaming.clear()
        self.resource_usage.clear()


# =============================================================================
# Doctor Command - System Diagnostics
# =============================================================================


class CheckStatus(Enum):
    """Status of a diagnostic check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class DoctorCheck:
    """Result of a single diagnostic check.

    Attributes:
        name: Human-readable check name
        status: Check result status
        message: Detailed message about the check result
        details: Additional details (versions, paths, etc.)
    """

    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details if self.details else None,
        }


@dataclass
class DoctorReport:
    """Complete diagnostic report from doctor command.

    Attributes:
        checks: List of all diagnostic check results
        summary: Summary of check statuses
        overall_status: Overall pass/warn/fail status
    """

    checks: list[DoctorCheck] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, int]:
        """Get count of checks by status."""
        counts: dict[str, int] = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
        for check in self.checks:
            counts[check.status.value] += 1
        return counts

    @property
    def overall_status(self) -> CheckStatus:
        """Get overall status (fail if any fail, warn if any warn)."""
        statuses = {c.status for c in self.checks}
        if CheckStatus.FAIL in statuses:
            return CheckStatus.FAIL
        if CheckStatus.WARN in statuses:
            return CheckStatus.WARN
        return CheckStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "overall_status": self.overall_status.value,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
        }


def _check_python_version() -> DoctorCheck:
    """Check Python version compatibility."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major != 3:
        return DoctorCheck(
            name="Python Version",
            status=CheckStatus.FAIL,
            message=f"Python 3.10+ required, found Python {version.major}",
            details={"version": version_str},
        )

    if version.minor < 10:
        return DoctorCheck(
            name="Python Version",
            status=CheckStatus.FAIL,
            message=f"Python 3.10+ required, found Python {version_str}",
            details={"version": version_str, "minimum": "3.10"},
        )

    if version.minor < 11:
        return DoctorCheck(
            name="Python Version",
            status=CheckStatus.WARN,
            message=f"Python {version_str} (3.11+ recommended for best performance)",
            details={"version": version_str, "recommended": "3.11+"},
        )

    return DoctorCheck(
        name="Python Version",
        status=CheckStatus.PASS,
        message=f"Python {version_str}",
        details={"version": version_str},
    )


def _check_required_deps() -> DoctorCheck:
    """Check that required dependencies are installed."""
    required = [
        ("faster_whisper", "faster-whisper"),
        ("ctranslate2", "ctranslate2"),
        ("numpy", "numpy"),
    ]

    missing = []
    versions = {}

    for module, package in required:
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "unknown")
            versions[package] = ver
        except ImportError:
            missing.append(package)

    if missing:
        return DoctorCheck(
            name="Required Dependencies",
            status=CheckStatus.FAIL,
            message=f"Missing required packages: {', '.join(missing)}",
            details={"missing": missing, "installed": versions},
        )

    return DoctorCheck(
        name="Required Dependencies",
        status=CheckStatus.PASS,
        message="All required dependencies installed",
        details={"versions": versions},
    )


def _check_optional_deps() -> DoctorCheck:
    """Check optional dependencies (diarization, emotion, etc.)."""
    optional = {
        "pyannote.audio": ("diarization", "pyannote-audio"),
        "transformers": ("emotion/diarization", "transformers"),
        "torch": ("GPU acceleration", "torch"),
        "torchaudio": ("audio processing", "torchaudio"),
    }

    available = {}
    missing = []

    for module, (feature, package) in optional.items():
        try:
            mod = __import__(module.split(".")[0])
            ver = getattr(mod, "__version__", "unknown")
            available[package] = {"version": ver, "feature": feature}
        except ImportError:
            missing.append(f"{package} ({feature})")

    if not available:
        return DoctorCheck(
            name="Optional Dependencies",
            status=CheckStatus.WARN,
            message="No optional dependencies installed",
            details={"missing": missing},
        )

    return DoctorCheck(
        name="Optional Dependencies",
        status=CheckStatus.PASS,
        message=f"{len(available)} optional packages available",
        details={"available": available, "missing": missing if missing else None},
    )


def _check_gpu() -> DoctorCheck:
    """Check GPU availability and CUDA version."""
    # Check CTranslate2 CUDA support
    try:
        import importlib

        ct2 = importlib.import_module("ctranslate2")
        get_count = getattr(ct2, "get_cuda_device_count", None)
        if get_count is None:
            ct2_cuda = False
            ct2_devices = 0
        else:
            ct2_devices = int(get_count())
            ct2_cuda = ct2_devices > 0
    except Exception:
        ct2_cuda = False
        ct2_devices = 0

    # Check PyTorch CUDA support
    torch_cuda = False
    torch_version = None
    cuda_version = None
    torch_devices = 0

    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.cuda.is_available()
        if torch_cuda:
            cuda_version = torch.version.cuda
            torch_devices = torch.cuda.device_count()
    except ImportError:
        pass

    details: dict[str, Any] = {
        "ctranslate2_cuda": ct2_cuda,
        "ctranslate2_devices": ct2_devices,
        "torch_cuda": torch_cuda,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "torch_devices": torch_devices,
    }

    if ct2_cuda and torch_cuda:
        return DoctorCheck(
            name="GPU Support",
            status=CheckStatus.PASS,
            message=f"CUDA available ({ct2_devices} device(s), CUDA {cuda_version})",
            details=details,
        )

    if ct2_cuda and not torch_cuda:
        return DoctorCheck(
            name="GPU Support",
            status=CheckStatus.WARN,
            message="CTranslate2 CUDA available, but PyTorch CUDA not available",
            details=details,
        )

    if not ct2_cuda and torch_cuda:
        return DoctorCheck(
            name="GPU Support",
            status=CheckStatus.WARN,
            message="PyTorch CUDA available, but CTranslate2 CUDA not available (ASR will use CPU)",
            details=details,
        )

    return DoctorCheck(
        name="GPU Support",
        status=CheckStatus.WARN,
        message="No GPU support detected (will use CPU)",
        details=details,
    )


def _check_ffmpeg() -> DoctorCheck:
    """Check ffmpeg availability."""
    import subprocess

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if not ffmpeg_path:
        return DoctorCheck(
            name="FFmpeg",
            status=CheckStatus.FAIL,
            message="ffmpeg not found in PATH (required for audio processing)",
            details={"ffmpeg": None, "ffprobe": ffprobe_path},
        )

    # Get ffmpeg version
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
    except Exception:
        version_line = "unknown"

    if not ffprobe_path:
        return DoctorCheck(
            name="FFmpeg",
            status=CheckStatus.WARN,
            message="ffmpeg found but ffprobe missing (some features may not work)",
            details={"ffmpeg": ffmpeg_path, "ffprobe": None, "version": version_line},
        )

    return DoctorCheck(
        name="FFmpeg",
        status=CheckStatus.PASS,
        message=f"ffmpeg available: {version_line}",
        details={"ffmpeg": ffmpeg_path, "ffprobe": ffprobe_path, "version": version_line},
    )


def _check_model_cache() -> DoctorCheck:
    """Check model cache status."""
    from .cache import CachePaths

    paths = CachePaths.from_env()

    cache_info: dict[str, Any] = {
        "root": str(paths.root),
        "exists": paths.root.exists(),
    }

    models_found = []

    # Check whisper models
    if paths.whisper_root.exists():
        whisper_models = list(paths.whisper_root.glob("*"))
        if whisper_models:
            cache_info["whisper_models"] = [m.name for m in whisper_models]
            models_found.extend([f"whisper/{m.name}" for m in whisper_models])

    # Check HF cache for common models
    hf_hub = paths.hf_home / "hub"
    if hf_hub.exists():
        hf_models = [
            d.name for d in hf_hub.iterdir() if d.is_dir() and d.name.startswith("models--")
        ]
        if hf_models:
            cache_info["hf_models_count"] = len(hf_models)
            models_found.append(f"{len(hf_models)} HF model(s)")

    if not paths.root.exists():
        return DoctorCheck(
            name="Model Cache",
            status=CheckStatus.WARN,
            message="Cache directory not yet created",
            details=cache_info,
        )

    if not models_found:
        return DoctorCheck(
            name="Model Cache",
            status=CheckStatus.WARN,
            message="Cache exists but no models downloaded yet",
            details=cache_info,
        )

    return DoctorCheck(
        name="Model Cache",
        status=CheckStatus.PASS,
        message=f"Cache at {paths.root}, {len(models_found)} model(s) found",
        details=cache_info,
    )


def _check_disk_space() -> DoctorCheck:
    """Check available disk space for cache."""
    from .cache import CachePaths

    paths = CachePaths.from_env()

    # Get disk usage for cache root (or home if not exists)
    check_path = paths.root if paths.root.exists() else Path.home()

    try:
        usage = shutil.disk_usage(check_path)
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        used_pct = (usage.used / usage.total) * 100

        details = {
            "path": str(check_path),
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "used_percent": round(used_pct, 1),
        }

        # Whisper large-v3 is ~3GB, warn if < 5GB free
        if free_gb < 5:
            return DoctorCheck(
                name="Disk Space",
                status=CheckStatus.WARN,
                message=f"Low disk space: {free_gb:.1f} GB free (need ~5GB for large models)",
                details=details,
            )

        return DoctorCheck(
            name="Disk Space",
            status=CheckStatus.PASS,
            message=f"{free_gb:.1f} GB free ({used_pct:.0f}% used)",
            details=details,
        )

    except Exception as e:
        return DoctorCheck(
            name="Disk Space",
            status=CheckStatus.WARN,
            message=f"Could not check disk space: {e}",
            details={"error": str(e)},
        )


def _check_memory() -> DoctorCheck:
    """Check available system memory."""
    try:
        import resource

        # Get memory limits
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        details: dict[str, Any] = {"soft_limit": soft, "hard_limit": hard}

    except Exception:
        details = {}

    # Try to get total system memory
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                        total_gb = total_kb / (1024**2)
                        details["total_gb"] = round(total_gb, 2)
                        break
                    if line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
                        avail_gb = avail_kb / (1024**2)
                        details["available_gb"] = round(avail_gb, 2)
        elif sys.platform == "darwin":
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                total_bytes = int(result.stdout.strip())
                total_gb = total_bytes / (1024**3)
                details["total_gb"] = round(total_gb, 2)
    except Exception:
        pass

    total_gb = details.get("total_gb", 0)
    avail_gb = details.get("available_gb", total_gb)

    # Large-v3 needs ~4GB RAM minimum
    if avail_gb > 0 and avail_gb < 4:
        return DoctorCheck(
            name="Memory",
            status=CheckStatus.WARN,
            message=f"Low memory: {avail_gb:.1f} GB available (4GB+ recommended)",
            details=details,
        )

    if total_gb > 0:
        return DoctorCheck(
            name="Memory",
            status=CheckStatus.PASS,
            message=f"Total: {total_gb:.1f} GB"
            + (f", Available: {avail_gb:.1f} GB" if avail_gb else ""),
            details=details,
        )

    return DoctorCheck(
        name="Memory",
        status=CheckStatus.WARN,
        message="Could not determine system memory",
        details=details,
    )


def _check_numpy_version() -> DoctorCheck:
    """Check for numpy version compatibility issues."""
    try:
        import numpy as np

        version = np.__version__
        major, minor = map(int, version.split(".")[:2])

        details = {"version": version}

        # numpy 2.x can cause issues with some packages
        if major >= 2:
            return DoctorCheck(
                name="NumPy Version",
                status=CheckStatus.WARN,
                message=f"NumPy {version} (v2.x may have compatibility issues with some packages)",
                details=details,
            )

        return DoctorCheck(
            name="NumPy Version",
            status=CheckStatus.PASS,
            message=f"NumPy {version}",
            details=details,
        )

    except ImportError:
        return DoctorCheck(
            name="NumPy Version",
            status=CheckStatus.FAIL,
            message="NumPy not installed",
            details={},
        )


def _check_hf_token() -> DoctorCheck:
    """Check for HuggingFace token (needed for some models)."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if token:
        # Mask token for display
        masked = token[:4] + "..." + token[-4:] if len(token) > 8 else "***"
        return DoctorCheck(
            name="HuggingFace Token",
            status=CheckStatus.PASS,
            message=f"Token set: {masked}",
            details={"token_set": True},
        )

    # Check if token file exists
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return DoctorCheck(
            name="HuggingFace Token",
            status=CheckStatus.PASS,
            message="Token found in cache file",
            details={"token_set": True, "source": "file"},
        )

    return DoctorCheck(
        name="HuggingFace Token",
        status=CheckStatus.WARN,
        message="No HF_TOKEN set (needed for pyannote diarization models)",
        details={"token_set": False},
    )


def _check_platform() -> DoctorCheck:
    """Check platform information."""
    details = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_implementation": platform.python_implementation(),
    }

    return DoctorCheck(
        name="Platform",
        status=CheckStatus.PASS,
        message=f"{platform.system()} {platform.release()} ({platform.machine()})",
        details=details,
    )


def run_doctor() -> DoctorReport:
    """Run all diagnostic checks and return a complete report.

    Returns:
        DoctorReport with all check results and summary.
    """
    report = DoctorReport()

    # Run all checks
    checks = [
        _check_platform,
        _check_python_version,
        _check_required_deps,
        _check_optional_deps,
        _check_numpy_version,
        _check_ffmpeg,
        _check_gpu,
        _check_hf_token,
        _check_model_cache,
        _check_disk_space,
        _check_memory,
    ]

    for check_fn in checks:
        try:
            result = check_fn()
            report.checks.append(result)
        except Exception as e:
            # If a check itself fails, record that
            report.checks.append(
                DoctorCheck(
                    name=check_fn.__name__.replace("_check_", "").replace("_", " ").title(),
                    status=CheckStatus.FAIL,
                    message=f"Check failed with error: {e}",
                    details={"error": str(e)},
                )
            )

    return report


def format_doctor_report(report: DoctorReport, use_color: bool = True) -> str:
    """Format doctor report for human-readable output.

    Args:
        report: DoctorReport to format
        use_color: Whether to use ANSI colors

    Returns:
        Formatted string for terminal output
    """
    from .color_utils import Colors, Symbols

    lines = []
    lines.append("")
    lines.append(Colors.bold("slower-whisper doctor") if use_color else "slower-whisper doctor")
    lines.append("=" * 50)
    lines.append("")

    # Status symbols
    symbols = {
        CheckStatus.PASS: (Colors.green(Symbols.check()) if use_color else Symbols.check()),
        CheckStatus.WARN: (Colors.yellow(Symbols.warn()) if use_color else Symbols.warn()),
        CheckStatus.FAIL: (Colors.red(Symbols.cross()) if use_color else Symbols.cross()),
        CheckStatus.SKIP: (Colors.dim(Symbols.bullet()) if use_color else Symbols.bullet()),
    }

    for check in report.checks:
        symbol = symbols[check.status]
        lines.append(f"  {symbol} {check.name}: {check.message}")

    lines.append("")
    lines.append("-" * 50)

    # Summary
    summary = report.summary
    overall = report.overall_status

    if overall == CheckStatus.PASS:
        overall_text = Colors.green("All checks passed") if use_color else "All checks passed"
    elif overall == CheckStatus.WARN:
        overall_text = Colors.yellow("Some warnings") if use_color else "Some warnings"
    else:
        overall_text = Colors.red("Some checks failed") if use_color else "Some checks failed"

    lines.append(f"  {overall_text}")
    lines.append(f"  Pass: {summary['pass']}, Warn: {summary['warn']}, Fail: {summary['fail']}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Prometheus Metrics Endpoint (for service mode)
# =============================================================================


@dataclass
class PrometheusMetrics:
    """Prometheus-compatible metrics for service mode.

    Tracks request counts, latency histograms, and error counts
    in a format suitable for /metrics endpoint.
    """

    # Request counters by endpoint and status
    request_count: dict[str, int] = field(default_factory=dict)

    # Latency sums and counts for calculating averages
    latency_sum_ms: dict[str, float] = field(default_factory=dict)
    latency_count: dict[str, int] = field(default_factory=dict)

    # Latency histogram buckets (in ms)
    latency_buckets: tuple[float, ...] = (50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000)
    latency_histogram: dict[str, dict[float, int]] = field(default_factory=dict)

    # Active sessions counter
    active_sessions: int = 0

    # Error counter by type
    error_count: dict[str, int] = field(default_factory=dict)

    def record_request(self, endpoint: str, status_code: int, latency_ms: float) -> None:
        """Record a request with its latency.

        Args:
            endpoint: API endpoint name
            status_code: HTTP status code
            latency_ms: Request latency in milliseconds
        """
        key = f"{endpoint}_{status_code}"

        # Increment counter
        self.request_count[key] = self.request_count.get(key, 0) + 1

        # Record latency for average
        self.latency_sum_ms[endpoint] = self.latency_sum_ms.get(endpoint, 0) + latency_ms
        self.latency_count[endpoint] = self.latency_count.get(endpoint, 0) + 1

        # Update histogram
        if endpoint not in self.latency_histogram:
            self.latency_histogram[endpoint] = dict.fromkeys(self.latency_buckets, 0)

        for bucket in self.latency_buckets:
            if latency_ms <= bucket:
                self.latency_histogram[endpoint][bucket] += 1

    def record_error(self, error_type: str) -> None:
        """Record an error by type.

        Args:
            error_type: Error type identifier
        """
        self.error_count[error_type] = self.error_count.get(error_type, 0) + 1

    def session_started(self) -> None:
        """Increment active session count."""
        self.active_sessions += 1

    def session_ended(self) -> None:
        """Decrement active session count."""
        self.active_sessions = max(0, self.active_sessions - 1)

    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus-compatible text format metrics
        """
        lines = []

        # Request count
        lines.append("# HELP slower_whisper_requests_total Total number of requests")
        lines.append("# TYPE slower_whisper_requests_total counter")
        for key, count in self.request_count.items():
            endpoint, status = key.rsplit("_", 1)
            lines.append(
                f'slower_whisper_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}'
            )

        # Latency histogram
        lines.append("# HELP slower_whisper_request_latency_ms Request latency in milliseconds")
        lines.append("# TYPE slower_whisper_request_latency_ms histogram")
        for endpoint, buckets in self.latency_histogram.items():
            cumulative = 0
            for bucket, count in sorted(buckets.items()):
                cumulative += count
                lines.append(
                    f'slower_whisper_request_latency_ms_bucket{{endpoint="{endpoint}",le="{bucket}"}} {cumulative}'
                )
            lines.append(
                f'slower_whisper_request_latency_ms_bucket{{endpoint="{endpoint}",le="+Inf"}} {cumulative}'
            )
            lines.append(
                f'slower_whisper_request_latency_ms_sum{{endpoint="{endpoint}"}} {self.latency_sum_ms.get(endpoint, 0)}'
            )
            lines.append(
                f'slower_whisper_request_latency_ms_count{{endpoint="{endpoint}"}} {self.latency_count.get(endpoint, 0)}'
            )

        # Active sessions
        lines.append("# HELP slower_whisper_active_sessions Number of active streaming sessions")
        lines.append("# TYPE slower_whisper_active_sessions gauge")
        lines.append(f"slower_whisper_active_sessions {self.active_sessions}")

        # Error count
        lines.append("# HELP slower_whisper_errors_total Total number of errors")
        lines.append("# TYPE slower_whisper_errors_total counter")
        for error_type, count in self.error_count.items():
            lines.append(f'slower_whisper_errors_total{{type="{error_type}"}} {count}')

        return "\n".join(lines)


# Global metrics instance for service mode
_global_metrics: PrometheusMetrics | None = None


def get_metrics() -> PrometheusMetrics:
    """Get or create the global metrics instance.

    Returns:
        Global PrometheusMetrics instance for service mode.
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = PrometheusMetrics()
    return _global_metrics


def reset_metrics() -> None:
    """Reset global metrics (mainly for testing)."""
    global _global_metrics
    _global_metrics = None


# =============================================================================
# Receipt Enhancement
# =============================================================================


def add_telemetry_to_receipt(
    receipt: dict[str, Any],
    collector: TelemetryCollector,
) -> dict[str, Any]:
    """Add telemetry data to an existing receipt/metadata dict.

    This function merges telemetry metrics into a transcript's metadata
    in a backward-compatible way. Old receipts without telemetry remain
    valid, and new receipts include telemetry data if collected.

    Args:
        receipt: Existing receipt/metadata dictionary
        collector: TelemetryCollector with metrics to add

    Returns:
        Updated receipt with telemetry field added (if collector has data)

    Example:
        >>> collector = TelemetryCollector()
        >>> collector.record_timing("asr_ms", 1500.0)
        >>> receipt = {"model_name": "large-v3", "device": "cuda"}
        >>> updated = add_telemetry_to_receipt(receipt, collector)
        >>> assert "telemetry" in updated
    """
    telemetry_data = collector.to_dict()

    if not telemetry_data:
        # No telemetry collected, return receipt unchanged
        return receipt

    # Create a copy to avoid mutating the original
    updated_receipt = dict(receipt)
    updated_receipt["telemetry"] = telemetry_data

    return updated_receipt
