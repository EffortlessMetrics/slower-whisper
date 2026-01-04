"""Device detection and resolution for ASR inference.

This module provides unified device selection logic with clear fallback
reasons and preflight banner formatting.

The key insight is that we check CTranslate2's CUDA availability (not torch's)
since faster-whisper uses CTranslate2 as its backend. A system may have CUDA
drivers but CTranslate2 compiled without CUDA support, or vice versa.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Valid device choices for CLI
DeviceChoice = Literal["auto", "cpu", "cuda"]

# Valid compute types for faster-whisper/CTranslate2
ComputeTypeChoice = Literal["auto", "float16", "float32", "int8", "int8_float16", "int8_float32"]


@dataclass
class ResolvedDevice:
    """Result of device resolution with fallback tracking.

    Attributes:
        device: The resolved device ("cuda" or "cpu")
        compute_type: The resolved compute type
        requested_device: What the user originally requested
        fallback_reason: If device differs from requested, explains why
        cuda_available: Whether CUDA is available in the backend
        cuda_device_count: Number of CUDA devices detected
    """

    device: str
    compute_type: str
    requested_device: str
    fallback_reason: str | None = None
    cuda_available: bool = False
    cuda_device_count: int = 0
    extra_info: dict[str, str] = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool:
        """True if we fell back from the requested device."""
        return self.fallback_reason is not None


def _detect_ctranslate2_cuda() -> tuple[bool, int, str | None]:
    """Detect CUDA availability via CTranslate2 (the actual ASR backend).

    Returns:
        Tuple of (cuda_available, device_count, error_reason)

    Note:
        Uses dynamic import via importlib to avoid mypy errors from untyped
        ctranslate2 package. This also gracefully handles missing installs.
    """
    import importlib

    try:
        ct2 = importlib.import_module("ctranslate2")
        get_count = getattr(ct2, "get_cuda_device_count", None)
        if get_count is None:
            return False, 0, "CTranslate2 missing get_cuda_device_count"
        device_count = int(get_count())
        if device_count > 0:
            return True, device_count, None
        else:
            return False, 0, "No CUDA devices found by CTranslate2"
    except ImportError:
        return False, 0, "CTranslate2 not installed"
    except Exception as e:
        return False, 0, f"CTranslate2 CUDA detection failed: {e}"


def _detect_torch_cuda() -> tuple[bool, str | None]:
    """Detect CUDA availability via PyTorch (for emotion/diarization).

    Returns:
        Tuple of (cuda_available, error_reason)
    """
    try:
        import torch

        if torch.cuda.is_available():
            return True, None
        else:
            return False, "torch.cuda.is_available() returned False"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"PyTorch CUDA detection failed: {e}"


def resolve_device(
    requested: DeviceChoice = "auto",
    *,
    allow_fallback: bool = True,
    backend: Literal["ctranslate2", "torch"] = "ctranslate2",
) -> ResolvedDevice:
    """Resolve requested device to actual device with fallback handling.

    Args:
        requested: User's device preference ("auto", "cpu", "cuda")
        allow_fallback: If True, fall back to CPU when CUDA unavailable.
                       If False, raise RuntimeError when CUDA requested but unavailable.
        backend: Which backend to check for CUDA availability.
                "ctranslate2" for ASR (Whisper), "torch" for emotion/diarization.

    Returns:
        ResolvedDevice with device, compute_type, and fallback info

    Raises:
        RuntimeError: If CUDA requested, unavailable, and allow_fallback=False
    """
    # Detect CUDA availability based on backend
    if backend == "ctranslate2":
        cuda_available, cuda_count, cuda_error = _detect_ctranslate2_cuda()
    else:
        cuda_available, cuda_error = _detect_torch_cuda()
        cuda_count = 0  # torch doesn't provide count in same way

    # Resolve "auto" to actual device
    if requested == "auto":
        if cuda_available:
            device = "cuda"
            fallback_reason = None
        else:
            device = "cpu"
            fallback_reason = cuda_error or "CUDA not available"
        requested_for_tracking = "auto"
    elif requested == "cuda":
        if cuda_available:
            device = "cuda"
            fallback_reason = None
        elif allow_fallback:
            device = "cpu"
            fallback_reason = cuda_error or "CUDA not available"
            logger.warning(
                "CUDA requested but unavailable (%s); falling back to CPU",
                fallback_reason,
            )
        else:
            msg = f"CUDA requested but unavailable: {cuda_error or 'CUDA not available'}"
            raise RuntimeError(msg)
        requested_for_tracking = "cuda"
    else:  # requested == "cpu"
        device = "cpu"
        fallback_reason = None
        requested_for_tracking = "cpu"

    # Auto-derive compute type
    compute_type = resolve_compute_type("auto", device)

    return ResolvedDevice(
        device=device,
        compute_type=compute_type,
        requested_device=requested_for_tracking,
        fallback_reason=fallback_reason,
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
    )


def resolve_compute_type(
    requested: ComputeTypeChoice | str | None,
    device: str,
) -> str:
    """Resolve compute type based on request and device.

    Args:
        requested: User's compute type preference or "auto"/None
        device: The resolved device ("cuda" or "cpu")

    Returns:
        Resolved compute type string
    """
    if requested is None or requested == "auto":
        # Sensible defaults: int8 for CPU (speed), float16 for CUDA (quality)
        return "int8" if device == "cpu" else "float16"
    return requested


def format_preflight_banner(
    resolved: ResolvedDevice,
    model_name: str,
    *,
    cache_dir: str | None = None,
    verbose: bool = False,
) -> str:
    """Format a preflight banner showing device configuration.

    Args:
        resolved: The resolved device configuration
        model_name: Whisper model name (e.g., "large-v3")
        cache_dir: Optional model cache directory
        verbose: If True, include additional details

    Returns:
        Formatted banner string (may be multi-line if verbose)
    """
    lines: list[str] = []

    # Main status line
    device_display = resolved.device.upper()
    if resolved.is_fallback:
        status = f"[Device] {device_display} (fallback from {resolved.requested_device})"
    else:
        status = f"[Device] {device_display}"

    main_line = f"{status} | compute_type={resolved.compute_type} | model={model_name}"
    lines.append(main_line)

    # Fallback reason if applicable
    if resolved.fallback_reason:
        lines.append(f"         └─ Reason: {resolved.fallback_reason}")

    # Verbose details
    if verbose:
        if resolved.cuda_available:
            lines.append(f"         └─ CUDA devices: {resolved.cuda_device_count}")
        if cache_dir:
            lines.append(f"         └─ Cache: {cache_dir}")
        for key, value in resolved.extra_info.items():
            lines.append(f"         └─ {key}: {value}")

    return "\n".join(lines)


def get_device_summary(resolved: ResolvedDevice) -> dict[str, str | int | bool | None]:
    """Get device resolution as a dict for metadata/logging.

    Args:
        resolved: The resolved device configuration

    Returns:
        Dict suitable for JSON serialization or logging
    """
    return {
        "device": resolved.device,
        "compute_type": resolved.compute_type,
        "requested_device": resolved.requested_device,
        "fallback_reason": resolved.fallback_reason,
        "cuda_available": resolved.cuda_available,
        "cuda_device_count": resolved.cuda_device_count,
    }
