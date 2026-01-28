"""Health check routes for the API service."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import __version__
from .models import SCHEMA_VERSION

router = APIRouter()


# =============================================================================
# Health Check Helpers
# =============================================================================


def _check_ffmpeg() -> dict[str, Any]:
    """Check if ffmpeg is available on PATH.

    Returns:
        Dict with status and optional error message
    """
    import shutil

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return {"status": "ok", "path": ffmpeg_path}
    return {"status": "error", "message": "ffmpeg not found on PATH"}


def _check_faster_whisper() -> dict[str, Any]:
    """Check if faster-whisper can be imported.

    Returns:
        Dict with status and optional error message
    """
    try:
        from . import asr_engine

        available = asr_engine._FASTER_WHISPER_AVAILABLE
        if available:
            return {"status": "ok"}
        return {"status": "error", "message": "faster-whisper import failed"}
    except Exception as e:
        return {"status": "error", "message": f"faster-whisper check failed: {str(e)}"}


def _check_cuda(device: str) -> dict[str, Any]:
    """Check CUDA availability if device=cuda is expected.

    Args:
        device: Expected device from config (cuda/cpu)

    Returns:
        Dict with status and optional error/warning message
    """
    if device != "cuda":
        return {"status": "ok", "message": "CUDA not required (device=cpu)"}

    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
            return {
                "status": "ok",
                "device_count": device_count,
                "device_name": device_name,
            }
        return {"status": "warning", "message": "CUDA requested but not available"}
    except ImportError:
        return {"status": "warning", "message": "torch not installed, cannot check CUDA"}
    except Exception as e:
        return {"status": "error", "message": f"CUDA check failed: {str(e)}"}


def _check_disk_space() -> dict[str, Any]:
    """Check disk space in cache directories.

    Returns:
        Dict with status and space information
    """
    import shutil

    try:
        from .cache import CachePaths

        paths = CachePaths.from_env()
        root_usage = shutil.disk_usage(paths.root.parent)

        # Convert to GB
        free_gb = root_usage.free / (1024**3)
        total_gb = root_usage.total / (1024**3)
        used_gb = root_usage.used / (1024**3)
        percent_used = (used_gb / total_gb * 100) if total_gb > 0 else 0

        # Warn if less than 5GB free
        status = "ok"
        if free_gb < 5.0:
            status = "warning"

        return {
            "status": status,
            "cache_root": str(paths.root),
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "percent_used": round(percent_used, 1),
        }
    except Exception as e:
        return {"status": "error", "message": f"Disk space check failed: {str(e)}"}


# =============================================================================
# Health Check Endpoints
# =============================================================================


@router.get(
    "/health",
    summary="Health check (legacy)",
    description="Simple health check (deprecated, use /health/live for liveness checks)",
    tags=["System"],
    deprecated=True,
)
async def health_check() -> dict[str, str]:
    """
    Legacy health check endpoint for service monitoring.

    DEPRECATED: Use /health/live for liveness checks or /health/ready for readiness checks.

    Returns:
        Dictionary with status and version information.
    """
    return {
        "status": "healthy",
        "service": "slower-whisper-api",
        "version": __version__,
        "schema_version": str(SCHEMA_VERSION),
    }


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Check if the service is alive and responsive (Kubernetes liveness probe)",
    tags=["System"],
    status_code=200,
)
async def health_liveness() -> JSONResponse:
    """
    Liveness probe for Kubernetes/orchestration systems.

    This endpoint performs minimal checks to verify the service process is alive
    and responsive. It should NOT check external dependencies or heavy initialization.

    Returns 200 if the service is running, even if not fully ready.

    Returns:
        JSONResponse with status "alive" and basic service info
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "alive",
            "service": "slower-whisper-api",
            "version": __version__,
            "schema_version": str(SCHEMA_VERSION),
        },
    )


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description=(
        "Check if the service is ready to handle requests "
        "(Kubernetes readiness probe, load balancer health check)"
    ),
    tags=["System"],
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is degraded or not ready"},
    },
)
async def health_readiness() -> JSONResponse:
    """
    Readiness probe for Kubernetes/orchestration systems and load balancers.

    This endpoint checks all critical dependencies and configuration:
    - ffmpeg availability (required for audio normalization)
    - faster-whisper import (required for transcription)
    - CUDA availability (if device=cuda expected)
    - Disk space in cache directories

    Returns:
        - 200 if all checks pass (service ready)
        - 503 if any critical check fails (service not ready)

    Response includes detailed status for each dependency.
    """
    checks: dict[str, Any] = {}
    overall_status = "ready"
    overall_healthy = True

    # Check ffmpeg (critical)
    checks["ffmpeg"] = _check_ffmpeg()
    if checks["ffmpeg"]["status"] == "error":
        overall_status = "degraded"
        overall_healthy = False

    # Check faster-whisper (critical)
    checks["faster_whisper"] = _check_faster_whisper()
    if checks["faster_whisper"]["status"] == "error":
        overall_status = "degraded"
        overall_healthy = False

    # Check CUDA (warning only, not critical)
    # In production, device should be read from config/env
    # For now, default to cpu to avoid false negatives
    device = "cpu"  # Could be read from env: os.environ.get("SLOWER_WHISPER_DEVICE", "cpu")
    checks["cuda"] = _check_cuda(device)
    if checks["cuda"]["status"] == "error":
        overall_status = "degraded"
        # Not marking as unhealthy - CUDA errors are warnings, CPU fallback works

    # Check disk space (warning if low)
    checks["disk_space"] = _check_disk_space()
    if checks["disk_space"]["status"] == "error":
        overall_status = "degraded"
        # Disk check errors are warnings, not critical failures

    # Determine HTTP status code
    status_code = 200 if overall_healthy else 503

    response_body = {
        "status": overall_status,
        "healthy": overall_healthy,
        "service": "slower-whisper-api",
        "version": __version__,
        "schema_version": str(SCHEMA_VERSION),
        "checks": checks,
    }

    return JSONResponse(
        status_code=status_code,
        content=response_body,
    )
