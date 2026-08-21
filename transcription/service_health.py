"""Health routes grounded in the service-owned ASR runtime."""

from __future__ import annotations

import shutil
from importlib import resources
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import __version__
from .exceptions import TranscriptionError
from .models import SCHEMA_VERSION

router = APIRouter()


def _check_ffmpeg() -> dict[str, Any]:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return {"status": "ok", "path": ffmpeg_path}
    return {"status": "error", "message": "ffmpeg not found on PATH"}


def _check_faster_whisper() -> dict[str, Any]:
    """Report importability; runtime state remains readiness authority."""
    try:
        from . import asr_engine

        if asr_engine._FASTER_WHISPER_AVAILABLE:
            return {"status": "ok"}
        return {"status": "error", "message": "faster-whisper import failed"}
    except Exception:  # noqa: BLE001 - never expose import internals remotely
        return {"status": "error", "message": "faster-whisper check failed"}


def _check_cuda(device: str) -> dict[str, Any]:
    if device != "cuda":
        return {"status": "ok", "message": f"CUDA not required (device={device})"}

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
    except Exception:  # noqa: BLE001 - sanitized health surface
        return {"status": "error", "message": "CUDA check failed"}


def _check_disk_space() -> dict[str, Any]:
    try:
        from .cache import CachePaths

        paths = CachePaths.from_env()
        root_usage = shutil.disk_usage(paths.root.parent)
        free_gb = root_usage.free / (1024**3)
        total_gb = root_usage.total / (1024**3)
        used_gb = root_usage.used / (1024**3)
        percent_used = (used_gb / total_gb * 100) if total_gb > 0 else 0
        status = "warning" if free_gb < 5.0 else "ok"
        return {
            "status": status,
            "cache_root": str(paths.root),
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "percent_used": round(percent_used, 1),
        }
    except Exception:  # noqa: BLE001 - sanitized health surface
        return {"status": "error", "message": "Disk space check failed"}


def _check_package_resources() -> dict[str, Any]:
    required = (
        "schemas/transcript-v2.schema.json",
        "schemas/stream_event.schema.json",
    )
    try:
        package_root = resources.files("transcription")
        missing = [name for name in required if not package_root.joinpath(name).is_file()]
    except Exception:  # noqa: BLE001 - sanitized health surface
        return {"status": "error", "message": "Package resource check failed"}
    if missing:
        return {
            "status": "error",
            "message": "Required package resources are missing",
            "missing": missing,
        }
    return {"status": "ok", "resources": list(required)}


def _runtime_check(request: Request) -> dict[str, Any]:
    runtime = getattr(request.app.state, "asr_runtime", None)
    if runtime is None:
        startup_error = getattr(request.app.state, "asr_startup_error", None)
        result: dict[str, Any] = {
            "status": "error",
            "state": "failed" if startup_error else "stopped",
            "ready": False,
        }
        if isinstance(startup_error, TranscriptionError):
            result["error"] = startup_error.public_details()
        return result

    status = runtime.status()
    return {
        "status": "ok" if status.get("ready") else "error",
        **status,
    }


def _runtime_device(runtime_check: dict[str, Any]) -> str:
    selected = runtime_check.get("selected") or {}
    profile = runtime_check.get("profile") or {}
    value = selected.get("device") or profile.get("device")
    return value if isinstance(value, str) and value else "cpu"


@router.get(
    "/health",
    summary="Health check (legacy)",
    description="Simple process health check (deprecated; use /health/live and /health/ready)",
    tags=["System"],
    deprecated=True,
)
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "slower-whisper-api",
        "version": __version__,
        "schema_version": str(SCHEMA_VERSION),
    }


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Process-only liveness probe; it performs no model work",
    tags=["System"],
    status_code=200,
)
async def health_liveness() -> JSONResponse:
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
    description="Ready only when the configured process-owned ASR runtime is usable",
    tags=["System"],
    responses={200: {"description": "Service is ready"}, 503: {"description": "Not ready"}},
)
async def health_readiness(request: Request) -> JSONResponse:
    runtime = _runtime_check(request)
    checks: dict[str, Any] = {
        "ffmpeg": _check_ffmpeg(),
        "resources": _check_package_resources(),
        "runtime": runtime,
        "faster_whisper": _check_faster_whisper(),
        "cuda": _check_cuda(_runtime_device(runtime)),
        "disk_space": _check_disk_space(),
    }

    critical_names = ("ffmpeg", "resources", "runtime")
    healthy = all(checks[name]["status"] == "ok" for name in critical_names)
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ready" if healthy else "degraded",
            "healthy": healthy,
            "service": "slower-whisper-api",
            "version": __version__,
            "schema_version": str(SCHEMA_VERSION),
            "checks": checks,
        },
    )
