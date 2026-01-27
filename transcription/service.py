"""
FastAPI service wrapper for slower-whisper transcription and enrichment.

This module provides a REST API interface to the slower-whisper pipeline,
exposing endpoints for transcription and audio enrichment as HTTP services.

Example usage:
    # Start the service (development mode)
    uvicorn transcription.service:app --reload --host 0.0.0.0 --port 8000

    # Start the service (production mode)
    uvicorn transcription.service:app --workers 4 --host 0.0.0.0 --port 8000

    # Using the API
    curl -X POST -F "audio=@interview.mp3" \
        "http://localhost:8000/transcribe?model=large-v3&language=en"

    curl -X POST -F "transcript=@transcript.json" -F "audio=@audio.wav" \
        "http://localhost:8000/enrich?enable_prosody=true&enable_emotion=true"
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .api import enrich_transcript as _enrich_transcript
from .api import load_transcript, transcribe_file
from .config import EnrichmentConfig, TranscriptionConfig, WhisperTask, validate_compute_type
from .exceptions import ConfigurationError, EnrichmentError, TranscriptionError
from .models import SCHEMA_VERSION, Transcript

# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants
# =============================================================================

# Maximum allowed file size in megabytes (configurable)
MAX_AUDIO_SIZE_MB = 500
MAX_TRANSCRIPT_SIZE_MB = 10  # JSON transcripts are typically small

# Streaming upload chunk size (1MB)
STREAMING_CHUNK_SIZE = 1024 * 1024

# Starlette renamed a couple status constants. Keep runtime compatibility
# without breaking mypy on older/lagging stubs.
HTTP_413_TOO_LARGE: int = getattr(
    status, "HTTP_413_CONTENT_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
)
HTTP_422_UNPROCESSABLE: int = getattr(
    status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY
)


# =============================================================================
# Error Handling Helpers
# =============================================================================


def create_error_response(
    status_code: int,
    error_type: str,
    message: str,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """
    Create a standardized error response.

    Args:
        status_code: HTTP status code
        error_type: Error type identifier (e.g., "validation_error", "transcription_error")
        message: Human-readable error message
        request_id: Optional request ID for tracing
        details: Optional additional error details

    Returns:
        JSONResponse with structured error format
    """
    error_data: dict[str, Any] = {
        "error": {
            "type": error_type,
            "message": message,
            "status_code": status_code,
        },
        # Backward compatibility: include "detail" field for legacy clients
        "detail": message,
    }

    if request_id:
        error_data["error"]["request_id"] = request_id

    if details:
        error_data["error"]["details"] = details

    return JSONResponse(
        status_code=status_code,
        content=error_data,
    )


def validate_file_size(
    file_content: bytes,
    max_size_mb: int,
    file_type: str = "file",
) -> None:
    """
    Validate that uploaded file size does not exceed the maximum allowed size.

    Args:
        file_content: Raw file content bytes
        max_size_mb: Maximum allowed size in megabytes
        file_type: Type of file for error messages (e.g., "audio", "transcript")

    Raises:
        HTTPException: 413 if file exceeds maximum size
    """
    file_size_bytes = len(file_content)
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_mb > max_size_mb:
        logger.warning(
            "File too large: %.2f MB (max: %d MB) for %s",
            file_size_mb,
            max_size_mb,
            file_type,
        )
        raise HTTPException(
            status_code=HTTP_413_TOO_LARGE,
            detail=(
                f"{file_type.capitalize()} file too large: {file_size_mb:.2f} MB. "
                f"Maximum allowed size is {max_size_mb} MB."
            ),
        )


async def save_upload_file_streaming(
    upload: UploadFile,
    dest: Path,
    *,
    max_bytes: int,
    file_type: str = "file",
) -> None:
    """
    Stream an uploaded file to disk in chunks to prevent memory exhaustion.

    This function writes the uploaded file to disk in chunks rather than reading
    the entire file into memory first. This prevents DoS attacks via large file uploads.

    Args:
        upload: The FastAPI UploadFile to read from.
        dest: Destination path to write the file to.
        max_bytes: Maximum allowed file size in bytes.
        file_type: Type of file for error messages (e.g., "audio", "transcript").

    Raises:
        HTTPException: 413 if file exceeds max_bytes.
        HTTPException: 500 if file write fails.
    """
    total = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await upload.read(STREAMING_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=HTTP_413_TOO_LARGE,
                        detail=f"{file_type.capitalize()} file too large: >{max_bytes // (1024 * 1024)} MB",
                    )
                f.write(chunk)
    except HTTPException:
        # Clean up partial file on size error
        if dest.exists():
            dest.unlink()
        raise
    except Exception as e:
        # Clean up partial file on other errors
        if dest.exists():
            dest.unlink()
        logger.error("Failed to save uploaded %s file", file_type, exc_info=e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded {file_type} file",
        ) from e


def validate_audio_format(audio_path: Path) -> None:
    """
    Validate that the uploaded file is a valid audio file.

    This performs a lightweight check by attempting to probe the file with ffmpeg.
    If ffmpeg cannot identify it as audio, we reject it early.
    When ffprobe is not available, we perform basic Python-based validation.

    Args:
        audio_path: Path to the audio file to validate

    Raises:
        HTTPException: 400 if file is not a valid audio format
    """
    import subprocess

    try:
        # Use ffprobe to check if file is valid audio
        # -v error: only show errors
        # -show_entries format=format_name: show format info
        # -of default=noprint_wrappers=1: simple output format
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=format_name,duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0 or not result.stdout.strip():
            logger.warning("Invalid audio file: ffprobe failed to identify format")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid audio file format. Please upload a valid audio file "
                    "(e.g., mp3, wav, m4a, flac, ogg)."
                ),
            )

        # Check that duration is present and reasonable
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            logger.warning("Invalid audio file: missing duration information")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid audio file: unable to determine audio duration.",
            )

        try:
            duration = float(lines[1])
            if duration <= 0:
                logger.warning("Invalid audio file: zero or negative duration")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid audio file: audio has zero or negative duration.",
                )
        except (ValueError, IndexError) as e:
            logger.warning("Invalid audio file: cannot parse duration: %s", e)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid audio file: unable to parse audio duration.",
            ) from e

    except subprocess.TimeoutExpired as e:
        logger.error("Audio validation timeout")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio validation timeout: file may be corrupted or invalid.",
        ) from e
    except FileNotFoundError:
        logger.error("ffprobe not found - using Python-based validation")
        # Security fix: When ffprobe is not available, perform basic Python validation
        # instead of accepting any file
        _validate_audio_format_python(audio_path)


def _validate_audio_format_python(audio_path: Path) -> None:
    """
    Basic Python-based audio file validation when ffprobe is not available.

    This is a fallback validation that checks file headers and basic properties
    to ensure the file is likely a valid audio file.

    Args:
        audio_path: Path to the audio file to validate

    Raises:
        HTTPException: 400 if file appears to be invalid
    """
    try:
        # Check file size (must be larger than 0)
        file_size = audio_path.stat().st_size
        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid audio file: file is empty.",
            )

        # Check file extension against allowed list
        allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
        if audio_path.suffix.lower() not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid audio file extension '{audio_path.suffix}'. "
                    f"Allowed extensions: {', '.join(sorted(allowed_extensions))}"
                ),
            )

        # Basic header validation for common formats
        with open(audio_path, "rb") as f:
            header = f.read(12)  # Read first 12 bytes for header check

            # WAV files start with "RIFF" and have "WAVE" at bytes 8-11
            if audio_path.suffix.lower() == ".wav":
                if len(header) < 12 or not header.startswith(b"RIFF") or header[8:12] != b"WAVE":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid WAV file: incorrect header format.",
                    )

            # MP3 files typically start with ID3 tag or MPEG frame sync
            elif audio_path.suffix.lower() == ".mp3":
                if len(header) < 3 or not (
                    header.startswith(b"ID3") or header.startswith(b"\xff\xfb")
                ):
                    # Not a definitive check, but catches obvious non-MP3 files
                    logger.warning("Possible invalid MP3 file: missing ID3 tag or MPEG sync")

            # M4A/AAC files start with "ftyp" box
            elif audio_path.suffix.lower() in (".m4a", ".aac"):
                if len(header) < 8 or header[4:8] != b"ftyp":
                    logger.warning("Possible invalid M4A/AAC file: missing ftyp box")

        # If we get here, basic checks passed
        logger.info(f"Basic validation passed for {audio_path.name} (ffprobe unavailable)")

    except HTTPException:
        # Re-raise our own validation errors
        raise
    except Exception as e:
        logger.error(f"Error during Python audio validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio file: validation failed.",
        ) from e


def validate_transcript_json(transcript_content: bytes) -> None:
    """
    Validate that the uploaded transcript is valid JSON with expected structure.

    Args:
        transcript_content: Raw transcript file content bytes

    Raises:
        HTTPException: 400 if transcript is not valid JSON or missing required fields
    """
    try:
        data = json.loads(transcript_content)
    except json.JSONDecodeError as e:
        logger.warning("Invalid transcript JSON: %s", e)
        # Security fix: Provide safe parse error location without raw exception
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transcript JSON: parse error at line {e.lineno}, column {e.colno}",
        ) from e

    # Validate required top-level fields
    required_fields = ["file_name", "segments"]
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        logger.warning("Transcript missing required fields: %s", missing_fields)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid transcript structure: missing required fields {missing_fields}. "
                "Expected a transcript JSON from /transcribe endpoint."
            ),
        )

    # Validate segments is a list
    if not isinstance(data.get("segments"), list):
        logger.warning("Transcript 'segments' field is not a list")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid transcript structure: 'segments' must be a list.",
        )

    # Validate segments is not empty
    if len(data["segments"]) == 0:
        logger.warning("Transcript has no segments")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid transcript: no segments found. Transcript must contain at least one segment.",
        )


# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title="Slower-Whisper API",
    description=(
        "REST API for local audio transcription and enrichment. "
        "Transcribe audio files with faster-whisper and optionally extract "
        "prosodic and emotional features from the audio waveform."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# =============================================================================
# Middleware
# =============================================================================


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all requests with timing and request ID.

    This middleware:
    - Generates a unique request_id for tracing
    - Logs request method, path, and query params
    - Measures request duration
    - Logs response status and timing
    - Attaches request_id to request.state for use in exception handlers
    - Records metrics for Prometheus endpoint
    """
    from .telemetry import get_metrics

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Log incoming request
    logger.info(
        "Request started: %s %s [request_id=%s]",
        request.method,
        request.url.path,
        request_id,
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
        },
    )

    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    # Log response
    logger.info(
        "Request completed: %s %s -> %d (%.2f ms) [request_id=%s]",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request_id,
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    # Record metrics for Prometheus
    # Skip metrics endpoint to avoid recursive counting
    if request.url.path != "/metrics":
        metrics = get_metrics()
        metrics.record_request(request.url.path, response.status_code, duration_ms)

    # Add request_id to response headers
    response.headers["X-Request-ID"] = request_id

    return response


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle request validation errors (422).

    FastAPI raises RequestValidationError when request data fails Pydantic validation
    (e.g., invalid query parameters, missing required fields, type mismatches).
    """
    request_id = getattr(request.state, "request_id", None)

    # Extract validation error details
    errors = exc.errors()
    logger.warning(
        "Validation error: %s %s [request_id=%s] - %d validation errors",
        request.method,
        request.url.path,
        request_id,
        len(errors),
        extra={
            "request_id": request_id,
            "validation_errors": errors,
        },
    )

    # Format validation errors for response
    formatted_errors = [
        {
            "loc": list(err.get("loc", [])),
            "msg": err.get("msg", "Validation error"),
            "type": err.get("type", "unknown"),
        }
        for err in errors
    ]

    return create_error_response(
        status_code=HTTP_422_UNPROCESSABLE,
        error_type="validation_error",
        message="Request validation failed",
        request_id=request_id,
        details={"validation_errors": formatted_errors},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle HTTPException (4xx/5xx errors raised by endpoint logic).

    This handler provides structured error responses for all HTTPException instances,
    including those raised explicitly in endpoints (400 Bad Request, 500 Internal Server Error).
    """
    request_id = getattr(request.state, "request_id", None)

    logger.warning(
        "HTTP exception: %s %s -> %d [request_id=%s] - %s",
        request.method,
        request.url.path,
        exc.status_code,
        request_id,
        exc.detail,
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )

    # Determine error type from status code
    error_type_map = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        413: "file_too_large",
        500: "internal_error",
        503: "service_unavailable",
    }
    error_type = error_type_map.get(exc.status_code, "http_error")

    return create_error_response(
        status_code=exc.status_code,
        error_type=error_type,
        message=str(exc.detail),
        request_id=request_id,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all unhandled exceptions (500 Internal Server Error).

    This is the global catch-all handler for any exception not caught by
    endpoint logic or other handlers. Logs full traceback and returns
    generic error to avoid leaking internal details.
    """
    request_id = getattr(request.state, "request_id", None)

    logger.exception(
        "Unhandled exception: %s %s [request_id=%s]",
        request.method,
        request.url.path,
        request_id,
        extra={
            "request_id": request_id,
            "exception_type": type(exc).__name__,
        },
        exc_info=exc,
    )

    # Security fix: Remove exception type information from client response
    # This prevents potential information disclosure about internal implementation
    # Exception details are still logged server-side for debugging
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type="internal_error",
        message="An unexpected internal error occurred",
        request_id=request_id,
        details={
            "hint": "Check server logs for details",
        },
    )


# =============================================================================
# Health Check Endpoints
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


@app.get(
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


@app.get(
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


@app.get(
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


# =============================================================================
# Metrics Endpoint
# =============================================================================


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Export metrics in Prometheus text format for monitoring and alerting.",
    tags=["System"],
    response_class=JSONResponse,
)
async def prometheus_metrics() -> JSONResponse:
    """
    Export metrics in Prometheus text format.

    This endpoint provides operational metrics compatible with Prometheus scraping:
    - Request counts by endpoint and status code
    - Request latency histograms
    - Active streaming session count
    - Error counts by type

    Returns:
        Plain text response in Prometheus exposition format
    """
    from .telemetry import get_metrics

    metrics = get_metrics()
    prometheus_text = metrics.to_prometheus_format()

    return JSONResponse(
        content={"metrics": prometheus_text},
        headers={"Content-Type": "text/plain; charset=utf-8"},
    )


# =============================================================================
# Transcription Endpoint
# =============================================================================


@app.post(
    "/transcribe",
    summary="Transcribe audio file",
    description=(
        "Upload an audio file and receive a transcription in JSON format. "
        "Supports any audio format that ffmpeg can decode (mp3, wav, m4a, etc.)."
    ),
    tags=["Transcription"],
    response_model=None,  # We return the raw Transcript dict
)
async def transcribe_audio(
    audio: Annotated[UploadFile, File(description="Audio file to transcribe")],
    model: Annotated[
        str,
        Query(
            description="Whisper model to use (tiny, base, small, medium, large-v3)",
            examples=["large-v3"],
        ),
    ] = "large-v3",
    language: Annotated[
        str | None,
        Query(
            description="Language code (e.g., 'en', 'es', 'fr'). If null, auto-detect is used.",
            examples=["en", "es", "fr"],
        ),
    ] = None,
    device: Annotated[
        str,
        Query(
            description="Device to use for inference ('cuda' or 'cpu')",
            examples=["cuda", "cpu"],
        ),
    ] = "cpu",
    compute_type: Annotated[
        str | None,
        Query(
            description=(
                "Compute precision override (float16, float32, int8). "
                "Leave empty to auto-select based on device."
            ),
            examples=["float16", "float32"],
        ),
    ] = None,
    task: Annotated[
        str,
        Query(
            description="Task to perform ('transcribe' or 'translate' to English)",
            examples=["transcribe", "translate"],
        ),
    ] = "transcribe",
    enable_diarization: Annotated[
        bool,
        Query(
            description="Run speaker diarization (pyannote.audio)",
            examples=[False, True],
        ),
    ] = False,
    diarization_device: Annotated[
        str,
        Query(
            description="Device for diarization ('cuda', 'cpu', or 'auto')",
            examples=["auto"],
        ),
    ] = "auto",
    min_speakers: Annotated[
        int | None,
        Query(
            description="Minimum number of speakers expected (hint to diarization model)",
            ge=1,
            examples=[2],
        ),
    ] = None,
    max_speakers: Annotated[
        int | None,
        Query(
            description="Maximum number of speakers expected (hint to diarization model)",
            ge=1,
            examples=[4],
        ),
    ] = None,
    overlap_threshold: Annotated[
        float | None,
        Query(
            description="Minimum overlap ratio (0.0-1.0) required to assign a speaker to a segment",
            ge=0.0,
            le=1.0,
            examples=[0.3],
        ),
    ] = None,
    word_timestamps: Annotated[
        bool,
        Query(
            description="Enable word-level timestamps in the response",
            examples=[False, True],
        ),
    ] = False,
) -> JSONResponse:
    """
    Transcribe an uploaded audio file using faster-whisper.

    This endpoint:
    1. Accepts an audio file upload
    2. Normalizes the audio to 16kHz mono WAV
    3. Transcribes using the specified Whisper model
    4. Returns the transcript in JSON format

    Args:
        audio: Uploaded audio file (any format supported by ffmpeg)
        model: Whisper model size (tiny, base, small, medium, large-v3)
        language: Language code for transcription, or None for auto-detect
        device: Device to use ('cuda' for GPU, 'cpu' for CPU)
        compute_type: Precision for model inference
        task: 'transcribe' or 'translate' (to English)
        enable_diarization: Whether to run speaker diarization (pyannote.audio)
        diarization_device: Device for diarization ('cuda', 'cpu', or 'auto')
        min_speakers: Minimum expected speaker count hint
        max_speakers: Maximum expected speaker count hint
        overlap_threshold: Minimum overlap ratio required to assign a speaker
        word_timestamps: Enable word-level timestamps in the response

    Returns:
        JSON response containing the Transcript object with segments and metadata

    Raises:
        400: Invalid configuration or unsupported audio format
        422: Validation error in request parameters
        500: Internal transcription error
    """
    # Validate task
    if task not in ("transcribe", "translate"):
        logger.warning("Invalid task parameter: %s", task)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Must be 'transcribe' or 'translate'.",
        )

    # Validate device
    if device not in ("cuda", "cpu"):
        logger.warning("Invalid device parameter: %s", device)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid device '{device}'. Must be 'cuda' or 'cpu'.",
        )

    # Validate diarization device early for clearer error messages
    if diarization_device not in ("cuda", "cpu", "auto"):
        logger.warning("Invalid diarization_device parameter: %s", diarization_device)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid diarization_device '{diarization_device}'. "
                "Must be 'cuda', 'cpu', or 'auto'."
            ),
        )

    try:
        normalized_compute_type = validate_compute_type(compute_type)
    except ConfigurationError as e:
        logger.warning("Invalid compute_type: %s", compute_type, exc_info=e)
        # Security fix: Do not leak raw exception to client
        raise HTTPException(
            status_code=400,
            detail=f"Invalid compute_type '{compute_type}'. See logs for details.",
        ) from e
    task_value = cast(WhisperTask, task)

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Security fix: Stream file to disk to prevent DoS (memory exhaustion)
        # Determine filename first
        safe_suffix = ""
        if audio.filename:
            # Extract and sanitize the file extension
            import re

            # Match only the last extension after the final dot
            ext_match = re.search(r"(\.[^.]+)$", audio.filename)
            if ext_match:
                ext = ext_match.group(1)
                # Only allow common audio extensions
                allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
                if ext.lower() in allowed_extensions:
                    safe_suffix = ext

        import secrets

        random_id = secrets.token_hex(16)
        audio_path = tmpdir_path / f"audio_{random_id}{safe_suffix}"

        try:
            total_size = 0
            # 1MB chunks
            CHUNK_SIZE = 1024 * 1024
            MAX_BYTES = MAX_AUDIO_SIZE_MB * 1024 * 1024

            with open(audio_path, "wb") as f:
                while True:
                    chunk = await audio.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > MAX_BYTES:
                        raise HTTPException(
                            status_code=HTTP_413_TOO_LARGE,
                            detail=f"File too large: >{MAX_AUDIO_SIZE_MB} MB",
                        )
                    f.write(chunk)

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to save uploaded audio file", exc_info=e)
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=500,
                detail="Failed to save uploaded audio file",
            ) from e

        # Validate audio format
        validate_audio_format(audio_path)

        # Create transcription config
        try:
            extra_kwargs: dict[str, Any] = {}
            if overlap_threshold is not None:
                extra_kwargs["overlap_threshold"] = overlap_threshold

            config = TranscriptionConfig(
                model=model,
                language=language,
                device=device,
                compute_type=normalized_compute_type,
                task=task_value,
                skip_existing_json=False,
                enable_diarization=enable_diarization,
                diarization_device=diarization_device,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                word_timestamps=word_timestamps,
                **extra_kwargs,
            )
        except (ValueError, TypeError) as e:
            logger.warning("Invalid transcription configuration", exc_info=e)
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=400,
                detail="Invalid transcription configuration. Check parameter values.",
            ) from e

        # Transcribe
        try:
            logger.info(
                "Starting transcription: model=%s, language=%s, device=%s, diarization=%s",
                model,
                language,
                device,
                enable_diarization,
            )
            transcript = transcribe_file(
                audio_path=audio_path,
                root=tmpdir_path,
                config=config,
            )
            logger.info(
                "Transcription completed successfully: %d segments", len(transcript.segments)
            )
        except ConfigurationError as e:
            logger.error("Configuration error during transcription", exc_info=e)
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=400,
                detail="Configuration error during transcription. Check parameter values.",
            ) from e
        except TranscriptionError as e:
            logger.error("Transcription failed", exc_info=e)
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=500,
                detail="Transcription failed",
            ) from e
        except Exception as e:
            logger.exception("Unexpected error during transcription")
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=500,
                detail="Unexpected error during transcription",
            ) from e

        # Convert Transcript to JSON-serializable dict
        return JSONResponse(
            content=_transcript_to_dict(transcript, include_words=word_timestamps),
            status_code=200,
        )


# =============================================================================
# SSE Streaming Transcription Endpoint
# =============================================================================


class SSEStreamingSession:
    """
    Manages state for an SSE streaming transcription session.

    Uses the same event envelope format as WebSocket streaming for
    consistency across the API.

    Attributes:
        stream_id: Unique identifier for this stream (sse-{uuid4})
        _event_id_counter: Monotonically increasing event ID
        _segment_seq: Segment sequence counter
    """

    def __init__(self) -> None:
        """Initialize a new SSE streaming session."""
        self.stream_id = f"sse-{uuid.uuid4()}"
        self._event_id_counter = 0
        self._segment_seq = 0
        self._start_time = time.time()
        self._chunks_processed = 0
        self._segments_partial = 0
        self._segments_finalized = 0
        self._errors = 0

    def _next_event_id(self) -> int:
        """Generate next monotonically increasing event ID."""
        self._event_id_counter += 1
        return self._event_id_counter

    def _next_segment_id(self) -> str:
        """Generate next segment ID."""
        seg_id = f"seg-{self._segment_seq}"
        self._segment_seq += 1
        return seg_id

    def _server_timestamp(self) -> int:
        """Get current server timestamp in milliseconds."""
        return int(time.time() * 1000)

    def create_envelope(
        self,
        event_type: str,
        payload: dict[str, Any],
        segment_id: str | None = None,
        ts_audio_start: float | None = None,
        ts_audio_end: float | None = None,
    ) -> dict[str, Any]:
        """Create an event envelope with current metadata."""
        result: dict[str, Any] = {
            "event_id": self._next_event_id(),
            "stream_id": self.stream_id,
            "type": event_type,
            "ts_server": self._server_timestamp(),
            "payload": payload,
        }
        if segment_id is not None:
            result["segment_id"] = segment_id
        if ts_audio_start is not None:
            result["ts_audio_start"] = ts_audio_start
        if ts_audio_end is not None:
            result["ts_audio_end"] = ts_audio_end
        return result

    def get_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        duration = time.time() - self._start_time
        return {
            "segments_partial": self._segments_partial,
            "segments_finalized": self._segments_finalized,
            "errors": self._errors,
            "duration_sec": round(duration, 3),
        }


async def _generate_sse_transcription(
    audio_path: Path,
    config: TranscriptionConfig,
    include_words: bool,
) -> AsyncGenerator[str, None]:
    """Generate SSE events during transcription using event envelope format.

    Uses a thread pool to run the synchronous transcription engine while
    yielding events as segments are produced. Events follow the same
    envelope structure as WebSocket streaming for API consistency.

    Event types emitted:
    - PARTIAL: Intermediate segment (emitted during transcription progress)
    - FINALIZED: Final segment with complete transcription
    - SESSION_ENDED: Stream complete with statistics
    - ERROR: Error occurred during processing

    Args:
        audio_path: Path to the normalized audio file
        config: Transcription configuration
        include_words: Whether to include word-level timestamps

    Yields:
        SSE-formatted event strings: "data: {event_envelope_json}\\n\\n"
    """
    import asyncio
    import queue
    import threading

    from .asr_engine import TranscriptionEngine
    from .config import AsrConfig
    from .diarization_orchestrator import _maybe_run_diarization
    from .transcription_helpers import _maybe_build_chunks

    # Create session for tracking state
    session = SSEStreamingSession()

    # Create a queue to communicate between threads
    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def run_transcription() -> None:
        """Run transcription in a background thread, pushing events to the queue."""
        try:
            # Create ASR configuration
            asr_cfg = AsrConfig(
                model_name=config.model,
                device=config.device,
                compute_type=config.compute_type,
                vad_min_silence_ms=config.vad_min_silence_ms,
                beam_size=config.beam_size,
                language=config.language,
                task=config.task,
                word_timestamps=config.word_timestamps,
            )

            # Initialize the transcription engine
            engine = TranscriptionEngine(asr_cfg)

            # Check if audio file exists
            if not audio_path.exists():
                session._errors += 1
                event_queue.put(
                    session.create_envelope(
                        "ERROR",
                        {
                            "code": "file_not_found",
                            "message": "Audio file not found",
                            "recoverable": False,
                        },
                    )
                )
                event_queue.put(None)
                return

            # Perform transcription using the engine
            logger.info("SSE: Starting transcription for %s", audio_path.name)

            transcript = engine.transcribe_file(audio_path)

            # Emit segments - first as PARTIAL, then final one as FINALIZED
            total_segments = len(transcript.segments)
            for idx, segment in enumerate(transcript.segments):
                seg_dict = _segment_to_dict(segment, include_words=include_words)
                segment_id = session._next_segment_id()
                is_last = idx == total_segments - 1

                # Emit PARTIAL for intermediate segments during processing
                if not is_last:
                    session._segments_partial += 1
                    event_queue.put(
                        session.create_envelope(
                            "PARTIAL",
                            {"segment": seg_dict},
                            segment_id=segment_id,
                            ts_audio_start=segment.start,
                            ts_audio_end=segment.end,
                        )
                    )
                else:
                    # Last segment gets FINALIZED
                    session._segments_finalized += 1
                    event_queue.put(
                        session.create_envelope(
                            "FINALIZED",
                            {"segment": seg_dict},
                            segment_id=segment_id,
                            ts_audio_start=segment.start,
                            ts_audio_end=segment.end,
                        )
                    )

            # Run diarization if enabled
            if config.enable_diarization:
                transcript = _maybe_run_diarization(transcript, audio_path, config)
                transcript = _maybe_build_chunks(transcript, config)

                # Re-emit segments with speaker info as FINALIZED updates
                for segment in transcript.segments:
                    if segment.speaker:
                        seg_dict = _segment_to_dict(segment, include_words=include_words)
                        segment_id = session._next_segment_id()
                        session._segments_finalized += 1
                        event_queue.put(
                            session.create_envelope(
                                "FINALIZED",
                                {"segment": seg_dict},
                                segment_id=segment_id,
                                ts_audio_start=segment.start,
                                ts_audio_end=segment.end,
                            )
                        )

            # Emit SESSION_ENDED event with metadata and statistics
            event_queue.put(
                session.create_envelope(
                    "SESSION_ENDED",
                    {
                        "stats": session.get_stats(),
                        "total_segments": len(transcript.segments),
                        "language": transcript.language,
                        "file_name": transcript.file_name,
                        "speakers": transcript.speakers,
                        "meta": transcript.meta,
                    },
                )
            )

        except Exception as e:
            logger.exception("SSE transcription error: %s", e)
            session._errors += 1
            event_queue.put(
                session.create_envelope(
                    "ERROR",
                    {
                        "code": "transcription_error",
                        "message": "Transcription failed",
                        "recoverable": False,
                    },
                )
            )

        finally:
            # Signal end of stream
            event_queue.put(None)

    # Start transcription in background thread
    thread = threading.Thread(target=run_transcription, daemon=True)
    thread.start()

    # Yield SSE events as they arrive
    try:
        while True:
            try:
                # Use asyncio.to_thread for non-blocking queue access
                event = await asyncio.to_thread(event_queue.get, timeout=0.1)
                if event is None:
                    break
                # SSE format: "data: {json}\n\n"
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # No event yet, continue polling
                continue
            except Exception:
                # Queue access error, break the loop
                break
    finally:
        # Ensure thread cleanup
        thread.join(timeout=1.0)


@app.post(
    "/transcribe/stream",
    summary="Transcribe audio with streaming results (SSE)",
    description=(
        "Upload an audio file and receive transcription results as a stream "
        "of Server-Sent Events (SSE). Events use the same envelope format as "
        "WebSocket streaming for API consistency. Each event contains segment "
        "data as it's transcribed, enabling progressive display of results."
    ),
    tags=["Transcription"],
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE stream of transcription events",
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"event_id": 1, "stream_id": "sse-abc123", "type": "PARTIAL", '
                        '"ts_server": 1705123456789, "segment_id": "seg-0", '
                        '"ts_audio_start": 0.0, "ts_audio_end": 2.5, '
                        '"payload": {"segment": {"id": 0, "start": 0.0, "end": 2.5, '
                        '"text": "Hello world"}}}\n\n'
                        'data: {"event_id": 2, "stream_id": "sse-abc123", "type": "FINALIZED", '
                        '"ts_server": 1705123456800, "segment_id": "seg-1", '
                        '"payload": {"segment": {"id": 1, "start": 2.5, "end": 5.0, '
                        '"text": "This is the final segment."}}}\n\n'
                        'data: {"event_id": 3, "stream_id": "sse-abc123", "type": "SESSION_ENDED", '
                        '"ts_server": 1705123456850, '
                        '"payload": {"stats": {"segments_finalized": 2}, "total_segments": 2}}\n\n'
                    )
                }
            },
        },
        400: {"description": "Invalid configuration or audio format"},
        413: {"description": "File too large"},
        500: {"description": "Internal server error"},
    },
)
async def transcribe_audio_streaming(
    audio: Annotated[UploadFile, File(description="Audio file to transcribe")],
    model: Annotated[
        str,
        Query(
            description="Whisper model to use (tiny, base, small, medium, large-v3)",
            examples=["large-v3"],
        ),
    ] = "large-v3",
    language: Annotated[
        str | None,
        Query(
            description="Language code (e.g., 'en', 'es', 'fr'). If null, auto-detect is used.",
            examples=["en", "es", "fr"],
        ),
    ] = None,
    device: Annotated[
        str,
        Query(
            description="Device to use for inference ('cuda' or 'cpu')",
            examples=["cuda", "cpu"],
        ),
    ] = "cpu",
    compute_type: Annotated[
        str | None,
        Query(
            description=(
                "Compute precision override (float16, float32, int8). "
                "Leave empty to auto-select based on device."
            ),
            examples=["float16", "float32"],
        ),
    ] = None,
    task: Annotated[
        str,
        Query(
            description="Task to perform ('transcribe' or 'translate' to English)",
            examples=["transcribe", "translate"],
        ),
    ] = "transcribe",
    enable_diarization: Annotated[
        bool,
        Query(
            description="Run speaker diarization (pyannote.audio)",
            examples=[False, True],
        ),
    ] = False,
    diarization_device: Annotated[
        str,
        Query(
            description="Device for diarization ('cuda', 'cpu', or 'auto')",
            examples=["auto"],
        ),
    ] = "auto",
    min_speakers: Annotated[
        int | None,
        Query(
            description="Minimum number of speakers expected (hint to diarization model)",
            ge=1,
            examples=[2],
        ),
    ] = None,
    max_speakers: Annotated[
        int | None,
        Query(
            description="Maximum number of speakers expected (hint to diarization model)",
            ge=1,
            examples=[4],
        ),
    ] = None,
    overlap_threshold: Annotated[
        float | None,
        Query(
            description="Minimum overlap ratio (0.0-1.0) required to assign a speaker to a segment",
            ge=0.0,
            le=1.0,
            examples=[0.3],
        ),
    ] = None,
    word_timestamps: Annotated[
        bool,
        Query(
            description="Enable word-level timestamps in the response",
            examples=[False, True],
        ),
    ] = False,
) -> StreamingResponse:
    """
    Stream transcription results as Server-Sent Events (SSE).

    This endpoint provides real-time streaming of transcription results,
    emitting events as each segment is transcribed. Events use the same
    envelope format as WebSocket streaming for API consistency.

    Event types (in envelope.type field):
    - `PARTIAL`: Intermediate segment during transcription
    - `FINALIZED`: Final segment with complete transcription
    - `SESSION_ENDED`: Stream complete with statistics and metadata
    - `ERROR`: Error occurred, includes code, message, and recoverable flag

    Event envelope format:
        {
            "event_id": 1,           // Monotonically increasing per stream
            "stream_id": "sse-...",  // Unique stream identifier
            "type": "PARTIAL",       // Event type
            "ts_server": 17051...,   // Server timestamp (ms)
            "segment_id": "seg-0",   // Segment ID (for segment events)
            "ts_audio_start": 0.0,   // Audio timestamp start (seconds)
            "ts_audio_end": 2.5,     // Audio timestamp end (seconds)
            "payload": {...}         // Event-specific data
        }

    Example SSE format:
        data: {"event_id": 1, "type": "PARTIAL", "payload": {"segment": {...}}}

        data: {"event_id": 2, "type": "FINALIZED", "payload": {"segment": {...}}}

        data: {"event_id": 3, "type": "SESSION_ENDED", "payload": {"stats": {...}}}

    Args:
        audio: Uploaded audio file (any format supported by ffmpeg)
        model: Whisper model size (tiny, base, small, medium, large-v3)
        language: Language code for transcription, or None for auto-detect
        device: Device to use ('cuda' for GPU, 'cpu' for CPU)
        compute_type: Precision for model inference
        task: 'transcribe' or 'translate' (to English)
        enable_diarization: Whether to run speaker diarization (pyannote.audio)
        diarization_device: Device for diarization ('cuda', 'cpu', or 'auto')
        min_speakers: Minimum expected speaker count hint
        max_speakers: Maximum expected speaker count hint
        overlap_threshold: Minimum overlap ratio required to assign a speaker
        word_timestamps: Enable word-level timestamps in the response

    Returns:
        StreamingResponse with Content-Type: text/event-stream

    Raises:
        400: Invalid configuration or unsupported audio format
        413: File too large
        422: Validation error in request parameters
        500: Internal transcription error
    """
    import re
    import secrets

    from .audio_io import normalize_single

    # Validate task
    if task not in ("transcribe", "translate"):
        logger.warning("Invalid task parameter: %s", task)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Must be 'transcribe' or 'translate'.",
        )

    # Validate device
    if device not in ("cuda", "cpu"):
        logger.warning("Invalid device parameter: %s", device)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid device '{device}'. Must be 'cuda' or 'cpu'.",
        )

    # Validate diarization device
    if diarization_device not in ("cuda", "cpu", "auto"):
        logger.warning("Invalid diarization_device parameter: %s", diarization_device)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid diarization_device '{diarization_device}'. "
                "Must be 'cuda', 'cpu', or 'auto'."
            ),
        )

    try:
        normalized_compute_type = validate_compute_type(compute_type)
    except ConfigurationError as e:
        logger.warning("Invalid compute_type: %s", compute_type, exc_info=e)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid compute_type '{compute_type}'. See logs for details.",
        ) from e

    task_value = cast(WhisperTask, task)

    # Create temporary directory for processing
    tmpdir = tempfile.mkdtemp()
    tmpdir_path = Path(tmpdir)

    try:
        # Security: Generate random filename with sanitized extension
        safe_suffix = ""
        if audio.filename:
            ext_match = re.search(r"(\.[^.]+)$", audio.filename)
            if ext_match:
                ext = ext_match.group(1)
                allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
                if ext.lower() in allowed_extensions:
                    safe_suffix = ext

        random_id = secrets.token_hex(16)
        audio_path = tmpdir_path / f"audio_{random_id}{safe_suffix}"

        # Stream file to disk to prevent DoS (memory exhaustion)
        await save_upload_file_streaming(
            audio,
            audio_path,
            max_bytes=MAX_AUDIO_SIZE_MB * 1024 * 1024,
            file_type="audio",
        )

        # Validate audio format
        validate_audio_format(audio_path)

        # Normalize audio to 16kHz mono WAV
        norm_path = tmpdir_path / f"norm_{random_id}.wav"
        try:
            normalize_single(audio_path, norm_path)
        except Exception as e:
            logger.error("Failed to normalize audio", exc_info=e)
            raise HTTPException(
                status_code=400,
                detail="Failed to normalize audio. Ensure ffmpeg is installed.",
            ) from e

        if not norm_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Audio normalization failed: output file not created.",
            )

        # Create transcription config
        extra_kwargs: dict[str, Any] = {}
        if overlap_threshold is not None:
            extra_kwargs["overlap_threshold"] = overlap_threshold

        config = TranscriptionConfig(
            model=model,
            language=language,
            device=device,
            compute_type=normalized_compute_type,
            task=task_value,
            skip_existing_json=False,
            enable_diarization=enable_diarization,
            diarization_device=diarization_device,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            word_timestamps=word_timestamps,
            **extra_kwargs,
        )

        # Create async generator for SSE events
        async def generate_with_cleanup() -> AsyncGenerator[str, None]:
            """Wrapper that ensures cleanup after streaming completes."""
            try:
                async for event in _generate_sse_transcription(norm_path, config, word_timestamps):
                    yield event
            finally:
                # Clean up temporary directory
                import shutil

                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception as cleanup_err:
                    logger.debug("Failed to clean up temp dir: %s", cleanup_err)

        logger.info(
            "Starting SSE streaming transcription: model=%s, device=%s",
            model,
            device,
        )

        return StreamingResponse(
            generate_with_cleanup(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    except HTTPException:
        # Clean up on HTTP errors
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    except Exception as e:
        # Clean up on unexpected errors
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.exception("Unexpected error setting up SSE stream")
        raise HTTPException(
            status_code=500,
            detail="Unexpected error setting up transcription stream",
        ) from e


# =============================================================================
# Enrichment Endpoint
# =============================================================================


@app.post(
    "/enrich",
    summary="Enrich transcript with audio features",
    description=(
        "Upload a transcript JSON and corresponding audio WAV to add "
        "prosodic and emotional features extracted from the audio waveform."
    ),
    tags=["Enrichment"],
    response_model=None,
)
async def enrich_audio(
    transcript: Annotated[
        UploadFile,
        File(description="Transcript JSON file (output from /transcribe)"),
    ],
    audio: Annotated[
        UploadFile,
        File(description="Audio WAV file (16kHz mono, matching the transcript)"),
    ],
    enable_prosody: Annotated[
        bool,
        Query(description="Extract prosodic features (pitch, energy, speech rate)"),
    ] = True,
    enable_emotion: Annotated[
        bool,
        Query(description="Extract dimensional emotion features (valence, arousal, dominance)"),
    ] = True,
    enable_categorical_emotion: Annotated[
        bool,
        Query(description="Extract categorical emotion labels (happy, sad, angry, etc.)"),
    ] = False,
    device: Annotated[
        str,
        Query(description="Device to use for emotion models ('cuda' or 'cpu')"),
    ] = "cpu",
) -> JSONResponse:
    """
    Enrich a transcript with audio-derived prosodic and emotional features.

    This endpoint:
    1. Accepts a transcript JSON (from /transcribe) and audio WAV
    2. Extracts prosodic features (pitch, energy, speech rate) from audio
    3. Extracts emotional features (valence, arousal, dominance) from audio
    4. Populates the 'audio_state' field in each segment
    5. Returns the enriched transcript

    Args:
        transcript: JSON transcript file (schema version 2)
        audio: WAV audio file (16kHz mono, matching the transcript)
        enable_prosody: Whether to extract prosodic features
        enable_emotion: Whether to extract dimensional emotion features
        enable_categorical_emotion: Whether to extract categorical emotion labels
        device: Device for emotion model inference ('cuda' or 'cpu')

    Returns:
        JSON response with enriched Transcript containing audio_state for each segment

    Raises:
        400: Invalid transcript JSON or configuration
        422: Validation error in request parameters
        500: Internal enrichment error
    """
    # Validate device
    if device not in ("cuda", "cpu"):
        logger.warning("Invalid device parameter for enrichment: %s", device)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid device '{device}'. Must be 'cuda' or 'cpu'.",
        )

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Read and validate uploaded transcript
        try:
            transcript_content = await transcript.read()
        except Exception as e:
            logger.error("Failed to read transcript file", exc_info=e)
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=500,
                detail="Failed to read transcript file",
            ) from e

        # Validate transcript file size
        validate_file_size(transcript_content, MAX_TRANSCRIPT_SIZE_MB, file_type="transcript")

        # Validate transcript JSON structure
        validate_transcript_json(transcript_content)

        # Save uploaded transcript
        transcript_path = tmpdir_path / "transcript.json"
        try:
            transcript_path.write_bytes(transcript_content)
        except Exception as e:
            logger.error("Failed to save transcript file", exc_info=e)
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=500,
                detail="Failed to save transcript file",
            ) from e

        # Security fix: Generate random filename to prevent directory traversal
        # Only preserve the file extension from the original filename
        import re
        import secrets

        safe_suffix = ".wav"  # Default to .wav for audio files
        if audio.filename:
            ext_match = re.search(r"(\.[^.]+)$", audio.filename)
            if ext_match:
                ext = ext_match.group(1)
                allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
                if ext.lower() in allowed_extensions:
                    safe_suffix = ext

        random_id = secrets.token_hex(16)
        audio_path = tmpdir_path / f"audio_{random_id}{safe_suffix}"

        # Stream file to disk to prevent DoS (memory exhaustion)
        await save_upload_file_streaming(
            audio,
            audio_path,
            max_bytes=MAX_AUDIO_SIZE_MB * 1024 * 1024,
            file_type="audio",
        )

        # Validate audio format
        validate_audio_format(audio_path)

        # Load transcript
        try:
            transcript_obj = load_transcript(transcript_path)
        except Exception as e:
            logger.warning("Invalid transcript JSON", exc_info=e)
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=400,
                detail="Invalid transcript JSON structure. Ensure valid JSON from /transcribe.",
            ) from e

        # Create enrichment config
        try:
            config = EnrichmentConfig(
                skip_existing=False,
                enable_prosody=enable_prosody,
                enable_emotion=enable_emotion,
                enable_categorical_emotion=enable_categorical_emotion,
                device=device,
            )
        except (ValueError, TypeError) as e:
            logger.warning("Invalid enrichment configuration", exc_info=e)
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=400,
                detail="Invalid enrichment configuration. Check parameter values.",
            ) from e

        # Enrich
        try:
            logger.info(
                "Starting enrichment: prosody=%s, emotion=%s, categorical_emotion=%s, device=%s",
                enable_prosody,
                enable_emotion,
                enable_categorical_emotion,
                device,
            )
            enriched = _enrich_transcript(
                transcript=transcript_obj,
                audio_path=audio_path,
                config=config,
            )
            logger.info(
                "Enrichment completed successfully: %d segments processed", len(enriched.segments)
            )
        except ConfigurationError as e:
            logger.error("Configuration error during enrichment", exc_info=e)
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=400,
                detail="Configuration error during enrichment. Check parameter values.",
            ) from e
        except EnrichmentError as e:
            logger.error("Enrichment failed", exc_info=e)
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=500,
                detail="Enrichment failed",
            ) from e
        except Exception as e:
            logger.exception("Unexpected error during enrichment")
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=500,
                detail="Unexpected error during enrichment",
            ) from e

        # Convert to dict and return
        return JSONResponse(
            content=_transcript_to_dict(enriched),
            status_code=200,
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _word_to_dict(word: Any) -> dict[str, Any]:
    """
    Convert a Word object to a JSON-serializable dictionary.

    Handles both Word dataclass instances and dict representations.
    Uses the canonical Word.to_dict() when available, otherwise
    extracts fields manually for compatibility.

    Args:
        word: Word object (dataclass or dict) to serialize

    Returns:
        Dictionary with word, start, end, probability, and optional speaker
    """
    # Use canonical to_dict() if available
    if hasattr(word, "to_dict"):
        result: dict[str, Any] = word.to_dict()
        return result

    # Handle dict representation directly
    if isinstance(word, dict):
        return word

    # Manual extraction for other object types
    out: dict[str, Any] = {}
    for key in ("word", "start", "end", "probability", "speaker"):
        val = getattr(word, key, None)
        if val is not None:
            out[key] = val

    # Some models use 'text' instead of 'word'
    if "word" not in out and hasattr(word, "text"):
        out["word"] = word.text

    return out


def _segment_to_dict(seg: Any, *, include_words: bool) -> dict[str, Any]:
    """
    Convert a Segment object to a JSON-serializable dictionary.

    Args:
        seg: Segment object to serialize
        include_words: If True, include word-level timestamps in output

    Returns:
        Dictionary representation of the segment
    """
    d: dict[str, Any] = {
        "id": seg.id,
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "speaker": seg.speaker,
        "tone": seg.tone,
        "audio_state": seg.audio_state,
    }

    # Include words only when requested and present
    if include_words and getattr(seg, "words", None):
        d["words"] = [_word_to_dict(w) for w in seg.words]

    return d


def _transcript_to_dict(
    transcript: Transcript,
    *,
    include_words: bool = False,
) -> dict[str, Any]:
    """
    Convert a Transcript dataclass to a JSON-serializable dictionary.

    Args:
        transcript: Transcript object to serialize
        include_words: If True, include word-level timestamps in segments

    Returns:
        Dictionary representation suitable for JSON response
    """
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "file_name": transcript.file_name,
        "language": transcript.language,
        "meta": transcript.meta or {},
        "segments": [
            _segment_to_dict(seg, include_words=include_words) for seg in transcript.segments
        ],
    }

    # Include optional diarization fields when present (v1.1+)
    if transcript.speakers is not None:
        data["speakers"] = transcript.speakers
    if transcript.turns is not None:
        data["turns"] = transcript.turns

    return data


# =============================================================================
# WebSocket Streaming Endpoint (v2.0.0)
# =============================================================================


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time audio streaming and transcription.

    This endpoint enables bidirectional communication for streaming audio
    transcription. Clients send audio chunks and receive transcription
    events in real-time.

    Protocol:
        1. Client connects to /stream
        2. Client sends START_SESSION with configuration
        3. Server responds with SESSION_STARTED
        4. Client sends AUDIO_CHUNK messages with base64-encoded audio
        5. Server sends PARTIAL and FINALIZED segment events
        6. Client sends END_SESSION to finalize
        7. Server sends SESSION_ENDED with statistics
        8. Connection closes

    Client Message Types:
        - START_SESSION: {type, config: {max_gap_sec, enable_prosody, ...}}
        - AUDIO_CHUNK: {type, data: base64_string, sequence: int}
        - END_SESSION: {type}
        - PING: {type, timestamp: int}

    Server Message Types:
        - SESSION_STARTED: {event_id, stream_id, type, ts_server, payload: {session_id}}
        - PARTIAL: {event_id, stream_id, segment_id, type, ts_server, ts_audio_*, payload: {segment}}
        - FINALIZED: {event_id, stream_id, segment_id, type, ts_server, ts_audio_*, payload: {segment}}
        - ERROR: {event_id, stream_id, type, ts_server, payload: {code, message, recoverable}}
        - SESSION_ENDED: {event_id, stream_id, type, ts_server, payload: {stats}}
        - PONG: {event_id, stream_id, type, ts_server, payload: {timestamp, server_timestamp}}

    Example (JavaScript):
        const ws = new WebSocket('ws://localhost:8000/stream');
        ws.onopen = () => {
            ws.send(JSON.stringify({
                type: 'START_SESSION',
                config: {max_gap_sec: 1.0, enable_prosody: true}
            }));
        };
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'FINALIZED') {
                console.log('Segment:', msg.payload.segment.text);
            }
        };
    """
    from .streaming_ws import (
        ClientMessageType,
        WebSocketSessionConfig,
        WebSocketStreamingSession,
        decode_audio_chunk,
        parse_client_message,
    )

    await websocket.accept()
    logger.info("WebSocket connection accepted")

    session: WebSocketStreamingSession | None = None

    try:
        while True:
            # Receive message from client
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                # Re-raise to be handled by outer exception handler
                raise
            except Exception as e:
                logger.warning("Failed to receive/parse WebSocket message: %s", e)
                if session:
                    error_event = session.create_error_event(
                        code="invalid_message",
                        message=f"Failed to parse message: {e}",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                continue

            # Parse message type and payload
            try:
                msg_type, payload = parse_client_message(data)
            except ValueError as e:
                logger.warning("Invalid client message: %s", e)
                if session:
                    error_event = session.create_error_event(
                        code="invalid_message_type",
                        message=str(e),
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                continue

            # Handle message based on type
            if msg_type == ClientMessageType.START_SESSION:
                if session is not None:
                    error_event = session.create_error_event(
                        code="session_already_started",
                        message="Session already started",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                    continue

                # Parse configuration from payload
                config_data = payload.get("config", {})
                config = WebSocketSessionConfig.from_dict(config_data)

                # Create and start session
                session = WebSocketStreamingSession(config=config)
                try:
                    start_event = await session.start()
                    await websocket.send_json(start_event.to_dict())
                    logger.info(
                        "WebSocket session started: stream_id=%s",
                        session.stream_id,
                    )
                except Exception as e:
                    logger.error("Failed to start session: %s", e, exc_info=True)
                    error_event = session.create_error_event(
                        code="session_start_failed",
                        message=f"Failed to start session: {e}",
                        recoverable=False,
                    )
                    await websocket.send_json(error_event.to_dict())
                    session = None

            elif msg_type == ClientMessageType.AUDIO_CHUNK:
                if session is None:
                    # Create temporary session just to send error
                    temp_session = WebSocketStreamingSession()
                    error_event = temp_session.create_error_event(
                        code="no_session",
                        message="No active session. Send START_SESSION first.",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                    continue

                try:
                    audio_bytes, sequence = decode_audio_chunk(payload)

                    # Check backpressure before processing
                    if session.check_backpressure():
                        # Drop partial events when under backpressure
                        dropped = session.drop_partial_events()
                        if dropped > 0:
                            # Emit buffer overflow error to inform client
                            overflow_event = session.create_buffer_overflow_error()
                            await websocket.send_json(overflow_event.to_dict())

                    events = await session.process_audio_chunk(audio_bytes, sequence)
                    for event in events:
                        await websocket.send_json(event.to_dict())
                except ValueError as e:
                    logger.warning("Invalid audio chunk: %s", e)
                    error_event = session.create_error_event(
                        code="invalid_audio_chunk",
                        message=str(e),
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                except Exception as e:
                    logger.error("Error processing audio chunk: %s", e, exc_info=True)
                    error_event = session.create_error_event(
                        code="processing_error",
                        message=f"Error processing audio: {e}",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())

            elif msg_type == ClientMessageType.END_SESSION:
                if session is None:
                    temp_session = WebSocketStreamingSession()
                    error_event = temp_session.create_error_event(
                        code="no_session",
                        message="No active session to end.",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                    continue

                try:
                    end_events = await session.end()
                    for event in end_events:
                        await websocket.send_json(event.to_dict())
                    logger.info(
                        "WebSocket session ended: stream_id=%s",
                        session.stream_id,
                    )
                except Exception as e:
                    logger.error("Error ending session: %s", e, exc_info=True)
                    error_event = session.create_error_event(
                        code="end_session_error",
                        message=f"Error ending session: {e}",
                        recoverable=False,
                    )
                    await websocket.send_json(error_event.to_dict())

                # Close connection after END_SESSION
                break

            elif msg_type == ClientMessageType.RESUME_SESSION:
                # Handle session resume after reconnection
                requested_session_id = payload.get("session_id")
                last_event_id = payload.get("last_event_id", 0)

                if session is None:
                    temp_session = WebSocketStreamingSession()
                    error_event = temp_session.create_error_event(
                        code="no_session",
                        message="No active session to resume. Send START_SESSION first.",
                        recoverable=True,
                    )
                    await websocket.send_json(error_event.to_dict())
                    continue

                # Validate session ID matches current session
                if requested_session_id != session.stream_id:
                    error_event = session.create_error_event(
                        code="session_mismatch",
                        message=(
                            f"Session ID mismatch: requested '{requested_session_id}' "
                            f"but current session is '{session.stream_id}'"
                        ),
                        recoverable=False,
                    )
                    await websocket.send_json(error_event.to_dict())
                    await websocket.close()
                    break

                # Get events for resume
                events_to_replay, gap_detected = session.get_events_for_resume(last_event_id)

                if gap_detected:
                    # Cannot resume - gap in event buffer
                    gap_error = session.create_resume_gap_error(last_event_id)
                    await websocket.send_json(gap_error.to_dict())
                    await websocket.close()
                    break

                # Replay missed events
                logger.info(
                    "Replaying %d events for resume: stream_id=%s, from_event_id=%d",
                    len(events_to_replay),
                    session.stream_id,
                    last_event_id,
                )
                for event in events_to_replay:
                    await websocket.send_json(event.to_dict())

                # Continue normal operation after replay

            elif msg_type == ClientMessageType.PING:
                if session is None:
                    temp_session = WebSocketStreamingSession()
                    pong_event = temp_session.create_pong_event(payload.get("timestamp", 0))
                    await websocket.send_json(pong_event.to_dict())
                else:
                    pong_event = session.create_pong_event(payload.get("timestamp", 0))
                    await websocket.send_json(pong_event.to_dict())

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected: stream_id=%s",
            session.stream_id if session else "no_session",
        )
        # Clean up session if it was active
        if session and session.state.value == "active":
            try:
                await session.end()
            except Exception as e:
                logger.warning("Error cleaning up session on disconnect: %s", e)

    except Exception:
        logger.exception("Unexpected error in WebSocket handler")
        if session:
            try:
                error_event = session.create_error_event(
                    code="internal_error",
                    message="An unexpected error occurred",
                    recoverable=False,
                )
                await websocket.send_json(error_event.to_dict())
            except Exception:
                pass  # Connection may already be closed

    finally:
        logger.info("WebSocket connection closed")


# =============================================================================
# WebSocket Session Management Endpoints (v2.0.0)
# =============================================================================


@app.get(
    "/stream/config",
    summary="Get default streaming configuration",
    description="Returns the default configuration for WebSocket streaming sessions.",
    tags=["Streaming"],
)
async def get_stream_config() -> JSONResponse:
    """
    Get default streaming configuration.

    Returns the default values for WebSocket streaming session configuration.
    Clients can use this to understand available options before connecting.

    Returns:
        JSONResponse with default configuration options.
    """
    from .streaming_ws import WebSocketSessionConfig

    default_config = WebSocketSessionConfig()
    return JSONResponse(
        status_code=200,
        content={
            "default_config": {
                "max_gap_sec": default_config.max_gap_sec,
                "enable_prosody": default_config.enable_prosody,
                "enable_emotion": default_config.enable_emotion,
                "enable_categorical_emotion": default_config.enable_categorical_emotion,
                "sample_rate": default_config.sample_rate,
                "audio_format": default_config.audio_format,
            },
            "supported_audio_formats": ["pcm_s16le"],
            "supported_sample_rates": [16000],
            "message_types": {
                "client": ["START_SESSION", "AUDIO_CHUNK", "END_SESSION", "PING"],
                "server": [
                    "SESSION_STARTED",
                    "PARTIAL",
                    "FINALIZED",
                    "SPEAKER_TURN",
                    "SEMANTIC_UPDATE",
                    "ERROR",
                    "SESSION_ENDED",
                    "PONG",
                ],
            },
        },
    )


# =============================================================================
# REST Session Management Endpoints (Issue #85)
# =============================================================================


@app.get(
    "/stream/sessions",
    summary="List active streaming sessions",
    description="Returns a list of all registered streaming sessions and their status.",
    tags=["Streaming"],
)
async def list_sessions() -> JSONResponse:
    """
    List all registered streaming sessions.

    Returns session info including status, config, and stats for each session.
    This endpoint provides visibility into active WebSocket sessions via REST.

    Returns:
        JSONResponse with list of sessions.
    """
    from .session_registry import get_registry

    registry = get_registry()
    sessions = registry.list_sessions()

    return JSONResponse(
        status_code=200,
        content={
            "sessions": [s.to_dict() for s in sessions],
            "count": len(sessions),
            "registry_stats": registry.get_stats(),
        },
    )


@app.post(
    "/stream/sessions",
    summary="Create a new streaming session",
    description="Creates a new streaming session for subsequent WebSocket connection.",
    tags=["Streaming"],
)
async def create_session(
    max_gap_sec: float = Query(default=1.0, ge=0.1, le=10.0),
    enable_prosody: bool = Query(default=False),
    enable_emotion: bool = Query(default=False),
    enable_diarization: bool = Query(default=False),
    sample_rate: int = Query(default=16000, ge=8000, le=48000),
) -> JSONResponse:
    """
    Create a new streaming session.

    Creates a session that can be connected to via WebSocket at /stream.
    The session_id returned should be passed in the WebSocket connection.

    Args:
        max_gap_sec: Gap threshold to finalize segment (0.1-10.0 seconds)
        enable_prosody: Extract prosodic features from audio
        enable_emotion: Extract dimensional emotion features
        enable_diarization: Enable incremental speaker diarization
        sample_rate: Expected audio sample rate (8000-48000 Hz)

    Returns:
        JSONResponse with session_id and WebSocket URL.
    """
    from .session_registry import get_registry
    from .streaming_ws import WebSocketSessionConfig, WebSocketStreamingSession

    config = WebSocketSessionConfig(
        max_gap_sec=max_gap_sec,
        enable_prosody=enable_prosody,
        enable_emotion=enable_emotion,
        enable_diarization=enable_diarization,
        sample_rate=sample_rate,
    )

    session = WebSocketStreamingSession(config=config)
    registry = get_registry()
    session_id = registry.register(session)

    logger.info("Created session via REST: %s", session_id)

    return JSONResponse(
        status_code=201,
        content={
            "session_id": session_id,
            "websocket_url": f"/stream?session_id={session_id}",
            "config": {
                "max_gap_sec": config.max_gap_sec,
                "enable_prosody": config.enable_prosody,
                "enable_emotion": config.enable_emotion,
                "enable_diarization": config.enable_diarization,
                "sample_rate": config.sample_rate,
                "audio_format": config.audio_format,
            },
        },
    )


@app.get(
    "/stream/sessions/{session_id}",
    summary="Get streaming session status",
    description="Returns detailed status and statistics for a specific session.",
    tags=["Streaming"],
)
async def get_session_status(session_id: str) -> JSONResponse:
    """
    Get status of a specific streaming session.

    Returns detailed information about the session including:
    - Current state (created, active, ending, ended, error, disconnected)
    - Configuration used
    - Runtime statistics
    - Last event ID (for resume)

    Args:
        session_id: Session ID to look up

    Returns:
        JSONResponse with session info.

    Raises:
        HTTPException: 404 if session not found.
    """
    from .session_registry import get_registry

    registry = get_registry()
    info = registry.get_info(session_id)

    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    return JSONResponse(
        status_code=200,
        content=info.to_dict(),
    )


@app.delete(
    "/stream/sessions/{session_id}",
    summary="Close streaming session",
    description="Force closes a streaming session and cleans up resources.",
    tags=["Streaming"],
)
async def close_session(session_id: str) -> JSONResponse:
    """
    Force close a streaming session.

    Ends the session if active, closes the WebSocket connection if connected,
    and removes the session from the registry.

    Args:
        session_id: Session ID to close

    Returns:
        JSONResponse confirming closure.

    Raises:
        HTTPException: 404 if session not found.
    """
    from .session_registry import get_registry

    registry = get_registry()

    # Check if session exists first
    info = registry.get_info(session_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    success = await registry.close_session(session_id)

    if success:
        logger.info("Closed session via REST: %s", session_id)
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Session {session_id} closed",
                "session_id": session_id,
            },
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to close session: {session_id}",
        )


# =============================================================================
# Main Entry Point (for development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "transcription.service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
