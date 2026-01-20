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
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"{file_type.capitalize()} file too large: {file_size_mb:.2f} MB. "
                f"Maximum allowed size is {max_size_mb} MB."
            ),
        )


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
    """
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
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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

        # Read and validate uploaded audio file
        try:
            content = await audio.read()
        except Exception as e:
            logger.error("Failed to read uploaded audio file", exc_info=e)
            raise HTTPException(
                status_code=500,
                detail="Failed to read uploaded audio file",
            ) from e

        # Validate file size before processing
        validate_file_size(content, MAX_AUDIO_SIZE_MB, file_type="audio")

        # Security fix: Generate random filename to prevent directory traversal
        # Only preserve the file extension from the original filename
        # Sanitize the extension to prevent path traversal
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

        # Generate a secure random filename with the sanitized extension
        import secrets

        random_id = secrets.token_hex(16)
        audio_path = tmpdir_path / f"audio_{random_id}{safe_suffix}"

        try:
            audio_path.write_bytes(content)
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

        # Read and validate uploaded audio
        try:
            audio_content = await audio.read()
        except Exception as e:
            logger.error("Failed to read audio file", exc_info=e)
            # Security fix: Do not leak exception details to client
            raise HTTPException(
                status_code=500,
                detail="Failed to read audio file",
            ) from e

        # Validate audio file size
        validate_file_size(audio_content, MAX_AUDIO_SIZE_MB, file_type="audio")

        # Security fix: Generate random filename to prevent directory traversal
        # Only preserve the file extension from the original filename
        # Sanitize the extension to prevent path traversal
        safe_suffix = ".wav"  # Default to .wav for audio files
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

        # Generate a secure random filename with the sanitized extension
        import secrets

        random_id = secrets.token_hex(16)
        audio_path = tmpdir_path / f"audio_{random_id}{safe_suffix}"

        try:
            audio_path.write_bytes(audio_content)
        except Exception as e:
            logger.error("Failed to save audio file", exc_info=e)
            raise HTTPException(
                status_code=500,
                detail="Failed to save audio file",
            ) from e

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
