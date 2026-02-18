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

from fastapi import FastAPI

from . import __version__
from . import models as _models
from . import service_enrich as _service_enrich
from . import service_errors as _service_errors
from . import service_health as _service_health
from . import service_serialization as _service_serialization
from . import service_settings as _service_settings
from . import service_transcribe as _service_transcribe
from .api import enrich_transcript as _enrich_transcript
from .api import load_transcript, transcribe_file
from .config import EnrichmentConfig, TranscriptionConfig, WhisperTask, validate_compute_type
from .exceptions import ConfigurationError, EnrichmentError, TranscriptionError
from .service_metrics import router as metrics_router
from .service_middleware import add_security_headers, log_requests
from .service_streaming import router as streaming_router
from .service_validation import (
    save_upload_file_streaming,
    validate_audio_format,
    validate_file_size,
    validate_transcript_json,
)

# =============================================================================
# Configuration Constants (re-exported for backwards compatibility)
# =============================================================================

MAX_AUDIO_SIZE_MB = _service_settings.MAX_AUDIO_SIZE_MB
MAX_TRANSCRIPT_SIZE_MB = _service_settings.MAX_TRANSCRIPT_SIZE_MB
STREAMING_CHUNK_SIZE = _service_settings.STREAMING_CHUNK_SIZE
HTTP_413_TOO_LARGE = _service_settings.HTTP_413_TOO_LARGE
HTTP_422_UNPROCESSABLE = _service_settings.HTTP_422_UNPROCESSABLE
SCHEMA_VERSION = _models.SCHEMA_VERSION
Transcript = _models.Transcript

# =============================================================================
# Backwards-compatible helpers (re-exported for tests)
# =============================================================================

create_error_response = _service_errors.create_error_response
register_exception_handlers = _service_errors.register_exception_handlers

_word_to_dict = _service_serialization._word_to_dict
_segment_to_dict = _service_serialization._segment_to_dict
_transcript_to_dict = _service_serialization._transcript_to_dict

# Health check helpers (re-exported for tests)
health_router = _service_health.router
_check_ffmpeg = _service_health._check_ffmpeg
_check_faster_whisper = _service_health._check_faster_whisper
_check_cuda = _service_health._check_cuda
_check_disk_space = _service_health._check_disk_space

# Transcription/enrichment routers + endpoint callables
transcribe_router = _service_transcribe.router
transcribe_audio = _service_transcribe.transcribe_audio
transcribe_audio_streaming = _service_transcribe.transcribe_audio_streaming

enrich_router = _service_enrich.router
enrich_audio = _service_enrich.enrich_audio

# =============================================================================
# FastAPI Application Setup
# =============================================================================

tags_metadata = [
    {
        "name": "Transcription",
        "description": "Core transcription endpoints using faster-whisper. Supports batch file upload and streaming SSE.",
    },
    {
        "name": "Enrichment",
        "description": "Audio analysis endpoints for extracting prosody, emotion, and speaker statistics.",
    },
    {
        "name": "Streaming",
        "description": "Real-time streaming endpoints (WebSocket) for live audio transcription.",
    },
    {
        "name": "System",
        "description": "Health checks and system metrics.",
    },
]

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
    openapi_tags=tags_metadata,
)

# =============================================================================
# Middleware & Exception Handlers
# =============================================================================

app.middleware("http")(log_requests)
app.middleware("http")(add_security_headers)
register_exception_handlers(app)

# =============================================================================
# Routers (system + streaming + core endpoints)
# =============================================================================

app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(streaming_router)
app.include_router(transcribe_router)
app.include_router(enrich_router)

__all__ = [
    "app",
    "MAX_AUDIO_SIZE_MB",
    "MAX_TRANSCRIPT_SIZE_MB",
    "STREAMING_CHUNK_SIZE",
    "HTTP_413_TOO_LARGE",
    "HTTP_422_UNPROCESSABLE",
    "SCHEMA_VERSION",
    "Transcript",
    "create_error_response",
    "register_exception_handlers",
    "_word_to_dict",
    "_segment_to_dict",
    "_transcript_to_dict",
    "save_upload_file_streaming",
    "validate_audio_format",
    "validate_file_size",
    "validate_transcript_json",
    "transcribe_file",
    "load_transcript",
    "_enrich_transcript",
    "transcribe_audio",
    "transcribe_audio_streaming",
    "enrich_audio",
    "_check_ffmpeg",
    "_check_faster_whisper",
    "_check_cuda",
    "_check_disk_space",
    "EnrichmentConfig",
    "TranscriptionConfig",
    "WhisperTask",
    "validate_compute_type",
    "ConfigurationError",
    "EnrichmentError",
    "TranscriptionError",
]


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
