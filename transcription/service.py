"""FastAPI service wrapper for slower-whisper.

The service owns one configured ASR runtime per process. Use one worker per
instance while model and streaming session state remain in process.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from . import __version__
from . import models as _models
from . import service_enrich as _service_enrich
from . import service_errors as _service_errors
from . import service_health as _service_health
from . import service_runtime_transcribe as _service_runtime_transcribe
from . import service_serialization as _service_serialization
from . import service_settings as _service_settings
from . import service_transcribe as _service_transcribe
from .api import enrich_transcript as _enrich_transcript
from .api import load_transcript, transcribe_file
from .config import EnrichmentConfig, TranscriptionConfig, WhisperTask, validate_compute_type
from .exceptions import (
    ConfigurationError,
    EnrichmentError,
    RuntimeNotReadyError,
    TranscriptionError,
)
from .service_metrics import router as metrics_router
from .service_middleware import add_security_headers, log_requests
from .service_runtime import ASRRuntime, RuntimeProfile
from .service_runtime_streaming import router as streaming_router
from .service_validation import (
    save_upload_file_streaming,
    validate_audio_format,
    validate_file_size,
    validate_transcript_json,
)

RuntimeFactory = Callable[[], ASRRuntime]

MAX_AUDIO_SIZE_MB = _service_settings.MAX_AUDIO_SIZE_MB
MAX_TRANSCRIPT_SIZE_MB = _service_settings.MAX_TRANSCRIPT_SIZE_MB
STREAMING_CHUNK_SIZE = _service_settings.STREAMING_CHUNK_SIZE
HTTP_413_TOO_LARGE = _service_settings.HTTP_413_TOO_LARGE
HTTP_422_UNPROCESSABLE = _service_settings.HTTP_422_UNPROCESSABLE
SCHEMA_VERSION = _models.SCHEMA_VERSION
Transcript = _models.Transcript

create_error_response = _service_errors.create_error_response
register_exception_handlers = _service_errors.register_exception_handlers

_word_to_dict = _service_serialization._word_to_dict
_segment_to_dict = _service_serialization._segment_to_dict
_transcript_to_dict = _service_serialization._transcript_to_dict

health_router = _service_health.router
_check_ffmpeg = _service_health._check_ffmpeg
_check_faster_whisper = _service_health._check_faster_whisper
_check_cuda = _service_health._check_cuda
_check_disk_space = _service_health._check_disk_space

transcribe_router = _service_runtime_transcribe.router
transcribe_audio = _service_runtime_transcribe.transcribe_audio
transcribe_audio_streaming = _service_transcribe.transcribe_audio_streaming

sse_transcribe_router = APIRouter()
for _route in _service_transcribe.router.routes:
    if getattr(_route, "path", None) != "/transcribe":
        sse_transcribe_router.routes.append(_route)

enrich_router = _service_enrich.router
enrich_audio = _service_enrich.enrich_audio


def build_service_runtime() -> ASRRuntime:
    """Build the process-owned runtime from the public environment config."""
    config = TranscriptionConfig.from_env()
    raw_limit = os.getenv("SLOWER_WHISPER_SERVICE_MAX_CONCURRENCY", "1")
    try:
        max_concurrency = int(raw_limit)
    except ValueError as exc:
        raise ConfigurationError(
            "SLOWER_WHISPER_SERVICE_MAX_CONCURRENCY must be an integer"
        ) from exc
    if max_concurrency < 1:
        raise ConfigurationError("SLOWER_WHISPER_SERVICE_MAX_CONCURRENCY must be at least 1")

    profile = RuntimeProfile.from_config(config)
    print(
        f"[preflight] model={profile.model} device={profile.device} "
        f"compute_type={profile.compute_type} max_concurrency={max_concurrency}",
        file=sys.stderr,
    )
    return ASRRuntime(profile, max_concurrency=max_concurrency)


def create_app(*, runtime_factory: RuntimeFactory | None = None) -> FastAPI:
    """Create a service app with an injectable process-runtime factory."""
    factory = runtime_factory or build_service_runtime

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime: ASRRuntime | None = None
        application.state.asr_runtime = None
        application.state.asr_startup_error = None
        try:
            runtime = factory()
            application.state.asr_runtime = runtime
            await runtime.start()
        except TranscriptionError as exc:
            application.state.asr_startup_error = exc
        except Exception as exc:  # noqa: BLE001 - app stays live but not ready
            startup_error = RuntimeNotReadyError(
                "The ASR service runtime could not be configured",
                context={"phase": "startup"},
            )
            startup_error.__cause__ = exc
            application.state.asr_startup_error = startup_error

        try:
            yield
        finally:
            if runtime is not None:
                await runtime.close()

    application = FastAPI(
        title="Slower-Whisper API",
        description=(
            "REST API for local audio transcription and enrichment. "
            "One configured ASR model lifecycle is owned per service process."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    application.state.runtime_factory = factory
    application.state.asr_runtime = None
    application.state.asr_startup_error = None

    application.middleware("http")(log_requests)
    application.middleware("http")(add_security_headers)
    register_exception_handlers(application)

    application.include_router(health_router)
    application.include_router(metrics_router)
    application.include_router(streaming_router)
    application.include_router(transcribe_router)
    application.include_router(sse_transcribe_router)
    application.include_router(enrich_router)
    return application


app = create_app()

__all__ = [
    "app",
    "create_app",
    "build_service_runtime",
    "ASRRuntime",
    "RuntimeProfile",
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "transcription.service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
