"""REST transcription through the process-owned ASR runtime.

The legacy ``service_transcribe`` module remains the authority for the public
file-orchestration patch points and for the separate SSE endpoint. This module
owns only the batch REST transaction that must reuse the service runtime.
"""

from __future__ import annotations

import logging
import secrets
import tempfile
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from . import service_transcribe as _legacy_transcribe
from .config import TranscriptionConfig, WhisperTask, validate_compute_type
from .exceptions import (
    ASRInferenceError,
    ASRModelLoadError,
    ASROutputError,
    ASRUnavailableError,
    ConfigurationError,
    RuntimeNotReadyError,
    RuntimeProfileMismatchError,
    TranscriptionError,
)
from .service_errors import create_error_response
from .service_serialization import _transcript_to_dict
from .service_settings import HTTP_413_TOO_LARGE, MAX_AUDIO_SIZE_MB
from .source_identity import safe_source_name

logger = logging.getLogger(__name__)
router = APIRouter()

_ALLOWED_AUDIO_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
}


def _request_runtime(request: Request) -> Any:
    """Return the process runtime or raise one sanitized typed failure."""
    runtime = getattr(request.app.state, "asr_runtime", None)
    startup_error = getattr(request.app.state, "asr_startup_error", None)
    if runtime is None:
        context: dict[str, Any] = {"state": "failed" if startup_error else "stopped"}
        if isinstance(startup_error, TranscriptionError):
            context["startup_error"] = startup_error.public_details()
        raise RuntimeNotReadyError(
            "The configured ASR runtime is not ready",
            context=context,
        )

    if not bool(getattr(runtime, "ready", False)):
        status = runtime.status()
        context = {"state": status.get("state", "unknown")}
        error = status.get("error")
        if isinstance(error, dict):
            context["startup_error"] = error
        raise RuntimeNotReadyError(
            "The configured ASR runtime is not ready",
            context=context,
        )
    return runtime


def _typed_error_response(request: Request, error: TranscriptionError) -> JSONResponse:
    """Preserve the stable domain reason while withholding provider detail."""
    if isinstance(error, RuntimeProfileMismatchError):
        status_code = 409
        message = "Request does not match the configured ASR profile"
    elif isinstance(
        error,
        (ASRUnavailableError, ASRModelLoadError, RuntimeNotReadyError),
    ):
        status_code = 503
        message = "ASR runtime is not ready"
    elif isinstance(error, (ASRInferenceError, ASROutputError)):
        status_code = 500
        message = "Transcription failed"
    else:
        status_code = 500
        message = "Transcription failed"

    return create_error_response(
        status_code=status_code,
        error_type=error.reason_code,
        message=message,
        request_id=getattr(request.state, "request_id", None),
        details=error.public_details(),
    )


def _safe_audio_suffix(filename: str | None) -> str:
    if not filename:
        return ""
    safe_name = safe_source_name(filename)
    suffix = Path(safe_name).suffix.lower()
    return suffix if suffix in _ALLOWED_AUDIO_SUFFIXES else ""


@router.post(
    "/transcribe",
    summary="Transcribe audio file",
    description=(
        "Upload audio and transcribe it through the process-owned ASR runtime. "
        "Explicit ASR overrides must match the configured process profile."
    ),
    tags=["Transcription"],
    response_model=None,
)
async def transcribe_audio(
    audio: Annotated[UploadFile, File(description="Audio file to transcribe")],
    request: Request,
    model: Annotated[str | None, Query(description="Configured Whisper model")] = None,
    language: Annotated[str | None, Query(description="Configured language hint")] = None,
    device: Annotated[
        str | None,
        Query(description="Configured ASR device ('cuda' or 'cpu')"),
    ] = None,
    compute_type: Annotated[
        str | None,
        Query(description="Configured compute precision"),
    ] = None,
    task: Annotated[str | None, Query(description="Configured ASR task")] = None,
    enable_diarization: Annotated[
        bool,
        Query(description="Run request-scoped speaker diarization"),
    ] = False,
    diarization_device: Annotated[
        str,
        Query(description="Diarization device ('cuda', 'cpu', or 'auto')"),
    ] = "auto",
    min_speakers: Annotated[int | None, Query(ge=1)] = None,
    max_speakers: Annotated[int | None, Query(ge=1)] = None,
    overlap_threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    word_timestamps: Annotated[
        bool | None,
        Query(description="Configured word-timestamp behavior"),
    ] = None,
) -> JSONResponse:
    """Transcribe one uploaded file through the service-owned runtime."""
    if task is not None and task not in {"transcribe", "translate"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid task. Must be 'transcribe' or 'translate'.",
        )
    if device is not None and device not in {"cuda", "cpu"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid device. Must be 'cuda' or 'cpu'.",
        )
    if diarization_device not in {"cuda", "cpu", "auto"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid diarization_device. Must be 'cuda', 'cpu', or 'auto'.",
        )

    try:
        normalized_compute_type = (
            validate_compute_type(compute_type) if compute_type is not None else None
        )
    except ConfigurationError as error:
        logger.warning("Invalid compute_type", exc_info=error)
        raise HTTPException(
            status_code=400,
            detail="Invalid compute_type. Check the supported values.",
        ) from error

    try:
        runtime = _request_runtime(request)
        profile = runtime.profile
        config_kwargs: dict[str, Any] = {}
        if overlap_threshold is not None:
            config_kwargs["overlap_threshold"] = overlap_threshold
        config = TranscriptionConfig(
            model=model if model is not None else profile.model,
            language=language if language is not None else profile.language,
            device=device if device is not None else profile.device,
            compute_type=(
                normalized_compute_type
                if normalized_compute_type is not None
                else profile.compute_type
            ),
            task=cast(
                WhisperTask,
                task if task is not None else profile.task,
            ),
            beam_size=profile.beam_size,
            vad_min_silence_ms=profile.vad_min_silence_ms,
            skip_existing_json=False,
            enable_diarization=enable_diarization,
            diarization_device=diarization_device,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            word_timestamps=(
                word_timestamps if word_timestamps is not None else profile.word_timestamps
            ),
            **config_kwargs,
        )
        runtime.assert_profile(config)
    except TranscriptionError as error:
        return _typed_error_response(request, error)
    except (TypeError, ValueError) as error:
        logger.warning("Invalid transcription configuration", exc_info=error)
        raise HTTPException(
            status_code=400,
            detail="Invalid transcription configuration. Check parameter values.",
        ) from error

    source_name = safe_source_name(audio.filename)
    audio_suffix = _safe_audio_suffix(source_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        audio_path = tmpdir_path / f"audio_{secrets.token_hex(16)}{audio_suffix}"
        try:
            await _legacy_transcribe.save_upload_file_streaming(
                audio,
                audio_path,
                max_bytes=MAX_AUDIO_SIZE_MB * 1024 * 1024,
                file_type="audio",
            )
        except HTTPException as error:
            if error.status_code == HTTP_413_TOO_LARGE:
                raise HTTPException(
                    status_code=HTTP_413_TOO_LARGE,
                    detail=f"File too large: >{MAX_AUDIO_SIZE_MB} MB",
                ) from error
            raise

        _legacy_transcribe.validate_audio_format(audio_path)

        try:
            transcript = await runtime.transcribe_file(
                _legacy_transcribe.transcribe_file,
                audio_path=audio_path,
                root=tmpdir_path,
                config=config,
            )
        except ConfigurationError as error:
            logger.error("Configuration error during transcription", exc_info=error)
            raise HTTPException(
                status_code=400,
                detail="Configuration error during transcription.",
            ) from error
        except TranscriptionError as error:
            logger.error("Transcription failed", exc_info=error)
            return _typed_error_response(request, error)
        except Exception as error:
            logger.exception("Unexpected error during transcription")
            raise HTTPException(
                status_code=500,
                detail="Unexpected error during transcription",
            ) from error

        transcript.file_name = source_name
        transcript.meta = dict(transcript.meta or {})
        transcript.meta["audio_file"] = source_name

        return JSONResponse(
            content=_transcript_to_dict(
                transcript,
                include_words=config.word_timestamps,
            ),
            status_code=200,
        )
