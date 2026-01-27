"""Transcription endpoints for the API service."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from .config import TranscriptionConfig, WhisperTask, validate_compute_type
from .exceptions import ConfigurationError, TranscriptionError
from .service_sse import _generate_sse_transcription

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_service_module():
    from . import service as _service

    return _service


# =============================================================================
# Transcription Endpoint
# =============================================================================


@router.post(
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
    service = _get_service_module()

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
            chunk_size = 1024 * 1024
            max_bytes = service.MAX_AUDIO_SIZE_MB * 1024 * 1024

            with open(audio_path, "wb") as f:
                while True:
                    chunk = await audio.read(chunk_size)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > max_bytes:
                        raise HTTPException(
                            status_code=service.HTTP_413_TOO_LARGE,
                            detail=f"File too large: >{service.MAX_AUDIO_SIZE_MB} MB",
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
        service.validate_audio_format(audio_path)

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
            transcript = service.transcribe_file(
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
            content=service._transcript_to_dict(transcript, include_words=word_timestamps),
            status_code=200,
        )


# =============================================================================
# SSE Streaming Transcription Endpoint
# =============================================================================


@router.post(
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

    service = _get_service_module()

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
        await service.save_upload_file_streaming(
            audio,
            audio_path,
            max_bytes=service.MAX_AUDIO_SIZE_MB * 1024 * 1024,
            file_type="audio",
        )

        # Validate audio format
        service.validate_audio_format(audio_path)

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
