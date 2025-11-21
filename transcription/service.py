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

import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from .api import enrich_transcript as _enrich_transcript
from .api import load_transcript, transcribe_file
from .config import EnrichmentConfig, TranscriptionConfig, validate_compute_type
from .exceptions import ConfigurationError, EnrichmentError, TranscriptionError
from .models import SCHEMA_VERSION, Transcript

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
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# =============================================================================
# Health Check Endpoint
# =============================================================================


@app.get(
    "/health",
    summary="Health check",
    description="Check if the service is running and responsive",
    tags=["System"],
)
async def health_check() -> dict[str, str]:
    """
    Health check endpoint for service monitoring and load balancers.

    Returns:
        Dictionary with status and version information.
    """
    return {
        "status": "healthy",
        "service": "slower-whisper-api",
        "version": "1.0.0",
        "schema_version": str(SCHEMA_VERSION),
    }


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
        str,
        Query(
            description="Compute precision (float16, float32, int8)",
            examples=["float16", "float32"],
        ),
    ] = "float32",
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

    Returns:
        JSON response containing the Transcript object with segments and metadata

    Raises:
        400: Invalid configuration or unsupported audio format
        422: Validation error in request parameters
        500: Internal transcription error
    """
    # Validate task
    if task not in ("transcribe", "translate"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Must be 'transcribe' or 'translate'.",
        )

    # Validate device
    if device not in ("cuda", "cpu"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid device '{device}'. Must be 'cuda' or 'cpu'.",
        )

    # Validate diarization device early for clearer error messages
    if diarization_device not in ("cuda", "cpu", "auto"):
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
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Save uploaded file
        audio_path = tmpdir_path / "uploaded_audio"
        if audio.filename:
            # Preserve extension if available
            suffix = Path(audio.filename).suffix
            audio_path = tmpdir_path / f"uploaded_audio{suffix}"

        try:
            content = await audio.read()
            audio_path.write_bytes(content)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save uploaded audio file: {str(e)}",
            ) from e

        # Create transcription config
        try:
            config_kwargs = {
                "model": model,
                "language": language,
                "device": device,
                "compute_type": normalized_compute_type,
                "task": task,  # type: ignore
                "skip_existing_json": False,
            }
            config_kwargs.update(
                {
                    "enable_diarization": enable_diarization,
                    "diarization_device": diarization_device,
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                }
            )
            if overlap_threshold is not None:
                config_kwargs["overlap_threshold"] = overlap_threshold

            config = TranscriptionConfig(**config_kwargs)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid configuration: {str(e)}",
            ) from e

        # Transcribe
        try:
            transcript = transcribe_file(
                audio_path=audio_path,
                root=tmpdir_path,
                config=config,
            )
        except ConfigurationError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration error: {str(e)}",
            ) from e
        except TranscriptionError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {str(e)}",
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error during transcription: {str(e)}",
            ) from e

        # Convert Transcript to JSON-serializable dict
        return JSONResponse(
            content=_transcript_to_dict(transcript),
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
        raise HTTPException(
            status_code=400,
            detail=f"Invalid device '{device}'. Must be 'cuda' or 'cpu'.",
        )

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Save uploaded transcript
        transcript_path = tmpdir_path / "transcript.json"
        try:
            transcript_content = await transcript.read()
            transcript_path.write_bytes(transcript_content)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save transcript file: {str(e)}",
            ) from e

        # Save uploaded audio
        audio_path = tmpdir_path / "audio.wav"
        try:
            audio_content = await audio.read()
            audio_path.write_bytes(audio_content)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save audio file: {str(e)}",
            ) from e

        # Load transcript
        try:
            transcript_obj = load_transcript(transcript_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transcript JSON: {str(e)}",
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
            raise HTTPException(
                status_code=400,
                detail=f"Invalid enrichment configuration: {str(e)}",
            ) from e

        # Enrich
        try:
            enriched = _enrich_transcript(
                transcript=transcript_obj,
                audio_path=audio_path,
                config=config,
            )
        except ConfigurationError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration error: {str(e)}",
            ) from e
        except EnrichmentError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Enrichment failed: {str(e)}",
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error during enrichment: {str(e)}",
            ) from e

        # Convert to dict and return
        return JSONResponse(
            content=_transcript_to_dict(enriched),
            status_code=200,
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _transcript_to_dict(transcript: Transcript) -> dict[str, Any]:
    """
    Convert a Transcript dataclass to a JSON-serializable dictionary.

    Args:
        transcript: Transcript object to serialize

    Returns:
        Dictionary representation suitable for JSON response
    """
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "file_name": transcript.file_name,
        "language": transcript.language,
        "meta": transcript.meta or {},
        "segments": [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": seg.speaker,
                "tone": seg.tone,
                "audio_state": seg.audio_state,
            }
            for seg in transcript.segments
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
