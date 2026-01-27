"""Enrichment endpoint for the API service."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from .api import enrich_transcript
from .config import EnrichmentConfig
from .exceptions import ConfigurationError, EnrichmentError
from .service_serialization import _transcript_to_dict
from .service_settings import MAX_AUDIO_SIZE_MB, MAX_TRANSCRIPT_SIZE_MB
from .service_validation import (
    save_upload_file_streaming,
    validate_audio_format,
    validate_file_size,
    validate_transcript_json,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Enrichment Endpoint
# =============================================================================


@router.post(
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
            from .api import load_transcript

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
            enriched = enrich_transcript(
                transcript=transcript_obj,
                audio_path=audio_path,
                config=config,
            )
            logger.info(
                "Enrichment completed successfully: %d segments processed",
                len(enriched.segments),
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
