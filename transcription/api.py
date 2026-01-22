"""
Public API for slower-whisper library.

This module provides a stable, high-level interface for transcription and
enrichment. It wraps the internal pipeline and audio_enrichment modules with
a clean API suitable for programmatic use.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import EnrichmentConfig, TranscriptionConfig

# --- Keep patch points stable for tests and downstream users ---
from .diarization_orchestrator import _maybe_run_diarization

# --- Real implementations live in orchestrators ---
from .enrichment_orchestrator import (
    _enrich_directory_impl,
    _enrich_transcript_impl,
    _run_semantic_annotator,
    _run_speaker_analytics,
)
from .models import Transcript
from .transcript_io import load_transcript, save_transcript
from .transcription_helpers import (
    _get_wav_duration_seconds,
    _maybe_build_chunks,
    _neutral_audio_state,
    _turns_have_metadata,
)
from .transcription_orchestrator import (
    _transcribe_bytes_impl,
    _transcribe_directory_impl,
    _transcribe_file_impl,
)

__all__ = [
    "transcribe_directory",
    "transcribe_file",
    "transcribe_bytes",
    "enrich_directory",
    "enrich_transcript",
    "load_transcript",
    "save_transcript",
]

logger = logging.getLogger(__name__)


def transcribe_directory(root: str | Path, config: TranscriptionConfig) -> list[Transcript]:
    return _transcribe_directory_impl(root=root, config=config)


def transcribe_file(
    audio_path: str | Path,
    root: str | Path,
    config: TranscriptionConfig,
) -> Transcript:
    # Pass helpers at call-time so patching transcription.api.* keeps working.
    return _transcribe_file_impl(
        audio_path=audio_path,
        root=root,
        config=config,
        get_wav_duration_seconds=_get_wav_duration_seconds,
        maybe_run_diarization=_maybe_run_diarization,
        maybe_build_chunks=_maybe_build_chunks,
    )


def transcribe_bytes(
    audio_bytes: bytes,
    config: TranscriptionConfig | None = None,
    *,
    file_name: str = "audio.wav",
) -> Transcript:
    """Transcribe audio from raw bytes.

    This is useful for API integrations where audio is received as bytes
    (e.g., from a REST endpoint, WebSocket, or memory buffer).

    Args:
        audio_bytes: Raw audio bytes (any ffmpeg-supported format)
        config: Transcription configuration. If None, uses defaults from env
            via TranscriptionConfig.from_sources().
        file_name: Filename to use in the transcript metadata. The extension
            is used to hint at the audio format (e.g., "audio.mp3", "clip.wav").
            Defaults to "audio.wav".

    Returns:
        Transcript object with transcription results

    Raises:
        TranscriptionError: If transcription fails due to invalid audio,
            missing dependencies, or other errors.

    Example:
        >>> from transcription import transcribe_bytes
        >>> with open("recording.wav", "rb") as f:
        ...     audio_data = f.read()
        >>> transcript = transcribe_bytes(audio_data)
        >>> print(transcript.full_text)

        >>> # With custom config
        >>> from transcription import TranscriptionConfig
        >>> config = TranscriptionConfig(model="base", device="cpu")
        >>> transcript = transcribe_bytes(audio_data, config, file_name="meeting.wav")
    """
    from pathlib import Path

    # Use default config from environment if not provided
    if config is None:
        config = TranscriptionConfig.from_sources()

    # Extract format hint from file_name extension
    suffix = Path(file_name).suffix
    format_hint = suffix.lstrip(".").lower() if suffix else "wav"

    transcript = _transcribe_bytes_impl(
        audio_data=audio_bytes,
        config=config,
        format=format_hint,
        get_wav_duration_seconds=_get_wav_duration_seconds,
        maybe_run_diarization=_maybe_run_diarization,
        maybe_build_chunks=_maybe_build_chunks,
    )

    # Set the file_name in the transcript metadata
    transcript.file_name = file_name

    return transcript


def enrich_directory(root: str | Path, config: EnrichmentConfig) -> list[Transcript]:
    return _enrich_directory_impl(
        root=root,
        config=config,
        turns_have_metadata=_turns_have_metadata,
        run_speaker_analytics=_run_speaker_analytics,
        run_semantic_annotator=_run_semantic_annotator,
    )


def enrich_transcript(
    transcript: Transcript,
    audio_path: str | Path,
    config: EnrichmentConfig,
) -> Transcript:
    return _enrich_transcript_impl(
        transcript=transcript,
        audio_path=audio_path,
        config=config,
        turns_have_metadata=_turns_have_metadata,
        neutral_audio_state=_neutral_audio_state,
        run_speaker_analytics=_run_speaker_analytics,
        run_semantic_annotator=_run_semantic_annotator,
    )
