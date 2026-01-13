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
    audio_data: bytes,
    config: TranscriptionConfig,
    format: str = "wav",
) -> Transcript:
    return _transcribe_bytes_impl(
        audio_data=audio_data,
        config=config,
        format=format,
        get_wav_duration_seconds=_get_wav_duration_seconds,
        maybe_run_diarization=_maybe_run_diarization,
        maybe_build_chunks=_maybe_build_chunks,
    )


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
