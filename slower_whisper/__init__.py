"""slower-whisper: Drop-in replacement for faster-whisper with enrichment.

This package provides a faster-whisper compatible API while transparently
enabling slower-whisper's enriched transcription features:

- Speaker diarization (who said what)
- Turn-based conversation modeling
- Audio enrichment (prosody, emotion, voice quality)
- Word-level timestamps with speaker alignment
- RAG-ready chunking for LLM pipelines

Usage (drop-in replacement):
    # Just change this import:
    # from faster_whisper import WhisperModel
    from slower_whisper import WhisperModel

    model = WhisperModel("base", device="auto")
    segments, info = model.transcribe("audio.wav")

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

Usage (pipeline API):
    from slower_whisper import transcribe_file, TranscriptionConfig

    cfg = TranscriptionConfig(model="base", device="auto", language="en")
    transcript = transcribe_file("meeting.wav", root=".", config=cfg)

Usage (with enrichment):
    from slower_whisper import WhisperModel

    model = WhisperModel("base")
    segments, info = model.transcribe(
        "meeting.wav",
        word_timestamps=True,
        diarize=True,  # Enable speaker diarization
        enrich=True,   # Enable audio enrichment
    )

    # Access enriched transcript
    transcript = model.last_transcript
    for turn in transcript.turns:
        print(f"Speaker {turn.speaker_id}: {turn.text}")

Compatibility:
    This module exports the same types as faster-whisper:
    - WhisperModel: Main transcription class
    - Segment: Transcribed segment (supports tuple unpacking)
    - Word: Word-level timestamp
    - TranscriptionInfo: Transcription metadata

    The Segment type supports both attribute access (segment.text) and
    tuple-style access (segment[4]) for backwards compatibility.

    Note: ``Segment`` and ``Word`` at this level are the faster-whisper compat
    wrappers. For the internal pipeline types, use
    ``from slower_whisper.pipeline.models import Segment``.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

# faster-whisper compat layer (top-level exports) — lightweight, no heavy deps
from .compat import Segment, TranscriptionInfo, Word
from .model import WhisperModel

# ---------------------------------------------------------------------------
# Lazy pipeline re-exports via PEP 562 (__getattr__)
#
# This avoids importing the heavy pipeline sub-package (and its transitive
# deps like soundfile) on bare ``import slower_whisper``.  The import only
# fires when a user actually accesses one of these names.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # pipeline.api
    "enrich_directory": (".pipeline.api", "enrich_directory"),
    "enrich_transcript": (".pipeline.api", "enrich_transcript"),
    "load_transcript": (".pipeline.api", "load_transcript"),
    "save_transcript": (".pipeline.api", "save_transcript"),
    "transcribe_bytes": (".pipeline.api", "transcribe_bytes"),
    "transcribe_directory": (".pipeline.api", "transcribe_directory"),
    "transcribe_file": (".pipeline.api", "transcribe_file"),
    # pipeline.config
    "EnrichmentConfig": (".pipeline.config", "EnrichmentConfig"),
    "TranscriptionConfig": (".pipeline.config", "TranscriptionConfig"),
    # pipeline.device
    "ResolvedDevice": (".pipeline.device", "ResolvedDevice"),
    "resolve_device": (".pipeline.device", "resolve_device"),
    # pipeline.exceptions
    "ConfigurationError": (".pipeline.exceptions", "ConfigurationError"),
    "EnrichmentError": (".pipeline.exceptions", "EnrichmentError"),
    "SlowerWhisperError": (".pipeline.exceptions", "SlowerWhisperError"),
    "TranscriptionError": (".pipeline.exceptions", "TranscriptionError"),
    # pipeline.models
    "Transcript": (".pipeline.models", "Transcript"),
    "Turn": (".pipeline.models", "Turn"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __package__)
        value = getattr(module, attr_name)
        # Cache in module globals so subsequent access is fast
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Keep eager imports visible to type checkers / IDE autocompletion
if TYPE_CHECKING:
    from .pipeline.api import (
        enrich_directory,
        enrich_transcript,
        load_transcript,
        save_transcript,
        transcribe_bytes,
        transcribe_directory,
        transcribe_file,
    )
    from .pipeline.config import EnrichmentConfig, TranscriptionConfig
    from .pipeline.device import ResolvedDevice, resolve_device
    from .pipeline.exceptions import (
        ConfigurationError,
        EnrichmentError,
        SlowerWhisperError,
        TranscriptionError,
    )
    from .pipeline.models import Transcript, Turn

__all__ = [
    # faster-whisper compat
    "WhisperModel",
    "Segment",
    "Word",
    "TranscriptionInfo",
    # Pipeline API
    "transcribe_file",
    "transcribe_directory",
    "transcribe_bytes",
    "enrich_transcript",
    "enrich_directory",
    "load_transcript",
    "save_transcript",
    # Configuration
    "TranscriptionConfig",
    "EnrichmentConfig",
    # Device
    "ResolvedDevice",
    "resolve_device",
    # Models
    "Transcript",
    "Turn",
    # Exceptions
    "SlowerWhisperError",
    "TranscriptionError",
    "EnrichmentError",
    "ConfigurationError",
]

try:
    __version__ = _pkg_version("slower-whisper")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
