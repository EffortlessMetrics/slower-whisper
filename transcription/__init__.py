"""
Local transcription pipeline package.

Public API:
    - transcribe_directory: Transcribe all audio in a project
    - transcribe_file: Transcribe a single file
    - enrich_directory: Enrich all transcripts with audio features
    - enrich_transcript: Enrich a single transcript
    - load_transcript: Load transcript from JSON
    - save_transcript: Save transcript to JSON

LLM Integration:
    - render_conversation_for_llm: Convert transcript to LLM-ready text
    - render_conversation_compact: Compact rendering for token-constrained contexts
    - render_segment: Render a single segment with speaker/audio cues
    - to_turn_view: Turn-level rendering with analytics metadata
    - to_speaker_summary: Summarize speaker stats for prompts

Configuration:
    - TranscriptionConfig: Stage 1 configuration
    - EnrichmentConfig: Stage 2 configuration

Streaming:
    - StreamConfig: Controls max gap for stitching partial segments
    - StreamingSession: Builds partial/final segments from post-ASR chunks
    - StreamChunk: Input chunk type (post-ASR)
    - StreamEvent: Output event with segment view
    - StreamingEnrichmentSession: Incremental audio enrichment for streaming (v1.7.0)
    - StreamingEnrichmentConfig: Configuration for streaming enrichment (v1.7.0)
    - LiveSemanticSession: Real-time semantic annotation for streaming (v1.7.0)
    - LiveSemanticsConfig: Configuration for live semantics (v1.7.0)
    - SemanticUpdatePayload: Structured payload for semantic updates (v1.7.0)

Models:
    - Transcript: Complete transcript with segments
    - Segment: Single transcribed segment
    - Word: Word-level timestamp with confidence (v1.8+)
    - Turn: Speaker turn with contiguous segments
    - TurnMeta: Metadata for speaker turns (question counts, interruptions)
    - AudioState: Audio enrichment container with prosody and emotion features
    - ProsodyState: Prosodic features (pitch, energy, rate, pauses)
    - EmotionState: Emotional features (valence, arousal)
    - ExtractionStatus: Feature extraction status tracking

Utilities:
    - turn_to_dict: Convert Turn objects to dictionaries
    - get_speaker_id: Extract speaker ID from various formats
    - get_speaker_label_or_id: Extract speaker label/ID with fallback

Exceptions:
    - SlowerWhisperError: Base exception for this library
    - TranscriptionError: Raised when transcription fails
    - EnrichmentError: Raised when audio enrichment fails
    - ConfigurationError: Raised when configuration is invalid
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .streaming_enrich import StreamingEnrichmentConfig, StreamingEnrichmentSession
    from .streaming_semantic import LiveSemanticsConfig, LiveSemanticSession, SemanticUpdatePayload

# Version of the transcription pipeline; included in JSON metadata.
# Must be defined before other imports to avoid circular imports
__version__ = "1.8.0"

# Configure global cache environment for all model downloads
# This ensures all HF/torch models are cached under SLOWER_WHISPER_CACHE_ROOT
# and reused across venvs and runs
from .cache import configure_global_cache_env

configure_global_cache_env()

# Public API exports
# ruff: noqa: E402 - imports below cache configuration intentionally
from .api import (
    enrich_directory,
    enrich_transcript,
    load_transcript,
    save_transcript,
    transcribe_directory,
    transcribe_file,
)
from .chunking import ChunkingConfig, build_chunks
from .config import AppConfig, AsrConfig, EnrichmentConfig, Paths, TranscriptionConfig
from .exceptions import (
    ConfigurationError,
    EnrichmentError,
    SlowerWhisperError,
    TranscriptionError,
)
from .exporters import export_transcript
from .llm_utils import (
    render_conversation_compact,
    render_conversation_for_llm,
    render_segment,
    to_speaker_summary,
    to_turn_view,
)
from .models import (
    BatchFileResult,
    BatchProcessingResult,
    Chunk,
    DiarizationMeta,
    EnrichmentBatchResult,
    EnrichmentFileResult,
    Segment,
    SpeakerStats,
    Transcript,
    Turn,
    TurnMeta,
    Word,
)
from .semantic import KeywordSemanticAnnotator, NoOpSemanticAnnotator, SemanticAnnotator
from .speaker_id import get_speaker_id, get_speaker_label_or_id
from .streaming import StreamChunk, StreamConfig, StreamEvent, StreamEventType, StreamingSession

# v1.7.0 streaming features are lazy-imported via __getattr__ below
# to avoid pulling in soundfile/heavy deps in base installs
from .turn_helpers import turn_to_dict
from .types_audio import AudioState, EmotionState, ExtractionStatus, ProsodyState
from .validation import validate_transcript_json

# Lazy imports for optional/heavy exports that must not break base installs.
# Accessing these names triggers an import; missing deps produce a helpful error.
# NOTE: These are NOT in __all__ because `from transcription import *` would fail
# on base installs when trying to resolve them.
_LAZY_OPTIONAL: dict[str, str] = {
    # v1.7.0 streaming features - require full extra (soundfile, etc.)
    "StreamingEnrichmentConfig": ".streaming_enrich",
    "StreamingEnrichmentSession": ".streaming_enrich",
    "LiveSemanticsConfig": ".streaming_semantic",
    "LiveSemanticSession": ".streaming_semantic",
    "SemanticUpdatePayload": ".streaming_semantic",
}

# Lazy imports for circular import avoidance (not due to missing deps)
_LAZY_CIRCULAR: dict[str, str] = {
    "run_pipeline": ".pipeline",
}


def __getattr__(name: str) -> Any:
    """Lazy import for optional/heavy exports and circular import avoidance."""
    # Handle optional deps that require 'full' extra
    if name in _LAZY_OPTIONAL:
        module_name = _LAZY_OPTIONAL[name]
        try:
            mod = importlib.import_module(module_name, __name__)
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                f"{name} requires optional dependencies. Install the 'full' extra "
                "(e.g., `uv sync --extra full` or `pip install 'slower-whisper[full]'`), "
                "or use the full Docker image."
            ) from exc
        value = getattr(mod, name)
        globals()[name] = value  # cache for future lookups
        return value

    # Handle circular import avoidance (re-raise original error if import fails)
    if name in _LAZY_CIRCULAR:
        module_name = _LAZY_CIRCULAR[name]
        mod = importlib.import_module(module_name, __name__)
        value = getattr(mod, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy imports in dir() output for discoverability."""
    lazy_names = list(_LAZY_OPTIONAL.keys()) + list(_LAZY_CIRCULAR.keys())
    return sorted(set(list(globals().keys()) + lazy_names))


__all__ = [
    # Public API functions
    "transcribe_directory",
    "transcribe_file",
    "enrich_directory",
    "enrich_transcript",
    "load_transcript",
    "save_transcript",
    "export_transcript",
    # LLM rendering utilities
    "render_conversation_for_llm",
    "render_conversation_compact",
    "render_segment",
    "to_turn_view",
    "to_speaker_summary",
    "build_chunks",
    "SemanticAnnotator",
    "NoOpSemanticAnnotator",
    "KeywordSemanticAnnotator",
    # Public configuration
    "TranscriptionConfig",
    "EnrichmentConfig",
    "ChunkingConfig",
    "StreamConfig",
    # NOTE: StreamingEnrichmentConfig and LiveSemanticsConfig are lazy imports
    # and excluded from __all__ to prevent `from transcription import *` failures
    # on base installs. Access them directly: transcription.StreamingEnrichmentConfig
    # Models
    "Chunk",
    "Segment",
    "Transcript",
    "Turn",
    "TurnMeta",
    "Word",
    "SpeakerStats",
    "DiarizationMeta",
    "BatchFileResult",
    "BatchProcessingResult",
    "EnrichmentFileResult",
    "EnrichmentBatchResult",
    "AudioState",
    "ProsodyState",
    "EmotionState",
    "ExtractionStatus",
    "validate_transcript_json",
    # Streaming
    "StreamChunk",
    "StreamEvent",
    "StreamingSession",
    # NOTE: StreamingEnrichmentSession, LiveSemanticSession, SemanticUpdatePayload
    # are lazy imports - see note above about __all__ exclusion
    # Utilities
    "turn_to_dict",
    "get_speaker_id",
    "get_speaker_label_or_id",
    # Exceptions
    "SlowerWhisperError",
    "TranscriptionError",
    "EnrichmentError",
    "ConfigurationError",
    # Legacy (for backward compatibility)
    "AppConfig",
    "AsrConfig",
    "Paths",
    # NOTE: run_pipeline is a lazy import for circular import avoidance
]
