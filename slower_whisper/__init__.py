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
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .compat import Segment, TranscriptionInfo, Word
from .model import WhisperModel

__all__ = [
    "WhisperModel",
    "Segment",
    "Word",
    "TranscriptionInfo",
]

try:
    __version__ = _pkg_version("slower-whisper")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
