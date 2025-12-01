"""
Streaming session management for slower-whisper (v2.0 prep).

This module provides typed interfaces for real-time streaming transcription.
Currently a skeleton with NotImplementedError bodies — implementation in v2.0.

See docs/STREAMING_ARCHITECTURE.md for the full RFC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from .models import Segment, Transcript


class StreamEventType(Enum):
    """WebSocket event types for streaming protocol."""

    READY = "ready"
    PARTIAL_SEGMENT = "partial_segment"
    SEGMENT_FINALIZED = "segment_finalized"
    TURN_UPDATED = "turn_updated"
    ANALYTICS_SNAPSHOT = "analytics_snapshot"
    ERROR = "error"
    DONE = "done"


@dataclass(slots=True)
class StreamConfig:
    """Configuration for a streaming session."""

    sample_rate: int = 16000
    language: str | None = None
    enable_diarization: bool = False
    enable_analytics: bool = False
    enable_semantics: bool = False


@dataclass(slots=True)
class StreamMeta:
    """Metadata block for streaming transcripts (meta.stream)."""

    session_id: str
    seq: int
    stream_version: int = 1
    features: dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True)
class PartialSegment:
    """A segment that may still change (partial=True)."""

    id: int
    start: float
    end: float
    text: str
    revision: int
    partial: bool = True
    speaker_id: str | None = None

    def finalize(self) -> Segment:
        """Convert to a finalized Segment (partial=False)."""
        return Segment(
            id=self.id,
            start=self.start,
            end=self.end,
            text=self.text,
            speaker={"id": self.speaker_id} if self.speaker_id else None,
        )


@dataclass(slots=True)
class StreamEvent:
    """A streaming event sent from server to client."""

    type: StreamEventType
    seq: int
    payload: dict | PartialSegment | Segment | None = None


class StreamingSessionProtocol(Protocol):
    """Protocol for streaming session implementations."""

    @property
    def session_id(self) -> str:
        """Unique session identifier."""
        ...

    @property
    def config(self) -> StreamConfig:
        """Session configuration."""
        ...

    def start(self) -> StreamEvent:
        """Initialize the session and return READY event."""
        ...

    def process_audio(self, audio_bytes: bytes, seq: int) -> list[StreamEvent]:
        """Process an audio chunk and return any events."""
        ...

    def stop(self) -> StreamEvent:
        """End the session and return DONE event."""
        ...

    def get_transcript(self) -> Transcript:
        """Get the current transcript state."""
        ...


class StreamingSession:
    """
    Streaming transcription session (v2.0 implementation placeholder).

    This class will manage:
    - Audio buffering and VAD
    - Incremental ASR with partial/final segments
    - Live diarization updates
    - Analytics snapshots

    Currently raises NotImplementedError — full implementation in v2.0.
    """

    def __init__(self, config: StreamConfig | None = None) -> None:
        self._config = config or StreamConfig()
        self._session_id: str = ""
        self._seq: int = 0
        self._segments: list[PartialSegment] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def config(self) -> StreamConfig:
        return self._config

    def start(self) -> StreamEvent:
        """Initialize the session."""
        raise NotImplementedError("Streaming implementation planned for v2.0")

    def process_audio(self, audio_bytes: bytes, seq: int) -> list[StreamEvent]:
        """Process audio chunk and emit events."""
        raise NotImplementedError("Streaming implementation planned for v2.0")

    def stop(self) -> StreamEvent:
        """Finalize session and return DONE event."""
        raise NotImplementedError("Streaming implementation planned for v2.0")

    def get_transcript(self) -> Transcript:
        """Get current transcript snapshot."""
        raise NotImplementedError("Streaming implementation planned for v2.0")


def apply_stream_event(transcript: Transcript, event: StreamEvent) -> Transcript:
    """
    Apply a streaming event to update a transcript.

    This function enables clients to rebuild state by applying events
    in sequence order.

    Args:
        transcript: Current transcript state
        event: Event to apply

    Returns:
        Updated transcript

    Raises:
        NotImplementedError: Implementation in v2.0
    """
    raise NotImplementedError("Streaming implementation planned for v2.0")
