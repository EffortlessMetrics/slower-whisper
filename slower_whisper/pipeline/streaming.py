"""
Streaming session management for post-ASR text chunks (v2.0).

This module implements a small, predictable state machine that turns
ordered text chunks into partial/final segment events. See
docs/STREAMING_ARCHITECTURE.md for the contract.

ID Contracts (v2.0):
- stream_id: Generated at session creation, format: `str-{uuid4}`
- event_id: Monotonically increasing positive integer per stream
- segment_id: Format: `seg-{seq}` where seq is zero-indexed
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict

from .ids import EventIdCounter, SegmentIdCounter, generate_stream_id


class StreamChunk(TypedDict):
    """Input chunk produced by upstream ASR."""

    start: float
    end: float
    text: str
    speaker_id: str | None


@dataclass(slots=True)
class StreamSegment:
    """Aggregated segment state.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        text: Aggregated transcribed text.
        speaker_id: Optional speaker identifier.
        audio_state: Optional audio enrichment data.
        segment_id: Optional segment ID (format: `seg-{seq}`).
    """

    start: float
    end: float
    text: str
    speaker_id: str | None = None
    audio_state: dict[str, Any] | None = None  # Optional audio enrichment data
    segment_id: str | None = None  # Optional segment ID (v2.0)


class StreamEventType(Enum):
    """Event types emitted by StreamingSession."""

    PARTIAL_SEGMENT = "partial_segment"
    FINAL_SEGMENT = "final_segment"
    SEMANTIC_UPDATE = "semantic_update"


@dataclass(slots=True)
class StreamEvent:
    """A streaming event containing the current segment view.

    Attributes:
        type: The type of event (PARTIAL_SEGMENT, FINAL_SEGMENT, SEMANTIC_UPDATE).
        segment: The segment associated with this event.
        semantic: Optional semantic update payload (only for SEMANTIC_UPDATE events).
                 This field is dynamically attached by LiveSemanticSession.
        event_id: Optional monotonically increasing event ID (v2.0).
        stream_id: Optional stream ID for correlation (v2.0).
    """

    type: StreamEventType
    segment: StreamSegment
    semantic: Any = None  # Optional SemanticUpdatePayload for SEMANTIC_UPDATE events
    event_id: int | None = None  # Optional event ID (v2.0)
    stream_id: str | None = None  # Optional stream ID (v2.0)


@dataclass(slots=True)
class StreamConfig:
    """Configuration for streaming state transitions.

    Attributes:
        max_gap_sec: Gap threshold in seconds to finalize a segment.
        emit_ids: Whether to emit IDs (event_id, stream_id, segment_id) in events.
    """

    max_gap_sec: float = 1.0
    emit_ids: bool = False  # Enable ID emission for v2.0 protocol


class StreamingSession:
    """
    Minimal streaming state machine for post-ASR chunks.

    Maintains a single in-flight partial segment and emits events as
    chunks arrive or when the stream closes.

    When emit_ids is enabled in config, the session generates stable IDs:
    - stream_id: Unique per session, format: `str-{uuid4}`
    - event_id: Monotonically increasing positive integer per stream
    - segment_id: Format: `seg-{seq}`, same ID for PARTIAL until FINALIZED

    Attributes:
        config: Session configuration.
        stream_id: Unique stream identifier (only set if emit_ids=True).
    """

    def __init__(self, config: StreamConfig | None = None) -> None:
        self.config = config or StreamConfig()
        self._current: StreamSegment | None = None

        # ID tracking (v2.0)
        self._event_counter: EventIdCounter | None
        self._segment_counter: SegmentIdCounter | None
        if self.config.emit_ids:
            self._stream_id: str | None = generate_stream_id()
            self._event_counter = EventIdCounter()
            self._segment_counter = SegmentIdCounter()
            self._current_segment_id: str | None = None
        else:
            self._stream_id = None
            self._event_counter = None
            self._segment_counter = None
            self._current_segment_id = None

    @property
    def stream_id(self) -> str | None:
        """Get the stream ID (None if emit_ids is disabled)."""
        return self._stream_id

    def ingest_chunk(self, chunk: StreamChunk) -> list[StreamEvent]:
        """Consume a chunk and return any events it produces."""
        self._validate_monotonic(chunk)
        events: list[StreamEvent] = []
        speaker_id = chunk.get("speaker_id")

        if self._current and not self._should_extend(chunk, speaker_id):
            # Finalize current segment
            events.append(self._create_event(StreamEventType.FINAL_SEGMENT, self._current))
            # Start new segment
            self._current = self._segment_from_chunk(chunk)
            self._assign_segment_id(self._current)
        elif self._current:
            self._extend_current(chunk)
        else:
            self._current = self._segment_from_chunk(chunk)
            self._assign_segment_id(self._current)

        events.append(self._create_event(StreamEventType.PARTIAL_SEGMENT, self._current))
        return events

    def end_of_stream(self) -> list[StreamEvent]:
        """Finalize the stream, emitting any remaining partial as final."""
        if not self._current:
            return []

        final_event = self._create_event(StreamEventType.FINAL_SEGMENT, self._current)
        self._current = None
        return [final_event]

    def _assign_segment_id(self, segment: StreamSegment) -> None:
        """Assign a new segment ID to a segment (if emit_ids is enabled)."""
        if self.config.emit_ids and self._segment_counter is not None:
            segment.segment_id = self._segment_counter.next()
            self._current_segment_id = segment.segment_id

    def _create_event(self, event_type: StreamEventType, segment: StreamSegment) -> StreamEvent:
        """Create a StreamEvent, optionally with IDs attached."""
        snapshot = self._snapshot(segment)

        if self.config.emit_ids and self._event_counter is not None:
            event_id = self._event_counter.next()
            return StreamEvent(
                type=event_type,
                segment=snapshot,
                event_id=event_id,
                stream_id=self._stream_id,
            )
        else:
            return StreamEvent(type=event_type, segment=snapshot)

    def _validate_monotonic(self, chunk: StreamChunk) -> None:
        if chunk["end"] < chunk["start"]:
            raise ValueError("Chunk end must be >= start")
        if self._current and chunk["start"] < self._current.end:
            raise ValueError("Chunks must arrive in non-decreasing time order")

    def _should_extend(self, chunk: StreamChunk, speaker_id: str | None) -> bool:
        assert self._current is not None
        same_speaker = self._current.speaker_id == speaker_id
        gap = chunk["start"] - self._current.end
        return same_speaker and gap <= self.config.max_gap_sec

    def _extend_current(self, chunk: StreamChunk) -> None:
        assert self._current is not None
        self._current.end = float(chunk["end"])
        if chunk["text"]:
            if self._current.text and not self._current.text.endswith(" "):
                self._current.text += " "
            self._current.text += chunk["text"]

    def _segment_from_chunk(self, chunk: StreamChunk) -> StreamSegment:
        return StreamSegment(
            start=float(chunk["start"]),
            end=float(chunk["end"]),
            text=chunk["text"],
            speaker_id=chunk.get("speaker_id"),
        )

    def _snapshot(self, segment: StreamSegment | None) -> StreamSegment:
        if segment is None:
            raise ValueError("Cannot snapshot a missing segment")
        return StreamSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker_id=segment.speaker_id,
        )
