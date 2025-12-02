"""
Streaming session management for post-ASR text chunks (v0.1).

This module implements a small, predictable state machine that turns
ordered text chunks into partial/final segment events. See
docs/STREAMING_ARCHITECTURE.md for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypedDict


class StreamChunk(TypedDict):
    """Input chunk produced by upstream ASR."""

    start: float
    end: float
    text: str
    speaker_id: str | None


@dataclass(slots=True)
class StreamSegment:
    """Aggregated segment state."""

    start: float
    end: float
    text: str
    speaker_id: str | None = None


class StreamEventType(Enum):
    """Event types emitted by StreamingSession."""

    PARTIAL_SEGMENT = "partial_segment"
    FINAL_SEGMENT = "final_segment"


@dataclass(slots=True)
class StreamEvent:
    """A streaming event containing the current segment view."""

    type: StreamEventType
    segment: StreamSegment


@dataclass(slots=True)
class StreamConfig:
    """Configuration for streaming state transitions."""

    max_gap_sec: float = 1.0


class StreamingSession:
    """
    Minimal streaming state machine for post-ASR chunks.

    Maintains a single in-flight partial segment and emits events as
    chunks arrive or when the stream closes.
    """

    def __init__(self, config: StreamConfig | None = None) -> None:
        self.config = config or StreamConfig()
        self._current: StreamSegment | None = None

    def ingest_chunk(self, chunk: StreamChunk) -> list[StreamEvent]:
        """Consume a chunk and return any events it produces."""
        self._validate_monotonic(chunk)
        events: list[StreamEvent] = []
        speaker_id = chunk.get("speaker_id")

        if self._current and not self._should_extend(chunk, speaker_id):
            events.append(StreamEvent(StreamEventType.FINAL_SEGMENT, self._snapshot(self._current)))
            self._current = self._segment_from_chunk(chunk)
        elif self._current:
            self._extend_current(chunk)
        else:
            self._current = self._segment_from_chunk(chunk)

        events.append(StreamEvent(StreamEventType.PARTIAL_SEGMENT, self._snapshot(self._current)))
        return events

    def end_of_stream(self) -> list[StreamEvent]:
        """Finalize the stream, emitting any remaining partial as final."""
        if not self._current:
            return []

        final_event = StreamEvent(StreamEventType.FINAL_SEGMENT, self._snapshot(self._current))
        self._current = None
        return [final_event]

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
