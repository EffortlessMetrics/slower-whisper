"""Event types for integration sinks.

Defines the event types and structures used by webhook sinks and other
integrations. Events follow a consistent envelope format for easy
processing by downstream systems.

Event Types:
- transcript.completed: Full transcript is ready
- segment.finalized: Single segment was finalized
- outcome.detected: Semantic outcome was detected
- session.started: Streaming session began
- session.ended: Streaming session ended
- error.occurred: Error occurred during processing
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import Segment, Transcript


class EventType(str, Enum):
    """Types of events that can be sent to integration sinks."""

    TRANSCRIPT_COMPLETED = "transcript.completed"
    SEGMENT_FINALIZED = "segment.finalized"
    OUTCOME_DETECTED = "outcome.detected"
    SESSION_STARTED = "session.started"
    SESSION_ENDED = "session.ended"
    ERROR_OCCURRED = "error.occurred"


@dataclass(slots=True)
class IntegrationEvent:
    """
    Event envelope for integration sinks.

    All events share a common envelope structure with type-specific payload.
    This enables consistent processing and routing across different sinks.

    Attributes:
        event_id: Unique identifier for this event (UUID v4).
        event_type: Type of event (from EventType enum).
        timestamp: Unix timestamp when event was created (seconds).
        source: Source identifier (e.g., session_id, file_name).
        payload: Type-specific event data.
        metadata: Optional additional context.
    """

    event_id: str
    event_type: EventType
    timestamp: float
    source: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to JSON-serializable dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntegrationEvent:
        """Create event from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=EventType(data["event_type"]),
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", "unknown"),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
        )


def _generate_event_id() -> str:
    """Generate a unique event ID."""
    return str(uuid.uuid4())


def _get_timestamp() -> float:
    """Get current Unix timestamp."""
    return time.time()


def create_transcript_event(
    transcript: Transcript,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> IntegrationEvent:
    """
    Create a transcript.completed event.

    Args:
        transcript: The completed transcript.
        source: Optional source identifier (defaults to file_name).
        metadata: Optional additional context.

    Returns:
        IntegrationEvent with transcript data in payload.
    """
    # Build segment summaries (not full data to keep payload reasonable)
    segments_summary = [
        {
            "id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "speaker": seg.speaker,
        }
        for seg in transcript.segments
    ]

    payload = {
        "file_name": transcript.file_name,
        "language": transcript.language,
        "duration": transcript.duration,
        "segment_count": len(transcript.segments),
        "word_count": transcript.word_count(),
        "speaker_ids": transcript.speaker_ids(),
        "is_enriched": transcript.is_enriched(),
        "segments": segments_summary,
    }

    # Include turns if available
    if transcript.turns:
        payload["turn_count"] = len(transcript.turns)

    # Include annotations if available
    if transcript.annotations:
        payload["annotations"] = transcript.annotations

    return IntegrationEvent(
        event_id=_generate_event_id(),
        event_type=EventType.TRANSCRIPT_COMPLETED,
        timestamp=_get_timestamp(),
        source=source or transcript.file_name,
        payload=payload,
        metadata=metadata or {},
    )


def create_segment_event(
    segment: Segment,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> IntegrationEvent:
    """
    Create a segment.finalized event.

    Args:
        segment: The finalized segment.
        source: Source identifier (e.g., session_id).
        metadata: Optional additional context.

    Returns:
        IntegrationEvent with segment data in payload.
    """
    payload = {
        "id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "speaker": segment.speaker,
    }

    # Include audio_state summary if enriched
    if segment.audio_state:
        audio_summary: dict[str, Any] = {}
        if "prosody" in segment.audio_state:
            prosody = segment.audio_state["prosody"]
            audio_summary["pitch_median_hz"] = prosody.get("pitch_median_hz")
            audio_summary["energy_median_db"] = prosody.get("energy_median_db")
        if "emotion" in segment.audio_state:
            audio_summary["emotion"] = segment.audio_state["emotion"]
        if audio_summary:
            payload["audio_state_summary"] = audio_summary

    return IntegrationEvent(
        event_id=_generate_event_id(),
        event_type=EventType.SEGMENT_FINALIZED,
        timestamp=_get_timestamp(),
        source=source,
        payload=payload,
        metadata=metadata or {},
    )


def create_outcome_event(
    outcomes: list[dict[str, Any]],
    source: str,
    segment_id: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> IntegrationEvent:
    """
    Create an outcome.detected event.

    Outcomes are semantic annotations like action items, decisions,
    questions, or other structured insights extracted from the transcript.

    Args:
        outcomes: List of detected outcomes.
        source: Source identifier.
        segment_id: Optional segment ID where outcomes were detected.
        metadata: Optional additional context.

    Returns:
        IntegrationEvent with outcomes in payload.
    """
    payload: dict[str, Any] = {
        "outcomes": outcomes,
        "outcome_count": len(outcomes),
    }
    if segment_id is not None:
        payload["segment_id"] = segment_id

    return IntegrationEvent(
        event_id=_generate_event_id(),
        event_type=EventType.OUTCOME_DETECTED,
        timestamp=_get_timestamp(),
        source=source,
        payload=payload,
        metadata=metadata or {},
    )


def create_session_started_event(
    session_id: str,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> IntegrationEvent:
    """
    Create a session.started event.

    Args:
        session_id: The streaming session identifier.
        config: Optional session configuration.
        metadata: Optional additional context.

    Returns:
        IntegrationEvent for session start.
    """
    payload: dict[str, Any] = {"session_id": session_id}
    if config:
        payload["config"] = config

    return IntegrationEvent(
        event_id=_generate_event_id(),
        event_type=EventType.SESSION_STARTED,
        timestamp=_get_timestamp(),
        source=session_id,
        payload=payload,
        metadata=metadata or {},
    )


def create_session_ended_event(
    session_id: str,
    stats: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> IntegrationEvent:
    """
    Create a session.ended event.

    Args:
        session_id: The streaming session identifier.
        stats: Optional session statistics.
        metadata: Optional additional context.

    Returns:
        IntegrationEvent for session end.
    """
    payload: dict[str, Any] = {"session_id": session_id}
    if stats:
        payload["stats"] = stats

    return IntegrationEvent(
        event_id=_generate_event_id(),
        event_type=EventType.SESSION_ENDED,
        timestamp=_get_timestamp(),
        source=session_id,
        payload=payload,
        metadata=metadata or {},
    )


def create_error_event(
    error_code: str,
    error_message: str,
    source: str,
    recoverable: bool = True,
    details: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> IntegrationEvent:
    """
    Create an error.occurred event.

    Args:
        error_code: Short error code (e.g., "ASR_FAILED").
        error_message: Human-readable error description.
        source: Source identifier.
        recoverable: Whether processing can continue.
        details: Optional error details.
        metadata: Optional additional context.

    Returns:
        IntegrationEvent for error.
    """
    payload: dict[str, Any] = {
        "code": error_code,
        "message": error_message,
        "recoverable": recoverable,
    }
    if details:
        payload["details"] = details

    return IntegrationEvent(
        event_id=_generate_event_id(),
        event_type=EventType.ERROR_OCCURRED,
        timestamp=_get_timestamp(),
        source=source,
        payload=payload,
        metadata=metadata or {},
    )
