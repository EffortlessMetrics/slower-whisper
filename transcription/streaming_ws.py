"""
WebSocket streaming protocol for real-time transcription (v2.0.0).

This module implements the WebSocket-based streaming API for real-time
audio transcription and enrichment. It provides event-driven communication
with clients using JSON message envelopes.

Key features:
- Event envelope with monotonically increasing event_id
- Client message types: START_SESSION, AUDIO_CHUNK, END_SESSION, PING
- Server message types: SESSION_STARTED, PARTIAL, FINALIZED, ERROR, etc.
- Session lifecycle management with configurable enrichment
- Backpressure handling: PARTIAL events can be dropped, FINALIZED never dropped

See docs/STREAMING_ARCHITECTURE.md section 6 for protocol specification.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# Message Type Enums
# =============================================================================


class ClientMessageType(str, Enum):
    """Message types sent from client to server."""

    START_SESSION = "START_SESSION"
    AUDIO_CHUNK = "AUDIO_CHUNK"
    END_SESSION = "END_SESSION"
    PING = "PING"


class ServerMessageType(str, Enum):
    """Message types sent from server to client."""

    SESSION_STARTED = "SESSION_STARTED"
    PARTIAL = "PARTIAL"
    FINALIZED = "FINALIZED"
    SPEAKER_TURN = "SPEAKER_TURN"
    SEMANTIC_UPDATE = "SEMANTIC_UPDATE"
    DIARIZATION_UPDATE = "DIARIZATION_UPDATE"
    ERROR = "ERROR"
    SESSION_ENDED = "SESSION_ENDED"
    PONG = "PONG"


# =============================================================================
# Event Envelope
# =============================================================================


@dataclass(slots=True)
class EventEnvelope:
    """
    Event envelope for WebSocket messages from server to client.

    The envelope provides consistent metadata for all server events,
    enabling clients to track event ordering, timing, and context.

    Attributes:
        event_id: Monotonically increasing ID per stream (never resets within session)
        stream_id: Unique stream identifier (format: str-{uuid4})
        segment_id: Segment identifier (format: seg-{seq}, None for non-segment events)
        type: Server message type
        ts_server: Server timestamp (Unix epoch milliseconds)
        ts_audio_start: Audio timestamp start (seconds, None for non-audio events)
        ts_audio_end: Audio timestamp end (seconds, None for non-audio events)
        payload: Message-specific payload data
    """

    event_id: int
    stream_id: str
    type: ServerMessageType
    ts_server: int
    payload: dict[str, Any]
    segment_id: str | None = None
    ts_audio_start: float | None = None
    ts_audio_end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert envelope to JSON-serializable dictionary."""
        result: dict[str, Any] = {
            "event_id": self.event_id,
            "stream_id": self.stream_id,
            "type": self.type.value,
            "ts_server": self.ts_server,
            "payload": self.payload,
        }
        if self.segment_id is not None:
            result["segment_id"] = self.segment_id
        if self.ts_audio_start is not None:
            result["ts_audio_start"] = self.ts_audio_start
        if self.ts_audio_end is not None:
            result["ts_audio_end"] = self.ts_audio_end
        return result


# =============================================================================
# Session Configuration
# =============================================================================


@dataclass(slots=True)
class WebSocketSessionConfig:
    """
    Configuration for a WebSocket streaming session.

    Sent by the client in START_SESSION message to configure
    the streaming behavior.

    Attributes:
        max_gap_sec: Gap threshold to finalize segment (default: 1.0s)
        enable_prosody: Extract prosodic features from audio (default: False)
        enable_emotion: Extract dimensional emotion features (default: False)
        enable_categorical_emotion: Extract categorical emotion labels (default: False)
        enable_diarization: Enable incremental speaker diarization (default: False)
        diarization_interval_sec: Interval for diarization updates in seconds (default: 30.0)
        sample_rate: Expected audio sample rate (default: 16000 Hz)
        audio_format: Audio encoding format (default: "pcm_s16le")
    """

    max_gap_sec: float = 1.0
    enable_prosody: bool = False
    enable_emotion: bool = False
    enable_categorical_emotion: bool = False
    enable_diarization: bool = False
    diarization_interval_sec: float = 30.0
    sample_rate: int = 16000
    audio_format: str = "pcm_s16le"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebSocketSessionConfig:
        """Create config from dictionary (e.g., from client message)."""
        return cls(
            max_gap_sec=float(data.get("max_gap_sec", 1.0)),
            enable_prosody=bool(data.get("enable_prosody", False)),
            enable_emotion=bool(data.get("enable_emotion", False)),
            enable_categorical_emotion=bool(data.get("enable_categorical_emotion", False)),
            enable_diarization=bool(data.get("enable_diarization", False)),
            diarization_interval_sec=float(data.get("diarization_interval_sec", 30.0)),
            sample_rate=int(data.get("sample_rate", 16000)),
            audio_format=str(data.get("audio_format", "pcm_s16le")),
        )


# =============================================================================
# Session State
# =============================================================================


class SessionState(str, Enum):
    """WebSocket session lifecycle states."""

    CREATED = "created"
    ACTIVE = "active"
    ENDING = "ending"
    ENDED = "ended"
    ERROR = "error"


@dataclass
class SessionStats:
    """Statistics tracked for a WebSocket streaming session."""

    chunks_received: int = 0
    bytes_received: int = 0
    segments_partial: int = 0
    segments_finalized: int = 0
    events_sent: int = 0
    events_dropped: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        duration = (self.end_time or time.time()) - self.start_time
        return {
            "chunks_received": self.chunks_received,
            "bytes_received": self.bytes_received,
            "segments_partial": self.segments_partial,
            "segments_finalized": self.segments_finalized,
            "events_sent": self.events_sent,
            "events_dropped": self.events_dropped,
            "errors": self.errors,
            "duration_sec": round(duration, 3),
        }


# =============================================================================
# Incremental Diarization Types
# =============================================================================


@dataclass(slots=True)
class SpeakerAssignment:
    """
    Speaker assignment for a time range.

    Represents the result of incremental diarization, mapping a time range
    to a speaker ID with optional confidence score.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        speaker_id: Speaker identifier (e.g., "spk_0", "SPEAKER_00").
        confidence: Optional confidence score (0.0-1.0).
    """

    start: float
    end: float
    speaker_id: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "speaker_id": self.speaker_id,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


class DiarizationHookProtocol(Protocol):
    """
    Protocol for incremental diarization hooks.

    A diarization hook receives accumulated audio and returns speaker
    assignments. This enables pluggable diarization backends.

    The hook is called periodically (based on diarization_interval_sec)
    with all audio accumulated so far.

    Example implementation:
        ```python
        async def my_diarization_hook(
            audio_buffer: bytes,
            sample_rate: int,
        ) -> list[SpeakerAssignment]:
            # Run diarization on audio_buffer
            # Return list of speaker assignments
            return [
                SpeakerAssignment(start=0.0, end=5.0, speaker_id="spk_0"),
                SpeakerAssignment(start=5.0, end=10.0, speaker_id="spk_1"),
            ]
        ```
    """

    def __call__(
        self,
        audio_buffer: bytes,
        sample_rate: int,
    ) -> Awaitable[list[SpeakerAssignment]]:
        """
        Run diarization on accumulated audio.

        Args:
            audio_buffer: Raw audio bytes (PCM format).
            sample_rate: Audio sample rate in Hz.

        Returns:
            List of SpeakerAssignment objects covering the audio duration.
        """
        ...


# Type alias for diarization hook callback
DiarizationHook = Callable[[bytes, int], Awaitable[list[SpeakerAssignment]]]


# =============================================================================
# WebSocket Streaming Session
# =============================================================================


class WebSocketStreamingSession:
    """
    Manages a single WebSocket streaming session.

    This class handles the server-side state for a WebSocket connection,
    coordinating audio chunk processing, event generation, and lifecycle
    management.

    The session integrates with the existing StreamingSession and optionally
    StreamingEnrichmentSession for real-time transcription and enrichment.

    Attributes:
        stream_id: Unique identifier for this stream (str-{uuid4})
        config: Session configuration from client
        state: Current session lifecycle state
        stats: Session statistics for monitoring

    Example:
        >>> session = WebSocketStreamingSession(config)
        >>> await session.start()
        >>> async for event in session.process_audio_chunk(audio_data, sequence=1):
        ...     await websocket.send_json(event.to_dict())
        >>> final_events = await session.end()
    """

    def __init__(
        self,
        config: WebSocketSessionConfig | None = None,
        diarization_hook: DiarizationHook | None = None,
    ) -> None:
        """
        Initialize a new WebSocket streaming session.

        Args:
            config: Session configuration. Defaults to WebSocketSessionConfig().
            diarization_hook: Optional async callback for incremental diarization.
                             When provided and enable_diarization is True, the hook
                             is called periodically with accumulated audio.
        """
        self.stream_id = f"str-{uuid.uuid4()}"
        self.config = config or WebSocketSessionConfig()
        self.state = SessionState.CREATED
        self.stats = SessionStats()

        # Event ID counter (monotonically increasing per stream)
        self._event_id_counter = 0

        # Segment sequence counter
        self._segment_seq = 0

        # Audio buffer for accumulating chunks
        self._audio_buffer = bytearray()

        # Last chunk sequence for ordering validation
        self._last_chunk_sequence: int | None = None

        # Queue for outgoing events (for backpressure management)
        self._event_queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=100)

        # Internal streaming session (lazy initialized on start)
        self._streaming_session: Any = None

        # Incremental diarization state
        self._diarization_hook = diarization_hook
        self._last_diarization_audio_len = 0  # Track audio length at last diarization
        self._current_speaker_assignments: list[SpeakerAssignment] = []
        self._diarization_update_count = 0

        logger.info(
            "Created WebSocket session: stream_id=%s, config=%s, diarization_hook=%s",
            self.stream_id,
            self.config,
            "enabled" if diarization_hook else "disabled",
        )

    def _next_event_id(self) -> int:
        """Generate next monotonically increasing event ID."""
        self._event_id_counter += 1
        return self._event_id_counter

    def _next_segment_id(self) -> str:
        """Generate next segment ID."""
        seg_id = f"seg-{self._segment_seq}"
        self._segment_seq += 1
        return seg_id

    def _server_timestamp(self) -> int:
        """Get current server timestamp in milliseconds."""
        return int(time.time() * 1000)

    def _create_envelope(
        self,
        msg_type: ServerMessageType,
        payload: dict[str, Any],
        segment_id: str | None = None,
        ts_audio_start: float | None = None,
        ts_audio_end: float | None = None,
    ) -> EventEnvelope:
        """Create an event envelope with current metadata."""
        return EventEnvelope(
            event_id=self._next_event_id(),
            stream_id=self.stream_id,
            type=msg_type,
            ts_server=self._server_timestamp(),
            payload=payload,
            segment_id=segment_id,
            ts_audio_start=ts_audio_start,
            ts_audio_end=ts_audio_end,
        )

    async def start(self) -> EventEnvelope:
        """
        Start the streaming session.

        Initializes internal state and returns a SESSION_STARTED event.

        Returns:
            EventEnvelope with SESSION_STARTED type.

        Raises:
            RuntimeError: If session is not in CREATED state.
        """
        if self.state != SessionState.CREATED:
            raise RuntimeError(f"Cannot start session in state {self.state}")

        # Import streaming components here to avoid circular imports
        from .streaming import StreamConfig, StreamingSession

        # Initialize base streaming session
        stream_config = StreamConfig(max_gap_sec=self.config.max_gap_sec)
        self._streaming_session = StreamingSession(stream_config)

        self.state = SessionState.ACTIVE
        logger.info("Started WebSocket session: stream_id=%s", self.stream_id)

        return self._create_envelope(
            ServerMessageType.SESSION_STARTED,
            {"session_id": self.stream_id},
        )

    async def process_audio_chunk(
        self,
        audio_data: bytes,
        sequence: int,
    ) -> list[EventEnvelope]:
        """
        Process an audio chunk and return any resulting events.

        This method decodes the audio chunk, feeds it through the ASR pipeline,
        and generates PARTIAL or FINALIZED events based on the streaming session
        state machine.

        Args:
            audio_data: Raw audio bytes (base64 decoded by caller)
            sequence: Chunk sequence number for ordering validation

        Returns:
            List of EventEnvelope objects (may be empty if no events generated).

        Raises:
            RuntimeError: If session is not in ACTIVE state.
            ValueError: If chunks arrive out of order.
        """
        if self.state != SessionState.ACTIVE:
            raise RuntimeError(f"Cannot process chunk in state {self.state}")

        # Validate sequence ordering
        if self._last_chunk_sequence is not None:
            if sequence <= self._last_chunk_sequence:
                raise ValueError(
                    f"Chunk sequence {sequence} <= last sequence {self._last_chunk_sequence}"
                )
        self._last_chunk_sequence = sequence

        # Update stats
        self.stats.chunks_received += 1
        self.stats.bytes_received += len(audio_data)

        # Accumulate audio data
        self._audio_buffer.extend(audio_data)

        # For now, we'll generate mock events since we don't have real-time ASR
        # In production, this would feed into faster-whisper's streaming API
        # or a VAD + chunked transcription approach
        events: list[EventEnvelope] = []

        # Calculate approximate audio timestamps based on buffer size
        # Assuming 16kHz 16-bit mono PCM (2 bytes per sample)
        bytes_per_second = self.config.sample_rate * 2
        buffer_duration = len(self._audio_buffer) / bytes_per_second

        # Emit a partial event periodically (every ~1 second of audio)
        if buffer_duration >= 1.0:
            segment_id = self._next_segment_id()
            self.stats.segments_partial += 1

            events.append(
                self._create_envelope(
                    ServerMessageType.PARTIAL,
                    {
                        "segment": {
                            "start": 0.0,
                            "end": buffer_duration,
                            "text": "[processing...]",
                            "speaker_id": None,
                        }
                    },
                    segment_id=segment_id,
                    ts_audio_start=0.0,
                    ts_audio_end=buffer_duration,
                )
            )

        # Check if we should trigger incremental diarization
        diarization_event = await self._maybe_trigger_diarization()
        if diarization_event:
            events.append(diarization_event)

        self.stats.events_sent += len(events)
        return events

    def _should_trigger_diarization(self) -> bool:
        """
        Check if incremental diarization should be triggered.

        Returns True if:
        - Diarization is enabled in config
        - A diarization hook is registered
        - Enough new audio has accumulated since last diarization
        """
        if not self.config.enable_diarization:
            return False
        if self._diarization_hook is None:
            return False

        # Calculate audio duration since last diarization
        bytes_per_second = self.config.sample_rate * 2  # 16-bit mono
        current_audio_len = len(self._audio_buffer)
        new_audio_bytes = current_audio_len - self._last_diarization_audio_len
        new_audio_duration = new_audio_bytes / bytes_per_second

        return new_audio_duration >= self.config.diarization_interval_sec

    async def _maybe_trigger_diarization(self) -> EventEnvelope | None:
        """
        Conditionally trigger incremental diarization and return update event.

        If diarization should be triggered (based on accumulated audio),
        calls the diarization hook and emits a DIARIZATION_UPDATE event
        if speaker assignments have changed.

        Returns:
            EventEnvelope with DIARIZATION_UPDATE type, or None if no update.
        """
        if not self._should_trigger_diarization():
            return None

        # _should_trigger_diarization confirms hook is not None
        assert self._diarization_hook is not None

        try:
            # Call the diarization hook
            audio_bytes = bytes(self._audio_buffer)
            new_assignments = await self._diarization_hook(
                audio_bytes,
                self.config.sample_rate,
            )

            # Track that we ran diarization at this point
            self._last_diarization_audio_len = len(self._audio_buffer)
            self._diarization_update_count += 1

            # Check if assignments changed
            if self._assignments_changed(new_assignments):
                self._current_speaker_assignments = new_assignments
                return self._create_diarization_update_event(new_assignments)

            return None

        except Exception as e:
            # Graceful degradation: log error but don't crash pipeline
            logger.warning(
                "Diarization hook failed (stream_id=%s): %s",
                self.stream_id,
                e,
            )
            return None

    def _assignments_changed(self, new_assignments: list[SpeakerAssignment]) -> bool:
        """
        Check if speaker assignments have meaningfully changed.

        Compares new assignments against current assignments to determine
        if a DIARIZATION_UPDATE event should be emitted.

        Args:
            new_assignments: New speaker assignments from diarization hook.

        Returns:
            True if assignments have changed, False otherwise.
        """
        if len(new_assignments) != len(self._current_speaker_assignments):
            return True

        for new, old in zip(new_assignments, self._current_speaker_assignments, strict=True):
            if (
                new.speaker_id != old.speaker_id
                or abs(new.start - old.start) > 0.1
                or abs(new.end - old.end) > 0.1
            ):
                return True

        return False

    def _create_diarization_update_event(
        self, assignments: list[SpeakerAssignment]
    ) -> EventEnvelope:
        """
        Create a DIARIZATION_UPDATE event envelope.

        Args:
            assignments: List of speaker assignments to include in payload.

        Returns:
            EventEnvelope with DIARIZATION_UPDATE type.
        """
        # Calculate audio duration for context
        bytes_per_second = self.config.sample_rate * 2
        audio_duration = len(self._audio_buffer) / bytes_per_second

        # Extract unique speaker IDs
        speaker_ids = sorted({a.speaker_id for a in assignments})

        return self._create_envelope(
            ServerMessageType.DIARIZATION_UPDATE,
            {
                "update_number": self._diarization_update_count,
                "audio_duration": round(audio_duration, 3),
                "num_speakers": len(speaker_ids),
                "speaker_ids": speaker_ids,
                "assignments": [a.to_dict() for a in assignments],
            },
            ts_audio_start=0.0,
            ts_audio_end=audio_duration,
        )

    async def _trigger_final_diarization(self) -> EventEnvelope | None:
        """
        Trigger final diarization at end of stream.

        Always runs diarization if enabled and hook is available,
        regardless of interval threshold.

        Returns:
            EventEnvelope with DIARIZATION_UPDATE type, or None if not available.
        """
        if not self.config.enable_diarization:
            return None
        if self._diarization_hook is None:
            return None
        if len(self._audio_buffer) == 0:
            return None

        try:
            audio_bytes = bytes(self._audio_buffer)
            new_assignments = await self._diarization_hook(
                audio_bytes,
                self.config.sample_rate,
            )

            self._diarization_update_count += 1
            self._current_speaker_assignments = new_assignments

            if new_assignments:
                return self._create_diarization_update_event(new_assignments)

            return None

        except Exception as e:
            logger.warning(
                "Final diarization hook failed (stream_id=%s): %s",
                self.stream_id,
                e,
            )
            return None

    def get_speaker_assignments(self) -> list[SpeakerAssignment]:
        """
        Get current speaker assignments.

        Returns the most recent speaker assignments from incremental
        diarization. Useful for applying speaker labels to finalized
        segments.

        Returns:
            List of SpeakerAssignment objects (may be empty).
        """
        return list(self._current_speaker_assignments)

    async def end(self) -> list[EventEnvelope]:
        """
        End the streaming session.

        Finalizes any pending segments, computes session statistics,
        and returns final events including SESSION_ENDED.

        Returns:
            List of EventEnvelope objects including final segments and SESSION_ENDED.

        Raises:
            RuntimeError: If session is not in ACTIVE state.
        """
        if self.state != SessionState.ACTIVE:
            raise RuntimeError(f"Cannot end session in state {self.state}")

        self.state = SessionState.ENDING
        events: list[EventEnvelope] = []

        # Trigger final diarization if enabled
        final_diarization = await self._trigger_final_diarization()
        if final_diarization:
            events.append(final_diarization)

        # Finalize any pending audio in buffer
        if len(self._audio_buffer) > 0:
            bytes_per_second = self.config.sample_rate * 2
            buffer_duration = len(self._audio_buffer) / bytes_per_second

            segment_id = self._next_segment_id()
            self.stats.segments_finalized += 1

            events.append(
                self._create_envelope(
                    ServerMessageType.FINALIZED,
                    {
                        "segment": {
                            "start": 0.0,
                            "end": buffer_duration,
                            "text": "[final segment]",
                            "speaker_id": None,
                            "audio_state": None,
                        }
                    },
                    segment_id=segment_id,
                    ts_audio_start=0.0,
                    ts_audio_end=buffer_duration,
                )
            )

        # Finalize stats
        self.stats.end_time = time.time()
        self.stats.events_sent += len(events) + 1  # +1 for SESSION_ENDED

        # Generate SESSION_ENDED event
        events.append(
            self._create_envelope(
                ServerMessageType.SESSION_ENDED,
                {"stats": self.stats.to_dict()},
            )
        )

        self.state = SessionState.ENDED
        self._audio_buffer.clear()

        logger.info(
            "Ended WebSocket session: stream_id=%s, stats=%s",
            self.stream_id,
            self.stats.to_dict(),
        )

        return events

    def create_error_event(
        self,
        code: str,
        message: str,
        recoverable: bool = True,
    ) -> EventEnvelope:
        """
        Create an error event.

        Args:
            code: Error code identifier
            message: Human-readable error message
            recoverable: Whether the client can continue after this error

        Returns:
            EventEnvelope with ERROR type.
        """
        self.stats.errors += 1
        if not recoverable:
            self.state = SessionState.ERROR

        return self._create_envelope(
            ServerMessageType.ERROR,
            {
                "code": code,
                "message": message,
                "recoverable": recoverable,
            },
        )

    def create_pong_event(self, client_timestamp: int) -> EventEnvelope:
        """
        Create a PONG response to client PING.

        Args:
            client_timestamp: Timestamp from client's PING message

        Returns:
            EventEnvelope with PONG type.
        """
        return self._create_envelope(
            ServerMessageType.PONG,
            {
                "timestamp": client_timestamp,
                "server_timestamp": self._server_timestamp(),
            },
        )


# =============================================================================
# Message Parsing Helpers
# =============================================================================


def parse_client_message(
    data: dict[str, Any],
) -> tuple[ClientMessageType, dict[str, Any]]:
    """
    Parse a client WebSocket message.

    Args:
        data: Parsed JSON message from client

    Returns:
        Tuple of (message_type, payload)

    Raises:
        ValueError: If message type is missing or invalid
    """
    msg_type_str = data.get("type")
    if not msg_type_str:
        raise ValueError("Missing 'type' field in client message")

    try:
        msg_type = ClientMessageType(msg_type_str)
    except ValueError as e:
        raise ValueError(f"Invalid client message type: {msg_type_str}") from e

    # Extract payload (everything except 'type')
    payload = {k: v for k, v in data.items() if k != "type"}

    return msg_type, payload


def decode_audio_chunk(payload: dict[str, Any]) -> tuple[bytes, int]:
    """
    Decode an AUDIO_CHUNK message payload.

    Args:
        payload: Message payload with 'data' (base64) and 'sequence' fields

    Returns:
        Tuple of (decoded_audio_bytes, sequence_number)

    Raises:
        ValueError: If required fields are missing or invalid
    """
    data_b64 = payload.get("data")
    if not data_b64:
        raise ValueError("Missing 'data' field in AUDIO_CHUNK")

    sequence = payload.get("sequence")
    if sequence is None:
        raise ValueError("Missing 'sequence' field in AUDIO_CHUNK")

    try:
        audio_bytes = base64.b64decode(data_b64)
    except Exception as e:
        raise ValueError(f"Invalid base64 audio data: {e}") from e

    return audio_bytes, int(sequence)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enums
    "ClientMessageType",
    "ServerMessageType",
    "SessionState",
    # Data classes
    "EventEnvelope",
    "WebSocketSessionConfig",
    "SessionStats",
    "SpeakerAssignment",
    # Diarization types
    "DiarizationHook",
    "DiarizationHookProtocol",
    # Session class
    "WebSocketStreamingSession",
    # Helpers
    "parse_client_message",
    "decode_audio_chunk",
]
