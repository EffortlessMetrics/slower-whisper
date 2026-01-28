"""
WebSocket streaming protocol for real-time transcription (v2.1.0).

This module implements the WebSocket-based streaming API for real-time
audio transcription and enrichment. It provides event-driven communication
with clients using JSON message envelopes.

Key features:
- Event envelope with monotonically increasing event_id
- Client message types: START_SESSION, AUDIO_CHUNK, END_SESSION, PING, TTS_STATE
- Server message types: SESSION_STARTED, PARTIAL, FINALIZED, ERROR, etc.
- v2.1 additions: PHYSICS_UPDATE, AUDIO_HEALTH, VAD_ACTIVITY, BARGE_IN, END_OF_TURN_HINT
- Session lifecycle management with configurable enrichment
- Backpressure handling: PARTIAL events can be dropped, FINALIZED never dropped
- TTS state tracking for barge-in detection

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
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .asr_engine import WhisperModelProtocol
    from .audio_health import AudioHealthAggregator
    from .conversation_physics import ConversationPhysicsTracker
    from .streaming_asr import StreamingASRAdapter, StreamingASRConfig
    from .streaming_callbacks import StreamCallbacks

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
    RESUME_SESSION = "RESUME_SESSION"  # Resume after reconnection
    TTS_STATE = "TTS_STATE"  # Client sends when TTS starts/stops playing


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
    # v2.1 additions
    PHYSICS_UPDATE = "PHYSICS_UPDATE"
    AUDIO_HEALTH = "AUDIO_HEALTH"
    VAD_ACTIVITY = "VAD_ACTIVITY"
    BARGE_IN = "BARGE_IN"
    END_OF_TURN_HINT = "END_OF_TURN_HINT"


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
        replay_buffer_size: Size of event replay buffer for resume (default: 100)
        backpressure_threshold: Queue size threshold for backpressure (default: 80)
        enable_conversation_physics: Enable conversation physics updates (default: False)
        enable_audio_health: Enable audio health monitoring (default: False)
        audio_health_interval_chunks: Interval for audio health updates in chunks (default: 10)
        enable_reflex_events: Enable reflex events like VAD_ACTIVITY and BARGE_IN (default: False)
        enable_tts_style: Enable TTS style hints in events (default: False)
        enable_correction_detection: Enable detection of speaker corrections (default: False)
        enable_commitment_tracking: Enable tracking of speaker commitments (default: False)
        correction_similarity_threshold: Similarity threshold for correction detection (default: 0.7)
    """

    max_gap_sec: float = 1.0
    enable_prosody: bool = False
    enable_emotion: bool = False
    enable_categorical_emotion: bool = False
    enable_diarization: bool = False
    diarization_interval_sec: float = 30.0
    sample_rate: int = 16000
    audio_format: str = "pcm_s16le"
    replay_buffer_size: int = 100
    backpressure_threshold: int = 80
    # v2.1 additions
    enable_conversation_physics: bool = False
    enable_audio_health: bool = False
    audio_health_interval_chunks: int = 10
    enable_reflex_events: bool = False
    enable_tts_style: bool = False
    enable_correction_detection: bool = False
    enable_commitment_tracking: bool = False
    correction_similarity_threshold: float = 0.7

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebSocketSessionConfig:
        """Create config from dictionary (e.g., from client message).

        Raises:
            ValueError: If sample_rate is not a positive integer.
        """
        sample_rate = int(data.get("sample_rate", 16000))
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        return cls(
            max_gap_sec=float(data.get("max_gap_sec", 1.0)),
            enable_prosody=bool(data.get("enable_prosody", False)),
            enable_emotion=bool(data.get("enable_emotion", False)),
            enable_categorical_emotion=bool(data.get("enable_categorical_emotion", False)),
            enable_diarization=bool(data.get("enable_diarization", False)),
            diarization_interval_sec=float(data.get("diarization_interval_sec", 30.0)),
            sample_rate=sample_rate,
            audio_format=str(data.get("audio_format", "pcm_s16le")),
            replay_buffer_size=int(data.get("replay_buffer_size", 100)),
            backpressure_threshold=int(data.get("backpressure_threshold", 80)),
            # v2.1 additions
            enable_conversation_physics=bool(data.get("enable_conversation_physics", False)),
            enable_audio_health=bool(data.get("enable_audio_health", False)),
            audio_health_interval_chunks=int(data.get("audio_health_interval_chunks", 10)),
            enable_reflex_events=bool(data.get("enable_reflex_events", False)),
            enable_tts_style=bool(data.get("enable_tts_style", False)),
            enable_correction_detection=bool(data.get("enable_correction_detection", False)),
            enable_commitment_tracking=bool(data.get("enable_commitment_tracking", False)),
            correction_similarity_threshold=float(data.get("correction_similarity_threshold", 0.7)),
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
    backpressure_events: int = 0
    resume_attempts: int = 0
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
            "backpressure_events": self.backpressure_events,
            "resume_attempts": self.resume_attempts,
            "duration_sec": round(duration, 3),
        }


@dataclass
class ReplayBuffer:
    """
    Circular buffer for storing events for resume capability.

    The replay buffer stores recent events so clients can resume
    after reconnection by replaying missed events.

    Attributes:
        max_size: Maximum number of events to store
        events: List of stored events (most recent last)
        oldest_event_id: Oldest event ID in buffer (for gap detection)
    """

    max_size: int = 100
    events: list[EventEnvelope] = field(default_factory=list)
    oldest_event_id: int = 0

    def add(self, event: EventEnvelope) -> None:
        """Add an event to the buffer."""
        self.events.append(event)
        if len(self.events) > self.max_size:
            removed = self.events.pop(0)
            self.oldest_event_id = self.events[0].event_id if self.events else removed.event_id + 1

    def get_events_since(self, last_event_id: int) -> tuple[list[EventEnvelope], bool]:
        """
        Get events since the given event ID.

        Args:
            last_event_id: Last event ID client received

        Returns:
            Tuple of (events to replay, gap_detected).
            gap_detected is True if some events were lost (not in buffer).
        """
        if not self.events:
            return [], False

        # Check for gap (client's last_event_id is older than our buffer)
        gap_detected = last_event_id < self.oldest_event_id

        # Find events to replay
        replay = [e for e in self.events if e.event_id > last_event_id]
        return replay, gap_detected

    def clear(self) -> None:
        """Clear the buffer."""
        self.events.clear()
        self.oldest_event_id = 0


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
        asr_model: WhisperModelProtocol | None = None,
        asr_config: StreamingASRConfig | None = None,
        callbacks: StreamCallbacks | None = None,
    ) -> None:
        """
        Initialize a new WebSocket streaming session.

        Args:
            config: Session configuration. Defaults to WebSocketSessionConfig().
            diarization_hook: Optional async callback for incremental diarization.
                             When provided and enable_diarization is True, the hook
                             is called periodically with accumulated audio.
            asr_model: Optional faster-whisper model instance for real transcription.
                      If not provided, the session will emit mock transcription events.
            asr_config: Optional StreamingASRConfig for ASR adapter behavior.
                       Only used when asr_model is provided.
            callbacks: Optional callbacks for streaming events (v2.1).
                      Used to emit physics, audio health, VAD, and barge-in events.
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

        # Replay buffer for resume capability
        self._replay_buffer = ReplayBuffer(max_size=self.config.replay_buffer_size)

        # Backpressure state
        self._backpressure_active = False

        # Internal streaming session (lazy initialized on start)
        self._streaming_session: Any = None

        # ASR adapter for real transcription (lazy initialized on start)
        self._asr_model = asr_model
        self._asr_config = asr_config
        self._asr_adapter: StreamingASRAdapter | None = None

        # Incremental diarization state
        self._diarization_hook = diarization_hook
        self._last_diarization_audio_len = 0  # Track audio length at last diarization
        self._current_speaker_assignments: list[SpeakerAssignment] = []
        self._diarization_update_count = 0

        # TTS state tracking for barge-in detection (v2.1)
        self._tts_playing: bool = False
        self._tts_start_time: float | None = None

        # v2.1 feature state (lazy initialized to avoid import cost when disabled)
        self._callbacks = callbacks
        self._audio_health_aggregator: AudioHealthAggregator | None = None
        self._physics_tracker: ConversationPhysicsTracker | None = None
        self._health_chunk_counter = 0  # Counter for interval-based health updates
        self._last_vad_is_speech: bool | None = None  # Track VAD state transitions
        self._silence_start_time: float | None = None  # Track silence duration

        # Initialize v2.1 trackers if features are enabled
        if self.config.enable_audio_health:
            from .audio_health import AudioHealthAggregator

            self._audio_health_aggregator = AudioHealthAggregator(window_size=10)

        if self.config.enable_conversation_physics:
            from .conversation_physics import ConversationPhysicsTracker

            self._physics_tracker = ConversationPhysicsTracker()

        logger.info(
            "Created WebSocket session: stream_id=%s, config=%s, diarization_hook=%s, asr=%s",
            self.stream_id,
            self.config,
            "enabled" if diarization_hook else "disabled",
            "enabled" if asr_model else "mock",
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
        envelope = EventEnvelope(
            event_id=self._next_event_id(),
            stream_id=self.stream_id,
            type=msg_type,
            ts_server=self._server_timestamp(),
            payload=payload,
            segment_id=segment_id,
            ts_audio_start=ts_audio_start,
            ts_audio_end=ts_audio_end,
        )
        # Add to replay buffer for resume capability
        self._replay_buffer.add(envelope)
        return envelope

    def check_backpressure(self) -> bool:
        """
        Check if backpressure should be applied.

        Returns True if the event queue is approaching capacity.
        Clients should slow down audio submission when this returns True.

        Returns:
            True if backpressure is active, False otherwise.
        """
        queue_size = self._event_queue.qsize()
        threshold = self.config.backpressure_threshold

        if queue_size >= threshold:
            if not self._backpressure_active:
                self._backpressure_active = True
                self.stats.backpressure_events += 1
                logger.warning(
                    "Backpressure activated: stream_id=%s, queue_size=%d",
                    self.stream_id,
                    queue_size,
                )
            return True

        if self._backpressure_active and queue_size < threshold * 0.5:
            self._backpressure_active = False
            logger.info(
                "Backpressure deactivated: stream_id=%s, queue_size=%d",
                self.stream_id,
                queue_size,
            )

        return self._backpressure_active

    def drop_partial_events(self) -> int:
        """
        Drop PARTIAL events from queue under backpressure.

        Implements the drop policy: PARTIAL events can be dropped,
        FINALIZED events are never dropped.

        Returns:
            Number of events dropped.
        """
        if self._event_queue.empty():
            return 0

        # Drain queue, keeping non-droppable events
        events_to_keep: list[EventEnvelope] = []
        dropped = 0

        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                # Never drop FINALIZED, SESSION_ENDED, or ERROR events
                if event.type in (
                    ServerMessageType.FINALIZED,
                    ServerMessageType.SESSION_ENDED,
                    ServerMessageType.ERROR,
                ):
                    events_to_keep.append(event)
                else:
                    dropped += 1
            except asyncio.QueueEmpty:
                break

        # Re-add kept events
        for event in events_to_keep:
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                # Should not happen as we just drained, but be safe
                dropped += 1

        if dropped > 0:
            self.stats.events_dropped += dropped
            logger.warning(
                "Dropped %d PARTIAL events due to backpressure: stream_id=%s",
                dropped,
                self.stream_id,
            )

        return dropped

    def get_events_for_resume(self, last_event_id: int) -> tuple[list[EventEnvelope], bool]:
        """
        Get events for resume after reconnection.

        Args:
            last_event_id: Last event ID client received before disconnect.

        Returns:
            Tuple of (events to replay, gap_detected).
            If gap_detected is True, some events were lost and cannot be replayed.
        """
        self.stats.resume_attempts += 1
        events, gap_detected = self._replay_buffer.get_events_since(last_event_id)

        if gap_detected:
            logger.warning(
                "Resume gap detected: stream_id=%s, last_event_id=%d, oldest_in_buffer=%d",
                self.stream_id,
                last_event_id,
                self._replay_buffer.oldest_event_id,
            )

        logger.info(
            "Resume request: stream_id=%s, last_event_id=%d, replaying=%d events, gap=%s",
            self.stream_id,
            last_event_id,
            len(events),
            gap_detected,
        )

        return events, gap_detected

    def create_resume_gap_error(self, last_event_id: int) -> EventEnvelope:
        """
        Create a RESUME_GAP error when replay buffer cannot satisfy resume.

        Args:
            last_event_id: Client's last event ID that caused the gap.

        Returns:
            EventEnvelope with ERROR type and RESUME_GAP code.
        """
        return self._create_envelope(
            ServerMessageType.ERROR,
            {
                "code": "RESUME_GAP",
                "message": (
                    f"Cannot resume from event {last_event_id}. "
                    f"Oldest available event is {self._replay_buffer.oldest_event_id}. "
                    "Client must restart session."
                ),
                "recoverable": False,
                "details": {
                    "requested_event_id": last_event_id,
                    "oldest_available": self._replay_buffer.oldest_event_id,
                    "current_event_id": self._event_id_counter,
                },
            },
        )

    def create_buffer_overflow_error(self) -> EventEnvelope:
        """
        Create a BUFFER_OVERFLOW error when backpressure is critical.

        Returns:
            EventEnvelope with ERROR type and BUFFER_OVERFLOW code.
        """
        self.stats.errors += 1
        return self._create_envelope(
            ServerMessageType.ERROR,
            {
                "code": "BUFFER_OVERFLOW",
                "message": (
                    "Event buffer overflow - client is not consuming events fast enough. "
                    "Some PARTIAL events have been dropped."
                ),
                "recoverable": True,
                "details": {
                    "events_dropped": self.stats.events_dropped,
                    "queue_size": self._event_queue.qsize(),
                },
            },
        )

    def set_tts_state(self, playing: bool) -> None:
        """
        Update the TTS playback state.

        Called when client sends TTS_STATE message to indicate TTS has
        started or stopped playing. This state is used for barge-in detection.

        Args:
            playing: True if TTS is currently playing, False if stopped.
        """
        if playing and not self._tts_playing:
            # TTS starting
            self._tts_playing = True
            self._tts_start_time = time.time()
            logger.debug(
                "TTS started: stream_id=%s, start_time=%s",
                self.stream_id,
                self._tts_start_time,
            )
        elif not playing and self._tts_playing:
            # TTS stopping
            duration = time.time() - (self._tts_start_time or time.time())
            self._tts_playing = False
            self._tts_start_time = None
            logger.debug(
                "TTS stopped: stream_id=%s, duration=%.3fs",
                self.stream_id,
                duration,
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

        # Initialize ASR adapter if model is provided
        if self._asr_model is not None:
            from .streaming_asr import StreamingASRAdapter, StreamingASRConfig

            # Use provided config or create default with session sample rate
            asr_config = self._asr_config or StreamingASRConfig(sample_rate=self.config.sample_rate)
            self._asr_adapter = StreamingASRAdapter(self._asr_model, asr_config)
            logger.info(
                "Initialized ASR adapter for session: stream_id=%s",
                self.stream_id,
            )

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

        When an ASR model is configured, the adapter performs VAD-triggered
        chunked transcription. Otherwise, mock events are generated for testing.

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

        # Accumulate audio data (for diarization and fallback)
        self._audio_buffer.extend(audio_data)

        events: list[EventEnvelope] = []

        # Use real ASR if adapter is available
        if self._asr_adapter is not None:
            events.extend(await self._process_audio_with_asr(audio_data))
        else:
            # Mock transcription for testing
            events.extend(self._generate_mock_events())

        # Check if we should trigger incremental diarization
        diarization_event = await self._maybe_trigger_diarization()
        if diarization_event:
            events.append(diarization_event)

        # Process v2.1 features (audio health, physics, reflex events)
        v21_events = self._process_v21_features(audio_data, events)
        events.extend(v21_events)

        self.stats.events_sent += len(events)
        return events

    async def _process_audio_with_asr(self, audio_data: bytes) -> list[EventEnvelope]:
        """Process audio through the real ASR adapter.

        Args:
            audio_data: Raw PCM audio bytes.

        Returns:
            List of EventEnvelope objects from transcription results.
        """
        if self._asr_adapter is None or self._streaming_session is None:
            return []

        events: list[EventEnvelope] = []

        try:
            # Feed audio to ASR adapter
            chunks = await self._asr_adapter.ingest_audio(audio_data)

            # Process each transcription chunk through the streaming session
            for chunk in chunks:
                stream_events = self._streaming_session.ingest_chunk(chunk)

                for stream_event in stream_events:
                    segment = stream_event.segment
                    segment_id = self._next_segment_id()

                    # Determine event type based on stream event type
                    from .streaming import StreamEventType

                    if stream_event.type == StreamEventType.FINAL_SEGMENT:
                        self.stats.segments_finalized += 1
                        events.append(
                            self._create_envelope(
                                ServerMessageType.FINALIZED,
                                {
                                    "segment": {
                                        "start": segment.start,
                                        "end": segment.end,
                                        "text": segment.text,
                                        "speaker_id": segment.speaker_id,
                                        "audio_state": None,
                                    }
                                },
                                segment_id=segment_id,
                                ts_audio_start=segment.start,
                                ts_audio_end=segment.end,
                            )
                        )
                    else:
                        # PARTIAL_SEGMENT
                        self.stats.segments_partial += 1
                        events.append(
                            self._create_envelope(
                                ServerMessageType.PARTIAL,
                                {
                                    "segment": {
                                        "start": segment.start,
                                        "end": segment.end,
                                        "text": segment.text,
                                        "speaker_id": segment.speaker_id,
                                    }
                                },
                                segment_id=segment_id,
                                ts_audio_start=segment.start,
                                ts_audio_end=segment.end,
                            )
                        )

        except Exception as e:
            # Log error but don't crash the pipeline (invariant 3)
            logger.warning(
                "ASR processing error (stream_id=%s): %s",
                self.stream_id,
                e,
            )
            self.stats.errors += 1

        return events

    def _generate_mock_events(self) -> list[EventEnvelope]:
        """Generate mock transcription events for testing.

        Used when no ASR model is configured.

        Returns:
            List of mock EventEnvelope objects.
        """
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

        return events

    def _process_v21_features(
        self,
        audio_data: bytes,
        transcription_events: list[EventEnvelope],
    ) -> list[EventEnvelope]:
        """Process v2.1 features and emit events.

        Handles audio health, VAD activity, barge-in detection, and
        conversation physics tracking based on config flags.

        Args:
            audio_data: Raw PCM audio bytes for the current chunk.
            transcription_events: Events generated by ASR (used for physics tracking).

        Returns:
            List of v2.1 EventEnvelope objects (may be empty if features disabled).
        """
        events: list[EventEnvelope] = []

        # --- Audio Health ---
        if self.config.enable_audio_health and self._audio_health_aggregator is not None:
            from .audio_health import analyze_chunk_health

            snapshot = analyze_chunk_health(audio_data, self.config.sample_rate)
            self._audio_health_aggregator.add_snapshot(snapshot)
            self._health_chunk_counter += 1

            # Emit at configured interval
            if self._health_chunk_counter >= self.config.audio_health_interval_chunks:
                self._health_chunk_counter = 0
                aggregate = self._audio_health_aggregator.get_aggregate()
                events.append(
                    self._create_envelope(
                        ServerMessageType.AUDIO_HEALTH,
                        {
                            "clipping_ratio": aggregate.clipping_ratio,
                            "rms_energy": aggregate.rms_energy,
                            "snr_proxy": aggregate.snr_proxy,
                            "spectral_centroid": aggregate.spectral_centroid,
                            "quality_score": aggregate.quality_score,
                            "is_speech_likely": aggregate.is_speech_likely,
                        },
                    )
                )
                # Invoke callback if available
                self._invoke_callback_safe("on_audio_health", aggregate)

        # --- VAD Activity & Barge-in ---
        if self.config.enable_reflex_events:
            # Use audio health snapshot for VAD if available, otherwise compute
            if self.config.enable_audio_health and self._audio_health_aggregator is not None:
                aggregate = self._audio_health_aggregator.get_aggregate()
                is_speech = aggregate.is_speech_likely
                energy = aggregate.rms_energy
            else:
                # Compute minimal VAD from raw audio
                from .audio_health import analyze_chunk_health

                snapshot = analyze_chunk_health(audio_data, self.config.sample_rate)
                is_speech = snapshot.is_speech_likely
                energy = snapshot.rms_energy

            # Track silence duration
            current_time = time.time()
            if is_speech:
                self._silence_start_time = None
                silence_duration = 0.0
            else:
                if self._silence_start_time is None:
                    self._silence_start_time = current_time
                silence_duration = current_time - self._silence_start_time

            # Emit VAD_ACTIVITY on state transition
            if self._last_vad_is_speech != is_speech:
                self._last_vad_is_speech = is_speech
                events.append(
                    self._create_envelope(
                        ServerMessageType.VAD_ACTIVITY,
                        {
                            "energy_level": energy,
                            "is_speech": is_speech,
                            "silence_duration_sec": silence_duration,
                        },
                    )
                )
                # Invoke callback
                from .streaming_callbacks import VADActivityPayload

                self._invoke_callback_safe(
                    "on_vad_activity",
                    VADActivityPayload(
                        energy_level=energy,
                        is_speech=is_speech,
                        silence_duration_sec=silence_duration,
                    ),
                )

            # Check for barge-in (speech detected while TTS playing)
            if is_speech and self._tts_playing and self._tts_start_time is not None:
                tts_elapsed = current_time - self._tts_start_time
                events.append(
                    self._create_envelope(
                        ServerMessageType.BARGE_IN,
                        {
                            "energy": energy,
                            "tts_elapsed_sec": tts_elapsed,
                        },
                    )
                )
                # Invoke callback
                from .streaming_callbacks import BargeInPayload

                self._invoke_callback_safe(
                    "on_barge_in",
                    BargeInPayload(energy=energy, tts_elapsed_sec=tts_elapsed),
                )
                # Auto-stop TTS state after barge-in detected
                self._tts_playing = False
                self._tts_start_time = None

        # --- Conversation Physics ---
        if self.config.enable_conversation_physics and self._physics_tracker is not None:
            # Record finalized segments to physics tracker
            for event in transcription_events:
                if event.type == ServerMessageType.FINALIZED:
                    segment = event.payload.get("segment", {})
                    start = segment.get("start", 0.0)
                    end = segment.get("end", 0.0)
                    speaker_id = segment.get("speaker_id") or "unknown"
                    self._physics_tracker.record_segment(speaker_id, start, end)

                    # Emit physics update after each finalized segment
                    physics_snapshot = self._physics_tracker.get_snapshot()
                    events.append(
                        self._create_envelope(
                            ServerMessageType.PHYSICS_UPDATE,
                            {
                                "speaker_talk_times": physics_snapshot.speaker_talk_times,
                                "total_duration_sec": physics_snapshot.total_duration_sec,
                                "interruption_count": physics_snapshot.interruption_count,
                                "interruption_rate": physics_snapshot.interruption_rate,
                                "mean_response_latency_sec": physics_snapshot.mean_response_latency_sec,
                                "speaker_transitions": physics_snapshot.speaker_transitions,
                                "overlap_duration_sec": physics_snapshot.overlap_duration_sec,
                            },
                        )
                    )
                    # Invoke callback
                    self._invoke_callback_safe("on_physics_update", physics_snapshot)

        return events

    def _invoke_callback_safe(self, method_name: str, payload: Any) -> None:
        """Invoke a callback method safely, catching exceptions.

        Per invariant #3, callbacks never crash the pipeline.

        Args:
            method_name: Name of the callback method to invoke.
            payload: Payload to pass to the callback.
        """
        if self._callbacks is None:
            return

        method = getattr(self._callbacks, method_name, None)
        if method is None:
            return

        try:
            method(payload)
        except Exception as e:
            logger.warning(
                "Callback %s raised exception (stream_id=%s): %s",
                method_name,
                self.stream_id,
                e,
            )
            # Try to invoke on_error if available
            try:
                from .streaming_callbacks import StreamingError

                error_handler = getattr(self._callbacks, "on_error", None)
                if error_handler:
                    error_handler(
                        StreamingError(
                            exception=e,
                            context=f"Callback {method_name} failed",
                            recoverable=True,
                        )
                    )
            except Exception:
                pass  # Don't let error handling errors crash us

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

        When ASR is enabled, flushes the ASR adapter to process any
        remaining buffered audio.

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

        # Flush ASR adapter if enabled
        if self._asr_adapter is not None:
            events.extend(await self._flush_asr_adapter())
        elif len(self._audio_buffer) > 0:
            # Mock finalization when no ASR
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

    async def _flush_asr_adapter(self) -> list[EventEnvelope]:
        """Flush the ASR adapter and finalize remaining audio.

        Returns:
            List of EventEnvelope objects from flushed transcription.
        """
        if self._asr_adapter is None or self._streaming_session is None:
            return []

        events: list[EventEnvelope] = []

        try:
            # Flush any remaining audio from the adapter
            final_chunks = await self._asr_adapter.flush()

            # Process each chunk through the streaming session
            for chunk in final_chunks:
                stream_events = self._streaming_session.ingest_chunk(chunk)
                for stream_event in stream_events:
                    segment = stream_event.segment
                    segment_id = self._next_segment_id()
                    self.stats.segments_finalized += 1

                    events.append(
                        self._create_envelope(
                            ServerMessageType.FINALIZED,
                            {
                                "segment": {
                                    "start": segment.start,
                                    "end": segment.end,
                                    "text": segment.text,
                                    "speaker_id": segment.speaker_id,
                                    "audio_state": None,
                                }
                            },
                            segment_id=segment_id,
                            ts_audio_start=segment.start,
                            ts_audio_end=segment.end,
                        )
                    )

            # Finalize any remaining partial in the streaming session
            end_stream_events = self._streaming_session.end_of_stream()
            for stream_event in end_stream_events:
                segment = stream_event.segment
                segment_id = self._next_segment_id()
                self.stats.segments_finalized += 1

                events.append(
                    self._create_envelope(
                        ServerMessageType.FINALIZED,
                        {
                            "segment": {
                                "start": segment.start,
                                "end": segment.end,
                                "text": segment.text,
                                "speaker_id": segment.speaker_id,
                                "audio_state": None,
                            }
                        },
                        segment_id=segment_id,
                        ts_audio_start=segment.start,
                        ts_audio_end=segment.end,
                    )
                )

        except Exception as e:
            logger.warning(
                "ASR flush error (stream_id=%s): %s",
                self.stream_id,
                e,
            )
            self.stats.errors += 1

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
    "ReplayBuffer",
    # Diarization types
    "DiarizationHook",
    "DiarizationHookProtocol",
    # Session class
    "WebSocketStreamingSession",
    # Helpers
    "parse_client_message",
    "decode_audio_chunk",
]
