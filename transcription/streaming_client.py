"""
Reference Python client for WebSocket streaming transcription.

This module provides a high-level client for interacting with the
slower-whisper streaming WebSocket API. It handles:
- Connection lifecycle management
- Message serialization/deserialization
- Event handling via callbacks or async iteration
- Reconnection with resume (best-effort)
- Audio chunking and sequencing

Example usage:
    async with StreamingClient("ws://localhost:8000/stream") as client:
        await client.start_session(max_gap_sec=1.0)

        async for chunk in audio_chunks:
            await client.send_audio(chunk)

        async for event in client.events():
            if event.type == "FINALIZED":
                print(f"Segment: {event.payload['segment']['text']}")

        await client.end_session()
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StreamingConfig:
    """Configuration for streaming session.

    Attributes:
        url: WebSocket URL to connect to.
        max_gap_sec: Gap threshold to finalize segment (default: 1.0s).
        enable_prosody: Extract prosodic features from audio.
        enable_emotion: Extract dimensional emotion features.
        enable_categorical_emotion: Extract categorical emotion labels.
        sample_rate: Expected audio sample rate (default: 16000 Hz).
        audio_format: Audio encoding format (default: "pcm_s16le").
        reconnect_attempts: Number of reconnection attempts on disconnect.
        reconnect_delay: Delay between reconnection attempts in seconds.
        ping_interval: Interval for sending ping messages (0 to disable).
        ping_timeout: Timeout for ping response before considering connection dead.
    """

    url: str = "ws://localhost:8000/stream"
    max_gap_sec: float = 1.0
    enable_prosody: bool = False
    enable_emotion: bool = False
    enable_categorical_emotion: bool = False
    sample_rate: int = 16000
    audio_format: str = "pcm_s16le"
    reconnect_attempts: int = 3
    reconnect_delay: float = 1.0
    ping_interval: float = 30.0
    ping_timeout: float = 10.0


# =============================================================================
# Events
# =============================================================================


class EventType(str, Enum):
    """Server event types."""

    SESSION_STARTED = "SESSION_STARTED"
    PARTIAL = "PARTIAL"
    FINALIZED = "FINALIZED"
    SPEAKER_TURN = "SPEAKER_TURN"
    SEMANTIC_UPDATE = "SEMANTIC_UPDATE"
    ERROR = "ERROR"
    SESSION_ENDED = "SESSION_ENDED"
    PONG = "PONG"


@dataclass
class StreamEvent:
    """Event received from the streaming server.

    Attributes:
        event_id: Monotonically increasing ID per stream.
        stream_id: Unique stream identifier.
        type: Server message type.
        ts_server: Server timestamp (Unix epoch milliseconds).
        payload: Message-specific payload data.
        segment_id: Segment identifier (for segment events).
        ts_audio_start: Audio timestamp start in seconds.
        ts_audio_end: Audio timestamp end in seconds.
    """

    event_id: int
    stream_id: str
    type: str
    ts_server: int
    payload: dict[str, Any]
    segment_id: str | None = None
    ts_audio_start: float | None = None
    ts_audio_end: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamEvent:
        """Create a StreamEvent from a dictionary.

        Args:
            data: Dictionary from JSON message.

        Returns:
            StreamEvent instance.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = ["event_id", "stream_id", "type", "ts_server", "payload"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        return cls(
            event_id=data["event_id"],
            stream_id=data["stream_id"],
            type=data["type"],
            ts_server=data["ts_server"],
            payload=data["payload"],
            segment_id=data.get("segment_id"),
            ts_audio_start=data.get("ts_audio_start"),
            ts_audio_end=data.get("ts_audio_end"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "event_id": self.event_id,
            "stream_id": self.stream_id,
            "type": self.type,
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

    @property
    def is_error(self) -> bool:
        """Check if this is an error event."""
        return self.type == EventType.ERROR.value

    @property
    def is_recoverable(self) -> bool:
        """Check if error is recoverable (only valid for ERROR events)."""
        if not self.is_error:
            return True
        return bool(self.payload.get("recoverable", True))


# =============================================================================
# Client State
# =============================================================================


class ClientState(str, Enum):
    """Client connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SESSION_ACTIVE = "session_active"
    SESSION_ENDING = "session_ending"
    CLOSED = "closed"


# =============================================================================
# Exceptions
# =============================================================================


class StreamingClientError(Exception):
    """Base exception for streaming client errors."""

    pass


class ConnectionError(StreamingClientError):
    """Raised when connection fails."""

    pass


class SessionError(StreamingClientError):
    """Raised when session operation fails."""

    pass


class ProtocolError(StreamingClientError):
    """Raised when protocol violation is detected."""

    pass


class ResumeGapError(SessionError):
    """Raised when session resume fails due to event gap.

    This indicates the server's replay buffer cannot satisfy the resume
    request because the client's last_event_id is too old. The client
    must restart the session.

    Attributes:
        requested_event_id: The event ID the client requested to resume from.
        oldest_available: The oldest event ID available in the server's buffer.
    """

    def __init__(
        self,
        message: str,
        requested_event_id: int | None = None,
        oldest_available: int | None = None,
    ) -> None:
        super().__init__(message)
        self.requested_event_id = requested_event_id
        self.oldest_available = oldest_available


# =============================================================================
# Statistics
# =============================================================================


@dataclass
class ClientStats:
    """Statistics tracked by the streaming client.

    Attributes:
        chunks_sent: Number of audio chunks sent.
        bytes_sent: Total audio bytes sent.
        events_received: Number of events received.
        partials_received: Number of PARTIAL events received.
        finalized_received: Number of FINALIZED events received.
        errors_received: Number of ERROR events received.
        reconnect_count: Number of reconnection attempts.
        ping_count: Number of pings sent.
        pong_count: Number of pongs received.
    """

    chunks_sent: int = 0
    bytes_sent: int = 0
    events_received: int = 0
    partials_received: int = 0
    finalized_received: int = 0
    errors_received: int = 0
    reconnect_count: int = 0
    ping_count: int = 0
    pong_count: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "chunks_sent": self.chunks_sent,
            "bytes_sent": self.bytes_sent,
            "events_received": self.events_received,
            "partials_received": self.partials_received,
            "finalized_received": self.finalized_received,
            "errors_received": self.errors_received,
            "reconnect_count": self.reconnect_count,
            "ping_count": self.ping_count,
            "pong_count": self.pong_count,
        }


# =============================================================================
# Streaming Client
# =============================================================================


class StreamingClient:
    """High-level client for WebSocket streaming transcription.

    Supports both callback-based and async iteration patterns.

    Example with callbacks:
        def on_finalized(event):
            print(f"Segment: {event.payload['segment']['text']}")

        client = StreamingClient(on_finalized=on_finalized)
        async with client:
            await client.start_session(max_gap_sec=1.0)
            await client.send_audio(audio_bytes)
            await client.end_session()

    Example with async iteration:
        async with StreamingClient() as client:
            await client.start_session()

            # Send audio in background
            async def send_audio():
                for chunk in audio_chunks:
                    await client.send_audio(chunk)
                await client.end_session()
            asyncio.create_task(send_audio())

            # Receive events
            async for event in client.events():
                if event.type == "FINALIZED":
                    print(event.payload["segment"]["text"])
    """

    def __init__(
        self,
        config: StreamingConfig | None = None,
        on_partial: Callable[[StreamEvent], None] | None = None,
        on_finalized: Callable[[StreamEvent], None] | None = None,
        on_error: Callable[[StreamEvent], None] | None = None,
        on_session_started: Callable[[StreamEvent], None] | None = None,
        on_session_ended: Callable[[StreamEvent], None] | None = None,
    ) -> None:
        """Initialize the streaming client.

        Args:
            config: Client configuration. Defaults to StreamingConfig().
            on_partial: Callback for PARTIAL events.
            on_finalized: Callback for FINALIZED events.
            on_error: Callback for ERROR events.
            on_session_started: Callback for SESSION_STARTED event.
            on_session_ended: Callback for SESSION_ENDED event.
        """
        self.config = config or StreamingConfig()
        self._on_partial = on_partial
        self._on_finalized = on_finalized
        self._on_error = on_error
        self._on_session_started = on_session_started
        self._on_session_ended = on_session_ended

        # Connection state
        self._state = ClientState.DISCONNECTED
        self._websocket: Any = None  # websockets.WebSocketClientProtocol
        self._stream_id: str | None = None

        # Sequencing
        self._sequence_counter = 0

        # Event ID tracking for resume support
        self._last_event_id: int = 0

        # Event queue for async iteration
        self._event_queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

        # Statistics
        self.stats = ClientStats()

        # Background tasks
        self._receive_task: asyncio.Task | None = None
        self._ping_task: asyncio.Task | None = None

        # Synchronization
        self._close_event = asyncio.Event()

        logger.debug("StreamingClient initialized with config: %s", self.config)

    @property
    def state(self) -> ClientState:
        """Get current client state."""
        return self._state

    @property
    def stream_id(self) -> str | None:
        """Get current stream ID (available after session start)."""
        return self._stream_id

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._state in (ClientState.CONNECTED, ClientState.SESSION_ACTIVE)

    @property
    def has_active_session(self) -> bool:
        """Check if there is an active session."""
        return self._state == ClientState.SESSION_ACTIVE

    @property
    def last_event_id(self) -> int:
        """Get the last received event ID.

        This can be used for resume after reconnection.
        """
        return self._last_event_id

    async def connect(self) -> None:
        """Establish WebSocket connection.

        Raises:
            ConnectionError: If connection fails.
            RuntimeError: If already connected.
        """
        if self._state not in (ClientState.DISCONNECTED, ClientState.CLOSED):
            raise RuntimeError(f"Cannot connect in state {self._state}")

        self._state = ClientState.CONNECTING

        try:
            # Import websockets here to allow the module to be imported
            # even if websockets is not installed
            try:
                import websockets
            except ImportError as e:
                raise ConnectionError(
                    "websockets package is required for StreamingClient. "
                    "Install it with: pip install websockets"
                ) from e

            logger.info("Connecting to %s", self.config.url)
            self._websocket = await websockets.connect(self.config.url)
            self._state = ClientState.CONNECTED
            self._close_event.clear()

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())

            logger.info("Connected to %s", self.config.url)

        except Exception as e:
            self._state = ClientState.DISCONNECTED
            logger.error("Connection failed: %s", e)
            raise ConnectionError(f"Failed to connect to {self.config.url}: {e}") from e

    async def start_session(self, **config_overrides: Any) -> StreamEvent:
        """Start a streaming session.

        Args:
            **config_overrides: Override config values for this session.
                Supported: max_gap_sec, enable_prosody, enable_emotion,
                enable_categorical_emotion, sample_rate, audio_format.

        Returns:
            SESSION_STARTED event.

        Raises:
            SessionError: If session start fails.
            RuntimeError: If not connected or session already active.
        """
        if self._state != ClientState.CONNECTED:
            raise RuntimeError(f"Cannot start session in state {self._state}")

        # Build session config
        session_config = {
            "max_gap_sec": config_overrides.get("max_gap_sec", self.config.max_gap_sec),
            "enable_prosody": config_overrides.get("enable_prosody", self.config.enable_prosody),
            "enable_emotion": config_overrides.get("enable_emotion", self.config.enable_emotion),
            "enable_categorical_emotion": config_overrides.get(
                "enable_categorical_emotion", self.config.enable_categorical_emotion
            ),
            "sample_rate": config_overrides.get("sample_rate", self.config.sample_rate),
            "audio_format": config_overrides.get("audio_format", self.config.audio_format),
        }

        # Send START_SESSION
        message = {
            "type": "START_SESSION",
            "config": session_config,
        }
        await self._send_message(message)
        logger.info("Sent START_SESSION with config: %s", session_config)

        # Wait for SESSION_STARTED response
        event = await self._wait_for_event_type("SESSION_STARTED", timeout=10.0)
        if event is None:
            raise SessionError("Timeout waiting for SESSION_STARTED")

        self._state = ClientState.SESSION_ACTIVE
        self._stream_id = event.stream_id
        self._sequence_counter = 0

        # Start ping task if configured
        if self.config.ping_interval > 0:
            self._ping_task = asyncio.create_task(self._ping_loop())

        logger.info("Session started: stream_id=%s", self._stream_id)

        # Invoke callback
        if self._on_session_started:
            try:
                self._on_session_started(event)
            except Exception as e:
                logger.warning("on_session_started callback raised: %s", e)

        return event

    async def resume_session(
        self,
        session_id: str,
        last_event_id: int | None = None,
    ) -> list[StreamEvent]:
        """Resume a session after reconnection.

        Attempts to reconnect to an existing session and replay any missed
        events since the specified last_event_id. If the server cannot provide
        the missed events (RESUME_GAP error), raises SessionError.

        Args:
            session_id: The stream_id of the session to resume.
            last_event_id: Last event ID received. If None, uses self.last_event_id.

        Returns:
            List of replayed events (may be empty if no events were missed).

        Raises:
            SessionError: If resume fails (RESUME_GAP or session mismatch).
            RuntimeError: If not connected.
        """
        if self._state != ClientState.CONNECTED:
            raise RuntimeError(f"Cannot resume session in state {self._state}")

        event_id = last_event_id if last_event_id is not None else self._last_event_id

        # Send RESUME_SESSION
        message = {
            "type": "RESUME_SESSION",
            "session_id": session_id,
            "last_event_id": event_id,
        }
        await self._send_message(message)
        logger.info(
            "Sent RESUME_SESSION: session_id=%s, last_event_id=%d",
            session_id,
            event_id,
        )

        # Collect replayed events until we get a non-replay event or error
        # The server replays events and then continues normal operation
        replayed_events: list[StreamEvent] = []

        # Give a short window for replayed events
        timeout = 5.0
        while True:
            event = await self._wait_for_event(timeout=timeout)
            if event is None:
                # No more events to replay within timeout
                break

            # Check for errors
            if event.is_error:
                error_code = event.payload.get("code", "")
                error_message = event.payload.get("message", "Unknown error")

                if error_code == "RESUME_GAP":
                    details = event.payload.get("details", {})
                    raise ResumeGapError(
                        f"Cannot resume session: {error_message}. Client must restart session.",
                        requested_event_id=details.get("requested_event_id"),
                        oldest_available=details.get("oldest_available"),
                    )
                elif error_code == "session_mismatch":
                    raise SessionError(f"Session mismatch: {error_message}")
                elif error_code == "no_session":
                    raise SessionError(f"No session to resume: {error_message}")
                elif not event.is_recoverable:
                    raise SessionError(f"Non-recoverable error during resume: {error_message}")
                else:
                    # Recoverable error, log and continue
                    logger.warning("Recoverable error during resume: %s", error_message)
                    continue

            replayed_events.append(event)

            # If this is a SESSION_STARTED, we're back in session
            if event.type == EventType.SESSION_STARTED.value:
                self._state = ClientState.SESSION_ACTIVE
                self._stream_id = event.stream_id
                break

            # Use shorter timeout for subsequent events during replay
            timeout = 1.0

        # Restore session state if we got replayed events
        if replayed_events and self._state != ClientState.SESSION_ACTIVE:
            self._state = ClientState.SESSION_ACTIVE
            self._stream_id = session_id

        # Start ping task if configured and we have an active session
        if self._state == ClientState.SESSION_ACTIVE and self.config.ping_interval > 0:
            if self._ping_task is None:
                self._ping_task = asyncio.create_task(self._ping_loop())

        logger.info(
            "Resume completed: session_id=%s, replayed=%d events",
            session_id,
            len(replayed_events),
        )

        return replayed_events

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Send an audio chunk.

        Handles sequencing automatically. Audio should be in the format
        specified in config (default: PCM 16-bit little-endian, 16kHz mono).

        Args:
            audio_bytes: Raw audio bytes.

        Raises:
            RuntimeError: If no active session.
            SessionError: If send fails.
        """
        if self._state != ClientState.SESSION_ACTIVE:
            raise RuntimeError(f"Cannot send audio in state {self._state}")

        self._sequence_counter += 1
        sequence = self._sequence_counter

        # Encode audio as base64
        data_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        message = {
            "type": "AUDIO_CHUNK",
            "data": data_b64,
            "sequence": sequence,
        }

        await self._send_message(message)

        # Update stats
        self.stats.chunks_sent += 1
        self.stats.bytes_sent += len(audio_bytes)

        logger.debug("Sent audio chunk: sequence=%d, bytes=%d", sequence, len(audio_bytes))

    async def end_session(self) -> list[StreamEvent]:
        """End session and return final events.

        Returns:
            List of final events (FINALIZED segments and SESSION_ENDED).

        Raises:
            RuntimeError: If no active session.
            SessionError: If end fails.
        """
        if self._state != ClientState.SESSION_ACTIVE:
            raise RuntimeError(f"Cannot end session in state {self._state}")

        self._state = ClientState.SESSION_ENDING

        # Stop ping task
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
            self._ping_task = None

        # Send END_SESSION
        message = {"type": "END_SESSION"}
        await self._send_message(message)
        logger.info("Sent END_SESSION")

        # Collect final events until SESSION_ENDED
        final_events: list[StreamEvent] = []
        while True:
            event = await self._wait_for_event(timeout=30.0)
            if event is None:
                raise SessionError("Timeout waiting for SESSION_ENDED")

            final_events.append(event)

            if event.type == EventType.SESSION_ENDED.value:
                break

        self._state = ClientState.CONNECTED
        logger.info("Session ended: collected %d final events", len(final_events))

        return final_events

    async def ping(self) -> StreamEvent:
        """Send ping and return pong event.

        Returns:
            PONG event with server timestamp.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected:
            raise RuntimeError(f"Cannot ping in state {self._state}")

        import time

        timestamp = int(time.time() * 1000)
        message = {
            "type": "PING",
            "timestamp": timestamp,
        }

        await self._send_message(message)
        self.stats.ping_count += 1

        # Wait for PONG
        event = await self._wait_for_event_type("PONG", timeout=self.config.ping_timeout)
        if event is None:
            raise SessionError("Timeout waiting for PONG")

        self.stats.pong_count += 1
        return event

    async def events(self) -> AsyncIterator[StreamEvent]:
        """Async iterator over received events.

        Yields events as they are received. The iterator ends when:
        - The session ends (SESSION_ENDED received)
        - The connection is closed
        - A non-recoverable error occurs

        Yields:
            StreamEvent objects.
        """
        while not self._close_event.is_set():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
                if event is None:
                    # Sentinel value indicating end of events
                    break
                yield event

                # Stop iteration on SESSION_ENDED
                if event.type == EventType.SESSION_ENDED.value:
                    break

                # Stop iteration on non-recoverable error
                if event.is_error and not event.is_recoverable:
                    break

            except TimeoutError:
                continue

    async def close(self) -> None:
        """Close the connection.

        Safe to call multiple times or in any state.
        """
        if self._state == ClientState.CLOSED:
            return

        logger.info("Closing connection")
        self._close_event.set()

        # Cancel background tasks
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
            self._ping_task = None

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        # Close websocket
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning("Error closing websocket: %s", e)
            self._websocket = None

        # Signal end of events
        await self._event_queue.put(None)

        self._state = ClientState.CLOSED
        logger.info("Connection closed")

    async def __aenter__(self) -> StreamingClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _send_message(self, message: dict[str, Any]) -> None:
        """Send a JSON message over the websocket."""
        if self._websocket is None:
            raise RuntimeError("Not connected")

        try:
            await self._websocket.send(json.dumps(message))
        except Exception as e:
            logger.error("Failed to send message: %s", e)
            raise SessionError(f"Failed to send message: {e}") from e

    async def _receive_loop(self) -> None:
        """Background task that receives messages from websocket."""
        try:
            async for message in self._websocket:
                if self._close_event.is_set():
                    break

                try:
                    data = json.loads(message)
                    event = StreamEvent.from_dict(data)
                    await self._handle_event(event)
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse message: %s", e)
                except ValueError as e:
                    logger.warning("Invalid event data: %s", e)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._close_event.is_set():
                logger.error("Receive loop error: %s", e)
                # Try to reconnect
                await self._handle_disconnect()

    async def _handle_event(self, event: StreamEvent) -> None:
        """Process a received event."""
        self.stats.events_received += 1

        # Track last event ID for resume capability
        self._last_event_id = event.event_id

        # Update type-specific stats
        if event.type == EventType.PARTIAL.value:
            self.stats.partials_received += 1
        elif event.type == EventType.FINALIZED.value:
            self.stats.finalized_received += 1
        elif event.type == EventType.ERROR.value:
            self.stats.errors_received += 1

        # Put in queue for async iteration
        await self._event_queue.put(event)

        # Invoke callbacks
        try:
            if event.type == EventType.PARTIAL.value and self._on_partial:
                self._on_partial(event)
            elif event.type == EventType.FINALIZED.value and self._on_finalized:
                self._on_finalized(event)
            elif event.type == EventType.ERROR.value and self._on_error:
                self._on_error(event)
            elif event.type == EventType.SESSION_ENDED.value and self._on_session_ended:
                self._on_session_ended(event)
        except Exception as e:
            logger.warning("Callback raised exception: %s", e)

        logger.debug(
            "Received event: type=%s, event_id=%d",
            event.type,
            event.event_id,
        )

    async def _wait_for_event(self, timeout: float = 10.0) -> StreamEvent | None:
        """Wait for any event from the queue."""
        try:
            event = await asyncio.wait_for(
                self._event_queue.get(),
                timeout=timeout,
            )
            return event
        except TimeoutError:
            return None

    async def _wait_for_event_type(
        self,
        event_type: str,
        timeout: float = 10.0,
    ) -> StreamEvent | None:
        """Wait for a specific event type."""
        start_time = asyncio.get_event_loop().time()

        while True:
            remaining = timeout - (asyncio.get_event_loop().time() - start_time)
            if remaining <= 0:
                return None

            event = await self._wait_for_event(timeout=remaining)
            if event is None:
                return None

            if event.type == event_type:
                return event

            # Check for error
            if event.is_error:
                if not event.is_recoverable:
                    raise SessionError(
                        f"Non-recoverable error: {event.payload.get('message', 'Unknown error')}"
                    )
                logger.warning("Received error while waiting for %s: %s", event_type, event.payload)

    async def _ping_loop(self) -> None:
        """Background task that sends periodic pings."""
        try:
            while not self._close_event.is_set():
                await asyncio.sleep(self.config.ping_interval)
                if self._state == ClientState.SESSION_ACTIVE:
                    try:
                        await self.ping()
                    except Exception as e:
                        logger.warning("Ping failed: %s", e)
        except asyncio.CancelledError:
            raise

    async def _handle_disconnect(self) -> None:
        """Handle unexpected disconnection with reconnection logic."""
        if self._close_event.is_set():
            return

        logger.warning("Handling disconnect")

        for attempt in range(self.config.reconnect_attempts):
            self.stats.reconnect_count += 1
            logger.info("Reconnection attempt %d/%d", attempt + 1, self.config.reconnect_attempts)

            await asyncio.sleep(self.config.reconnect_delay)

            try:
                self._state = ClientState.DISCONNECTED
                await self.connect()

                # Note: Session state is lost on reconnect.
                # The caller needs to restart the session if needed.
                logger.info("Reconnected successfully")
                return

            except Exception as e:
                logger.warning("Reconnection attempt %d failed: %s", attempt + 1, e)

        logger.error("All reconnection attempts failed")
        self._state = ClientState.CLOSED
        await self._event_queue.put(None)


# =============================================================================
# Helper Functions
# =============================================================================


def create_client(
    url: str = "ws://localhost:8000/stream",
    **kwargs: Any,
) -> StreamingClient:
    """Create a StreamingClient with common defaults.

    This is a convenience function for creating a client with
    a URL and optional configuration overrides.

    Args:
        url: WebSocket URL.
        **kwargs: Configuration overrides (see StreamingConfig).

    Returns:
        StreamingClient instance.
    """
    config = StreamingConfig(url=url, **kwargs)
    return StreamingClient(config=config)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Main classes
    "StreamingClient",
    "StreamingConfig",
    "StreamEvent",
    "ClientStats",
    # Enums
    "EventType",
    "ClientState",
    # Exceptions
    "StreamingClientError",
    "ConnectionError",
    "SessionError",
    "ProtocolError",
    "ResumeGapError",
    # Factory function
    "create_client",
]
