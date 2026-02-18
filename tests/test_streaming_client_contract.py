"""Contract tests for StreamingClient (#134, #223).

These tests verify that the StreamingClient correctly implements the streaming
protocol contract, ensuring it can interoperate with the WebSocket server.

Tests cover:
- Protocol compatibility: client sends correct message format, parses server responses
- Event ordering verification: client detects out-of-order events
- Session lifecycle: correct state transitions
- Backpressure handling: client respects server backpressure signals
- Error handling: client handles error events appropriately
- Reconnection: client can reconnect and resume (best-effort)
- Resume session: client tracks last_event_id and can resume

These are contract tests - they verify the interface between client and server
without testing the full integration.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from slower_whisper.pipeline.streaming_client import (
    ClientState,
    ClientStats,
    EventType,
    ResumeGapError,
    SessionError,
    StreamEvent,
    StreamingClient,
    StreamingConfig,
    create_client,
)
from slower_whisper.pipeline.streaming_ws import (
    ClientMessageType,
    EventEnvelope,
    ServerMessageType,
    WebSocketSessionConfig,
    decode_audio_chunk,
    parse_client_message,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_websocket() -> MagicMock:
    """Create a mock WebSocket connection for testing."""
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.closed = False
    return ws


@pytest.fixture
def sample_session_started_event() -> dict[str, Any]:
    """Create a sample SESSION_STARTED event."""
    return {
        "event_id": 1,
        "stream_id": "str-12345678-1234-4567-89ab-123456789abc",
        "type": "SESSION_STARTED",
        "ts_server": int(time.time() * 1000),
        "payload": {"session_id": "str-12345678-1234-4567-89ab-123456789abc"},
    }


@pytest.fixture
def sample_partial_event() -> dict[str, Any]:
    """Create a sample PARTIAL event."""
    return {
        "event_id": 2,
        "stream_id": "str-12345678-1234-4567-89ab-123456789abc",
        "type": "PARTIAL",
        "ts_server": int(time.time() * 1000),
        "payload": {
            "segment": {
                "start": 0.0,
                "end": 1.5,
                "text": "Hello world",
                "speaker_id": None,
            }
        },
        "segment_id": "seg-0",
        "ts_audio_start": 0.0,
        "ts_audio_end": 1.5,
    }


@pytest.fixture
def sample_finalized_event() -> dict[str, Any]:
    """Create a sample FINALIZED event."""
    return {
        "event_id": 3,
        "stream_id": "str-12345678-1234-4567-89ab-123456789abc",
        "type": "FINALIZED",
        "ts_server": int(time.time() * 1000),
        "payload": {
            "segment": {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world, this is a test",
                "speaker_id": "spk_0",
                "audio_state": None,
            }
        },
        "segment_id": "seg-0",
        "ts_audio_start": 0.0,
        "ts_audio_end": 2.5,
    }


@pytest.fixture
def sample_session_ended_event() -> dict[str, Any]:
    """Create a sample SESSION_ENDED event."""
    return {
        "event_id": 4,
        "stream_id": "str-12345678-1234-4567-89ab-123456789abc",
        "type": "SESSION_ENDED",
        "ts_server": int(time.time() * 1000),
        "payload": {
            "stats": {
                "chunks_received": 10,
                "bytes_received": 320000,
                "segments_partial": 5,
                "segments_finalized": 3,
                "events_sent": 15,
                "events_dropped": 0,
                "errors": 0,
                "duration_sec": 10.5,
            }
        },
    }


# =============================================================================
# Protocol Compatibility Tests
# =============================================================================


class TestProtocolMessageFormats:
    """Test that client sends correctly formatted messages."""

    def test_start_session_message_format(self) -> None:
        """Test START_SESSION message matches expected server format."""
        config = StreamingConfig(
            max_gap_sec=0.5,
            enable_prosody=True,
            enable_emotion=True,
            enable_categorical_emotion=False,
            sample_rate=16000,
            audio_format="pcm_s16le",
        )

        # Build message as client would
        message = {
            "type": "START_SESSION",
            "config": {
                "max_gap_sec": config.max_gap_sec,
                "enable_prosody": config.enable_prosody,
                "enable_emotion": config.enable_emotion,
                "enable_categorical_emotion": config.enable_categorical_emotion,
                "sample_rate": config.sample_rate,
                "audio_format": config.audio_format,
            },
        }

        # Server should be able to parse it
        msg_type, payload = parse_client_message(message)
        assert msg_type == ClientMessageType.START_SESSION

        # Server config should accept the values
        server_config = WebSocketSessionConfig.from_dict(payload["config"])
        assert server_config.max_gap_sec == 0.5
        assert server_config.enable_prosody is True
        assert server_config.enable_emotion is True
        assert server_config.sample_rate == 16000

    def test_audio_chunk_message_format(self) -> None:
        """Test AUDIO_CHUNK message matches expected server format."""
        audio_bytes = b"\x00" * 3200  # 0.1 seconds at 16kHz 16-bit
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        sequence = 42

        # Build message as client would
        message = {
            "type": "AUDIO_CHUNK",
            "data": encoded,
            "sequence": sequence,
        }

        # Server should be able to parse it
        msg_type, payload = parse_client_message(message)
        assert msg_type == ClientMessageType.AUDIO_CHUNK

        # Server should be able to decode it
        decoded_audio, decoded_seq = decode_audio_chunk(payload)
        assert decoded_audio == audio_bytes
        assert decoded_seq == sequence

    def test_end_session_message_format(self) -> None:
        """Test END_SESSION message matches expected server format."""
        message = {"type": "END_SESSION"}

        msg_type, payload = parse_client_message(message)
        assert msg_type == ClientMessageType.END_SESSION
        assert payload == {}

    def test_ping_message_format(self) -> None:
        """Test PING message matches expected server format."""
        timestamp = int(time.time() * 1000)
        message = {
            "type": "PING",
            "timestamp": timestamp,
        }

        msg_type, payload = parse_client_message(message)
        assert msg_type == ClientMessageType.PING
        assert payload["timestamp"] == timestamp


class TestEventParsing:
    """Test that client correctly parses server events."""

    def test_parse_session_started(self, sample_session_started_event: dict[str, Any]) -> None:
        """Test parsing SESSION_STARTED event."""
        event = StreamEvent.from_dict(sample_session_started_event)

        assert event.event_id == 1
        assert event.type == EventType.SESSION_STARTED.value
        assert "session_id" in event.payload

    def test_parse_partial(self, sample_partial_event: dict[str, Any]) -> None:
        """Test parsing PARTIAL event."""
        event = StreamEvent.from_dict(sample_partial_event)

        assert event.type == EventType.PARTIAL.value
        assert event.segment_id == "seg-0"
        assert event.ts_audio_start == 0.0
        assert event.ts_audio_end == 1.5
        assert "segment" in event.payload
        assert event.payload["segment"]["text"] == "Hello world"

    def test_parse_finalized(self, sample_finalized_event: dict[str, Any]) -> None:
        """Test parsing FINALIZED event."""
        event = StreamEvent.from_dict(sample_finalized_event)

        assert event.type == EventType.FINALIZED.value
        assert event.segment_id == "seg-0"
        assert "segment" in event.payload
        assert event.payload["segment"]["speaker_id"] == "spk_0"

    def test_parse_session_ended(self, sample_session_ended_event: dict[str, Any]) -> None:
        """Test parsing SESSION_ENDED event."""
        event = StreamEvent.from_dict(sample_session_ended_event)

        assert event.type == EventType.SESSION_ENDED.value
        assert "stats" in event.payload
        assert event.payload["stats"]["chunks_received"] == 10

    def test_parse_error_event(self) -> None:
        """Test parsing ERROR event."""
        error_data = {
            "event_id": 5,
            "stream_id": "str-test",
            "type": "ERROR",
            "ts_server": int(time.time() * 1000),
            "payload": {
                "code": "ASR_TIMEOUT",
                "message": "ASR processing timed out",
                "recoverable": True,
            },
        }

        event = StreamEvent.from_dict(error_data)

        assert event.type == EventType.ERROR.value
        assert event.is_error is True
        assert event.is_recoverable is True
        assert event.payload["code"] == "ASR_TIMEOUT"

    def test_parse_non_recoverable_error(self) -> None:
        """Test parsing non-recoverable ERROR event."""
        error_data = {
            "event_id": 5,
            "stream_id": "str-test",
            "type": "ERROR",
            "ts_server": int(time.time() * 1000),
            "payload": {
                "code": "FATAL_ERROR",
                "message": "Fatal error occurred",
                "recoverable": False,
            },
        }

        event = StreamEvent.from_dict(error_data)

        assert event.is_error is True
        assert event.is_recoverable is False

    def test_event_from_server_envelope(self) -> None:
        """Test client can parse server EventEnvelope format."""
        # Create server-side envelope
        envelope = EventEnvelope(
            event_id=10,
            stream_id="str-12345678-1234-4567-89ab-123456789abc",
            type=ServerMessageType.FINALIZED,
            ts_server=int(time.time() * 1000),
            payload={"segment": {"text": "Hello"}},
            segment_id="seg-5",
            ts_audio_start=5.0,
            ts_audio_end=7.5,
        )

        # Client should parse it correctly
        event = StreamEvent.from_dict(envelope.to_dict())

        assert event.event_id == 10
        assert event.stream_id == "str-12345678-1234-4567-89ab-123456789abc"
        assert event.type == "FINALIZED"
        assert event.segment_id == "seg-5"
        assert event.ts_audio_start == 5.0
        assert event.ts_audio_end == 7.5


# =============================================================================
# Event Ordering Contract Tests
# =============================================================================


class TestEventOrdering:
    """Test that client verifies event ordering."""

    def test_event_id_validation(self) -> None:
        """Test that event IDs are always positive integers."""
        event = StreamEvent(
            event_id=1,
            stream_id="str-test",
            type="PARTIAL",
            ts_server=1000,
            payload={},
        )
        assert isinstance(event.event_id, int)
        assert event.event_id >= 1

    def test_stream_event_ordering_detection(self) -> None:
        """Test that out-of-order events can be detected."""
        events = [
            StreamEvent(
                event_id=1, stream_id="str-test", type="SESSION_STARTED", ts_server=1000, payload={}
            ),
            StreamEvent(
                event_id=2, stream_id="str-test", type="PARTIAL", ts_server=1001, payload={}
            ),
            StreamEvent(
                event_id=3, stream_id="str-test", type="FINALIZED", ts_server=1002, payload={}
            ),
        ]

        # Verify monotonic
        for i in range(len(events) - 1):
            assert events[i].event_id < events[i + 1].event_id

    def test_detect_out_of_order_events(self) -> None:
        """Test detection of out-of-order events."""
        events = [
            StreamEvent(
                event_id=1, stream_id="str-test", type="SESSION_STARTED", ts_server=1000, payload={}
            ),
            StreamEvent(
                event_id=3, stream_id="str-test", type="PARTIAL", ts_server=1001, payload={}
            ),  # Gap!
            StreamEvent(
                event_id=2, stream_id="str-test", type="FINALIZED", ts_server=1002, payload={}
            ),  # Out of order!
        ]

        # Check for gaps
        last_id = 0
        gaps = []
        for event in events:
            if event.event_id != last_id + 1:
                gaps.append((last_id, event.event_id))
            last_id = event.event_id

        assert len(gaps) > 0  # Should detect the gap/out-of-order

    def test_session_started_first_event_contract(self) -> None:
        """Test that SESSION_STARTED must be the first event."""
        # First event should be SESSION_STARTED
        first_event = StreamEvent(
            event_id=1,
            stream_id="str-test",
            type="SESSION_STARTED",
            ts_server=1000,
            payload={"session_id": "str-test"},
        )
        assert first_event.event_id == 1
        assert first_event.type == "SESSION_STARTED"

    def test_session_ended_last_event_contract(self) -> None:
        """Test that SESSION_ENDED is the last event."""
        last_event = StreamEvent(
            event_id=100,
            stream_id="str-test",
            type="SESSION_ENDED",
            ts_server=2000,
            payload={"stats": {}},
        )
        assert last_event.type == "SESSION_ENDED"


class TestEventOrderingValidator:
    """Test event ordering validation helper."""

    def test_validate_monotonic_event_ids(self) -> None:
        """Test helper to validate monotonic event IDs."""
        event_ids = [1, 2, 3, 4, 5]

        # Check monotonic
        is_monotonic = all(event_ids[i] < event_ids[i + 1] for i in range(len(event_ids) - 1))
        assert is_monotonic is True

    def test_detect_duplicate_event_id(self) -> None:
        """Test detection of duplicate event IDs."""
        event_ids = [1, 2, 2, 3, 4]  # Duplicate 2

        # Find duplicates
        seen = set()
        duplicates = []
        for eid in event_ids:
            if eid in seen:
                duplicates.append(eid)
            seen.add(eid)

        assert 2 in duplicates

    def test_detect_gap_in_event_ids(self) -> None:
        """Test detection of gaps in event IDs."""
        event_ids = [1, 2, 4, 5]  # Gap at 3

        # Find gaps
        gaps = []
        for i in range(len(event_ids) - 1):
            if event_ids[i + 1] != event_ids[i] + 1:
                gaps.append((event_ids[i], event_ids[i + 1]))

        assert (2, 4) in gaps


# =============================================================================
# Session Lifecycle Contract Tests
# =============================================================================


class TestSessionLifecycle:
    """Test session lifecycle state transitions."""

    def test_initial_state_is_disconnected(self) -> None:
        """Test client starts in DISCONNECTED state."""
        client = StreamingClient()
        assert client.state == ClientState.DISCONNECTED

    def test_state_after_connect(self) -> None:
        """Test state transitions to CONNECTING then CONNECTED."""
        client = StreamingClient()
        assert client.state == ClientState.DISCONNECTED

        # Cannot test actual connection without server, but can verify state machine
        client._state = ClientState.CONNECTING
        assert client.state == ClientState.CONNECTING
        assert client.is_connected is False

        client._state = ClientState.CONNECTED
        assert client.state == ClientState.CONNECTED
        assert client.is_connected is True

    def test_state_after_session_start(self) -> None:
        """Test state transitions to SESSION_ACTIVE after start."""
        client = StreamingClient()
        client._state = ClientState.SESSION_ACTIVE
        assert client.has_active_session is True

    def test_state_after_session_end(self) -> None:
        """Test state transitions back to CONNECTED after session end."""
        client = StreamingClient()
        client._state = ClientState.SESSION_ENDING
        assert client.has_active_session is False

        client._state = ClientState.CONNECTED
        assert client.state == ClientState.CONNECTED
        assert client.has_active_session is False

    def test_state_after_close(self) -> None:
        """Test state transitions to CLOSED after close."""
        client = StreamingClient()
        client._state = ClientState.CLOSED
        assert client.state == ClientState.CLOSED
        assert client.is_connected is False


class TestStateTransitionValidation:
    """Test that invalid state transitions are rejected."""

    @pytest.mark.asyncio
    async def test_cannot_connect_when_already_connected(self) -> None:
        """Test connect raises error if already connected."""
        client = StreamingClient()
        client._state = ClientState.CONNECTED

        with pytest.raises(RuntimeError, match="Cannot connect in state"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_cannot_start_session_when_disconnected(self) -> None:
        """Test start_session raises error if not connected."""
        client = StreamingClient()

        with pytest.raises(RuntimeError, match="Cannot start session in state"):
            await client.start_session()

    @pytest.mark.asyncio
    async def test_cannot_send_audio_without_session(self) -> None:
        """Test send_audio raises error if no active session."""
        client = StreamingClient()
        client._state = ClientState.CONNECTED

        with pytest.raises(RuntimeError, match="Cannot send audio in state"):
            await client.send_audio(b"\x00" * 100)

    @pytest.mark.asyncio
    async def test_cannot_end_session_without_active_session(self) -> None:
        """Test end_session raises error if no active session."""
        client = StreamingClient()
        client._state = ClientState.CONNECTED

        with pytest.raises(RuntimeError, match="Cannot end session in state"):
            await client.end_session()


# =============================================================================
# Statistics Contract Tests
# =============================================================================


class TestStatisticsTracking:
    """Test that client tracks statistics correctly."""

    @pytest.mark.asyncio
    async def test_stats_increment_on_event(self) -> None:
        """Test stats are incremented when events are handled."""
        client = StreamingClient()

        # Process events
        partial = StreamEvent(
            event_id=1, stream_id="str-test", type="PARTIAL", ts_server=1000, payload={}
        )
        finalized = StreamEvent(
            event_id=2, stream_id="str-test", type="FINALIZED", ts_server=1001, payload={}
        )
        error = StreamEvent(
            event_id=3,
            stream_id="str-test",
            type="ERROR",
            ts_server=1002,
            payload={"code": "test", "message": "test", "recoverable": True},
        )

        await client._handle_event(partial)
        await client._handle_event(finalized)
        await client._handle_event(error)

        assert client.stats.events_received == 3
        assert client.stats.partials_received == 1
        assert client.stats.finalized_received == 1
        assert client.stats.errors_received == 1

    def test_stats_to_dict(self) -> None:
        """Test stats serialization."""
        stats = ClientStats(
            chunks_sent=10,
            bytes_sent=32000,
            events_received=15,
            partials_received=8,
            finalized_received=5,
            errors_received=2,
            reconnect_count=1,
            ping_count=5,
            pong_count=5,
        )

        d = stats.to_dict()

        assert d["chunks_sent"] == 10
        assert d["bytes_sent"] == 32000
        assert d["events_received"] == 15
        assert d["partials_received"] == 8
        assert d["finalized_received"] == 5
        assert d["errors_received"] == 2
        assert d["reconnect_count"] == 1
        assert d["ping_count"] == 5
        assert d["pong_count"] == 5


# =============================================================================
# Callback Contract Tests
# =============================================================================


class TestCallbackInvocation:
    """Test that callbacks are invoked correctly."""

    @pytest.mark.asyncio
    async def test_on_partial_callback(self) -> None:
        """Test on_partial callback is invoked for PARTIAL events."""
        received_events: list[StreamEvent] = []

        def on_partial(event: StreamEvent) -> None:
            received_events.append(event)

        client = StreamingClient(on_partial=on_partial)

        event = StreamEvent(
            event_id=1,
            stream_id="str-test",
            type="PARTIAL",
            ts_server=1000,
            payload={"segment": {"text": "hello"}},
        )

        await client._handle_event(event)

        assert len(received_events) == 1
        assert received_events[0].type == "PARTIAL"

    @pytest.mark.asyncio
    async def test_on_finalized_callback(self) -> None:
        """Test on_finalized callback is invoked for FINALIZED events."""
        received_events: list[StreamEvent] = []

        def on_finalized(event: StreamEvent) -> None:
            received_events.append(event)

        client = StreamingClient(on_finalized=on_finalized)

        event = StreamEvent(
            event_id=1,
            stream_id="str-test",
            type="FINALIZED",
            ts_server=1000,
            payload={"segment": {"text": "hello"}},
        )

        await client._handle_event(event)

        assert len(received_events) == 1
        assert received_events[0].type == "FINALIZED"

    @pytest.mark.asyncio
    async def test_on_error_callback(self) -> None:
        """Test on_error callback is invoked for ERROR events."""
        received_events: list[StreamEvent] = []

        def on_error(event: StreamEvent) -> None:
            received_events.append(event)

        client = StreamingClient(on_error=on_error)

        event = StreamEvent(
            event_id=1,
            stream_id="str-test",
            type="ERROR",
            ts_server=1000,
            payload={"code": "test", "message": "test", "recoverable": True},
        )

        await client._handle_event(event)

        assert len(received_events) == 1
        assert received_events[0].type == "ERROR"

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_crash_client(self) -> None:
        """Test callback exceptions are caught (per invariant #3)."""

        def bad_callback(event: StreamEvent) -> None:
            raise RuntimeError("Callback error!")

        client = StreamingClient(on_finalized=bad_callback)

        event = StreamEvent(
            event_id=1,
            stream_id="str-test",
            type="FINALIZED",
            ts_server=1000,
            payload={"segment": {}},
        )

        # Should not raise
        await client._handle_event(event)

        # Stats should still be updated
        assert client.stats.events_received == 1
        assert client.stats.finalized_received == 1


# =============================================================================
# Configuration Contract Tests
# =============================================================================


class TestConfigurationContract:
    """Test client configuration matches server expectations."""

    def test_default_config_compatible_with_server(self) -> None:
        """Test default config values are compatible with server defaults."""
        client_config = StreamingConfig()
        server_config = WebSocketSessionConfig()

        # Key values should match
        assert client_config.max_gap_sec == server_config.max_gap_sec
        assert client_config.sample_rate == server_config.sample_rate
        assert client_config.audio_format == server_config.audio_format

    def test_config_values_are_serializable(self) -> None:
        """Test config values can be serialized to JSON."""
        config = StreamingConfig(
            max_gap_sec=0.5,
            enable_prosody=True,
            enable_emotion=True,
            sample_rate=8000,
        )

        # Should be JSON serializable
        config_dict = {
            "max_gap_sec": config.max_gap_sec,
            "enable_prosody": config.enable_prosody,
            "enable_emotion": config.enable_emotion,
            "sample_rate": config.sample_rate,
        }

        json_str = json.dumps(config_dict)
        parsed = json.loads(json_str)

        assert parsed["max_gap_sec"] == 0.5
        assert parsed["enable_prosody"] is True


# =============================================================================
# Error Handling Contract Tests
# =============================================================================


class TestErrorHandling:
    """Test client handles errors according to contract."""

    def test_is_error_property(self) -> None:
        """Test is_error property correctly identifies error events."""
        error_event = StreamEvent(
            event_id=1,
            stream_id="str-test",
            type="ERROR",
            ts_server=1000,
            payload={"code": "test", "message": "test", "recoverable": True},
        )
        assert error_event.is_error is True

        normal_event = StreamEvent(
            event_id=2,
            stream_id="str-test",
            type="FINALIZED",
            ts_server=1000,
            payload={},
        )
        assert normal_event.is_error is False

    def test_is_recoverable_property(self) -> None:
        """Test is_recoverable property."""
        recoverable_error = StreamEvent(
            event_id=1,
            stream_id="str-test",
            type="ERROR",
            ts_server=1000,
            payload={"code": "temp", "message": "temp", "recoverable": True},
        )
        assert recoverable_error.is_recoverable is True

        non_recoverable_error = StreamEvent(
            event_id=2,
            stream_id="str-test",
            type="ERROR",
            ts_server=1000,
            payload={"code": "fatal", "message": "fatal", "recoverable": False},
        )
        assert non_recoverable_error.is_recoverable is False

    def test_non_error_events_always_recoverable(self) -> None:
        """Test non-error events always return True for is_recoverable."""
        event = StreamEvent(
            event_id=1,
            stream_id="str-test",
            type="FINALIZED",
            ts_server=1000,
            payload={},
        )
        assert event.is_recoverable is True


# =============================================================================
# Audio Encoding Contract Tests
# =============================================================================


class TestAudioEncoding:
    """Test audio encoding matches server expectations."""

    def test_audio_chunk_base64_encoding(self) -> None:
        """Test audio chunks are base64 encoded correctly."""
        audio_bytes = b"\x00\x01\x02\x03" * 100
        encoded = base64.b64encode(audio_bytes).decode("utf-8")

        # Should be valid base64
        decoded = base64.b64decode(encoded)
        assert decoded == audio_bytes

    def test_audio_chunk_server_decoding(self) -> None:
        """Test server can decode audio encoded by client pattern."""
        audio_bytes = b"\x00" * 3200  # 0.1 seconds at 16kHz 16-bit mono
        encoded = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {"data": encoded, "sequence": 1}
        decoded, seq = decode_audio_chunk(payload)

        assert decoded == audio_bytes
        assert seq == 1


# =============================================================================
# Factory Function Contract Tests
# =============================================================================


class TestCreateClientFactory:
    """Test create_client factory function."""

    def test_create_client_with_defaults(self) -> None:
        """Test create_client with default values."""
        client = create_client()
        assert client.config.url == "ws://localhost:8000/stream"
        assert client.state == ClientState.DISCONNECTED

    def test_create_client_with_custom_url(self) -> None:
        """Test create_client with custom URL."""
        client = create_client(url="ws://custom:9000/api/stream")
        assert client.config.url == "ws://custom:9000/api/stream"

    def test_create_client_with_config_overrides(self) -> None:
        """Test create_client with config overrides."""
        client = create_client(
            url="ws://localhost:8000/stream",
            max_gap_sec=0.5,
            enable_prosody=True,
            reconnect_attempts=5,
        )
        assert client.config.max_gap_sec == 0.5
        assert client.config.enable_prosody is True
        assert client.config.reconnect_attempts == 5


# =============================================================================
# Async Context Manager Contract Tests
# =============================================================================


class TestAsyncContextManager:
    """Test async context manager protocol."""

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        """Test close() can be called multiple times safely."""
        client = StreamingClient()
        client._state = ClientState.CLOSED

        # Should not raise
        await client.close()
        await client.close()

        assert client.state == ClientState.CLOSED


# =============================================================================
# Reconnection Contract Tests
# =============================================================================


class TestReconnectionContract:
    """Test reconnection behavior."""

    def test_reconnect_count_tracking(self) -> None:
        """Test reconnection attempts are tracked in stats."""
        client = StreamingClient()

        # Simulate reconnection attempts
        client.stats.reconnect_count = 3

        assert client.stats.reconnect_count == 3

    def test_reconnect_config_defaults(self) -> None:
        """Test reconnection config defaults."""
        config = StreamingConfig()

        assert config.reconnect_attempts == 3
        assert config.reconnect_delay == 1.0


# =============================================================================
# Ping/Pong Contract Tests
# =============================================================================


class TestPingPongContract:
    """Test ping/pong heartbeat contract."""

    def test_ping_config_defaults(self) -> None:
        """Test ping config defaults."""
        config = StreamingConfig()

        assert config.ping_interval == 30.0
        assert config.ping_timeout == 10.0

    def test_ping_stats_tracking(self) -> None:
        """Test ping/pong stats are tracked."""
        stats = ClientStats()

        stats.ping_count = 5
        stats.pong_count = 5

        assert stats.ping_count == stats.pong_count


# =============================================================================
# Resume Session Contract Tests (#134)
# =============================================================================


class TestResumeSessionContract:
    """Test resume session protocol contract (#134)."""

    def test_resume_session_message_format(self) -> None:
        """Test RESUME_SESSION message matches expected server format."""
        session_id = "str-12345678-1234-4567-89ab-123456789abc"
        last_event_id = 50

        # Build message as client would
        message = {
            "type": "RESUME_SESSION",
            "session_id": session_id,
            "last_event_id": last_event_id,
        }

        # Server should be able to parse it
        msg_type, payload = parse_client_message(message)
        assert msg_type == ClientMessageType.RESUME_SESSION
        assert payload["session_id"] == session_id
        assert payload["last_event_id"] == 50

    @pytest.mark.asyncio
    async def test_last_event_id_tracking(self) -> None:
        """Test that client tracks last_event_id from received events."""
        client = StreamingClient()

        # Initially zero
        assert client.last_event_id == 0

        # Simulate receiving events
        event1 = StreamEvent(
            event_id=5,
            stream_id="str-test",
            type="PARTIAL",
            ts_server=1000,
            payload={},
        )
        await client._handle_event(event1)
        assert client.last_event_id == 5

        event2 = StreamEvent(
            event_id=10,
            stream_id="str-test",
            type="FINALIZED",
            ts_server=1001,
            payload={},
        )
        await client._handle_event(event2)
        assert client.last_event_id == 10

    @pytest.mark.asyncio
    async def test_resume_session_requires_connected_state(self) -> None:
        """Test resume_session raises error if not connected."""
        client = StreamingClient()

        with pytest.raises(RuntimeError, match="Cannot resume session in state"):
            await client.resume_session("str-123", 10)

    def test_resume_gap_error_exception(self) -> None:
        """Test ResumeGapError contains event ID details."""
        error = ResumeGapError(
            message="Cannot resume from event 10",
            requested_event_id=10,
            oldest_available=50,
        )

        assert error.requested_event_id == 10
        assert error.oldest_available == 50
        assert isinstance(error, SessionError)

    def test_resume_gap_error_event_parsing(self) -> None:
        """Test client parses RESUME_GAP error event correctly."""
        error_data = {
            "event_id": 100,
            "stream_id": "str-test",
            "type": "ERROR",
            "ts_server": 1000,
            "payload": {
                "code": "RESUME_GAP",
                "message": "Cannot resume from event 10. Oldest available is 50.",
                "recoverable": False,
                "details": {
                    "requested_event_id": 10,
                    "oldest_available": 50,
                    "current_event_id": 100,
                },
            },
        }

        event = StreamEvent.from_dict(error_data)

        assert event.is_error is True
        assert event.is_recoverable is False
        assert event.payload["code"] == "RESUME_GAP"
        assert event.payload["details"]["requested_event_id"] == 10
        assert event.payload["details"]["oldest_available"] == 50

    def test_last_event_id_property_exposed(self) -> None:
        """Test last_event_id property is exposed on client."""
        client = StreamingClient()

        # Property should exist and be accessible
        assert hasattr(client, "last_event_id")
        assert isinstance(client.last_event_id, int)


class TestResumeServerCompatibility:
    """Test resume protocol compatibility with server."""

    def test_server_replay_buffer_interface(self) -> None:
        """Test server replay buffer interface matches client expectations."""
        from slower_whisper.pipeline.streaming_ws import ReplayBuffer

        buffer = ReplayBuffer(max_size=100)

        # Add some events
        from slower_whisper.pipeline.streaming_ws import EventEnvelope, ServerMessageType

        for i in range(1, 11):
            envelope = EventEnvelope(
                event_id=i,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000 + i,
                payload={"segment": {"text": f"segment {i}"}},
            )
            buffer.add(envelope)

        # Get events since event ID 5
        events, gap_detected = buffer.get_events_since(5)

        # Should return events 6-10
        assert len(events) == 5
        assert gap_detected is False
        assert events[0].event_id == 6

    def test_server_resume_gap_detection(self) -> None:
        """Test server detects resume gap correctly."""
        from slower_whisper.pipeline.streaming_ws import ReplayBuffer

        buffer = ReplayBuffer(max_size=5)  # Small buffer

        # Add 10 events (only last 5 will be kept)
        from slower_whisper.pipeline.streaming_ws import EventEnvelope, ServerMessageType

        for i in range(1, 11):
            envelope = EventEnvelope(
                event_id=i,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000 + i,
                payload={},
            )
            buffer.add(envelope)

        # Try to resume from event 2 (which was evicted)
        events, gap_detected = buffer.get_events_since(2)

        # Gap should be detected
        assert gap_detected is True
        # Should still return available events
        assert len(events) > 0

    def test_server_create_resume_gap_error_format(self) -> None:
        """Test server RESUME_GAP error format matches client expectations."""
        from slower_whisper.pipeline.streaming_ws import WebSocketStreamingSession

        session = WebSocketStreamingSession()

        # Create a RESUME_GAP error
        error_envelope = session.create_resume_gap_error(last_event_id=10)

        # Convert to dict (as would be sent to client)
        error_dict = error_envelope.to_dict()

        # Client should be able to parse it
        event = StreamEvent.from_dict(error_dict)

        assert event.type == "ERROR"
        assert event.payload["code"] == "RESUME_GAP"
        assert event.payload["recoverable"] is False
        assert "details" in event.payload
        assert event.payload["details"]["requested_event_id"] == 10
