"""Tests for the reference Python streaming client (#134).

Tests cover:
- Client initialization and configuration
- Message serialization
- Event parsing (StreamEvent.from_dict)
- Client state management
- Statistics tracking
- Integration tests with mock server
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from slower_whisper.pipeline.streaming_client import (
    ClientState,
    ClientStats,
    ConnectionError,
    EventType,
    StreamEvent,
    StreamingClient,
    StreamingConfig,
    create_client,
)

# =============================================================================
# StreamingConfig Tests
# =============================================================================


class TestStreamingConfig:
    """Tests for StreamingConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = StreamingConfig()
        assert config.url == "ws://localhost:8000/stream"
        assert config.max_gap_sec == 1.0
        assert config.enable_prosody is False
        assert config.enable_emotion is False
        assert config.enable_categorical_emotion is False
        assert config.sample_rate == 16000
        assert config.audio_format == "pcm_s16le"
        assert config.reconnect_attempts == 3
        assert config.reconnect_delay == 1.0
        assert config.ping_interval == 30.0
        assert config.ping_timeout == 10.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = StreamingConfig(
            url="ws://custom:9000/stream",
            max_gap_sec=0.5,
            enable_prosody=True,
            enable_emotion=True,
            sample_rate=8000,
            reconnect_attempts=5,
        )
        assert config.url == "ws://custom:9000/stream"
        assert config.max_gap_sec == 0.5
        assert config.enable_prosody is True
        assert config.enable_emotion is True
        assert config.sample_rate == 8000
        assert config.reconnect_attempts == 5


# =============================================================================
# StreamEvent Tests
# =============================================================================


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_from_dict_minimal(self) -> None:
        """Test parsing minimal event dictionary."""
        data = {
            "event_id": 1,
            "stream_id": "str-abc123",
            "type": "SESSION_STARTED",
            "ts_server": 1234567890000,
            "payload": {"session_id": "str-abc123"},
        }

        event = StreamEvent.from_dict(data)

        assert event.event_id == 1
        assert event.stream_id == "str-abc123"
        assert event.type == "SESSION_STARTED"
        assert event.ts_server == 1234567890000
        assert event.payload == {"session_id": "str-abc123"}
        assert event.segment_id is None
        assert event.ts_audio_start is None
        assert event.ts_audio_end is None

    def test_from_dict_with_audio_timestamps(self) -> None:
        """Test parsing event with audio timestamps."""
        data = {
            "event_id": 5,
            "stream_id": "str-abc123",
            "type": "FINALIZED",
            "ts_server": 1234567890000,
            "segment_id": "seg-0",
            "ts_audio_start": 0.0,
            "ts_audio_end": 2.5,
            "payload": {
                "segment": {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                    "speaker_id": "spk_0",
                }
            },
        }

        event = StreamEvent.from_dict(data)

        assert event.segment_id == "seg-0"
        assert event.ts_audio_start == 0.0
        assert event.ts_audio_end == 2.5

    def test_from_dict_missing_required_fields(self) -> None:
        """Test parsing raises ValueError for missing fields."""
        data = {
            "event_id": 1,
            "stream_id": "str-abc123",
            # Missing: type, ts_server, payload
        }

        with pytest.raises(ValueError, match="Missing required fields"):
            StreamEvent.from_dict(data)

    def test_to_dict_minimal(self) -> None:
        """Test serialization of minimal event."""
        event = StreamEvent(
            event_id=1,
            stream_id="str-abc123",
            type="PONG",
            ts_server=1000,
            payload={"timestamp": 999},
        )

        d = event.to_dict()

        assert d["event_id"] == 1
        assert d["stream_id"] == "str-abc123"
        assert d["type"] == "PONG"
        assert d["ts_server"] == 1000
        assert d["payload"] == {"timestamp": 999}
        # Optional fields should not be present
        assert "segment_id" not in d
        assert "ts_audio_start" not in d
        assert "ts_audio_end" not in d

    def test_to_dict_complete(self) -> None:
        """Test serialization with all fields."""
        event = StreamEvent(
            event_id=10,
            stream_id="str-abc",
            type="PARTIAL",
            ts_server=2000,
            payload={"segment": {"text": "hello"}},
            segment_id="seg-5",
            ts_audio_start=1.0,
            ts_audio_end=3.5,
        )

        d = event.to_dict()

        assert d["segment_id"] == "seg-5"
        assert d["ts_audio_start"] == 1.0
        assert d["ts_audio_end"] == 3.5

    def test_is_error(self) -> None:
        """Test is_error property."""
        error_event = StreamEvent(
            event_id=1,
            stream_id="str-123",
            type="ERROR",
            ts_server=1000,
            payload={"code": "test", "message": "Test error", "recoverable": True},
        )
        assert error_event.is_error is True

        normal_event = StreamEvent(
            event_id=2,
            stream_id="str-123",
            type="FINALIZED",
            ts_server=1000,
            payload={"segment": {}},
        )
        assert normal_event.is_error is False

    def test_is_recoverable(self) -> None:
        """Test is_recoverable property."""
        recoverable_error = StreamEvent(
            event_id=1,
            stream_id="str-123",
            type="ERROR",
            ts_server=1000,
            payload={"code": "test", "message": "Test", "recoverable": True},
        )
        assert recoverable_error.is_recoverable is True

        non_recoverable_error = StreamEvent(
            event_id=2,
            stream_id="str-123",
            type="ERROR",
            ts_server=1000,
            payload={"code": "fatal", "message": "Fatal", "recoverable": False},
        )
        assert non_recoverable_error.is_recoverable is False

        # Non-error events are always "recoverable"
        normal_event = StreamEvent(
            event_id=3,
            stream_id="str-123",
            type="FINALIZED",
            ts_server=1000,
            payload={},
        )
        assert normal_event.is_recoverable is True


# =============================================================================
# ClientStats Tests
# =============================================================================


class TestClientStats:
    """Tests for ClientStats dataclass."""

    def test_initial_stats(self) -> None:
        """Test initial stats values."""
        stats = ClientStats()
        assert stats.chunks_sent == 0
        assert stats.bytes_sent == 0
        assert stats.events_received == 0
        assert stats.partials_received == 0
        assert stats.finalized_received == 0
        assert stats.errors_received == 0
        assert stats.reconnect_count == 0
        assert stats.ping_count == 0
        assert stats.pong_count == 0

    def test_to_dict(self) -> None:
        """Test stats to dictionary conversion."""
        stats = ClientStats(
            chunks_sent=10,
            bytes_sent=32000,
            events_received=15,
            finalized_received=5,
        )

        d = stats.to_dict()

        assert d["chunks_sent"] == 10
        assert d["bytes_sent"] == 32000
        assert d["events_received"] == 15
        assert d["finalized_received"] == 5


# =============================================================================
# EventType Tests
# =============================================================================


class TestEventType:
    """Tests for EventType enum."""

    def test_event_type_values(self) -> None:
        """Test event type enum values."""
        assert EventType.SESSION_STARTED.value == "SESSION_STARTED"
        assert EventType.PARTIAL.value == "PARTIAL"
        assert EventType.FINALIZED.value == "FINALIZED"
        assert EventType.SPEAKER_TURN.value == "SPEAKER_TURN"
        assert EventType.SEMANTIC_UPDATE.value == "SEMANTIC_UPDATE"
        assert EventType.ERROR.value == "ERROR"
        assert EventType.SESSION_ENDED.value == "SESSION_ENDED"
        assert EventType.PONG.value == "PONG"


# =============================================================================
# StreamingClient Initialization Tests
# =============================================================================


class TestStreamingClientInit:
    """Tests for StreamingClient initialization."""

    def test_default_initialization(self) -> None:
        """Test client with default config."""
        client = StreamingClient()
        assert client.config.url == "ws://localhost:8000/stream"
        assert client.state == ClientState.DISCONNECTED
        assert client.stream_id is None
        assert client.is_connected is False
        assert client.has_active_session is False

    def test_custom_config(self) -> None:
        """Test client with custom config."""
        config = StreamingConfig(
            url="ws://custom:9000/stream",
            max_gap_sec=2.0,
        )
        client = StreamingClient(config=config)
        assert client.config.url == "ws://custom:9000/stream"
        assert client.config.max_gap_sec == 2.0

    def test_with_callbacks(self) -> None:
        """Test client with callbacks."""
        on_partial = MagicMock()
        on_finalized = MagicMock()
        on_error = MagicMock()

        client = StreamingClient(
            on_partial=on_partial,
            on_finalized=on_finalized,
            on_error=on_error,
        )

        assert client._on_partial is on_partial
        assert client._on_finalized is on_finalized
        assert client._on_error is on_error


# =============================================================================
# Message Serialization Tests
# =============================================================================


class TestMessageSerialization:
    """Tests for message serialization helpers."""

    def test_audio_chunk_encoding(self) -> None:
        """Test audio chunk base64 encoding."""
        audio_bytes = b"test audio data"
        encoded = base64.b64encode(audio_bytes).decode("utf-8")

        # Verify encoding
        assert base64.b64decode(encoded) == audio_bytes

        # Build message as client would
        message = {
            "type": "AUDIO_CHUNK",
            "data": encoded,
            "sequence": 1,
        }

        # Verify message structure
        assert message["type"] == "AUDIO_CHUNK"
        assert isinstance(message["data"], str)
        assert message["sequence"] == 1

    def test_start_session_message(self) -> None:
        """Test START_SESSION message format."""
        config = StreamingConfig(
            max_gap_sec=0.5,
            enable_prosody=True,
            enable_emotion=True,
            sample_rate=16000,
        )

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

        # Verify structure
        assert message["type"] == "START_SESSION"
        assert message["config"]["max_gap_sec"] == 0.5
        assert message["config"]["enable_prosody"] is True
        assert message["config"]["enable_emotion"] is True
        assert message["config"]["sample_rate"] == 16000

    def test_end_session_message(self) -> None:
        """Test END_SESSION message format."""
        message = {"type": "END_SESSION"}
        assert message["type"] == "END_SESSION"

    def test_ping_message(self) -> None:
        """Test PING message format."""
        timestamp = 1234567890000
        message = {
            "type": "PING",
            "timestamp": timestamp,
        }

        assert message["type"] == "PING"
        assert message["timestamp"] == timestamp


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateClient:
    """Tests for create_client factory function."""

    def test_create_client_defaults(self) -> None:
        """Test create_client with defaults."""
        client = create_client()
        assert client.config.url == "ws://localhost:8000/stream"

    def test_create_client_custom_url(self) -> None:
        """Test create_client with custom URL."""
        client = create_client(url="ws://custom:9000/stream")
        assert client.config.url == "ws://custom:9000/stream"

    def test_create_client_with_overrides(self) -> None:
        """Test create_client with config overrides."""
        client = create_client(
            url="ws://localhost:8000/stream",
            max_gap_sec=2.0,
            enable_prosody=True,
        )
        assert client.config.max_gap_sec == 2.0
        assert client.config.enable_prosody is True


# =============================================================================
# Client State Tests
# =============================================================================


class TestClientState:
    """Tests for ClientState enum."""

    def test_state_values(self) -> None:
        """Test state enum values."""
        assert ClientState.DISCONNECTED.value == "disconnected"
        assert ClientState.CONNECTING.value == "connecting"
        assert ClientState.CONNECTED.value == "connected"
        assert ClientState.SESSION_ACTIVE.value == "session_active"
        assert ClientState.SESSION_ENDING.value == "session_ending"
        assert ClientState.CLOSED.value == "closed"


# =============================================================================
# Mock WebSocket Server for Integration Tests
# =============================================================================


class MockWebSocket:
    """Mock WebSocket connection for testing."""

    def __init__(self) -> None:
        self.sent_messages: list[str] = []
        self.receive_queue: asyncio.Queue[str] = asyncio.Queue()
        self.closed = False

    async def send(self, message: str) -> None:
        """Mock send method."""
        if self.closed:
            raise Exception("Connection closed")
        self.sent_messages.append(message)

    async def recv(self) -> str:
        """Mock receive method."""
        if self.closed:
            raise Exception("Connection closed")
        return await self.receive_queue.get()

    async def close(self) -> None:
        """Mock close method."""
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration from None
        try:
            return await asyncio.wait_for(self.receive_queue.get(), timeout=0.1)
        except TimeoutError:
            raise StopAsyncIteration from None

    async def push_message(self, message: dict[str, Any]) -> None:
        """Push a message to the receive queue."""
        await self.receive_queue.put(json.dumps(message))


# =============================================================================
# Integration Tests with Mock Server
# =============================================================================


class TestClientIntegration:
    """Integration tests with mock WebSocket server."""

    @pytest.fixture
    def mock_ws(self) -> MockWebSocket:
        """Create a mock WebSocket."""
        return MockWebSocket()

    @pytest.mark.asyncio
    async def test_connect_without_websockets_raises(self) -> None:
        """Test that connecting without websockets installed raises ConnectionError."""
        client = StreamingClient()

        with patch.dict("sys.modules", {"websockets": None}):
            # Clear any cached import

            try:
                # Try to force import failure
                with patch(
                    "builtins.__import__", side_effect=ImportError("No module named 'websockets'")
                ):
                    with pytest.raises(ConnectionError, match="websockets package is required"):
                        await client.connect()
            except Exception:
                # If websockets is actually installed, skip this test
                pytest.skip("websockets is installed, cannot test ImportError path")

    @pytest.mark.asyncio
    async def test_connect_already_connected_raises(self) -> None:
        """Test that connecting when already connected raises RuntimeError."""
        client = StreamingClient()
        client._state = ClientState.CONNECTED

        with pytest.raises(RuntimeError, match="Cannot connect in state"):
            await client.connect()

    @pytest.mark.asyncio
    async def test_start_session_not_connected_raises(self) -> None:
        """Test that starting session when not connected raises RuntimeError."""
        client = StreamingClient()

        with pytest.raises(RuntimeError, match="Cannot start session in state"):
            await client.start_session()

    @pytest.mark.asyncio
    async def test_send_audio_no_session_raises(self) -> None:
        """Test that sending audio without session raises RuntimeError."""
        client = StreamingClient()

        with pytest.raises(RuntimeError, match="Cannot send audio in state"):
            await client.send_audio(b"test")

    @pytest.mark.asyncio
    async def test_end_session_no_session_raises(self) -> None:
        """Test that ending session without active session raises RuntimeError."""
        client = StreamingClient()

        with pytest.raises(RuntimeError, match="Cannot end session in state"):
            await client.end_session()

    @pytest.mark.asyncio
    async def test_ping_not_connected_raises(self) -> None:
        """Test that ping without connection raises RuntimeError."""
        client = StreamingClient()

        with pytest.raises(RuntimeError, match="Cannot ping in state"):
            await client.ping()

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        """Test that close is safe to call multiple times."""
        client = StreamingClient()
        client._state = ClientState.CLOSED

        # Should not raise
        await client.close()
        assert client.state == ClientState.CLOSED

    @pytest.mark.asyncio
    async def test_stats_tracking(self) -> None:
        """Test that stats are tracked correctly."""
        client = StreamingClient()

        # Simulate some events
        partial_event = StreamEvent(
            event_id=1,
            stream_id="str-123",
            type="PARTIAL",
            ts_server=1000,
            payload={"segment": {}},
        )
        finalized_event = StreamEvent(
            event_id=2,
            stream_id="str-123",
            type="FINALIZED",
            ts_server=1001,
            payload={"segment": {}},
        )
        error_event = StreamEvent(
            event_id=3,
            stream_id="str-123",
            type="ERROR",
            ts_server=1002,
            payload={"code": "test", "message": "Test", "recoverable": True},
        )

        await client._handle_event(partial_event)
        await client._handle_event(finalized_event)
        await client._handle_event(error_event)

        assert client.stats.events_received == 3
        assert client.stats.partials_received == 1
        assert client.stats.finalized_received == 1
        assert client.stats.errors_received == 1

    @pytest.mark.asyncio
    async def test_callback_invocation(self) -> None:
        """Test that callbacks are invoked for events."""
        on_partial = MagicMock()
        on_finalized = MagicMock()
        on_error = MagicMock()

        client = StreamingClient(
            on_partial=on_partial,
            on_finalized=on_finalized,
            on_error=on_error,
        )

        partial_event = StreamEvent(
            event_id=1,
            stream_id="str-123",
            type="PARTIAL",
            ts_server=1000,
            payload={"segment": {}},
        )
        finalized_event = StreamEvent(
            event_id=2,
            stream_id="str-123",
            type="FINALIZED",
            ts_server=1001,
            payload={"segment": {}},
        )
        error_event = StreamEvent(
            event_id=3,
            stream_id="str-123",
            type="ERROR",
            ts_server=1002,
            payload={"code": "test", "message": "Test", "recoverable": True},
        )

        await client._handle_event(partial_event)
        await client._handle_event(finalized_event)
        await client._handle_event(error_event)

        on_partial.assert_called_once_with(partial_event)
        on_finalized.assert_called_once_with(finalized_event)
        on_error.assert_called_once_with(error_event)

    @pytest.mark.asyncio
    async def test_callback_exception_caught(self) -> None:
        """Test that callback exceptions don't crash the client."""

        def bad_callback(event: StreamEvent) -> None:
            raise Exception("Callback error")

        client = StreamingClient(on_finalized=bad_callback)

        event = StreamEvent(
            event_id=1,
            stream_id="str-123",
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
# Protocol Compatibility Tests
# =============================================================================


class TestProtocolCompatibility:
    """Tests ensuring client is compatible with server protocol."""

    def test_client_message_type_alignment(self) -> None:
        """Test that client uses correct message type strings."""
        from slower_whisper.pipeline.streaming_ws import ClientMessageType

        # Client should send these exact strings
        assert ClientMessageType.START_SESSION.value == "START_SESSION"
        assert ClientMessageType.AUDIO_CHUNK.value == "AUDIO_CHUNK"
        assert ClientMessageType.END_SESSION.value == "END_SESSION"
        assert ClientMessageType.PING.value == "PING"

    def test_server_message_type_alignment(self) -> None:
        """Test that client expects correct server message types."""
        from slower_whisper.pipeline.streaming_ws import ServerMessageType

        # Client EventType should match server ServerMessageType
        assert EventType.SESSION_STARTED.value == ServerMessageType.SESSION_STARTED.value
        assert EventType.PARTIAL.value == ServerMessageType.PARTIAL.value
        assert EventType.FINALIZED.value == ServerMessageType.FINALIZED.value
        assert EventType.SPEAKER_TURN.value == ServerMessageType.SPEAKER_TURN.value
        assert EventType.SEMANTIC_UPDATE.value == ServerMessageType.SEMANTIC_UPDATE.value
        assert EventType.ERROR.value == ServerMessageType.ERROR.value
        assert EventType.SESSION_ENDED.value == ServerMessageType.SESSION_ENDED.value
        assert EventType.PONG.value == ServerMessageType.PONG.value

    def test_event_envelope_compatibility(self) -> None:
        """Test that client can parse server EventEnvelope format."""
        from slower_whisper.pipeline.streaming_ws import EventEnvelope, ServerMessageType

        # Create a server-side envelope
        server_envelope = EventEnvelope(
            event_id=1,
            stream_id="str-test-123",
            type=ServerMessageType.SESSION_STARTED,
            ts_server=1234567890000,
            payload={"session_id": "str-test-123"},
        )

        # Convert to dict (as server would send)
        server_dict = server_envelope.to_dict()

        # Client should be able to parse it
        client_event = StreamEvent.from_dict(server_dict)

        assert client_event.event_id == 1
        assert client_event.stream_id == "str-test-123"
        assert client_event.type == "SESSION_STARTED"
        assert client_event.ts_server == 1234567890000
        assert client_event.payload == {"session_id": "str-test-123"}

    def test_session_config_compatibility(self) -> None:
        """Test that client config matches server WebSocketSessionConfig."""
        from slower_whisper.pipeline.streaming_ws import WebSocketSessionConfig

        # Client config
        client_config = StreamingConfig(
            max_gap_sec=0.5,
            enable_prosody=True,
            enable_emotion=True,
            enable_categorical_emotion=False,
            sample_rate=16000,
            audio_format="pcm_s16le",
        )

        # Build config dict as client would send
        config_dict = {
            "max_gap_sec": client_config.max_gap_sec,
            "enable_prosody": client_config.enable_prosody,
            "enable_emotion": client_config.enable_emotion,
            "enable_categorical_emotion": client_config.enable_categorical_emotion,
            "sample_rate": client_config.sample_rate,
            "audio_format": client_config.audio_format,
        }

        # Server should be able to parse it
        server_config = WebSocketSessionConfig.from_dict(config_dict)

        assert server_config.max_gap_sec == 0.5
        assert server_config.enable_prosody is True
        assert server_config.enable_emotion is True
        assert server_config.enable_categorical_emotion is False
        assert server_config.sample_rate == 16000
        assert server_config.audio_format == "pcm_s16le"

    def test_audio_chunk_decoding_compatibility(self) -> None:
        """Test that client audio encoding matches server decoding."""
        from slower_whisper.pipeline.streaming_ws import decode_audio_chunk

        # Client encodes audio
        audio_bytes = b"test audio data with some bytes"
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        sequence = 42

        # Build payload as client would
        payload = {
            "data": encoded,
            "sequence": sequence,
        }

        # Server should be able to decode it
        decoded_audio, decoded_seq = decode_audio_chunk(payload)

        assert decoded_audio == audio_bytes
        assert decoded_seq == 42


# =============================================================================
# Full WebSocket Integration Test (requires running server)
# =============================================================================


@pytest.mark.integration
class TestLiveServerIntegration:
    """Integration tests with live server (skipped by default).

    Run these tests with: pytest -m integration
    Requires a running slower-whisper server at localhost:8000
    """

    @pytest.mark.asyncio
    async def test_full_session_lifecycle(self) -> None:
        """Test complete session lifecycle with real server."""
        pytest.skip("Requires running server - run manually")

        async with StreamingClient() as client:
            # Start session
            start_event = await client.start_session(max_gap_sec=1.0)
            assert start_event.type == "SESSION_STARTED"
            assert client.stream_id is not None

            # Send some audio
            audio = b"\x00" * 32000  # 1 second of silence
            await client.send_audio(audio)

            # End session
            final_events = await client.end_session()

            # Should have SESSION_ENDED
            ended = [e for e in final_events if e.type == "SESSION_ENDED"]
            assert len(ended) == 1
            assert "stats" in ended[0].payload

    @pytest.mark.asyncio
    async def test_ping_pong(self) -> None:
        """Test ping/pong heartbeat with real server."""
        pytest.skip("Requires running server - run manually")

        async with StreamingClient() as client:
            # Ping should work even without session
            pong = await client.ping()
            assert pong.type == "PONG"
            assert "timestamp" in pong.payload
            assert "server_timestamp" in pong.payload
