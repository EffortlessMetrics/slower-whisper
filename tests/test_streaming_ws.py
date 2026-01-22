"""Tests for WebSocket streaming protocol (v2.0.0).

Tests cover:
- Event envelope structure and serialization
- Session lifecycle (create, start, process, end)
- Message parsing and validation
- Audio chunk decoding
- Error handling
- WebSocket endpoint integration
- Incremental diarization hook (#86)
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from transcription.streaming_ws import (
    ClientMessageType,
    EventEnvelope,
    ServerMessageType,
    SessionState,
    SessionStats,
    WebSocketSessionConfig,
    WebSocketStreamingSession,
    decode_audio_chunk,
    parse_client_message,
)

# =============================================================================
# Event Envelope Tests
# =============================================================================


class TestEventEnvelope:
    """Tests for EventEnvelope dataclass."""

    def test_envelope_creation(self) -> None:
        """Test basic envelope creation with required fields."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-test-123",
            type=ServerMessageType.SESSION_STARTED,
            ts_server=1234567890000,
            payload={"session_id": "str-test-123"},
        )
        assert envelope.event_id == 1
        assert envelope.stream_id == "str-test-123"
        assert envelope.type == ServerMessageType.SESSION_STARTED
        assert envelope.ts_server == 1234567890000
        assert envelope.payload == {"session_id": "str-test-123"}
        assert envelope.segment_id is None
        assert envelope.ts_audio_start is None
        assert envelope.ts_audio_end is None

    def test_envelope_with_audio_timestamps(self) -> None:
        """Test envelope creation with audio timestamps."""
        envelope = EventEnvelope(
            event_id=5,
            stream_id="str-test-123",
            type=ServerMessageType.FINALIZED,
            ts_server=1234567890000,
            payload={"segment": {}},
            segment_id="seg-0",
            ts_audio_start=0.0,
            ts_audio_end=2.5,
        )
        assert envelope.segment_id == "seg-0"
        assert envelope.ts_audio_start == 0.0
        assert envelope.ts_audio_end == 2.5

    def test_envelope_to_dict_minimal(self) -> None:
        """Test envelope serialization without optional fields."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-test",
            type=ServerMessageType.PONG,
            ts_server=1000,
            payload={"timestamp": 999},
        )
        d = envelope.to_dict()
        assert d == {
            "event_id": 1,
            "stream_id": "str-test",
            "type": "PONG",
            "ts_server": 1000,
            "payload": {"timestamp": 999},
        }
        # Optional fields should not be in dict when None
        assert "segment_id" not in d
        assert "ts_audio_start" not in d
        assert "ts_audio_end" not in d

    def test_envelope_to_dict_complete(self) -> None:
        """Test envelope serialization with all fields."""
        envelope = EventEnvelope(
            event_id=10,
            stream_id="str-abc",
            type=ServerMessageType.PARTIAL,
            ts_server=2000,
            payload={"segment": {"text": "hello"}},
            segment_id="seg-5",
            ts_audio_start=1.0,
            ts_audio_end=3.5,
        )
        d = envelope.to_dict()
        assert d["event_id"] == 10
        assert d["stream_id"] == "str-abc"
        assert d["type"] == "PARTIAL"
        assert d["ts_server"] == 2000
        assert d["payload"] == {"segment": {"text": "hello"}}
        assert d["segment_id"] == "seg-5"
        assert d["ts_audio_start"] == 1.0
        assert d["ts_audio_end"] == 3.5


# =============================================================================
# Session Configuration Tests
# =============================================================================


class TestWebSocketSessionConfig:
    """Tests for WebSocketSessionConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = WebSocketSessionConfig()
        assert config.max_gap_sec == 1.0
        assert config.enable_prosody is False
        assert config.enable_emotion is False
        assert config.enable_categorical_emotion is False
        assert config.sample_rate == 16000
        assert config.audio_format == "pcm_s16le"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = WebSocketSessionConfig(
            max_gap_sec=0.5,
            enable_prosody=True,
            enable_emotion=True,
            sample_rate=8000,
        )
        assert config.max_gap_sec == 0.5
        assert config.enable_prosody is True
        assert config.enable_emotion is True
        assert config.sample_rate == 8000

    def test_from_dict_defaults(self) -> None:
        """Test config creation from empty dict uses defaults."""
        config = WebSocketSessionConfig.from_dict({})
        assert config.max_gap_sec == 1.0
        assert config.enable_prosody is False
        assert config.sample_rate == 16000

    def test_from_dict_partial(self) -> None:
        """Test config creation from partial dict."""
        config = WebSocketSessionConfig.from_dict(
            {
                "max_gap_sec": 2.0,
                "enable_prosody": True,
            }
        )
        assert config.max_gap_sec == 2.0
        assert config.enable_prosody is True
        assert config.enable_emotion is False  # default

    def test_from_dict_type_coercion(self) -> None:
        """Test config handles string/int type coercion."""
        config = WebSocketSessionConfig.from_dict(
            {
                "max_gap_sec": "1.5",
                "sample_rate": "16000",
                "enable_prosody": 1,  # truthy
            }
        )
        assert config.max_gap_sec == 1.5
        assert config.sample_rate == 16000
        assert config.enable_prosody is True


# =============================================================================
# Session Stats Tests
# =============================================================================


class TestSessionStats:
    """Tests for SessionStats."""

    def test_initial_stats(self) -> None:
        """Test initial stats values."""
        stats = SessionStats()
        assert stats.chunks_received == 0
        assert stats.bytes_received == 0
        assert stats.segments_partial == 0
        assert stats.segments_finalized == 0
        assert stats.events_sent == 0
        assert stats.events_dropped == 0
        assert stats.errors == 0
        assert stats.end_time is None
        assert stats.start_time > 0

    def test_stats_to_dict(self) -> None:
        """Test stats serialization."""
        stats = SessionStats()
        stats.chunks_received = 10
        stats.bytes_received = 32000
        stats.segments_finalized = 5
        stats.end_time = stats.start_time + 10.0

        d = stats.to_dict()
        assert d["chunks_received"] == 10
        assert d["bytes_received"] == 32000
        assert d["segments_finalized"] == 5
        assert d["duration_sec"] == pytest.approx(10.0, rel=0.01)


# =============================================================================
# WebSocket Session Tests
# =============================================================================


class TestWebSocketStreamingSession:
    """Tests for WebSocketStreamingSession."""

    def test_session_creation(self) -> None:
        """Test session initialization."""
        session = WebSocketStreamingSession()
        assert session.stream_id.startswith("str-")
        assert session.state == SessionState.CREATED
        assert session.config.max_gap_sec == 1.0

    def test_session_with_custom_config(self) -> None:
        """Test session with custom configuration."""
        config = WebSocketSessionConfig(max_gap_sec=0.5, enable_prosody=True)
        session = WebSocketStreamingSession(config=config)
        assert session.config.max_gap_sec == 0.5
        assert session.config.enable_prosody is True

    @pytest.mark.asyncio
    async def test_session_start(self) -> None:
        """Test session start returns SESSION_STARTED event."""
        session = WebSocketStreamingSession()
        event = await session.start()

        assert session.state == SessionState.ACTIVE
        assert event.type == ServerMessageType.SESSION_STARTED
        assert event.event_id == 1
        assert event.stream_id == session.stream_id
        assert event.payload["session_id"] == session.stream_id

    @pytest.mark.asyncio
    async def test_session_cannot_start_twice(self) -> None:
        """Test starting an already started session raises error."""
        session = WebSocketStreamingSession()
        await session.start()

        with pytest.raises(RuntimeError, match="Cannot start session"):
            await session.start()

    @pytest.mark.asyncio
    async def test_process_audio_chunk(self) -> None:
        """Test processing audio chunks."""
        session = WebSocketStreamingSession()
        await session.start()

        # 1 second of audio at 16kHz 16-bit mono = 32000 bytes
        audio_data = b"\x00" * 32000
        events = await session.process_audio_chunk(audio_data, sequence=1)

        assert session.stats.chunks_received == 1
        assert session.stats.bytes_received == 32000
        assert len(events) >= 1  # Should emit at least one partial

    @pytest.mark.asyncio
    async def test_chunk_sequence_validation(self) -> None:
        """Test chunks must arrive in order."""
        session = WebSocketStreamingSession()
        await session.start()

        await session.process_audio_chunk(b"\x00" * 100, sequence=1)
        await session.process_audio_chunk(b"\x00" * 100, sequence=2)

        with pytest.raises(ValueError, match="sequence"):
            await session.process_audio_chunk(b"\x00" * 100, sequence=2)  # duplicate

        with pytest.raises(ValueError, match="sequence"):
            await session.process_audio_chunk(b"\x00" * 100, sequence=1)  # out of order

    @pytest.mark.asyncio
    async def test_session_end(self) -> None:
        """Test ending session returns FINALIZED and SESSION_ENDED events."""
        session = WebSocketStreamingSession()
        await session.start()
        await session.process_audio_chunk(b"\x00" * 32000, sequence=1)

        events = await session.end()

        assert session.state == SessionState.ENDED
        assert any(e.type == ServerMessageType.FINALIZED for e in events)
        assert any(e.type == ServerMessageType.SESSION_ENDED for e in events)

        # Check SESSION_ENDED has stats
        ended_event = next(e for e in events if e.type == ServerMessageType.SESSION_ENDED)
        assert "stats" in ended_event.payload
        assert "duration_sec" in ended_event.payload["stats"]

    @pytest.mark.asyncio
    async def test_cannot_process_after_end(self) -> None:
        """Test processing chunks after end raises error."""
        session = WebSocketStreamingSession()
        await session.start()
        await session.end()

        with pytest.raises(RuntimeError, match="Cannot process chunk"):
            await session.process_audio_chunk(b"\x00" * 100, sequence=1)

    def test_create_error_event(self) -> None:
        """Test error event creation."""
        session = WebSocketStreamingSession()
        event = session.create_error_event(
            code="test_error",
            message="Test error message",
            recoverable=True,
        )

        assert event.type == ServerMessageType.ERROR
        assert event.payload["code"] == "test_error"
        assert event.payload["message"] == "Test error message"
        assert event.payload["recoverable"] is True
        assert session.stats.errors == 1

    def test_create_error_event_non_recoverable(self) -> None:
        """Test non-recoverable error sets session state to ERROR."""
        session = WebSocketStreamingSession()
        session.create_error_event(
            code="fatal_error",
            message="Fatal error",
            recoverable=False,
        )

        assert session.state == SessionState.ERROR

    def test_create_pong_event(self) -> None:
        """Test PONG event creation."""
        session = WebSocketStreamingSession()
        client_ts = 1234567890000

        event = session.create_pong_event(client_ts)

        assert event.type == ServerMessageType.PONG
        assert event.payload["timestamp"] == client_ts
        assert "server_timestamp" in event.payload

    @pytest.mark.asyncio
    async def test_event_id_monotonic(self) -> None:
        """Test event IDs are monotonically increasing."""
        session = WebSocketStreamingSession()
        event1 = await session.start()
        event2 = session.create_pong_event(0)
        event3 = session.create_error_event("test", "test", True)

        assert event1.event_id == 1
        assert event2.event_id == 2
        assert event3.event_id == 3


# =============================================================================
# Message Parsing Tests
# =============================================================================


class TestMessageParsing:
    """Tests for message parsing helpers."""

    def test_parse_start_session(self) -> None:
        """Test parsing START_SESSION message."""
        msg = {"type": "START_SESSION", "config": {"max_gap_sec": 0.5}}
        msg_type, payload = parse_client_message(msg)

        assert msg_type == ClientMessageType.START_SESSION
        assert payload == {"config": {"max_gap_sec": 0.5}}

    def test_parse_audio_chunk(self) -> None:
        """Test parsing AUDIO_CHUNK message."""
        msg = {"type": "AUDIO_CHUNK", "data": "dGVzdA==", "sequence": 1}
        msg_type, payload = parse_client_message(msg)

        assert msg_type == ClientMessageType.AUDIO_CHUNK
        assert payload["data"] == "dGVzdA=="
        assert payload["sequence"] == 1

    def test_parse_end_session(self) -> None:
        """Test parsing END_SESSION message."""
        msg = {"type": "END_SESSION"}
        msg_type, payload = parse_client_message(msg)

        assert msg_type == ClientMessageType.END_SESSION
        assert payload == {}

    def test_parse_ping(self) -> None:
        """Test parsing PING message."""
        msg = {"type": "PING", "timestamp": 1234567890000}
        msg_type, payload = parse_client_message(msg)

        assert msg_type == ClientMessageType.PING
        assert payload["timestamp"] == 1234567890000

    def test_parse_missing_type(self) -> None:
        """Test parsing message without type field."""
        with pytest.raises(ValueError, match="Missing 'type'"):
            parse_client_message({"config": {}})

    def test_parse_invalid_type(self) -> None:
        """Test parsing message with invalid type."""
        with pytest.raises(ValueError, match="Invalid client message type"):
            parse_client_message({"type": "INVALID_TYPE"})


class TestAudioChunkDecoding:
    """Tests for audio chunk decoding."""

    def test_decode_valid_chunk(self) -> None:
        """Test decoding valid audio chunk."""
        audio_bytes = b"test audio data"
        encoded = base64.b64encode(audio_bytes).decode()

        payload = {"data": encoded, "sequence": 5}
        decoded, seq = decode_audio_chunk(payload)

        assert decoded == audio_bytes
        assert seq == 5

    def test_decode_empty_audio_rejected(self) -> None:
        """Test decoding empty audio chunk is rejected."""
        # Empty base64 encoding results in empty string, which is rejected
        payload = {"data": "", "sequence": 1}
        with pytest.raises(ValueError, match="Missing 'data'"):
            decode_audio_chunk(payload)

    def test_decode_missing_data(self) -> None:
        """Test decoding chunk without data field."""
        with pytest.raises(ValueError, match="Missing 'data'"):
            decode_audio_chunk({"sequence": 1})

    def test_decode_missing_sequence(self) -> None:
        """Test decoding chunk without sequence field."""
        with pytest.raises(ValueError, match="Missing 'sequence'"):
            decode_audio_chunk({"data": "dGVzdA=="})

    def test_decode_invalid_base64(self) -> None:
        """Test decoding chunk with invalid base64."""
        with pytest.raises(ValueError, match="Invalid base64"):
            decode_audio_chunk({"data": "not-valid-base64!!!", "sequence": 1})


# =============================================================================
# Enum Tests
# =============================================================================


class TestMessageEnums:
    """Tests for message type enums."""

    def test_client_message_types(self) -> None:
        """Test client message type values."""
        assert ClientMessageType.START_SESSION.value == "START_SESSION"
        assert ClientMessageType.AUDIO_CHUNK.value == "AUDIO_CHUNK"
        assert ClientMessageType.END_SESSION.value == "END_SESSION"
        assert ClientMessageType.PING.value == "PING"

    def test_server_message_types(self) -> None:
        """Test server message type values."""
        assert ServerMessageType.SESSION_STARTED.value == "SESSION_STARTED"
        assert ServerMessageType.PARTIAL.value == "PARTIAL"
        assert ServerMessageType.FINALIZED.value == "FINALIZED"
        assert ServerMessageType.SPEAKER_TURN.value == "SPEAKER_TURN"
        assert ServerMessageType.SEMANTIC_UPDATE.value == "SEMANTIC_UPDATE"
        assert ServerMessageType.ERROR.value == "ERROR"
        assert ServerMessageType.SESSION_ENDED.value == "SESSION_ENDED"
        assert ServerMessageType.PONG.value == "PONG"

    def test_session_states(self) -> None:
        """Test session state values."""
        assert SessionState.CREATED.value == "created"
        assert SessionState.ACTIVE.value == "active"
        assert SessionState.ENDING.value == "ending"
        assert SessionState.ENDED.value == "ended"
        assert SessionState.ERROR.value == "error"

    def test_diarization_update_type(self) -> None:
        """Test DIARIZATION_UPDATE message type exists."""
        assert ServerMessageType.DIARIZATION_UPDATE.value == "DIARIZATION_UPDATE"


# =============================================================================
# Incremental Diarization Tests
# =============================================================================


class TestSpeakerAssignment:
    """Tests for SpeakerAssignment dataclass."""

    def test_speaker_assignment_creation(self) -> None:
        """Test basic SpeakerAssignment creation."""
        from transcription.streaming_ws import SpeakerAssignment

        assignment = SpeakerAssignment(
            start=0.0,
            end=5.0,
            speaker_id="spk_0",
            confidence=0.95,
        )
        assert assignment.start == 0.0
        assert assignment.end == 5.0
        assert assignment.speaker_id == "spk_0"
        assert assignment.confidence == 0.95

    def test_speaker_assignment_to_dict(self) -> None:
        """Test SpeakerAssignment serialization."""
        from transcription.streaming_ws import SpeakerAssignment

        assignment = SpeakerAssignment(
            start=1.0,
            end=3.5,
            speaker_id="spk_1",
            confidence=0.87,
        )
        d = assignment.to_dict()
        assert d == {
            "start": 1.0,
            "end": 3.5,
            "speaker_id": "spk_1",
            "confidence": 0.87,
        }

    def test_speaker_assignment_to_dict_no_confidence(self) -> None:
        """Test SpeakerAssignment serialization without confidence."""
        from transcription.streaming_ws import SpeakerAssignment

        assignment = SpeakerAssignment(
            start=0.0,
            end=2.0,
            speaker_id="spk_0",
        )
        d = assignment.to_dict()
        assert d == {
            "start": 0.0,
            "end": 2.0,
            "speaker_id": "spk_0",
        }
        assert "confidence" not in d


class TestIncrementalDiarization:
    """Tests for incremental diarization hook functionality."""

    def test_config_diarization_defaults(self) -> None:
        """Test diarization config defaults."""
        config = WebSocketSessionConfig()
        assert config.enable_diarization is False
        assert config.diarization_interval_sec == 30.0

    def test_config_diarization_from_dict(self) -> None:
        """Test diarization config from dict."""
        config = WebSocketSessionConfig.from_dict(
            {
                "enable_diarization": True,
                "diarization_interval_sec": 15.0,
            }
        )
        assert config.enable_diarization is True
        assert config.diarization_interval_sec == 15.0

    def test_session_with_diarization_hook(self) -> None:
        """Test session creation with diarization hook."""
        from transcription.streaming_ws import SpeakerAssignment

        async def mock_hook(audio_buffer: bytes, sample_rate: int) -> list[SpeakerAssignment]:
            return [SpeakerAssignment(start=0.0, end=1.0, speaker_id="spk_0")]

        config = WebSocketSessionConfig(enable_diarization=True)
        session = WebSocketStreamingSession(config=config, diarization_hook=mock_hook)
        assert session._diarization_hook is not None

    def test_session_without_diarization_hook(self) -> None:
        """Test session creation without diarization hook (graceful degradation)."""
        config = WebSocketSessionConfig(enable_diarization=True)
        session = WebSocketStreamingSession(config=config)
        assert session._diarization_hook is None

    @pytest.mark.asyncio
    async def test_diarization_not_triggered_when_disabled(self) -> None:
        """Test diarization is not triggered when disabled in config."""
        from transcription.streaming_ws import SpeakerAssignment

        hook_called = False

        async def mock_hook(audio_buffer: bytes, sample_rate: int) -> list[SpeakerAssignment]:
            nonlocal hook_called
            hook_called = True
            return []

        config = WebSocketSessionConfig(enable_diarization=False)
        session = WebSocketStreamingSession(config=config, diarization_hook=mock_hook)
        await session.start()

        # Send enough audio to pass the interval
        audio_data = b"\x00" * 32000 * 35  # 35 seconds of 16kHz 16-bit audio
        await session.process_audio_chunk(audio_data, sequence=1)

        assert hook_called is False

    @pytest.mark.asyncio
    async def test_diarization_not_triggered_without_hook(self) -> None:
        """Test diarization is not triggered when hook is not provided."""
        config = WebSocketSessionConfig(
            enable_diarization=True,
            diarization_interval_sec=1.0,
        )
        session = WebSocketStreamingSession(config=config)
        await session.start()

        # Send enough audio to pass the interval
        audio_data = b"\x00" * 32000 * 2  # 2 seconds of audio
        events = await session.process_audio_chunk(audio_data, sequence=1)

        # No DIARIZATION_UPDATE should be in events
        diarization_events = [e for e in events if e.type == ServerMessageType.DIARIZATION_UPDATE]
        assert len(diarization_events) == 0

    @pytest.mark.asyncio
    async def test_diarization_triggered_at_interval(self) -> None:
        """Test diarization is triggered after interval threshold."""
        from transcription.streaming_ws import SpeakerAssignment

        hook_calls = []

        async def mock_hook(audio_buffer: bytes, sample_rate: int) -> list[SpeakerAssignment]:
            hook_calls.append(len(audio_buffer))
            return [
                SpeakerAssignment(start=0.0, end=1.0, speaker_id="spk_0"),
                SpeakerAssignment(start=1.0, end=2.0, speaker_id="spk_1"),
            ]

        config = WebSocketSessionConfig(
            enable_diarization=True,
            diarization_interval_sec=1.0,  # Trigger every 1 second
        )
        session = WebSocketStreamingSession(config=config, diarization_hook=mock_hook)
        await session.start()

        # Send 2 seconds of audio (should trigger diarization)
        audio_data = b"\x00" * 32000 * 2
        events = await session.process_audio_chunk(audio_data, sequence=1)

        assert len(hook_calls) == 1
        diarization_events = [e for e in events if e.type == ServerMessageType.DIARIZATION_UPDATE]
        assert len(diarization_events) == 1

        # Check event payload
        event = diarization_events[0]
        assert event.payload["update_number"] == 1
        assert event.payload["num_speakers"] == 2
        assert event.payload["speaker_ids"] == ["spk_0", "spk_1"]
        assert len(event.payload["assignments"]) == 2

    @pytest.mark.asyncio
    async def test_diarization_not_triggered_before_interval(self) -> None:
        """Test diarization is not triggered before interval threshold."""
        from transcription.streaming_ws import SpeakerAssignment

        hook_calls = []

        async def mock_hook(audio_buffer: bytes, sample_rate: int) -> list[SpeakerAssignment]:
            hook_calls.append(len(audio_buffer))
            return []

        config = WebSocketSessionConfig(
            enable_diarization=True,
            diarization_interval_sec=30.0,  # 30 second interval
        )
        session = WebSocketStreamingSession(config=config, diarization_hook=mock_hook)
        await session.start()

        # Send only 10 seconds of audio (should not trigger)
        audio_data = b"\x00" * 32000 * 10
        await session.process_audio_chunk(audio_data, sequence=1)

        assert len(hook_calls) == 0

    @pytest.mark.asyncio
    async def test_final_diarization_on_end(self) -> None:
        """Test final diarization is triggered on session end."""
        from transcription.streaming_ws import SpeakerAssignment

        hook_calls = []

        async def mock_hook(audio_buffer: bytes, sample_rate: int) -> list[SpeakerAssignment]:
            hook_calls.append(len(audio_buffer))
            return [SpeakerAssignment(start=0.0, end=5.0, speaker_id="spk_0")]

        config = WebSocketSessionConfig(
            enable_diarization=True,
            diarization_interval_sec=60.0,  # High interval
        )
        session = WebSocketStreamingSession(config=config, diarization_hook=mock_hook)
        await session.start()

        # Send audio but not enough to trigger interval
        audio_data = b"\x00" * 32000 * 5  # 5 seconds
        await session.process_audio_chunk(audio_data, sequence=1)
        assert len(hook_calls) == 0  # Not triggered during processing

        # End session should trigger final diarization
        events = await session.end()
        assert len(hook_calls) == 1  # Triggered on end

        # Check for DIARIZATION_UPDATE in end events
        diarization_events = [e for e in events if e.type == ServerMessageType.DIARIZATION_UPDATE]
        assert len(diarization_events) == 1

    @pytest.mark.asyncio
    async def test_diarization_hook_failure_graceful(self) -> None:
        """Test diarization hook failure doesn't crash pipeline."""

        async def failing_hook(audio_buffer: bytes, sample_rate: int) -> list:
            raise RuntimeError("Diarization failed!")

        config = WebSocketSessionConfig(
            enable_diarization=True,
            diarization_interval_sec=1.0,
        )
        session = WebSocketStreamingSession(config=config, diarization_hook=failing_hook)
        await session.start()

        # Send enough audio to trigger diarization
        audio_data = b"\x00" * 32000 * 2
        # Should not raise, graceful degradation
        events = await session.process_audio_chunk(audio_data, sequence=1)

        # No DIARIZATION_UPDATE event due to failure
        diarization_events = [e for e in events if e.type == ServerMessageType.DIARIZATION_UPDATE]
        assert len(diarization_events) == 0

    @pytest.mark.asyncio
    async def test_get_speaker_assignments(self) -> None:
        """Test getting current speaker assignments."""
        from transcription.streaming_ws import SpeakerAssignment

        expected_assignments = [
            SpeakerAssignment(start=0.0, end=2.0, speaker_id="spk_0"),
            SpeakerAssignment(start=2.0, end=4.0, speaker_id="spk_1"),
        ]

        async def mock_hook(audio_buffer: bytes, sample_rate: int) -> list[SpeakerAssignment]:
            return expected_assignments

        config = WebSocketSessionConfig(
            enable_diarization=True,
            diarization_interval_sec=1.0,
        )
        session = WebSocketStreamingSession(config=config, diarization_hook=mock_hook)
        await session.start()

        # Initially empty
        assert session.get_speaker_assignments() == []

        # Trigger diarization
        audio_data = b"\x00" * 32000 * 2
        await session.process_audio_chunk(audio_data, sequence=1)

        # Should have assignments now
        assignments = session.get_speaker_assignments()
        assert len(assignments) == 2
        assert assignments[0].speaker_id == "spk_0"
        assert assignments[1].speaker_id == "spk_1"

    @pytest.mark.asyncio
    async def test_diarization_update_only_on_change(self) -> None:
        """Test DIARIZATION_UPDATE only emitted when assignments change."""
        from transcription.streaming_ws import SpeakerAssignment

        call_count = 0

        async def mock_hook(audio_buffer: bytes, sample_rate: int) -> list[SpeakerAssignment]:
            nonlocal call_count
            call_count += 1
            # Always return same assignments
            return [SpeakerAssignment(start=0.0, end=1.0, speaker_id="spk_0")]

        config = WebSocketSessionConfig(
            enable_diarization=True,
            diarization_interval_sec=1.0,
        )
        session = WebSocketStreamingSession(config=config, diarization_hook=mock_hook)
        await session.start()

        # First call - should emit event
        audio_data = b"\x00" * 32000 * 2
        events1 = await session.process_audio_chunk(audio_data, sequence=1)
        diar_events1 = [e for e in events1 if e.type == ServerMessageType.DIARIZATION_UPDATE]
        assert len(diar_events1) == 1

        # Second call - same assignments, should not emit
        events2 = await session.process_audio_chunk(audio_data, sequence=2)
        diar_events2 = [e for e in events2 if e.type == ServerMessageType.DIARIZATION_UPDATE]
        assert len(diar_events2) == 0

        # Hook was still called
        assert call_count == 2


# =============================================================================
# WebSocket Endpoint Integration Tests
# =============================================================================


class TestWebSocketEndpoint:
    """Integration tests for the /stream WebSocket endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create FastAPI test client."""
        from transcription.service import app

        return TestClient(app)

    def test_websocket_full_session(self, client: TestClient) -> None:
        """Test complete WebSocket session lifecycle."""
        with client.websocket_connect("/stream") as websocket:
            # Send START_SESSION
            websocket.send_json(
                {
                    "type": "START_SESSION",
                    "config": {"max_gap_sec": 1.0},
                }
            )

            # Receive SESSION_STARTED
            msg = websocket.receive_json()
            assert msg["type"] == "SESSION_STARTED"
            assert "session_id" in msg["payload"]
            assert msg["stream_id"].startswith("str-")

            # Send audio chunk
            audio_data = base64.b64encode(b"\x00" * 32000).decode()
            websocket.send_json(
                {
                    "type": "AUDIO_CHUNK",
                    "data": audio_data,
                    "sequence": 1,
                }
            )

            # May receive PARTIAL event(s)
            # Receive events until we're ready for END_SESSION
            received_events = []
            while True:
                try:
                    # Use timeout to avoid hanging
                    msg = websocket.receive_json()
                    received_events.append(msg)
                    if msg["type"] in ("PARTIAL", "ERROR"):
                        break
                except Exception:
                    break

            # Send END_SESSION
            websocket.send_json({"type": "END_SESSION"})

            # Receive FINALIZED and SESSION_ENDED
            final_events = []
            while True:
                try:
                    msg = websocket.receive_json()
                    final_events.append(msg)
                    if msg["type"] == "SESSION_ENDED":
                        break
                except Exception:
                    break

            # Verify we got SESSION_ENDED
            assert any(e["type"] == "SESSION_ENDED" for e in final_events)

            # Verify stats in SESSION_ENDED
            ended_event = next(e for e in final_events if e["type"] == "SESSION_ENDED")
            assert "stats" in ended_event["payload"]
            assert ended_event["payload"]["stats"]["chunks_received"] >= 1

    def test_websocket_ping_pong(self, client: TestClient) -> None:
        """Test PING/PONG heartbeat."""
        with client.websocket_connect("/stream") as websocket:
            # Send PING without starting session
            websocket.send_json(
                {
                    "type": "PING",
                    "timestamp": 1234567890000,
                }
            )

            # Receive PONG
            msg = websocket.receive_json()
            assert msg["type"] == "PONG"
            assert msg["payload"]["timestamp"] == 1234567890000
            assert "server_timestamp" in msg["payload"]

    def test_websocket_error_no_session(self, client: TestClient) -> None:
        """Test sending AUDIO_CHUNK without START_SESSION returns error."""
        with client.websocket_connect("/stream") as websocket:
            websocket.send_json(
                {
                    "type": "AUDIO_CHUNK",
                    "data": base64.b64encode(b"\x00" * 100).decode(),
                    "sequence": 1,
                }
            )

            msg = websocket.receive_json()
            assert msg["type"] == "ERROR"
            assert msg["payload"]["code"] == "no_session"

    def test_websocket_invalid_message_type(self, client: TestClient) -> None:
        """Test invalid message type handling."""
        with client.websocket_connect("/stream") as websocket:
            # Start session first
            websocket.send_json({"type": "START_SESSION", "config": {}})
            websocket.receive_json()  # SESSION_STARTED

            # Send invalid message type
            websocket.send_json({"type": "INVALID_TYPE"})

            msg = websocket.receive_json()
            assert msg["type"] == "ERROR"
            assert msg["payload"]["code"] == "invalid_message_type"


class TestStreamConfigEndpoint:
    """Tests for /stream/config endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create FastAPI test client."""
        from transcription.service import app

        return TestClient(app)

    def test_get_stream_config(self, client: TestClient) -> None:
        """Test getting default streaming configuration."""
        response = client.get("/stream/config")
        assert response.status_code == 200

        data = response.json()
        assert "default_config" in data
        assert "supported_audio_formats" in data
        assert "supported_sample_rates" in data
        assert "message_types" in data

        # Check default config values
        config = data["default_config"]
        assert config["max_gap_sec"] == 1.0
        assert config["sample_rate"] == 16000
        assert config["audio_format"] == "pcm_s16le"

        # Check message types
        assert "START_SESSION" in data["message_types"]["client"]
        assert "SESSION_STARTED" in data["message_types"]["server"]
