"""Contract tests for streaming protocol (Issue #222).

These tests verify that the streaming protocol implementation conforms to
the documented contract in docs/schemas/event_envelope_v2.json.

Tests cover:
- EventEnvelope serialization validates against JSON Schema
- Monotonic event IDs within a session
- Required fields always present
- Timestamp format correctness
- Stream ID and segment ID format validation
- Resume protocol (RESUME_SESSION handling)
- Backpressure handling (drop policy, BUFFER_OVERFLOW errors)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from transcription.streaming_ws import (
    ClientMessageType,
    EventEnvelope,
    ReplayBuffer,
    ServerMessageType,
    SessionState,
    WebSocketSessionConfig,
    WebSocketStreamingSession,
    parse_client_message,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def schema_path() -> Path:
    """Return path to the event envelope JSON schema."""
    return Path(__file__).parent.parent / "docs" / "schemas" / "event_envelope_v2.json"


@pytest.fixture
def event_envelope_schema(schema_path: Path) -> dict[str, Any]:
    """Load the event envelope JSON schema."""
    with open(schema_path) as f:
        return json.load(f)


@pytest.fixture
def jsonschema_validator(event_envelope_schema: dict[str, Any]):
    """Create a JSON Schema validator if jsonschema is available."""
    try:
        from jsonschema import Draft202012Validator

        return Draft202012Validator(event_envelope_schema)
    except ImportError:
        pytest.skip("jsonschema package not installed")


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestSchemaExists:
    """Verify schema files exist and are valid JSON."""

    def test_event_envelope_schema_exists(self, schema_path: Path) -> None:
        """Test that the event envelope schema file exists."""
        assert schema_path.exists(), f"Schema file not found: {schema_path}"

    def test_event_envelope_schema_is_valid_json(self, schema_path: Path) -> None:
        """Test that the schema is valid JSON."""
        with open(schema_path) as f:
            data = json.load(f)
        assert "$schema" in data
        assert "properties" in data

    def test_schema_has_required_fields_definition(
        self, event_envelope_schema: dict[str, Any]
    ) -> None:
        """Test schema defines required fields."""
        required = event_envelope_schema.get("required", [])
        assert "event_id" in required
        assert "stream_id" in required
        assert "type" in required
        assert "ts_server" in required
        assert "payload" in required


# =============================================================================
# Event Envelope Contract Tests
# =============================================================================


class TestEventEnvelopeContract:
    """Test EventEnvelope conforms to the documented contract."""

    def test_required_fields_always_present(self) -> None:
        """Test that serialized envelopes always have required fields."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-12345678-1234-4567-89ab-123456789abc",
            type=ServerMessageType.SESSION_STARTED,
            ts_server=1706300000000,
            payload={"session_id": "str-12345678-1234-4567-89ab-123456789abc"},
        )
        d = envelope.to_dict()

        # All required fields must be present
        assert "event_id" in d
        assert "stream_id" in d
        assert "type" in d
        assert "ts_server" in d
        assert "payload" in d

    def test_event_id_is_positive_integer(self) -> None:
        """Test event_id is always a positive integer."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-test",
            type=ServerMessageType.PONG,
            ts_server=1000,
            payload={},
        )
        d = envelope.to_dict()
        assert isinstance(d["event_id"], int)
        assert d["event_id"] >= 1

    def test_stream_id_format(self) -> None:
        """Test stream_id follows str-{uuid4} format."""
        session = WebSocketStreamingSession()
        # Stream ID should match the pattern str-{uuid4}
        pattern = r"^str-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        assert re.match(pattern, session.stream_id), f"Invalid stream_id: {session.stream_id}"

    def test_segment_id_format(self) -> None:
        """Test segment_id follows seg-{seq} format."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-test",
            type=ServerMessageType.PARTIAL,
            ts_server=1000,
            payload={"segment": {}},
            segment_id="seg-5",
        )
        d = envelope.to_dict()
        pattern = r"^seg-\d+$"
        assert re.match(pattern, d["segment_id"]), f"Invalid segment_id: {d['segment_id']}"

    def test_ts_server_is_milliseconds(self) -> None:
        """Test ts_server is Unix epoch milliseconds."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-test",
            type=ServerMessageType.PONG,
            ts_server=1706300000000,  # 2024-01-26 in ms
            payload={},
        )
        d = envelope.to_dict()
        # Should be milliseconds (13+ digits for recent dates)
        assert d["ts_server"] > 1000000000000, "ts_server should be in milliseconds"

    def test_type_is_valid_enum_value(self) -> None:
        """Test type is a valid ServerMessageType value."""
        valid_types = [
            "SESSION_STARTED",
            "PARTIAL",
            "FINALIZED",
            "SPEAKER_TURN",
            "SEMANTIC_UPDATE",
            "DIARIZATION_UPDATE",
            "ERROR",
            "SESSION_ENDED",
            "PONG",
        ]
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-test",
            type=ServerMessageType.SESSION_STARTED,
            ts_server=1000,
            payload={},
        )
        d = envelope.to_dict()
        assert d["type"] in valid_types

    def test_optional_fields_excluded_when_none(self) -> None:
        """Test optional fields are excluded from dict when None."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-test",
            type=ServerMessageType.PONG,
            ts_server=1000,
            payload={},
        )
        d = envelope.to_dict()
        assert "segment_id" not in d
        assert "ts_audio_start" not in d
        assert "ts_audio_end" not in d

    def test_optional_fields_included_when_set(self) -> None:
        """Test optional fields are included when set."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-test",
            type=ServerMessageType.FINALIZED,
            ts_server=1000,
            payload={"segment": {}},
            segment_id="seg-0",
            ts_audio_start=0.0,
            ts_audio_end=2.5,
        )
        d = envelope.to_dict()
        assert d["segment_id"] == "seg-0"
        assert d["ts_audio_start"] == 0.0
        assert d["ts_audio_end"] == 2.5


# =============================================================================
# JSON Schema Validation Tests
# =============================================================================


class TestJsonSchemaValidation:
    """Test that serialized events validate against the JSON schema."""

    def test_session_started_validates(self, jsonschema_validator) -> None:
        """Test SESSION_STARTED event validates against schema."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-12345678-1234-4567-89ab-123456789abc",
            type=ServerMessageType.SESSION_STARTED,
            ts_server=1706300000000,
            payload={"session_id": "str-12345678-1234-4567-89ab-123456789abc"},
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_partial_validates(self, jsonschema_validator) -> None:
        """Test PARTIAL event validates against schema."""
        envelope = EventEnvelope(
            event_id=2,
            stream_id="str-12345678-1234-4567-89ab-123456789abc",
            type=ServerMessageType.PARTIAL,
            ts_server=1706300001000,
            payload={
                "segment": {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "Hello world",
                    "speaker_id": None,
                }
            },
            segment_id="seg-0",
            ts_audio_start=0.0,
            ts_audio_end=1.5,
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_finalized_validates(self, jsonschema_validator) -> None:
        """Test FINALIZED event validates against schema."""
        envelope = EventEnvelope(
            event_id=3,
            stream_id="str-12345678-1234-4567-89ab-123456789abc",
            type=ServerMessageType.FINALIZED,
            ts_server=1706300002000,
            payload={
                "segment": {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world",
                    "speaker_id": "spk_0",
                    "audio_state": None,
                }
            },
            segment_id="seg-0",
            ts_audio_start=0.0,
            ts_audio_end=2.0,
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_error_validates(self, jsonschema_validator) -> None:
        """Test ERROR event validates against schema."""
        envelope = EventEnvelope(
            event_id=4,
            stream_id="str-12345678-1234-4567-89ab-123456789abc",
            type=ServerMessageType.ERROR,
            ts_server=1706300003000,
            payload={
                "code": "ASR_TIMEOUT",
                "message": "ASR processing timed out",
                "recoverable": True,
            },
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_session_ended_validates(self, jsonschema_validator) -> None:
        """Test SESSION_ENDED event validates against schema."""
        envelope = EventEnvelope(
            event_id=10,
            stream_id="str-12345678-1234-4567-89ab-123456789abc",
            type=ServerMessageType.SESSION_ENDED,
            ts_server=1706300010000,
            payload={
                "stats": {
                    "chunks_received": 5,
                    "bytes_received": 160000,
                    "segments_partial": 3,
                    "segments_finalized": 2,
                    "events_sent": 10,
                    "events_dropped": 0,
                    "errors": 0,
                    "duration_sec": 10.0,
                }
            },
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_pong_validates(self, jsonschema_validator) -> None:
        """Test PONG event validates against schema."""
        envelope = EventEnvelope(
            event_id=5,
            stream_id="str-12345678-1234-4567-89ab-123456789abc",
            type=ServerMessageType.PONG,
            ts_server=1706300004000,
            payload={
                "timestamp": 1706300003999,
                "server_timestamp": 1706300004000,
            },
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_diarization_update_validates(self, jsonschema_validator) -> None:
        """Test DIARIZATION_UPDATE event validates against schema."""
        envelope = EventEnvelope(
            event_id=6,
            stream_id="str-12345678-1234-4567-89ab-123456789abc",
            type=ServerMessageType.DIARIZATION_UPDATE,
            ts_server=1706300005000,
            payload={
                "update_number": 1,
                "audio_duration": 30.0,
                "num_speakers": 2,
                "speaker_ids": ["spk_0", "spk_1"],
                "assignments": [
                    {"start": 0.0, "end": 15.0, "speaker_id": "spk_0"},
                    {"start": 15.0, "end": 30.0, "speaker_id": "spk_1"},
                ],
            },
            ts_audio_start=0.0,
            ts_audio_end=30.0,
        )
        jsonschema_validator.validate(envelope.to_dict())


# =============================================================================
# Monotonic Event ID Tests
# =============================================================================


class TestMonotonicEventIds:
    """Test event IDs are monotonically increasing within a session."""

    @pytest.mark.asyncio
    async def test_event_ids_start_at_one(self) -> None:
        """Test event IDs start at 1."""
        session = WebSocketStreamingSession()
        event = await session.start()
        assert event.event_id == 1

    @pytest.mark.asyncio
    async def test_event_ids_monotonically_increase(self) -> None:
        """Test event IDs increase monotonically."""
        session = WebSocketStreamingSession()
        event1 = await session.start()  # ID 1
        event2 = session.create_pong_event(0)  # ID 2
        event3 = session.create_error_event("test", "test", True)  # ID 3

        assert event1.event_id == 1
        assert event2.event_id == 2
        assert event3.event_id == 3
        assert event1.event_id < event2.event_id < event3.event_id

    @pytest.mark.asyncio
    async def test_event_ids_never_reset(self) -> None:
        """Test event IDs never reset within a session."""
        session = WebSocketStreamingSession()
        await session.start()

        # Process multiple chunks
        audio_data = b"\x00" * 32000
        all_event_ids = [1]  # Start event

        for i in range(5):
            events = await session.process_audio_chunk(audio_data, sequence=i + 1)
            for event in events:
                all_event_ids.append(event.event_id)

        # End session
        end_events = await session.end()
        for event in end_events:
            all_event_ids.append(event.event_id)

        # All IDs should be strictly increasing
        for i in range(len(all_event_ids) - 1):
            assert all_event_ids[i] < all_event_ids[i + 1], (
                f"Event IDs not monotonic: {all_event_ids}"
            )

    @pytest.mark.asyncio
    async def test_event_ids_no_gaps_in_normal_operation(self) -> None:
        """Test no gaps in event IDs during normal operation."""
        session = WebSocketStreamingSession()
        start_event = await session.start()
        pong_event = session.create_pong_event(0)
        error_event = session.create_error_event("test", "test", True)

        # Should have no gaps
        assert start_event.event_id == 1
        assert pong_event.event_id == 2
        assert error_event.event_id == 3


# =============================================================================
# Session Lifecycle Contract Tests
# =============================================================================


class TestSessionLifecycleContract:
    """Test session lifecycle follows documented contract."""

    def test_session_starts_in_created_state(self) -> None:
        """Test new sessions start in CREATED state."""
        session = WebSocketStreamingSession()
        assert session.state == SessionState.CREATED

    @pytest.mark.asyncio
    async def test_start_transitions_to_active(self) -> None:
        """Test start() transitions to ACTIVE state."""
        session = WebSocketStreamingSession()
        await session.start()
        assert session.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_end_transitions_through_ending_to_ended(self) -> None:
        """Test end() transitions through ENDING to ENDED."""
        session = WebSocketStreamingSession()
        await session.start()
        await session.end()
        assert session.state == SessionState.ENDED

    def test_non_recoverable_error_transitions_to_error(self) -> None:
        """Test non-recoverable error transitions to ERROR state."""
        session = WebSocketStreamingSession()
        session.create_error_event("fatal", "Fatal error", recoverable=False)
        assert session.state == SessionState.ERROR

    @pytest.mark.asyncio
    async def test_cannot_process_in_created_state(self) -> None:
        """Test cannot process audio in CREATED state."""
        session = WebSocketStreamingSession()
        with pytest.raises(RuntimeError):
            await session.process_audio_chunk(b"\x00" * 100, sequence=1)

    @pytest.mark.asyncio
    async def test_cannot_process_after_ended(self) -> None:
        """Test cannot process audio after ENDED state."""
        session = WebSocketStreamingSession()
        await session.start()
        await session.end()
        with pytest.raises(RuntimeError):
            await session.process_audio_chunk(b"\x00" * 100, sequence=1)


# =============================================================================
# Ordering Contract Tests
# =============================================================================


class TestOrderingContract:
    """Test event ordering follows documented contract."""

    @pytest.mark.asyncio
    async def test_session_started_is_first_event(self) -> None:
        """Test SESSION_STARTED is always the first event."""
        session = WebSocketStreamingSession()
        event = await session.start()
        assert event.type == ServerMessageType.SESSION_STARTED
        assert event.event_id == 1

    @pytest.mark.asyncio
    async def test_session_ended_is_last_event(self) -> None:
        """Test SESSION_ENDED is always the last event."""
        session = WebSocketStreamingSession()
        await session.start()
        await session.process_audio_chunk(b"\x00" * 32000, sequence=1)
        end_events = await session.end()

        # Last event should be SESSION_ENDED
        assert end_events[-1].type == ServerMessageType.SESSION_ENDED

    @pytest.mark.asyncio
    async def test_finalized_comes_after_partials(self) -> None:
        """Test FINALIZED events come after related PARTIAL events."""
        session = WebSocketStreamingSession()
        await session.start()

        # Send audio to generate events
        await session.process_audio_chunk(b"\x00" * 32000, sequence=1)
        end_events = await session.end()

        # Find segment events
        segment_events = [
            e
            for e in end_events
            if e.type in (ServerMessageType.PARTIAL, ServerMessageType.FINALIZED)
        ]

        # If we have both partial and finalized, partial should come first
        partial_ids = [e.event_id for e in segment_events if e.type == ServerMessageType.PARTIAL]
        finalized_ids = [
            e.event_id for e in segment_events if e.type == ServerMessageType.FINALIZED
        ]

        if partial_ids and finalized_ids:
            assert min(partial_ids) < max(finalized_ids)


# =============================================================================
# Configuration Contract Tests
# =============================================================================


class TestConfigurationContract:
    """Test configuration follows documented defaults and constraints."""

    def test_default_config_matches_documented_defaults(self) -> None:
        """Test default config matches documented values."""
        config = WebSocketSessionConfig()

        # These values should match docs/STREAMING_ARCHITECTURE.md
        assert config.max_gap_sec == 1.0
        assert config.enable_prosody is False
        assert config.enable_emotion is False
        assert config.enable_categorical_emotion is False
        assert config.enable_diarization is False
        assert config.sample_rate == 16000
        assert config.audio_format == "pcm_s16le"
        assert config.replay_buffer_size == 100
        assert config.backpressure_threshold == 80

    def test_config_rejects_invalid_sample_rate(self) -> None:
        """Test config rejects invalid sample rate."""
        with pytest.raises(ValueError):
            WebSocketSessionConfig.from_dict({"sample_rate": 0})

        with pytest.raises(ValueError):
            WebSocketSessionConfig.from_dict({"sample_rate": -1})


# =============================================================================
# Replay Buffer Contract Tests
# =============================================================================


class TestReplayBufferContract:
    """Test replay buffer for resume capability."""

    def test_replay_buffer_stores_events(self) -> None:
        """Test replay buffer stores events correctly."""
        from transcription.streaming_ws import ReplayBuffer

        buffer = ReplayBuffer(max_size=10)

        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-test",
            type=ServerMessageType.PARTIAL,
            ts_server=1000,
            payload={},
        )
        buffer.add(envelope)

        assert len(buffer.events) == 1
        assert buffer.events[0].event_id == 1

    def test_replay_buffer_respects_max_size(self) -> None:
        """Test replay buffer drops oldest events when full."""
        from transcription.streaming_ws import ReplayBuffer

        buffer = ReplayBuffer(max_size=3)

        for i in range(5):
            envelope = EventEnvelope(
                event_id=i + 1,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000,
                payload={},
            )
            buffer.add(envelope)

        assert len(buffer.events) == 3
        assert buffer.events[0].event_id == 3
        assert buffer.events[-1].event_id == 5
        assert buffer.oldest_event_id == 3

    def test_get_events_since_returns_newer_events(self) -> None:
        """Test getting events since a given event ID."""
        from transcription.streaming_ws import ReplayBuffer

        buffer = ReplayBuffer(max_size=10)

        for i in range(5):
            envelope = EventEnvelope(
                event_id=i + 1,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000,
                payload={},
            )
            buffer.add(envelope)

        events, gap = buffer.get_events_since(3)

        assert len(events) == 2
        assert events[0].event_id == 4
        assert events[1].event_id == 5
        assert gap is False

    def test_get_events_since_detects_gap(self) -> None:
        """Test gap detection when events are lost."""
        from transcription.streaming_ws import ReplayBuffer

        buffer = ReplayBuffer(max_size=3)

        # Add events 1-5, buffer only keeps 3-5
        for i in range(5):
            envelope = EventEnvelope(
                event_id=i + 1,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000,
                payload={},
            )
            buffer.add(envelope)

        # Client requesting from event 1 will have a gap
        events, gap = buffer.get_events_since(1)

        assert gap is True  # Events 2 were lost
        assert len(events) == 3  # Can only provide 3, 4, 5


# =============================================================================
# Backpressure Contract Tests
# =============================================================================


class TestBackpressureContract:
    """Test backpressure handling."""

    @pytest.mark.asyncio
    async def test_check_backpressure_returns_false_initially(self) -> None:
        """Test backpressure is not active initially."""
        session = WebSocketStreamingSession()
        await session.start()

        assert session.check_backpressure() is False

    @pytest.mark.asyncio
    async def test_drop_partial_preserves_finalized(self) -> None:
        """Test dropping partials preserves FINALIZED events."""
        session = WebSocketStreamingSession()
        await session.start()

        # Manually add events to queue
        partial = session._create_envelope(
            ServerMessageType.PARTIAL,
            {"segment": {"text": "partial"}},
        )
        finalized = session._create_envelope(
            ServerMessageType.FINALIZED,
            {"segment": {"text": "final"}},
        )

        session._event_queue.put_nowait(partial)
        session._event_queue.put_nowait(finalized)

        dropped = session.drop_partial_events()

        assert dropped == 1
        assert session._event_queue.qsize() == 1

        # Remaining event should be FINALIZED
        remaining = session._event_queue.get_nowait()
        assert remaining.type == ServerMessageType.FINALIZED


# =============================================================================
# Resume Protocol Contract Tests
# =============================================================================


class TestResumeProtocolContract:
    """Test resume protocol for reconnection."""

    @pytest.mark.asyncio
    async def test_get_events_for_resume(self) -> None:
        """Test getting events for resume."""
        session = WebSocketStreamingSession()
        await session.start()

        # Generate some events
        session.create_pong_event(0)
        session.create_pong_event(0)
        session.create_pong_event(0)

        # Resume from event 2
        events, gap = session.get_events_for_resume(2)

        assert len(events) == 2  # Events 3 and 4
        assert gap is False
        assert session.stats.resume_attempts == 1

    @pytest.mark.asyncio
    async def test_resume_gap_error(self) -> None:
        """Test resume gap error creation."""
        config = WebSocketSessionConfig(replay_buffer_size=2)
        session = WebSocketStreamingSession(config=config)
        await session.start()

        # Generate events to fill and overflow buffer
        for _ in range(5):
            session.create_pong_event(0)

        # Try to resume from very old event
        events, gap = session.get_events_for_resume(1)
        assert gap is True

        # Create gap error
        error = session.create_resume_gap_error(1)
        assert error.type == ServerMessageType.ERROR
        assert error.payload["code"] == "RESUME_GAP"
        assert error.payload["recoverable"] is False

    @pytest.mark.asyncio
    async def test_buffer_overflow_error(self) -> None:
        """Test buffer overflow error creation."""
        session = WebSocketStreamingSession()
        await session.start()

        error = session.create_buffer_overflow_error()

        assert error.type == ServerMessageType.ERROR
        assert error.payload["code"] == "BUFFER_OVERFLOW"
        assert error.payload["recoverable"] is True
        assert session.stats.errors == 1


# =============================================================================
# RESUME_SESSION Message Parsing Tests
# =============================================================================


class TestResumeSessionMessageParsing:
    """Test parsing of RESUME_SESSION client messages."""

    def test_parse_resume_session_message(self) -> None:
        """Test parsing a valid RESUME_SESSION message."""
        data = {
            "type": "RESUME_SESSION",
            "session_id": "str-12345678-1234-4567-89ab-123456789abc",
            "last_event_id": 15,
        }
        msg_type, payload = parse_client_message(data)

        assert msg_type == ClientMessageType.RESUME_SESSION
        assert payload["session_id"] == "str-12345678-1234-4567-89ab-123456789abc"
        assert payload["last_event_id"] == 15

    def test_resume_session_enum_value(self) -> None:
        """Test RESUME_SESSION enum value exists."""
        assert ClientMessageType.RESUME_SESSION.value == "RESUME_SESSION"


# =============================================================================
# Resume Protocol Integration Tests
# =============================================================================


class TestResumeProtocolIntegration:
    """Integration tests for resume protocol behavior."""

    @pytest.mark.asyncio
    async def test_resume_with_valid_last_event_id(self) -> None:
        """Test successful resume with valid last_event_id."""
        session = WebSocketStreamingSession()
        await session.start()

        # Generate some events
        session.create_pong_event(100)  # event 2
        session.create_pong_event(200)  # event 3
        session.create_pong_event(300)  # event 4

        # Resume from event 2 should return events 3 and 4
        events, gap = session.get_events_for_resume(2)

        assert gap is False
        assert len(events) == 2
        assert events[0].event_id == 3
        assert events[1].event_id == 4

    @pytest.mark.asyncio
    async def test_resume_with_current_event_id(self) -> None:
        """Test resume when client is already up-to-date."""
        session = WebSocketStreamingSession()
        await session.start()

        # Generate some events
        session.create_pong_event(100)  # event 2
        session.create_pong_event(200)  # event 3

        # Resume from event 3 (current) should return empty
        events, gap = session.get_events_for_resume(3)

        assert gap is False
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_resume_gap_detection_with_small_buffer(self) -> None:
        """Test gap detection when buffer is too small."""
        config = WebSocketSessionConfig(replay_buffer_size=3)
        session = WebSocketStreamingSession(config=config)
        await session.start()  # event 1

        # Generate events 2-7 (6 events, buffer only holds 3)
        for i in range(6):
            session.create_pong_event(i * 100)

        # Buffer should only have events 5, 6, 7
        # Resume from event 2 should detect gap
        events, gap = session.get_events_for_resume(2)

        assert gap is True
        # Should still return what we have (events newer than 2)
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_resume_increments_stats(self) -> None:
        """Test that resume attempts are tracked in stats."""
        session = WebSocketStreamingSession()
        await session.start()

        assert session.stats.resume_attempts == 0

        session.get_events_for_resume(0)
        assert session.stats.resume_attempts == 1

        session.get_events_for_resume(0)
        assert session.stats.resume_attempts == 2

    @pytest.mark.asyncio
    async def test_resume_gap_error_format(self) -> None:
        """Test RESUME_GAP error has correct format."""
        config = WebSocketSessionConfig(replay_buffer_size=2)
        session = WebSocketStreamingSession(config=config)
        await session.start()

        # Fill buffer
        for _ in range(5):
            session.create_pong_event(0)

        error = session.create_resume_gap_error(1)

        assert error.type == ServerMessageType.ERROR
        assert error.payload["code"] == "RESUME_GAP"
        assert error.payload["recoverable"] is False
        assert "details" in error.payload
        assert error.payload["details"]["requested_event_id"] == 1
        assert "oldest_available" in error.payload["details"]
        assert "current_event_id" in error.payload["details"]


# =============================================================================
# Backpressure Integration Tests
# =============================================================================


class TestBackpressureIntegration:
    """Integration tests for backpressure handling."""

    @pytest.mark.asyncio
    async def test_backpressure_activates_at_threshold(self) -> None:
        """Test backpressure activates when queue reaches threshold."""
        config = WebSocketSessionConfig(backpressure_threshold=5)
        session = WebSocketStreamingSession(config=config)
        await session.start()

        # Initially no backpressure
        assert session.check_backpressure() is False
        assert session._backpressure_active is False

        # Fill queue to threshold
        for _ in range(5):
            partial = session._create_envelope(
                ServerMessageType.PARTIAL,
                {"segment": {"text": "test"}},
            )
            session._event_queue.put_nowait(partial)

        # Now backpressure should be active
        assert session.check_backpressure() is True
        assert session._backpressure_active is True
        assert session.stats.backpressure_events == 1

    @pytest.mark.asyncio
    async def test_backpressure_deactivates_at_low_watermark(self) -> None:
        """Test backpressure deactivates when queue drains to 50%."""
        config = WebSocketSessionConfig(backpressure_threshold=10)
        session = WebSocketStreamingSession(config=config)
        await session.start()

        # Fill queue to threshold
        for _ in range(10):
            partial = session._create_envelope(
                ServerMessageType.PARTIAL,
                {"segment": {"text": "test"}},
            )
            session._event_queue.put_nowait(partial)

        # Activate backpressure
        session.check_backpressure()
        assert session._backpressure_active is True

        # Drain queue to below 50% (5 events)
        session._event_queue = __import__("asyncio").Queue(maxsize=100)
        for _ in range(4):  # 4 events < 50% of 10
            partial = session._create_envelope(
                ServerMessageType.PARTIAL,
                {"segment": {"text": "test"}},
            )
            session._event_queue.put_nowait(partial)

        # Check backpressure - should deactivate
        session.check_backpressure()
        assert session._backpressure_active is False

    @pytest.mark.asyncio
    async def test_drop_partial_preserves_critical_events(self) -> None:
        """Test that dropping partials preserves FINALIZED, ERROR, SESSION_ENDED."""
        session = WebSocketStreamingSession()
        await session.start()

        # Add mix of event types
        partial1 = session._create_envelope(ServerMessageType.PARTIAL, {"segment": {"text": "p1"}})
        finalized = session._create_envelope(
            ServerMessageType.FINALIZED, {"segment": {"text": "final"}}
        )
        partial2 = session._create_envelope(ServerMessageType.PARTIAL, {"segment": {"text": "p2"}})
        error = session._create_envelope(
            ServerMessageType.ERROR,
            {"code": "test", "message": "test", "recoverable": True},
        )
        partial3 = session._create_envelope(ServerMessageType.PARTIAL, {"segment": {"text": "p3"}})
        ended = session._create_envelope(ServerMessageType.SESSION_ENDED, {"stats": {}})

        session._event_queue.put_nowait(partial1)
        session._event_queue.put_nowait(finalized)
        session._event_queue.put_nowait(partial2)
        session._event_queue.put_nowait(error)
        session._event_queue.put_nowait(partial3)
        session._event_queue.put_nowait(ended)

        dropped = session.drop_partial_events()

        assert dropped == 3  # Three partials dropped
        assert session._event_queue.qsize() == 3  # Three critical events remain
        assert session.stats.events_dropped == 3

        # Verify remaining events are the critical ones
        remaining_types = []
        while not session._event_queue.empty():
            event = session._event_queue.get_nowait()
            remaining_types.append(event.type)

        assert ServerMessageType.FINALIZED in remaining_types
        assert ServerMessageType.ERROR in remaining_types
        assert ServerMessageType.SESSION_ENDED in remaining_types

    @pytest.mark.asyncio
    async def test_buffer_overflow_error_includes_details(self) -> None:
        """Test BUFFER_OVERFLOW error includes useful details."""
        session = WebSocketStreamingSession()
        await session.start()

        # Simulate some dropped events
        session.stats.events_dropped = 15

        # Add some events to queue
        for _ in range(5):
            partial = session._create_envelope(
                ServerMessageType.PARTIAL, {"segment": {"text": "test"}}
            )
            session._event_queue.put_nowait(partial)

        error = session.create_buffer_overflow_error()

        assert error.payload["code"] == "BUFFER_OVERFLOW"
        assert error.payload["recoverable"] is True
        assert "details" in error.payload
        assert error.payload["details"]["events_dropped"] == 15
        assert error.payload["details"]["queue_size"] == 5


# =============================================================================
# Replay Buffer Edge Cases
# =============================================================================


class TestReplayBufferEdgeCases:
    """Test edge cases in replay buffer behavior."""

    def test_empty_buffer_returns_empty_list(self) -> None:
        """Test getting events from empty buffer."""
        buffer = ReplayBuffer(max_size=10)
        events, gap = buffer.get_events_since(0)

        assert events == []
        assert gap is False

    def test_buffer_oldest_event_id_tracking(self) -> None:
        """Test that oldest_event_id is tracked correctly."""
        buffer = ReplayBuffer(max_size=3)

        # Add events 1-5
        for i in range(5):
            envelope = EventEnvelope(
                event_id=i + 1,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000,
                payload={},
            )
            buffer.add(envelope)

        # oldest_event_id should be 3 (events 1, 2 were evicted)
        assert buffer.oldest_event_id == 3
        assert len(buffer.events) == 3
        assert buffer.events[0].event_id == 3
        assert buffer.events[-1].event_id == 5

    def test_buffer_clear_resets_state(self) -> None:
        """Test that clear() resets buffer state."""
        buffer = ReplayBuffer(max_size=10)

        for i in range(5):
            envelope = EventEnvelope(
                event_id=i + 1,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000,
                payload={},
            )
            buffer.add(envelope)

        buffer.clear()

        assert len(buffer.events) == 0
        assert buffer.oldest_event_id == 0

    def test_get_events_since_filters_correctly(self) -> None:
        """Test that get_events_since only returns events newer than threshold."""
        buffer = ReplayBuffer(max_size=10)

        for i in range(5):
            envelope = EventEnvelope(
                event_id=i + 1,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000,
                payload={},
            )
            buffer.add(envelope)

        # Get events since event 3
        events, gap = buffer.get_events_since(3)

        assert len(events) == 2
        assert events[0].event_id == 4
        assert events[1].event_id == 5
        assert gap is False

    def test_get_events_since_zero_returns_all(self) -> None:
        """Test that last_event_id=0 returns all events."""
        buffer = ReplayBuffer(max_size=10)

        for i in range(3):
            envelope = EventEnvelope(
                event_id=i + 1,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000,
                payload={},
            )
            buffer.add(envelope)

        events, gap = buffer.get_events_since(0)

        assert len(events) == 3
        assert gap is False


# =============================================================================
# WebSocket Handler Resume Protocol Tests (Mock-based)
# =============================================================================


class TestWebSocketHandlerResumeProtocol:
    """Tests for WebSocket handler resume protocol behavior."""

    @pytest.mark.asyncio
    async def test_session_mismatch_error(self) -> None:
        """Test that mismatched session IDs produce correct error."""
        session = WebSocketStreamingSession()
        await session.start()

        # Simulate requesting a different session
        requested_id = "str-different-session-id"
        current_id = session.stream_id

        assert requested_id != current_id

        # The handler would check this and create error
        error = session.create_error_event(
            code="session_mismatch",
            message=f"Session ID mismatch: requested '{requested_id}' but current session is '{current_id}'",
            recoverable=False,
        )

        assert error.type == ServerMessageType.ERROR
        assert error.payload["code"] == "session_mismatch"
        assert error.payload["recoverable"] is False

    @pytest.mark.asyncio
    async def test_resume_without_session_error(self) -> None:
        """Test resume attempt without active session."""
        temp_session = WebSocketStreamingSession()

        # The handler would create this error when no session exists
        error = temp_session.create_error_event(
            code="no_session",
            message="No active session to resume. Send START_SESSION first.",
            recoverable=True,
        )

        assert error.type == ServerMessageType.ERROR
        assert error.payload["code"] == "no_session"
        assert error.payload["recoverable"] is True


# =============================================================================
# Event Ordering Guarantees Tests (Issue #222)
# =============================================================================


class TestEventOrderingGuarantees:
    """Test event ordering guarantees per streaming protocol contract."""

    @pytest.mark.asyncio
    async def test_event_ids_are_monotonically_increasing(self) -> None:
        """Test that event_id always increases (never decreases or repeats)."""
        session = WebSocketStreamingSession()
        start_event = await session.start()

        all_event_ids = [start_event.event_id]

        # Generate various event types
        pong_event = session.create_pong_event(100)
        all_event_ids.append(pong_event.event_id)

        error_event = session.create_error_event("test", "test error", recoverable=True)
        all_event_ids.append(error_event.event_id)

        # Process audio chunks
        audio_data = b"\x00" * 32000
        for i in range(3):
            events = await session.process_audio_chunk(audio_data, sequence=i + 1)
            for event in events:
                all_event_ids.append(event.event_id)

        # End session
        end_events = await session.end()
        for event in end_events:
            all_event_ids.append(event.event_id)

        # Verify strict monotonic increase
        for i in range(len(all_event_ids) - 1):
            assert all_event_ids[i] < all_event_ids[i + 1], (
                f"Event IDs not strictly monotonic at index {i}: "
                f"{all_event_ids[i]} >= {all_event_ids[i + 1]}"
            )

    @pytest.mark.asyncio
    async def test_finalized_events_follow_partial_events(self) -> None:
        """Test that FINALIZED for a segment comes after all its PARTIAL events."""
        session = WebSocketStreamingSession()
        await session.start()

        # Process enough audio to generate segments
        audio_data = b"\x00" * 32000  # 2 seconds at 16kHz
        events = await session.process_audio_chunk(audio_data, sequence=1)

        # End session to finalize
        end_events = await session.end()

        # Collect all segment events by segment_id
        segment_events: dict[str | None, list[EventEnvelope]] = {}
        for event in events + end_events:
            if event.type in (ServerMessageType.PARTIAL, ServerMessageType.FINALIZED):
                seg_id = event.segment_id
                if seg_id not in segment_events:
                    segment_events[seg_id] = []
                segment_events[seg_id].append(event)

        # For each segment, verify FINALIZED comes after all PARTIALs
        for seg_id, seg_events in segment_events.items():
            partial_ids = [e.event_id for e in seg_events if e.type == ServerMessageType.PARTIAL]
            finalized_ids = [
                e.event_id for e in seg_events if e.type == ServerMessageType.FINALIZED
            ]

            if partial_ids and finalized_ids:
                # All PARTIAL event_ids should be less than all FINALIZED event_ids
                assert max(partial_ids) < min(finalized_ids), (
                    f"Segment {seg_id}: PARTIAL event_ids {partial_ids} "
                    f"not all before FINALIZED event_ids {finalized_ids}"
                )

    @pytest.mark.asyncio
    async def test_session_ended_is_last_event(self) -> None:
        """Test that SESSION_ENDED is always the final event in a session."""
        session = WebSocketStreamingSession()
        await session.start()

        # Process some audio
        audio_data = b"\x00" * 32000
        await session.process_audio_chunk(audio_data, sequence=1)

        # End session
        end_events = await session.end()

        # The last event must be SESSION_ENDED
        assert len(end_events) > 0
        assert end_events[-1].type == ServerMessageType.SESSION_ENDED

        # SESSION_ENDED should have the highest event_id
        session_ended_id = end_events[-1].event_id
        for event in end_events[:-1]:
            assert event.event_id < session_ended_id, (
                f"Event {event.type} (id={event.event_id}) has id >= SESSION_ENDED (id={session_ended_id})"
            )

    @pytest.mark.asyncio
    async def test_no_events_after_session_ended(self) -> None:
        """Test that no events can be generated after SESSION_ENDED."""
        session = WebSocketStreamingSession()
        await session.start()

        # End the session
        end_events = await session.end()

        # Verify SESSION_ENDED was sent
        assert end_events[-1].type == ServerMessageType.SESSION_ENDED
        assert session.state == SessionState.ENDED

        # Attempting to process audio should fail
        with pytest.raises(RuntimeError, match="Cannot process chunk in state"):
            await session.process_audio_chunk(b"\x00" * 100, sequence=1)

        # Attempting to end again should fail
        with pytest.raises(RuntimeError, match="Cannot end session in state"):
            await session.end()


# =============================================================================
# Backpressure Behavior Tests (Issue #222)
# =============================================================================


class TestBackpressureBehavior:
    """Test backpressure handling per streaming protocol contract."""

    @pytest.mark.asyncio
    async def test_partial_events_dropped_under_backpressure(self) -> None:
        """Test that PARTIAL events are dropped when queue is full."""
        session = WebSocketStreamingSession()
        await session.start()

        # Fill the queue with PARTIAL events
        for i in range(10):
            partial = session._create_envelope(
                ServerMessageType.PARTIAL,
                {"segment": {"text": f"partial-{i}"}},
                segment_id=f"seg-{i}",
            )
            session._event_queue.put_nowait(partial)

        initial_queue_size = session._event_queue.qsize()
        assert initial_queue_size == 10

        # Drop partial events
        dropped = session.drop_partial_events()

        # All should be dropped
        assert dropped == 10
        assert session._event_queue.qsize() == 0
        assert session.stats.events_dropped == 10

    @pytest.mark.asyncio
    async def test_finalized_events_never_dropped(self) -> None:
        """Test that FINALIZED events survive backpressure drops."""
        session = WebSocketStreamingSession()
        await session.start()

        # Add mix of PARTIAL and FINALIZED events
        for i in range(5):
            session._event_queue.put_nowait(
                session._create_envelope(
                    ServerMessageType.PARTIAL,
                    {"segment": {"text": f"partial-{i}"}},
                )
            )

        # Add FINALIZED event in the middle
        finalized = session._create_envelope(
            ServerMessageType.FINALIZED,
            {"segment": {"text": "finalized"}},
            segment_id="seg-final",
        )
        session._event_queue.put_nowait(finalized)

        # Add more PARTIAL events
        for i in range(5, 8):
            session._event_queue.put_nowait(
                session._create_envelope(
                    ServerMessageType.PARTIAL,
                    {"segment": {"text": f"partial-{i}"}},
                )
            )

        initial_size = session._event_queue.qsize()
        assert initial_size == 9  # 8 PARTIAL + 1 FINALIZED

        # Drop partial events
        dropped = session.drop_partial_events()

        assert dropped == 8  # All PARTIAL dropped
        assert session._event_queue.qsize() == 1  # Only FINALIZED remains

        # Verify it's the FINALIZED event
        remaining = session._event_queue.get_nowait()
        assert remaining.type == ServerMessageType.FINALIZED

    @pytest.mark.asyncio
    async def test_error_events_never_dropped(self) -> None:
        """Test that ERROR events survive backpressure drops."""
        session = WebSocketStreamingSession()
        await session.start()

        # Add PARTIAL events
        for i in range(5):
            session._event_queue.put_nowait(
                session._create_envelope(
                    ServerMessageType.PARTIAL,
                    {"segment": {"text": f"partial-{i}"}},
                )
            )

        # Add ERROR event
        error = session._create_envelope(
            ServerMessageType.ERROR,
            {"code": "TEST_ERROR", "message": "Test error", "recoverable": True},
        )
        session._event_queue.put_nowait(error)

        # Drop partial events
        dropped = session.drop_partial_events()

        assert dropped == 5
        assert session._event_queue.qsize() == 1

        # Verify it's the ERROR event
        remaining = session._event_queue.get_nowait()
        assert remaining.type == ServerMessageType.ERROR
        assert remaining.payload["code"] == "TEST_ERROR"

    @pytest.mark.asyncio
    async def test_session_ended_never_dropped(self) -> None:
        """Test that SESSION_ENDED events survive backpressure drops."""
        session = WebSocketStreamingSession()
        await session.start()

        # Add PARTIAL events
        for i in range(5):
            session._event_queue.put_nowait(
                session._create_envelope(
                    ServerMessageType.PARTIAL,
                    {"segment": {"text": f"partial-{i}"}},
                )
            )

        # Add SESSION_ENDED event
        ended = session._create_envelope(
            ServerMessageType.SESSION_ENDED,
            {"stats": {"total_events": 10}},
        )
        session._event_queue.put_nowait(ended)

        # Drop partial events
        dropped = session.drop_partial_events()

        assert dropped == 5
        assert session._event_queue.qsize() == 1

        # Verify it's the SESSION_ENDED event
        remaining = session._event_queue.get_nowait()
        assert remaining.type == ServerMessageType.SESSION_ENDED


# =============================================================================
# Resume Protocol Tests (Issue #222)
# =============================================================================


class TestResumeProtocol:
    """Test resume protocol for reconnection handling."""

    @pytest.mark.asyncio
    async def test_resume_session_replays_events_from_last_event_id(self) -> None:
        """Test that events after last_event_id are replayed on resume."""
        session = WebSocketStreamingSession()
        await session.start()  # event_id=1

        # Generate several events
        session.create_pong_event(100)  # event_id=2
        session.create_pong_event(200)  # event_id=3
        session.create_pong_event(300)  # event_id=4
        session.create_pong_event(400)  # event_id=5

        # Resume from event_id=2 (should replay events 3, 4, 5)
        events, gap = session.get_events_for_resume(2)

        assert gap is False
        assert len(events) == 3
        assert events[0].event_id == 3
        assert events[1].event_id == 4
        assert events[2].event_id == 5

    @pytest.mark.asyncio
    async def test_resume_with_invalid_session_id_returns_error(self) -> None:
        """Test that resuming with wrong session_id returns SESSION_MISMATCH error."""
        session = WebSocketStreamingSession()
        await session.start()

        # Simulate handler checking session_id mismatch
        requested_session_id = "str-wrong-session-id"
        actual_session_id = session.stream_id

        # This is how the handler would create the error
        error = session.create_error_event(
            code="SESSION_MISMATCH",
            message=f"Session ID mismatch: requested {requested_session_id}, current is {actual_session_id}",
            recoverable=False,
        )

        assert error.type == ServerMessageType.ERROR
        assert error.payload["code"] == "SESSION_MISMATCH"
        assert error.payload["recoverable"] is False

    @pytest.mark.asyncio
    async def test_resume_with_gap_returns_resume_gap_error(self) -> None:
        """Test that if events were purged from buffer, RESUME_GAP error is returned."""
        # Use small buffer to force eviction
        config = WebSocketSessionConfig(replay_buffer_size=3)
        session = WebSocketStreamingSession(config=config)
        await session.start()  # event_id=1

        # Generate many events to overflow buffer
        for i in range(10):
            session.create_pong_event(i * 100)

        # Buffer should only have last 3 events (9, 10, 11)
        # Trying to resume from event_id=1 should detect gap
        events, gap = session.get_events_for_resume(1)

        assert gap is True

        # Create the gap error
        error = session.create_resume_gap_error(1)

        assert error.type == ServerMessageType.ERROR
        assert error.payload["code"] == "RESUME_GAP"
        assert error.payload["recoverable"] is False
        assert error.payload["details"]["requested_event_id"] == 1

    @pytest.mark.asyncio
    async def test_resume_continues_transcription(self) -> None:
        """Test that after resume, new audio continues normally."""
        session = WebSocketStreamingSession()
        await session.start()

        # Process some audio before "disconnect"
        audio_data = b"\x00" * 32000
        await session.process_audio_chunk(audio_data, sequence=1)

        # Simulate resume - get events since last known
        last_event_id = session._event_id_counter
        events, gap = session.get_events_for_resume(last_event_id)

        # No new events since we haven't processed anything
        assert len(events) == 0
        assert gap is False

        # Continue processing audio after resume
        events_after = await session.process_audio_chunk(audio_data, sequence=2)

        # New events should have higher event_ids
        for event in events_after:
            assert event.event_id > last_event_id


# =============================================================================
# Replay Buffer Contract Tests (Issue #222)
# =============================================================================


class TestReplayBufferContracts:
    """Test replay buffer behavior contracts."""

    def test_replay_buffer_stores_events_up_to_capacity(self) -> None:
        """Test that buffer stores events up to max_size."""
        buffer = ReplayBuffer(max_size=5)

        # Add exactly max_size events
        for i in range(5):
            envelope = EventEnvelope(
                event_id=i + 1,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000 + i,
                payload={},
            )
            buffer.add(envelope)

        assert len(buffer.events) == 5
        assert buffer.events[0].event_id == 1
        assert buffer.events[-1].event_id == 5

    def test_replay_buffer_evicts_oldest_on_overflow(self) -> None:
        """Test circular buffer behavior - oldest events evicted first."""
        buffer = ReplayBuffer(max_size=3)

        # Add 5 events to a buffer of size 3
        for i in range(5):
            envelope = EventEnvelope(
                event_id=i + 1,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000 + i,
                payload={"index": i},
            )
            buffer.add(envelope)

        # Should only have events 3, 4, 5
        assert len(buffer.events) == 3
        assert buffer.events[0].event_id == 3
        assert buffer.events[1].event_id == 4
        assert buffer.events[2].event_id == 5
        assert buffer.oldest_event_id == 3

    def test_replay_buffer_gap_detection(self) -> None:
        """Test that buffer detects when requested event_id is too old."""
        buffer = ReplayBuffer(max_size=3)

        # Add events 1-5, buffer keeps only 3-5
        for i in range(5):
            envelope = EventEnvelope(
                event_id=i + 1,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000,
                payload={},
            )
            buffer.add(envelope)

        # Request events since id=1 (which was evicted)
        events, gap = buffer.get_events_since(1)

        # Gap should be detected
        assert gap is True
        # Should still return available events
        assert len(events) == 3
        assert events[0].event_id == 3

    def test_replay_buffer_no_gap_for_recent_events(self) -> None:
        """Test no gap detection when requested event is in buffer."""
        buffer = ReplayBuffer(max_size=5)

        for i in range(5):
            envelope = EventEnvelope(
                event_id=i + 1,
                stream_id="str-test",
                type=ServerMessageType.PARTIAL,
                ts_server=1000,
                payload={},
            )
            buffer.add(envelope)

        # Request events since id=3 (which is in buffer)
        events, gap = buffer.get_events_since(3)

        assert gap is False
        assert len(events) == 2  # Events 4 and 5
        assert events[0].event_id == 4
        assert events[1].event_id == 5


# =============================================================================
# Session Lifecycle Contract Tests (Issue #222)
# =============================================================================


class TestSessionLifecycleContracts:
    """Test session lifecycle state transitions."""

    @pytest.mark.asyncio
    async def test_session_transitions_created_to_active_on_start(self) -> None:
        """Test session transitions from CREATED to ACTIVE on start()."""
        session = WebSocketStreamingSession()
        assert session.state == SessionState.CREATED

        await session.start()

        assert session.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_session_transitions_active_to_ending_on_end(self) -> None:
        """Test session transitions through ENDING state on end()."""
        session = WebSocketStreamingSession()
        await session.start()
        assert session.state == SessionState.ACTIVE

        # End transitions through ENDING to ENDED
        await session.end()
        assert session.state == SessionState.ENDED

    @pytest.mark.asyncio
    async def test_session_transitions_ending_to_ended_after_final_events(self) -> None:
        """Test that end() returns SESSION_ENDED as the final event."""
        session = WebSocketStreamingSession()
        await session.start()

        # Process some audio so there's something to finalize
        await session.process_audio_chunk(b"\x00" * 32000, sequence=1)

        # End session
        end_events = await session.end()

        # State should be ENDED
        assert session.state == SessionState.ENDED

        # Last event should be SESSION_ENDED
        assert len(end_events) > 0
        assert end_events[-1].type == ServerMessageType.SESSION_ENDED

    def test_non_recoverable_error_transitions_to_error_state(self) -> None:
        """Test that non-recoverable error transitions to ERROR state."""
        session = WebSocketStreamingSession()
        assert session.state == SessionState.CREATED

        # Create non-recoverable error
        error = session.create_error_event(
            code="FATAL_ERROR",
            message="Fatal error occurred",
            recoverable=False,
        )

        assert session.state == SessionState.ERROR
        assert error.payload["recoverable"] is False

    @pytest.mark.asyncio
    async def test_recoverable_error_does_not_change_state(self) -> None:
        """Test that recoverable error does not change session state."""
        session = WebSocketStreamingSession()
        await session.start()

        initial_state = session.state
        assert initial_state == SessionState.ACTIVE

        # Create recoverable error
        error = session.create_error_event(
            code="TEMP_ERROR",
            message="Temporary error",
            recoverable=True,
        )

        # State should remain ACTIVE
        assert session.state == SessionState.ACTIVE
        assert error.payload["recoverable"] is True


# =============================================================================
# Session Registry Disconnection Tests (Issue #222)
# =============================================================================


class TestSessionRegistryDisconnection:
    """Test session registry disconnection and reconnection handling."""

    @pytest.fixture
    def registry(self):
        """Create a fresh session registry for testing."""
        from transcription.session_registry import SessionRegistry

        # Reset singleton
        SessionRegistry.reset()
        reg = SessionRegistry()
        reg.configure(
            idle_timeout_sec=60.0,
            disconnected_ttl_sec=10.0,  # Short TTL for testing
            cleanup_interval_sec=5.0,
        )
        yield reg
        SessionRegistry.reset()

    @pytest.mark.asyncio
    async def test_disconnected_session_can_resume_within_ttl(self, registry) -> None:
        """Test that a disconnected session can resume within TTL window."""
        from transcription.session_registry import SessionStatus

        session = WebSocketStreamingSession()
        await session.start()

        # Register session
        session_id = registry.register(session)
        registry.mark_active(session_id)

        # Verify active
        info = registry.get_info(session_id)
        assert info is not None
        assert info.status == SessionStatus.ACTIVE

        # Mark disconnected
        registry.mark_disconnected(session_id)

        info = registry.get_info(session_id)
        assert info is not None
        assert info.status == SessionStatus.DISCONNECTED

        # Session should still be retrievable for resume
        registered = registry.get(session_id)
        assert registered is not None
        assert registered.session is session

        # Simulate reconnection
        mock_websocket = object()  # Dummy websocket
        registry.update_websocket(session_id, mock_websocket)

        info = registry.get_info(session_id)
        assert info is not None
        assert info.status == SessionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_disconnected_session_expires_after_ttl(self, registry) -> None:
        """Test that disconnected session expires after TTL."""
        import time

        session = WebSocketStreamingSession()
        await session.start()

        # Register and activate
        session_id = registry.register(session)
        registry.mark_active(session_id)

        # Mark disconnected
        registry.mark_disconnected(session_id)

        # Manipulate disconnected_at to simulate TTL expiry
        registered = registry.get(session_id)
        assert registered is not None
        registered.disconnected_at = time.time() - 15.0  # 15 seconds ago, TTL is 10

        # Run cleanup
        cleaned = await registry.cleanup_expired_sessions()

        # Session should be cleaned up
        assert cleaned == 1
        assert registry.get(session_id) is None


# =============================================================================
# Comprehensive Integration Tests (Issue #222)
# =============================================================================


class TestStreamingProtocolIntegration:
    """Integration tests for the complete streaming protocol."""

    @pytest.mark.asyncio
    async def test_full_session_lifecycle_event_ordering(self) -> None:
        """Test complete session lifecycle with proper event ordering."""
        session = WebSocketStreamingSession()

        all_events: list[EventEnvelope] = []

        # Start session
        start_event = await session.start()
        all_events.append(start_event)
        assert start_event.type == ServerMessageType.SESSION_STARTED

        # Process multiple audio chunks
        audio_data = b"\x00" * 32000
        for i in range(3):
            events = await session.process_audio_chunk(audio_data, sequence=i + 1)
            all_events.extend(events)

        # End session
        end_events = await session.end()
        all_events.extend(end_events)

        # Verify event ordering
        event_ids = [e.event_id for e in all_events]

        # All event IDs strictly increasing
        for i in range(len(event_ids) - 1):
            assert event_ids[i] < event_ids[i + 1]

        # First event is SESSION_STARTED
        assert all_events[0].type == ServerMessageType.SESSION_STARTED

        # Last event is SESSION_ENDED
        assert all_events[-1].type == ServerMessageType.SESSION_ENDED

    @pytest.mark.asyncio
    async def test_backpressure_preserves_critical_events_ordering(self) -> None:
        """Test that critical events maintain ordering after backpressure drops."""
        session = WebSocketStreamingSession()
        await session.start()

        # Add events in specific order
        events_added: list[tuple[int, ServerMessageType]] = []

        # Add some partials
        for i in range(3):
            partial = session._create_envelope(
                ServerMessageType.PARTIAL,
                {"segment": {"text": f"partial-{i}"}},
            )
            session._event_queue.put_nowait(partial)
            events_added.append((partial.event_id, partial.type))

        # Add finalized
        finalized = session._create_envelope(
            ServerMessageType.FINALIZED,
            {"segment": {"text": "final"}},
        )
        session._event_queue.put_nowait(finalized)
        events_added.append((finalized.event_id, finalized.type))

        # Add more partials
        for i in range(3, 5):
            partial = session._create_envelope(
                ServerMessageType.PARTIAL,
                {"segment": {"text": f"partial-{i}"}},
            )
            session._event_queue.put_nowait(partial)
            events_added.append((partial.event_id, partial.type))

        # Add error
        error = session._create_envelope(
            ServerMessageType.ERROR,
            {"code": "test", "message": "test", "recoverable": True},
        )
        session._event_queue.put_nowait(error)
        events_added.append((error.event_id, error.type))

        # Drop partials
        dropped = session.drop_partial_events()
        assert dropped == 5  # All 5 partials dropped

        # Remaining events should maintain their relative ordering
        remaining = []
        while not session._event_queue.empty():
            event = session._event_queue.get_nowait()
            remaining.append(event)

        assert len(remaining) == 2  # FINALIZED and ERROR

        # Verify order preserved (FINALIZED before ERROR by event_id)
        assert remaining[0].type == ServerMessageType.FINALIZED
        assert remaining[1].type == ServerMessageType.ERROR
        assert remaining[0].event_id < remaining[1].event_id

    @pytest.mark.asyncio
    async def test_replay_buffer_integration_with_session(self) -> None:
        """Test replay buffer integration with session event generation."""
        config = WebSocketSessionConfig(replay_buffer_size=10)
        session = WebSocketStreamingSession(config=config)

        await session.start()  # event 1

        # Generate several events
        for i in range(5):
            session.create_pong_event(i * 100)  # events 2-6

        # All events should be in replay buffer
        events, gap = session.get_events_for_resume(0)

        assert gap is False
        assert len(events) == 6  # start + 5 pongs

        # Resume from middle
        events, gap = session.get_events_for_resume(3)

        assert gap is False
        assert len(events) == 3  # events 4, 5, 6
        assert all(e.event_id > 3 for e in events)
