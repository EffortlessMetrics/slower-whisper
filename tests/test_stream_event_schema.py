"""Tests for stream event envelope schema validation (Issue #133).

This module tests that the EventEnvelope implementation conforms to the
JSON schema specification at transcription/schemas/stream_event.schema.json.

Tests cover:
- Schema file exists and is valid JSON Schema
- EventEnvelope serialization matches schema requirements
- All event types validate against their payload schemas
- ID format contracts (stream_id, event_id, segment_id)
- v2.1 event types (PHYSICS_UPDATE, AUDIO_HEALTH, VAD_ACTIVITY, BARGE_IN, END_OF_TURN_HINT)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from slower_whisper.pipeline.streaming_ws import (
    EventEnvelope,
    ServerMessageType,
    WebSocketStreamingSession,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def schema_path() -> Path:
    """Return path to the stream event JSON schema."""
    return (
        Path(__file__).parent.parent
        / "slower_whisper"
        / "pipeline"
        / "schemas"
        / "stream_event.schema.json"
    )


@pytest.fixture
def stream_event_schema(schema_path: Path) -> dict[str, Any]:
    """Load the stream event JSON schema."""
    with open(schema_path) as f:
        return json.load(f)


@pytest.fixture
def jsonschema_validator(stream_event_schema: dict[str, Any]):
    """Create a JSON Schema validator if jsonschema is available."""
    try:
        from jsonschema import Draft202012Validator

        return Draft202012Validator(stream_event_schema)
    except ImportError:
        pytest.skip("jsonschema package not installed")


@pytest.fixture
def valid_stream_id() -> str:
    """Return a valid stream_id for testing."""
    return "str-12345678-1234-4567-89ab-123456789abc"


@pytest.fixture
def valid_ts_server() -> int:
    """Return a valid server timestamp for testing."""
    return 1706300000000


# =============================================================================
# Schema File Tests
# =============================================================================


class TestSchemaFile:
    """Tests for the schema file itself."""

    def test_schema_file_exists(self, schema_path: Path) -> None:
        """Test that the schema file exists at the expected location."""
        assert schema_path.exists(), f"Schema file not found: {schema_path}"

    def test_schema_is_valid_json(self, schema_path: Path) -> None:
        """Test that the schema file is valid JSON."""
        with open(schema_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_schema_has_required_structure(self, stream_event_schema: dict[str, Any]) -> None:
        """Test that the schema has the required JSON Schema structure."""
        assert "$schema" in stream_event_schema
        assert "properties" in stream_event_schema
        assert "required" in stream_event_schema
        assert "$defs" in stream_event_schema

    def test_schema_defines_required_fields(self, stream_event_schema: dict[str, Any]) -> None:
        """Test that the schema defines the required envelope fields."""
        required = stream_event_schema.get("required", [])
        assert "event_id" in required
        assert "stream_id" in required
        assert "type" in required
        assert "ts_server" in required
        assert "payload" in required

    def test_schema_includes_v21_event_types(self, stream_event_schema: dict[str, Any]) -> None:
        """Test that the schema includes v2.1 event types."""
        server_message_types = stream_event_schema["$defs"]["ServerMessageType"]["enum"]
        v21_types = [
            "PHYSICS_UPDATE",
            "AUDIO_HEALTH",
            "VAD_ACTIVITY",
            "BARGE_IN",
            "END_OF_TURN_HINT",
        ]
        for event_type in v21_types:
            assert event_type in server_message_types, f"Missing v2.1 event type: {event_type}"

    def test_schema_defines_v21_payloads(self, stream_event_schema: dict[str, Any]) -> None:
        """Test that the schema defines payload schemas for v2.1 events."""
        defs = stream_event_schema["$defs"]
        v21_payloads = [
            "PhysicsUpdatePayload",
            "AudioHealthPayload",
            "VADActivityPayload",
            "BargeInPayload",
            "EndOfTurnHintPayload",
        ]
        for payload in v21_payloads:
            assert payload in defs, f"Missing v2.1 payload definition: {payload}"


# =============================================================================
# ID Format Contract Tests
# =============================================================================


class TestIdFormatContracts:
    """Tests for ID format contracts (stream_id, event_id, segment_id)."""

    def test_stream_id_format_pattern(self) -> None:
        """Test that generated stream_id matches the str-{uuid4} pattern."""
        session = WebSocketStreamingSession()
        pattern = r"^str-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        assert re.match(pattern, session.stream_id), (
            f"Invalid stream_id format: {session.stream_id}"
        )

    def test_event_id_starts_at_one(self) -> None:
        """Test that event_id starts at 1 (not 0)."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-12345678-1234-4567-89ab-123456789abc",
            type=ServerMessageType.SESSION_STARTED,
            ts_server=1706300000000,
            payload={"session_id": "str-12345678-1234-4567-89ab-123456789abc"},
        )
        d = envelope.to_dict()
        assert d["event_id"] >= 1

    def test_segment_id_format_pattern(self) -> None:
        """Test that segment_id follows the seg-{seq} pattern."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-12345678-1234-4567-89ab-123456789abc",
            type=ServerMessageType.PARTIAL,
            ts_server=1706300000000,
            payload={"segment": {"start": 0.0, "end": 1.0, "text": "test"}},
            segment_id="seg-0",
        )
        d = envelope.to_dict()
        pattern = r"^seg-\d+$"
        assert re.match(pattern, d["segment_id"]), f"Invalid segment_id format: {d['segment_id']}"

    def test_segment_id_can_be_null(self) -> None:
        """Test that segment_id can be null for non-segment events."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id="str-12345678-1234-4567-89ab-123456789abc",
            type=ServerMessageType.PONG,
            ts_server=1706300000000,
            payload={"timestamp": 1000, "server_timestamp": 1001},
        )
        d = envelope.to_dict()
        assert "segment_id" not in d  # Null fields are excluded from serialization


# =============================================================================
# Envelope Serialization Tests
# =============================================================================


class TestEnvelopeSerialization:
    """Tests for EventEnvelope serialization."""

    def test_to_dict_includes_required_fields(
        self, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test that to_dict() includes all required fields."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id=valid_stream_id,
            type=ServerMessageType.SESSION_STARTED,
            ts_server=valid_ts_server,
            payload={"session_id": valid_stream_id},
        )
        d = envelope.to_dict()

        assert "event_id" in d
        assert "stream_id" in d
        assert "type" in d
        assert "ts_server" in d
        assert "payload" in d

    def test_to_dict_type_is_string_value(self, valid_stream_id: str, valid_ts_server: int) -> None:
        """Test that type is serialized as string, not Enum."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id=valid_stream_id,
            type=ServerMessageType.SESSION_STARTED,
            ts_server=valid_ts_server,
            payload={},
        )
        d = envelope.to_dict()
        assert isinstance(d["type"], str)
        assert d["type"] == "SESSION_STARTED"

    def test_to_dict_excludes_none_optional_fields(
        self, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test that None optional fields are excluded from serialization."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id=valid_stream_id,
            type=ServerMessageType.PONG,
            ts_server=valid_ts_server,
            payload={},
            segment_id=None,
            ts_audio_start=None,
            ts_audio_end=None,
        )
        d = envelope.to_dict()

        assert "segment_id" not in d
        assert "ts_audio_start" not in d
        assert "ts_audio_end" not in d

    def test_to_dict_includes_set_optional_fields(
        self, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test that set optional fields are included in serialization."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id=valid_stream_id,
            type=ServerMessageType.FINALIZED,
            ts_server=valid_ts_server,
            payload={"segment": {"start": 0.0, "end": 2.5, "text": "Hello world"}},
            segment_id="seg-0",
            ts_audio_start=0.0,
            ts_audio_end=2.5,
        )
        d = envelope.to_dict()

        assert "segment_id" in d
        assert d["segment_id"] == "seg-0"
        assert "ts_audio_start" in d
        assert d["ts_audio_start"] == 0.0
        assert "ts_audio_end" in d
        assert d["ts_audio_end"] == 2.5


# =============================================================================
# JSON Schema Validation - Core Event Types
# =============================================================================


class TestCoreEventTypeValidation:
    """Tests that core event types validate against the JSON schema."""

    def test_session_started_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test SESSION_STARTED event validates against schema."""
        envelope = EventEnvelope(
            event_id=1,
            stream_id=valid_stream_id,
            type=ServerMessageType.SESSION_STARTED,
            ts_server=valid_ts_server,
            payload={"session_id": valid_stream_id},
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_partial_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test PARTIAL event validates against schema."""
        envelope = EventEnvelope(
            event_id=2,
            stream_id=valid_stream_id,
            type=ServerMessageType.PARTIAL,
            ts_server=valid_ts_server,
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

    def test_finalized_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test FINALIZED event validates against schema."""
        envelope = EventEnvelope(
            event_id=3,
            stream_id=valid_stream_id,
            type=ServerMessageType.FINALIZED,
            ts_server=valid_ts_server,
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

    def test_error_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test ERROR event validates against schema."""
        envelope = EventEnvelope(
            event_id=4,
            stream_id=valid_stream_id,
            type=ServerMessageType.ERROR,
            ts_server=valid_ts_server,
            payload={
                "code": "ASR_TIMEOUT",
                "message": "ASR processing timed out",
                "recoverable": True,
            },
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_session_ended_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test SESSION_ENDED event validates against schema."""
        envelope = EventEnvelope(
            event_id=10,
            stream_id=valid_stream_id,
            type=ServerMessageType.SESSION_ENDED,
            ts_server=valid_ts_server,
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

    def test_pong_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test PONG event validates against schema."""
        envelope = EventEnvelope(
            event_id=5,
            stream_id=valid_stream_id,
            type=ServerMessageType.PONG,
            ts_server=valid_ts_server,
            payload={
                "timestamp": valid_ts_server - 1,
                "server_timestamp": valid_ts_server,
            },
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_diarization_update_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test DIARIZATION_UPDATE event validates against schema."""
        envelope = EventEnvelope(
            event_id=6,
            stream_id=valid_stream_id,
            type=ServerMessageType.DIARIZATION_UPDATE,
            ts_server=valid_ts_server,
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
# JSON Schema Validation - v2.1 Event Types
# =============================================================================


class TestV21EventTypeValidation:
    """Tests that v2.1 event types validate against the JSON schema."""

    def test_physics_update_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test PHYSICS_UPDATE event validates against schema."""
        envelope = EventEnvelope(
            event_id=7,
            stream_id=valid_stream_id,
            type=ServerMessageType.PHYSICS_UPDATE,
            ts_server=valid_ts_server,
            payload={
                "speaker_talk_times": {"spk_0": 15.5, "spk_1": 12.3},
                "total_duration_sec": 30.0,
                "interruption_count": 2,
                "interruption_rate": 4.0,
                "mean_response_latency_sec": 0.85,
                "speaker_transitions": 5,
                "overlap_duration_sec": 1.2,
            },
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_audio_health_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test AUDIO_HEALTH event validates against schema."""
        envelope = EventEnvelope(
            event_id=8,
            stream_id=valid_stream_id,
            type=ServerMessageType.AUDIO_HEALTH,
            ts_server=valid_ts_server,
            payload={
                "clipping_ratio": 0.01,
                "rms_energy": -22.5,
                "snr_proxy": 25.0,
                "spectral_centroid": 2500.0,
                "quality_score": 0.85,
                "is_speech_likely": True,
            },
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_vad_activity_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test VAD_ACTIVITY event validates against schema."""
        envelope = EventEnvelope(
            event_id=9,
            stream_id=valid_stream_id,
            type=ServerMessageType.VAD_ACTIVITY,
            ts_server=valid_ts_server,
            payload={
                "energy_level": -18.5,
                "is_speech": True,
                "silence_duration_sec": 0.0,
            },
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_barge_in_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test BARGE_IN event validates against schema."""
        envelope = EventEnvelope(
            event_id=10,
            stream_id=valid_stream_id,
            type=ServerMessageType.BARGE_IN,
            ts_server=valid_ts_server,
            payload={
                "energy": -15.0,
                "tts_elapsed_sec": 2.3,
            },
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_end_of_turn_hint_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test END_OF_TURN_HINT event validates against schema."""
        envelope = EventEnvelope(
            event_id=11,
            stream_id=valid_stream_id,
            type=ServerMessageType.END_OF_TURN_HINT,
            ts_server=valid_ts_server,
            payload={
                "confidence": 0.85,
                "silence_duration_sec": 1.2,
                "prosodic_cues": {
                    "falling_intonation": True,
                    "final_lengthening": False,
                },
            },
        )
        jsonschema_validator.validate(envelope.to_dict())

    def test_end_of_turn_hint_minimal_validates(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test END_OF_TURN_HINT with minimal fields validates."""
        envelope = EventEnvelope(
            event_id=12,
            stream_id=valid_stream_id,
            type=ServerMessageType.END_OF_TURN_HINT,
            ts_server=valid_ts_server,
            payload={
                "confidence": 0.75,
                "silence_duration_sec": 0.8,
            },
        )
        jsonschema_validator.validate(envelope.to_dict())


# =============================================================================
# Ordering Guarantee Tests
# =============================================================================


class TestOrderingGuarantees:
    """Tests for event ordering guarantees documented in the specification."""

    @pytest.mark.asyncio
    async def test_event_ids_are_monotonically_increasing(self) -> None:
        """Test that event_id always increases (Guarantee 1)."""
        session = WebSocketStreamingSession()
        event1 = await session.start()
        event2 = session.create_pong_event(0)
        event3 = session.create_pong_event(0)

        assert event1.event_id < event2.event_id < event3.event_id

    @pytest.mark.asyncio
    async def test_session_started_has_event_id_one(self) -> None:
        """Test that SESSION_STARTED is event_id=1."""
        session = WebSocketStreamingSession()
        event = await session.start()

        assert event.type == ServerMessageType.SESSION_STARTED
        assert event.event_id == 1

    @pytest.mark.asyncio
    async def test_session_ended_is_last_event(self) -> None:
        """Test that SESSION_ENDED is always the last event."""
        session = WebSocketStreamingSession()
        await session.start()
        await session.process_audio_chunk(b"\x00" * 32000, sequence=1)
        end_events = await session.end()

        assert len(end_events) > 0
        assert end_events[-1].type == ServerMessageType.SESSION_ENDED

        # SESSION_ENDED should have the highest event_id
        session_ended_id = end_events[-1].event_id
        for event in end_events[:-1]:
            assert event.event_id < session_ended_id


# =============================================================================
# Backpressure Contract Tests
# =============================================================================


class TestBackpressureContract:
    """Tests for backpressure drop policy documented in the specification."""

    @pytest.mark.asyncio
    async def test_finalized_never_dropped(self) -> None:
        """Test that FINALIZED events are never dropped (invariant)."""
        session = WebSocketStreamingSession()
        await session.start()

        # Add PARTIAL and FINALIZED events
        for i in range(5):
            session._event_queue.put_nowait(
                session._create_envelope(
                    ServerMessageType.PARTIAL,
                    {"segment": {"text": f"partial-{i}"}},
                )
            )

        finalized = session._create_envelope(
            ServerMessageType.FINALIZED,
            {"segment": {"text": "final", "start": 0.0, "end": 1.0}},
        )
        session._event_queue.put_nowait(finalized)

        # Drop partials
        dropped = session.drop_partial_events()

        assert dropped == 5  # All partials dropped
        assert session._event_queue.qsize() == 1

        # Verify the remaining event is FINALIZED
        remaining = session._event_queue.get_nowait()
        assert remaining.type == ServerMessageType.FINALIZED

    @pytest.mark.asyncio
    async def test_error_never_dropped(self) -> None:
        """Test that ERROR events are never dropped."""
        session = WebSocketStreamingSession()
        await session.start()

        # Add PARTIAL and ERROR events
        for i in range(3):
            session._event_queue.put_nowait(
                session._create_envelope(
                    ServerMessageType.PARTIAL,
                    {"segment": {"text": f"partial-{i}"}},
                )
            )

        error = session._create_envelope(
            ServerMessageType.ERROR,
            {"code": "ASR_TIMEOUT", "message": "Timeout", "recoverable": True},
        )
        session._event_queue.put_nowait(error)

        # Drop partials
        dropped = session.drop_partial_events()

        assert dropped == 3
        assert session._event_queue.qsize() == 1

        remaining = session._event_queue.get_nowait()
        assert remaining.type == ServerMessageType.ERROR


# =============================================================================
# Resume Contract Tests
# =============================================================================


class TestResumeContract:
    """Tests for resume protocol documented in the specification."""

    @pytest.mark.asyncio
    async def test_replay_events_after_last_event_id(self) -> None:
        """Test that resume returns events after last_event_id."""
        session = WebSocketStreamingSession()
        await session.start()  # event 1

        session.create_pong_event(0)  # event 2
        session.create_pong_event(0)  # event 3
        session.create_pong_event(0)  # event 4

        # Resume from event 2
        events, gap = session.get_events_for_resume(2)

        assert gap is False
        assert len(events) == 2
        assert events[0].event_id == 3
        assert events[1].event_id == 4

    @pytest.mark.asyncio
    async def test_resume_gap_when_events_evicted(self) -> None:
        """Test that gap is detected when events are evicted from buffer."""
        from slower_whisper.pipeline.streaming_ws import WebSocketSessionConfig

        # Small buffer to force eviction
        config = WebSocketSessionConfig(replay_buffer_size=3)
        session = WebSocketStreamingSession(config=config)
        await session.start()  # event 1

        # Generate events to overflow buffer
        for _ in range(5):
            session.create_pong_event(0)

        # Resume from event 1 (which was evicted)
        events, gap = session.get_events_for_resume(1)

        assert gap is True


# =============================================================================
# Schema Validation Error Tests
# =============================================================================


class TestSchemaValidationErrors:
    """Tests that invalid events fail schema validation."""

    def test_invalid_event_id_zero_fails(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test that event_id=0 fails validation (must be >= 1)."""
        from jsonschema import ValidationError

        invalid_event = {
            "event_id": 0,  # Invalid: must be >= 1
            "stream_id": valid_stream_id,
            "type": "SESSION_STARTED",
            "ts_server": valid_ts_server,
            "payload": {"session_id": valid_stream_id},
        }

        with pytest.raises(ValidationError):
            jsonschema_validator.validate(invalid_event)

    def test_invalid_stream_id_format_fails(
        self, jsonschema_validator, valid_ts_server: int
    ) -> None:
        """Test that invalid stream_id format fails validation."""
        from jsonschema import ValidationError

        invalid_event = {
            "event_id": 1,
            "stream_id": "invalid-stream-id",  # Invalid: doesn't match pattern
            "type": "SESSION_STARTED",
            "ts_server": valid_ts_server,
            "payload": {"session_id": "str-test"},
        }

        with pytest.raises(ValidationError):
            jsonschema_validator.validate(invalid_event)

    def test_invalid_type_fails(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test that invalid event type fails validation."""
        from jsonschema import ValidationError

        invalid_event = {
            "event_id": 1,
            "stream_id": valid_stream_id,
            "type": "INVALID_TYPE",  # Invalid: not in enum
            "ts_server": valid_ts_server,
            "payload": {},
        }

        with pytest.raises(ValidationError):
            jsonschema_validator.validate(invalid_event)

    def test_missing_required_field_fails(
        self, jsonschema_validator, valid_stream_id: str, valid_ts_server: int
    ) -> None:
        """Test that missing required field fails validation."""
        from jsonschema import ValidationError

        invalid_event = {
            "event_id": 1,
            "stream_id": valid_stream_id,
            # Missing "type" field
            "ts_server": valid_ts_server,
            "payload": {},
        }

        with pytest.raises(ValidationError):
            jsonschema_validator.validate(invalid_event)
