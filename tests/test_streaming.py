"""Tests for streaming module skeleton (v2.0 prep).

These tests verify the type structure and interfaces are correct,
even though the implementation raises NotImplementedError.
"""

from __future__ import annotations

import pytest

from transcription.models import Segment, Transcript
from transcription.streaming import (
    PartialSegment,
    StreamConfig,
    StreamEvent,
    StreamEventType,
    StreamingSession,
    StreamMeta,
    apply_stream_event,
)


class TestStreamEventType:
    """Test StreamEventType enum."""

    def test_all_event_types_defined(self) -> None:
        """Verify all expected event types exist."""
        expected = {
            "READY",
            "PARTIAL_SEGMENT",
            "SEGMENT_FINALIZED",
            "TURN_UPDATED",
            "ANALYTICS_SNAPSHOT",
            "ERROR",
            "DONE",
        }
        actual = {e.name for e in StreamEventType}
        assert actual == expected

    def test_event_type_values(self) -> None:
        """Verify event type values match protocol."""
        assert StreamEventType.READY.value == "ready"
        assert StreamEventType.PARTIAL_SEGMENT.value == "partial_segment"
        assert StreamEventType.DONE.value == "done"


class TestStreamConfig:
    """Test StreamConfig dataclass."""

    def test_default_values(self) -> None:
        """Verify default configuration."""
        config = StreamConfig()
        assert config.sample_rate == 16000
        assert config.language is None
        assert config.enable_diarization is False
        assert config.enable_analytics is False
        assert config.enable_semantics is False

    def test_custom_values(self) -> None:
        """Verify custom configuration."""
        config = StreamConfig(
            sample_rate=48000,
            language="en",
            enable_diarization=True,
            enable_analytics=True,
            enable_semantics=True,
        )
        assert config.sample_rate == 48000
        assert config.language == "en"
        assert config.enable_diarization is True


class TestStreamMeta:
    """Test StreamMeta dataclass."""

    def test_required_fields(self) -> None:
        """Verify required fields."""
        meta = StreamMeta(session_id="test_123", seq=42)
        assert meta.session_id == "test_123"
        assert meta.seq == 42
        assert meta.stream_version == 1
        assert meta.features == {}

    def test_features_dict(self) -> None:
        """Verify features dictionary."""
        meta = StreamMeta(
            session_id="test",
            seq=0,
            features={"diarization": True, "analytics": False},
        )
        assert meta.features["diarization"] is True
        assert meta.features["analytics"] is False


class TestPartialSegment:
    """Test PartialSegment dataclass."""

    def test_creation(self) -> None:
        """Verify partial segment creation."""
        seg = PartialSegment(
            id=0,
            start=0.0,
            end=2.5,
            text="Hello world",
            revision=1,
        )
        assert seg.partial is True
        assert seg.revision == 1
        assert seg.speaker_id is None

    def test_finalize(self) -> None:
        """Verify conversion to Segment."""
        partial = PartialSegment(
            id=0,
            start=0.0,
            end=2.5,
            text="Hello",
            revision=3,
            speaker_id="spk_0",
        )
        final = partial.finalize()
        assert isinstance(final, Segment)
        assert final.id == 0
        assert final.text == "Hello"
        assert final.speaker == {"id": "spk_0"}

    def test_finalize_no_speaker(self) -> None:
        """Verify finalize without speaker."""
        partial = PartialSegment(id=1, start=0.0, end=1.0, text="Hi", revision=1)
        final = partial.finalize()
        assert final.speaker is None


class TestStreamEvent:
    """Test StreamEvent dataclass."""

    def test_creation(self) -> None:
        """Verify event creation."""
        event = StreamEvent(
            type=StreamEventType.READY,
            seq=0,
            payload={"session_id": "test"},
        )
        assert event.type == StreamEventType.READY
        assert event.seq == 0
        assert event.payload == {"session_id": "test"}

    def test_event_with_segment(self) -> None:
        """Verify event with segment payload."""
        seg = PartialSegment(id=0, start=0.0, end=1.0, text="Hi", revision=1)
        event = StreamEvent(
            type=StreamEventType.PARTIAL_SEGMENT,
            seq=1,
            payload=seg,
        )
        assert isinstance(event.payload, PartialSegment)


class TestStreamingSession:
    """Test StreamingSession class."""

    def test_creation(self) -> None:
        """Verify session creation."""
        session = StreamingSession()
        assert session.config.sample_rate == 16000

    def test_custom_config(self) -> None:
        """Verify session with custom config."""
        config = StreamConfig(language="en", enable_diarization=True)
        session = StreamingSession(config)
        assert session.config.language == "en"
        assert session.config.enable_diarization is True

    def test_start_not_implemented(self) -> None:
        """Verify start raises NotImplementedError."""
        session = StreamingSession()
        with pytest.raises(NotImplementedError, match="v2.0"):
            session.start()

    def test_process_audio_not_implemented(self) -> None:
        """Verify process_audio raises NotImplementedError."""
        session = StreamingSession()
        with pytest.raises(NotImplementedError, match="v2.0"):
            session.process_audio(b"audio data", seq=0)

    def test_stop_not_implemented(self) -> None:
        """Verify stop raises NotImplementedError."""
        session = StreamingSession()
        with pytest.raises(NotImplementedError, match="v2.0"):
            session.stop()

    def test_get_transcript_not_implemented(self) -> None:
        """Verify get_transcript raises NotImplementedError."""
        session = StreamingSession()
        with pytest.raises(NotImplementedError, match="v2.0"):
            session.get_transcript()


class TestApplyStreamEvent:
    """Test apply_stream_event function."""

    def test_not_implemented(self) -> None:
        """Verify apply raises NotImplementedError."""
        transcript = Transcript(
            file_name="test.wav",
            language="en",
            segments=[],
        )
        event = StreamEvent(type=StreamEventType.DONE, seq=0)
        with pytest.raises(NotImplementedError, match="v2.0"):
            apply_stream_event(transcript, event)


class TestProtocolCompliance:
    """Test that StreamingSession satisfies the protocol."""

    def test_session_has_protocol_methods(self) -> None:
        """Verify session has all protocol methods."""
        session = StreamingSession()
        # These should exist (even if they raise)
        assert hasattr(session, "session_id")
        assert hasattr(session, "config")
        assert hasattr(session, "start")
        assert hasattr(session, "process_audio")
        assert hasattr(session, "stop")
        assert hasattr(session, "get_transcript")
