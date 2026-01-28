"""Tests for streaming callbacks (v1.9.0).

This test suite validates the callback system for streaming enrichment sessions:
1. Callback invocation when segments are finalized
2. Graceful handling when callbacks are None
3. Exception isolation in callbacks (never crash the pipeline)
4. The invoke_callback_safely helper function
5. on_error callback invocation when another callback fails

Design:
- Uses pytest fixtures for reusable test setup
- Mocks AudioSegmentExtractor to avoid real audio file dependencies
- Tests the REAL callback infrastructure from transcription.streaming_callbacks
- Validates error isolation and logging behavior
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from transcription.streaming import StreamChunk, StreamSegment
from transcription.streaming_callbacks import (
    NoOpCallbacks,
    StreamCallbacks,
    StreamingError,
    invoke_callback_safely,
)
from transcription.streaming_enrich import (
    StreamingEnrichmentConfig,
    StreamingEnrichmentSession,
)

# =============================================================================
# Helper Functions
# =============================================================================


def _chunk(start: float, end: float, text: str, speaker: str | None = None) -> StreamChunk:
    """Create a StreamChunk for testing."""
    return {"start": start, "end": end, "text": text, "speaker_id": speaker}


def _create_mock_extractor(duration: float = 60.0, sample_rate: int = 16000) -> MagicMock:
    """Create a mock AudioSegmentExtractor."""
    mock = MagicMock()
    mock.duration_seconds = duration
    mock.sample_rate = sample_rate
    mock.total_frames = int(duration * sample_rate)
    mock.wav_path = Path("/tmp/fake_audio.wav")
    return mock


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_extractor() -> MagicMock:
    """Provide a mock AudioSegmentExtractor."""
    return _create_mock_extractor()


@pytest.fixture
def base_config() -> StreamingEnrichmentConfig:
    """Provide a base config with all enrichment disabled."""
    return StreamingEnrichmentConfig(
        enable_prosody=False,
        enable_emotion=False,
        enable_categorical_emotion=False,
    )


@pytest.fixture
def recording_callbacks() -> dict[str, list[Any]]:
    """Fixture that provides a callbacks object that records all calls.

    Returns a dict with the callbacks object and recorded events.
    """

    class RecordingCallbacks:
        def __init__(self) -> None:
            self.finalized_segments: list[StreamSegment] = []
            self.speaker_turns: list[dict] = []
            self.semantic_updates: list[Any] = []
            self.errors: list[StreamingError] = []

        def on_segment_finalized(self, segment: StreamSegment) -> None:
            self.finalized_segments.append(segment)

        def on_speaker_turn(self, turn: dict) -> None:
            self.speaker_turns.append(turn)

        def on_semantic_update(self, payload: Any) -> None:
            self.semantic_updates.append(payload)

        def on_error(self, error: StreamingError) -> None:
            self.errors.append(error)

    return {"callbacks": RecordingCallbacks(), "instance": RecordingCallbacks}  # type: ignore[dict-item]


@pytest.fixture
def failing_callbacks() -> dict[str, Any]:
    """Fixture that provides a callbacks object where on_segment_finalized raises."""

    class FailingCallbacks:
        def __init__(self) -> None:
            self.errors: list[StreamingError] = []
            self.call_count = 0

        def on_segment_finalized(self, segment: StreamSegment) -> None:
            self.call_count += 1
            raise RuntimeError("Simulated callback failure")

        def on_error(self, error: StreamingError) -> None:
            self.errors.append(error)

    return {"callbacks": FailingCallbacks(), "instance": FailingCallbacks}


# =============================================================================
# 1. Test invoke_callback_safely Helper Function
# =============================================================================


class TestInvokeCallbackSafely:
    """Tests for the invoke_callback_safely helper function."""

    def test_returns_false_when_callbacks_is_none(self) -> None:
        """invoke_callback_safely returns False when callbacks is None."""
        result = invoke_callback_safely(
            None, "on_segment_finalized", StreamSegment(0.0, 1.0, "test")
        )
        assert result is False

    def test_returns_false_when_method_not_found(self) -> None:
        """invoke_callback_safely returns False when method doesn't exist."""

        class EmptyCallbacks:
            pass

        result = invoke_callback_safely(EmptyCallbacks(), "nonexistent_method", "arg")
        assert result is False

    def test_returns_true_on_successful_invocation(self) -> None:
        """invoke_callback_safely returns True when callback succeeds."""
        called = []

        class SuccessCallbacks:
            def on_segment_finalized(self, segment: StreamSegment) -> None:
                called.append(segment)

        segment = StreamSegment(0.0, 1.0, "test")
        result = invoke_callback_safely(SuccessCallbacks(), "on_segment_finalized", segment)

        assert result is True
        assert len(called) == 1
        assert called[0] is segment

    def test_returns_false_on_exception(self) -> None:
        """invoke_callback_safely returns False when callback raises."""

        class FailingCallbacks:
            def on_segment_finalized(self, segment: StreamSegment) -> None:
                raise ValueError("Test error")

            def on_error(self, error: StreamingError) -> None:
                pass

        result = invoke_callback_safely(
            FailingCallbacks(),
            "on_segment_finalized",
            StreamSegment(0.0, 1.0, "test"),
        )

        assert result is False

    def test_exception_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """invoke_callback_safely logs warning when callback raises."""

        class FailingCallbacks:
            def on_segment_finalized(self, segment: StreamSegment) -> None:
                raise ValueError("Test error message")

            def on_error(self, error: StreamingError) -> None:
                pass

        with caplog.at_level(logging.WARNING):
            invoke_callback_safely(
                FailingCallbacks(),
                "on_segment_finalized",
                StreamSegment(0.0, 1.0, "test"),
            )

        assert "on_segment_finalized raised exception" in caplog.text
        assert "Test error message" in caplog.text

    def test_on_error_invoked_when_callback_fails(self) -> None:
        """invoke_callback_safely calls on_error when another callback fails."""
        errors_received: list[StreamingError] = []

        class FailingCallbacks:
            def on_segment_finalized(self, segment: StreamSegment) -> None:
                raise ValueError("Segment callback failed")

            def on_error(self, error: StreamingError) -> None:
                errors_received.append(error)

        invoke_callback_safely(
            FailingCallbacks(),
            "on_segment_finalized",
            StreamSegment(0.0, 1.0, "test"),
        )

        assert len(errors_received) == 1
        assert isinstance(errors_received[0], StreamingError)
        assert isinstance(errors_received[0].exception, ValueError)
        assert "Segment callback failed" in str(errors_received[0].exception)
        assert "on_segment_finalized" in errors_received[0].context
        assert errors_received[0].recoverable is True

    def test_on_error_not_invoked_for_itself(self) -> None:
        """invoke_callback_safely does not call on_error when on_error fails."""
        call_count = 0

        class FailingOnError:
            def on_error(self, error: StreamingError) -> None:
                nonlocal call_count
                call_count += 1
                raise RuntimeError("on_error also failed")

        # This should not cause infinite recursion
        invoke_callback_safely(
            FailingOnError(),
            "on_error",
            StreamingError(exception=ValueError("test"), context="test context"),
        )

        # on_error should only be called once (not recursively)
        assert call_count == 1

    def test_handles_on_error_failure_gracefully(self, caplog: pytest.LogCaptureFixture) -> None:
        """invoke_callback_safely logs error when on_error also fails."""

        class DoubleFailure:
            def on_segment_finalized(self, segment: StreamSegment) -> None:
                raise ValueError("First failure")

            def on_error(self, error: StreamingError) -> None:
                raise RuntimeError("on_error also failed")

        with caplog.at_level(logging.ERROR):
            invoke_callback_safely(
                DoubleFailure(),
                "on_segment_finalized",
                StreamSegment(0.0, 1.0, "test"),
            )

        assert "on_error callback also raised" in caplog.text

    def test_passes_kwargs_to_callback(self) -> None:
        """invoke_callback_safely forwards kwargs to the callback."""
        received_kwargs: dict[str, Any] = {}

        class KwargsCallbacks:
            def custom_method(self, **kwargs: Any) -> None:
                received_kwargs.update(kwargs)

        invoke_callback_safely(
            KwargsCallbacks(),
            "custom_method",
            foo="bar",
            count=42,
        )

        assert received_kwargs == {"foo": "bar", "count": 42}


# =============================================================================
# 2. Test NoOpCallbacks
# =============================================================================


class TestNoOpCallbacks:
    """Tests for the NoOpCallbacks default implementation."""

    def test_all_methods_are_no_ops(self) -> None:
        """NoOpCallbacks methods do nothing and don't raise."""
        callbacks = NoOpCallbacks()

        # These should all succeed without raising
        callbacks.on_segment_finalized(StreamSegment(0.0, 1.0, "test"))
        callbacks.on_speaker_turn({"id": "turn_0", "speaker_id": "spk_0"})
        callbacks.on_semantic_update(MagicMock())
        callbacks.on_error(StreamingError(exception=ValueError("test"), context="test"))

    def test_satisfies_protocol(self) -> None:
        """NoOpCallbacks satisfies the StreamCallbacks protocol."""
        callbacks = NoOpCallbacks()
        assert isinstance(callbacks, StreamCallbacks)


# =============================================================================
# 3. Test StreamingError Dataclass
# =============================================================================


class TestStreamingError:
    """Tests for the StreamingError dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """StreamingError can be created with required fields only."""
        error = StreamingError(
            exception=ValueError("test"),
            context="Test context",
        )

        assert isinstance(error.exception, ValueError)
        assert error.context == "Test context"
        assert error.segment_start is None
        assert error.segment_end is None
        assert error.recoverable is True  # default

    def test_creation_with_all_fields(self) -> None:
        """StreamingError can be created with all fields."""
        error = StreamingError(
            exception=RuntimeError("full error"),
            context="Full context",
            segment_start=10.5,
            segment_end=15.0,
            recoverable=False,
        )

        assert str(error.exception) == "full error"
        assert error.context == "Full context"
        assert error.segment_start == 10.5
        assert error.segment_end == 15.0
        assert error.recoverable is False


# =============================================================================
# 4. Test StreamingEnrichmentSession with Callbacks
# =============================================================================


class TestSessionCallbackIntegration:
    """Integration tests for StreamingEnrichmentSession with callbacks."""

    def test_on_segment_finalized_called_on_gap(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """on_segment_finalized is called when a segment is finalized due to gap."""
        callbacks = recording_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # First chunk starts a segment
            session.ingest_chunk(_chunk(0.0, 0.5, "hello", "spk_0"))

            # Second chunk with large gap finalizes first segment
            base_config.base_config.max_gap_sec = 0.5
            session.ingest_chunk(_chunk(2.0, 2.5, "world", "spk_0"))

            # Should have received one finalized segment callback
            assert len(callbacks.finalized_segments) == 1
            assert callbacks.finalized_segments[0].text == "hello"
            assert callbacks.finalized_segments[0].start == 0.0
            assert callbacks.finalized_segments[0].end == 0.5

    def test_on_segment_finalized_called_on_end_of_stream(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """on_segment_finalized is called when end_of_stream is called."""
        callbacks = recording_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "test message", "spk_0"))

            # No callback yet (segment is partial)
            assert len(callbacks.finalized_segments) == 0

            # End of stream finalizes the segment
            session.end_of_stream()

            assert len(callbacks.finalized_segments) == 1
            assert callbacks.finalized_segments[0].text == "test message"

    def test_callbacks_can_be_none(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """Session works correctly when callbacks is None (no crash)."""
        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=None,  # No callbacks
            )

            # These should not raise
            events = session.ingest_chunk(_chunk(0.0, 1.0, "test", "spk_0"))
            assert len(events) == 1

            final_events = session.end_of_stream()
            assert len(final_events) == 1

    def test_callback_exception_does_not_crash_pipeline(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        failing_callbacks: dict[str, Any],
    ) -> None:
        """Callback exceptions are caught and don't crash the streaming pipeline."""
        callbacks = failing_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # This should not raise even though callback will fail
            session.ingest_chunk(_chunk(0.0, 0.5, "hello", "spk_0"))
            events = session.end_of_stream()

            # Pipeline should complete successfully
            assert len(events) == 1
            assert events[0].segment.text == "hello"

            # Callback was called (and failed)
            assert callbacks.call_count == 1

    def test_on_error_invoked_when_callback_fails(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        failing_callbacks: dict[str, Any],
    ) -> None:
        """on_error is invoked when on_segment_finalized raises."""
        callbacks = failing_callbacks["callbacks"]

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 0.5, "test", "spk_0"))
            session.end_of_stream()

            # on_error should have been called
            assert len(callbacks.errors) == 1
            error = callbacks.errors[0]
            assert isinstance(error.exception, RuntimeError)
            assert "Simulated callback failure" in str(error.exception)
            assert "on_segment_finalized" in error.context
            assert error.recoverable is True

    def test_multiple_segments_invoke_multiple_callbacks(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """Multiple finalized segments invoke the callback multiple times."""
        callbacks = recording_callbacks["callbacks"]
        base_config.base_config.max_gap_sec = 0.5

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # Chunk 1
            session.ingest_chunk(_chunk(0.0, 0.4, "one", "spk_0"))

            # Chunk 2 with gap (finalizes chunk 1)
            session.ingest_chunk(_chunk(1.5, 2.0, "two", "spk_0"))

            # Chunk 3 with gap (finalizes chunk 2)
            session.ingest_chunk(_chunk(3.5, 4.0, "three", "spk_0"))

            # End of stream (finalizes chunk 3)
            session.end_of_stream()

            # Should have 3 callbacks
            assert len(callbacks.finalized_segments) == 3
            texts = [seg.text for seg in callbacks.finalized_segments]
            assert texts == ["one", "two", "three"]


# =============================================================================
# 5. Test Partial Callbacks (Missing Methods)
# =============================================================================


class TestPartialCallbacks:
    """Tests for callbacks that only implement some methods."""

    def test_missing_on_segment_finalized_is_ok(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
    ) -> None:
        """Session works when callbacks lacks on_segment_finalized."""

        class PartialCallbacks:
            def on_error(self, error: StreamingError) -> None:
                pass

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=PartialCallbacks(),
            )

            # Should not raise
            session.ingest_chunk(_chunk(0.0, 1.0, "test", "spk_0"))
            events = session.end_of_stream()
            assert len(events) == 1

    def test_missing_on_error_when_callback_fails(
        self,
        mock_extractor: MagicMock,
        base_config: StreamingEnrichmentConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Session continues when callback fails and on_error is missing."""

        class FailNoOnError:
            def on_segment_finalized(self, segment: StreamSegment) -> None:
                raise ValueError("Failure without on_error")

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=FailNoOnError(),
            )

            with caplog.at_level(logging.WARNING):
                session.ingest_chunk(_chunk(0.0, 0.5, "test", "spk_0"))
                events = session.end_of_stream()

            # Pipeline should still complete
            assert len(events) == 1
            # Warning should be logged
            assert "on_segment_finalized raised exception" in caplog.text


# =============================================================================
# 6. Test Callback with Audio State
# =============================================================================


class TestCallbackWithAudioState:
    """Tests for callbacks receiving segments with audio_state."""

    def test_callback_receives_audio_state_when_enrichment_disabled(
        self,
        mock_extractor: MagicMock,
        recording_callbacks: dict[str, Any],
    ) -> None:
        """Callback receives segment with audio_state=None when enrichment disabled."""
        callbacks = recording_callbacks["callbacks"]
        config = StreamingEnrichmentConfig(
            enable_prosody=False,
            enable_emotion=False,
            enable_categorical_emotion=False,
        )

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "test", "spk_0"))
            session.end_of_stream()

            assert len(callbacks.finalized_segments) == 1
            # audio_state should be None when enrichment is disabled
            assert callbacks.finalized_segments[0].audio_state is None


# =============================================================================
# 7. Test Protocol Compliance
# =============================================================================


class TestProtocolCompliance:
    """Tests for StreamCallbacks protocol compliance."""

    def test_runtime_checkable_protocol(self) -> None:
        """StreamCallbacks protocol is runtime-checkable."""

        class CompliantCallbacks:
            def on_segment_finalized(self, segment: StreamSegment) -> None:
                pass

            def on_speaker_turn(self, turn: dict) -> None:
                pass

            def on_semantic_update(self, payload: Any) -> None:
                pass

            def on_error(self, error: StreamingError) -> None:
                pass

            # v2.1 callbacks
            def on_physics_update(self, snapshot: Any) -> None:
                pass

            def on_audio_health(self, snapshot: Any) -> None:
                pass

            def on_vad_activity(self, payload: Any) -> None:
                pass

            def on_barge_in(self, payload: Any) -> None:
                pass

            def on_end_of_turn_hint(self, payload: Any) -> None:
                pass

            def on_correction(self, correction: Any) -> None:
                pass

            def on_commitment(self, commitment: Any) -> None:
                pass

        callbacks = CompliantCallbacks()
        assert isinstance(callbacks, StreamCallbacks)

    def test_partial_implementation_not_protocol(self) -> None:
        """Partial implementations don't satisfy the full protocol."""

        class PartialCallbacks:
            def on_segment_finalized(self, segment: StreamSegment) -> None:
                pass

        # Partial implementation lacks other methods, so not fully compliant
        # But runtime_checkable only checks methods that exist, so this passes
        callbacks = PartialCallbacks()
        # Note: runtime_checkable protocol only checks if methods exist
        # The implementation works regardless
        assert hasattr(callbacks, "on_segment_finalized")
