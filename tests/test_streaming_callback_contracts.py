"""Callback contract tests for streaming enrichment (v1.9.0 #107).

This module defines and tests the behavioral contracts for the streaming callback
system. These are invariants that downstream consumers can rely on.

## Contracts Tested

1. **Exactly-Once Delivery**: Each finalized segment triggers exactly one
   `on_segment_finalized` callback. No duplicates, no omissions.

2. **Monotonic Ordering**: Callbacks are delivered in chronological order.
   If segment A ends before segment B starts, A's callback fires before B's.

3. **End-of-Stream Flush**: Calling `end_of_stream()` flushes any buffered
   partial segment and triggers its callback.

4. **Speaker Turn Boundaries**: Speaker changes (A→B→A) trigger `on_speaker_turn`
   events at each transition. Turn boundaries are detected correctly.

5. **Callback Ordering Within Segment**: `on_segment_finalized` is called
   before `on_speaker_turn` for the segment that completes the turn.

6. **Error Isolation**: If a callback raises an exception, subsequent callbacks
   still fire. The `on_error` callback is invoked for the exception.

7. **Payload Integrity**: Callback payloads contain all required fields with
   correct types and values.

These tests use deterministic fixtures and mock the audio extractor to avoid
real audio file dependencies. All timing is controlled via explicit chunk
timestamps.

See also: streaming_callbacks.py, streaming_enrich.py, STREAMING_ARCHITECTURE.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from transcription.streaming import StreamChunk, StreamSegment
from transcription.streaming_callbacks import StreamingError
from transcription.streaming_enrich import (
    StreamingEnrichmentConfig,
    StreamingEnrichmentSession,
)

# =============================================================================
# Test Fixtures
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


@dataclass
class CallbackEvent:
    """Recorded callback event for ordering verification."""

    event_type: str
    timestamp: float  # start time for ordering
    payload: Any


@dataclass
class ContractCallbacks:
    """Recording callbacks for contract verification.

    Tracks the order and content of all callback invocations.
    """

    events: list[CallbackEvent] = field(default_factory=list)
    finalized_segments: list[StreamSegment] = field(default_factory=list)
    speaker_turns: list[dict] = field(default_factory=list)
    errors: list[StreamingError] = field(default_factory=list)

    should_raise_on_segment: bool = False
    raise_after_n_segments: int = -1  # -1 means never

    def on_segment_finalized(self, segment: StreamSegment) -> None:
        self.events.append(CallbackEvent("segment_finalized", segment.start, segment))
        self.finalized_segments.append(segment)

        # Optionally raise to test error isolation
        if self.should_raise_on_segment:
            if (
                self.raise_after_n_segments < 0
                or len(self.finalized_segments) <= self.raise_after_n_segments
            ):
                raise RuntimeError("Simulated segment callback failure")

    def on_speaker_turn(self, turn: dict) -> None:
        self.events.append(CallbackEvent("speaker_turn", turn["start"], turn))
        self.speaker_turns.append(turn)

    def on_error(self, error: StreamingError) -> None:
        self.events.append(CallbackEvent("error", error.segment_start or 0.0, error))
        self.errors.append(error)


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


# =============================================================================
# Contract 1: Exactly-Once Delivery
# =============================================================================


class TestExactlyOnceDelivery:
    """Contract: Each finalized segment triggers exactly one callback."""

    def test_single_segment_single_callback(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """One segment finalized = exactly one on_segment_finalized call."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "hello", "spk_0"))
            session.end_of_stream()

        # Exactly one callback
        assert len(callbacks.finalized_segments) == 1
        assert callbacks.finalized_segments[0].text == "hello"

    def test_multiple_segments_one_callback_each(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """N segments finalized = exactly N on_segment_finalized calls."""
        callbacks = ContractCallbacks()
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

            # 3 segments with gaps > max_gap_sec
            session.ingest_chunk(_chunk(0.0, 0.4, "one", "spk_0"))
            session.ingest_chunk(_chunk(1.0, 1.4, "two", "spk_0"))  # gap=0.6 > 0.5
            session.ingest_chunk(_chunk(2.5, 2.9, "three", "spk_0"))  # gap=1.1 > 0.5
            session.end_of_stream()

        # Exactly 3 callbacks
        assert len(callbacks.finalized_segments) == 3
        texts = [seg.text for seg in callbacks.finalized_segments]
        assert texts == ["one", "two", "three"]

    def test_no_duplicate_callbacks_on_end_of_stream(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """end_of_stream() doesn't re-fire callbacks for already-finalized segments."""
        callbacks = ContractCallbacks()
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

            # First segment finalized by gap
            session.ingest_chunk(_chunk(0.0, 0.4, "first", "spk_0"))
            session.ingest_chunk(_chunk(1.0, 1.4, "second", "spk_0"))  # gap finalizes first

            count_before_eos = len(callbacks.finalized_segments)
            session.end_of_stream()
            count_after_eos = len(callbacks.finalized_segments)

        # First was already finalized, second finalized by EOS
        assert count_before_eos == 1  # "first" finalized
        assert count_after_eos == 2  # "second" finalized by EOS
        # No duplicates
        texts = [seg.text for seg in callbacks.finalized_segments]
        assert texts == ["first", "second"]


# =============================================================================
# Contract 2: Monotonic Ordering
# =============================================================================


class TestMonotonicOrdering:
    """Contract: Callbacks are delivered in chronological order."""

    def test_segments_callback_order_matches_time_order(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Segments are finalized in chronological order (by start time)."""
        callbacks = ContractCallbacks()
        base_config.base_config.max_gap_sec = 0.3

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # Segments in chronological order
            session.ingest_chunk(_chunk(0.0, 0.2, "first", "spk_0"))
            session.ingest_chunk(_chunk(1.0, 1.2, "second", "spk_0"))
            session.ingest_chunk(_chunk(2.0, 2.2, "third", "spk_0"))
            session.end_of_stream()

        # Verify ordering
        times = [seg.start for seg in callbacks.finalized_segments]
        assert times == sorted(times), "Callbacks not in chronological order"

    def test_callback_events_are_ordered(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """All callback events (segment + turn) are delivered in order."""
        callbacks = ContractCallbacks()
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

            # A→B speaker change
            session.ingest_chunk(_chunk(0.0, 0.4, "hello", "spk_A"))
            session.ingest_chunk(_chunk(1.0, 1.4, "world", "spk_B"))
            session.end_of_stream()

        # Extract all event timestamps
        event_times = [e.timestamp for e in callbacks.events]

        # Verify all events are in non-decreasing order
        for i in range(1, len(event_times)):
            assert event_times[i] >= event_times[i - 1], (
                f"Event {i} ({event_times[i]}) before event {i - 1} ({event_times[i - 1]})"
            )


# =============================================================================
# Contract 3: End-of-Stream Flush
# =============================================================================


class TestEndOfStreamFlush:
    """Contract: end_of_stream() flushes buffered segments."""

    def test_partial_segment_flushed_on_eos(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """A partial segment in buffer is finalized when end_of_stream is called."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # Single chunk, no gap - stays as partial
            session.ingest_chunk(_chunk(0.0, 1.0, "partial content", "spk_0"))

            # No callback yet
            assert len(callbacks.finalized_segments) == 0

            # EOS flushes the partial
            session.end_of_stream()

        assert len(callbacks.finalized_segments) == 1
        assert callbacks.finalized_segments[0].text == "partial content"

    def test_empty_stream_eos_no_callback(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """end_of_stream() with no chunks does not fire any callbacks."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.end_of_stream()

        assert len(callbacks.finalized_segments) == 0
        assert len(callbacks.speaker_turns) == 0


# =============================================================================
# Contract 4: Speaker Turn Boundaries
# =============================================================================


class TestSpeakerTurnBoundaries:
    """Contract: Speaker changes trigger on_speaker_turn at boundaries."""

    def test_speaker_change_triggers_turn_callback(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """A→B speaker change triggers on_speaker_turn for A's turn."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "hello from A", "spk_A"))
            session.ingest_chunk(_chunk(1.5, 2.5, "hello from B", "spk_B"))
            session.end_of_stream()

        # Two turns: one for A, one for B
        assert len(callbacks.speaker_turns) == 2
        assert callbacks.speaker_turns[0]["speaker_id"] == "spk_A"
        assert callbacks.speaker_turns[1]["speaker_id"] == "spk_B"

    def test_a_b_a_yields_three_turns(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """A→B→A speaker sequence yields three turn callbacks."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "first from A", "spk_A"))
            session.ingest_chunk(_chunk(1.5, 2.5, "from B", "spk_B"))
            session.ingest_chunk(_chunk(3.0, 4.0, "back to A", "spk_A"))
            session.end_of_stream()

        # Three turns
        assert len(callbacks.speaker_turns) == 3
        speakers = [t["speaker_id"] for t in callbacks.speaker_turns]
        assert speakers == ["spk_A", "spk_B", "spk_A"]

    def test_same_speaker_contiguous_is_one_turn(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Multiple segments from same speaker = one turn."""
        callbacks = ContractCallbacks()
        base_config.base_config.max_gap_sec = 0.3

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # Two segments from same speaker with gap
            session.ingest_chunk(_chunk(0.0, 0.2, "part one", "spk_A"))
            session.ingest_chunk(_chunk(0.6, 0.8, "part two", "spk_A"))
            session.end_of_stream()

        # One turn containing both segments
        assert len(callbacks.speaker_turns) == 1
        assert callbacks.speaker_turns[0]["speaker_id"] == "spk_A"
        assert "part one" in callbacks.speaker_turns[0]["text"]
        assert "part two" in callbacks.speaker_turns[0]["text"]

    def test_turn_has_required_fields(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Turn dict contains all required fields."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "hello world", "spk_A"))
            session.end_of_stream()

        assert len(callbacks.speaker_turns) == 1
        turn = callbacks.speaker_turns[0]

        # Required fields per StreamCallbacks.on_speaker_turn docstring
        assert "id" in turn
        assert "speaker_id" in turn
        assert "start" in turn
        assert "end" in turn
        assert "segment_ids" in turn
        assert "text" in turn

        # Validate types
        assert isinstance(turn["id"], str)
        assert isinstance(turn["speaker_id"], str)
        assert isinstance(turn["start"], (int, float))
        assert isinstance(turn["end"], (int, float))
        assert isinstance(turn["segment_ids"], list)
        assert isinstance(turn["text"], str)


# =============================================================================
# Contract 5: Callback Ordering Within Segment
# =============================================================================


class TestCallbackOrderingWithinSegment:
    """Contract: on_segment_finalized fires before on_speaker_turn."""

    def test_segment_callback_before_turn_callback(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """on_segment_finalized is called before on_speaker_turn."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "hello", "spk_A"))
            session.end_of_stream()

        # Both callbacks fired
        assert len(callbacks.finalized_segments) == 1
        assert len(callbacks.speaker_turns) == 1

        # Find event indices
        event_types = [e.event_type for e in callbacks.events]
        segment_idx = event_types.index("segment_finalized")
        turn_idx = event_types.index("speaker_turn")

        assert segment_idx < turn_idx, "on_segment_finalized should fire before on_speaker_turn"

    def test_segment_before_turn_with_speaker_change(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """During speaker change, segment callback fires before turn callback."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # A speaks, then B speaks (triggers turn for A)
            session.ingest_chunk(_chunk(0.0, 1.0, "from A", "spk_A"))
            session.ingest_chunk(_chunk(1.5, 2.5, "from B", "spk_B"))
            session.end_of_stream()

        # Check ordering: A's segment finalized before A's turn
        event_types = [e.event_type for e in callbacks.events]

        # Find first segment_finalized and first speaker_turn
        first_segment_idx = event_types.index("segment_finalized")
        first_turn_idx = event_types.index("speaker_turn")

        assert first_segment_idx < first_turn_idx


# =============================================================================
# Contract 6: Error Isolation
# =============================================================================


class TestErrorIsolation:
    """Contract: Callback exceptions don't crash the pipeline."""

    def test_exception_does_not_stop_pipeline(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Pipeline continues after callback raises."""
        callbacks = ContractCallbacks()
        callbacks.should_raise_on_segment = True
        base_config.base_config.max_gap_sec = 0.3

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # First segment triggers error, second should still be processed
            session.ingest_chunk(_chunk(0.0, 0.2, "first", "spk_0"))
            session.ingest_chunk(_chunk(0.6, 0.8, "second", "spk_0"))
            session.ingest_chunk(_chunk(1.2, 1.4, "third", "spk_0"))
            events = session.end_of_stream()

        # Pipeline completed (returned events)
        assert len(events) == 1  # Last segment finalized by EOS

    def test_on_error_invoked_for_callback_exception(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """on_error is called when another callback raises."""
        callbacks = ContractCallbacks()
        callbacks.should_raise_on_segment = True

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "test", "spk_0"))
            session.end_of_stream()

        # on_error was called
        assert len(callbacks.errors) >= 1
        error = callbacks.errors[0]
        assert isinstance(error.exception, RuntimeError)
        assert "on_segment_finalized" in error.context
        assert error.recoverable is True

    def test_later_callbacks_fire_after_exception(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Subsequent callbacks fire even after one raises."""
        callbacks = ContractCallbacks()
        callbacks.should_raise_on_segment = True
        callbacks.raise_after_n_segments = 1  # Only first segment raises

        base_config.base_config.max_gap_sec = 0.3

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # Three segments, first one's callback raises
            session.ingest_chunk(_chunk(0.0, 0.2, "first", "spk_0"))
            session.ingest_chunk(_chunk(0.6, 0.8, "second", "spk_0"))
            session.ingest_chunk(_chunk(1.2, 1.4, "third", "spk_0"))
            session.end_of_stream()

        # All three segment callbacks were invoked
        assert len(callbacks.finalized_segments) == 3

        # Error was recorded for the first one
        assert len(callbacks.errors) == 1


# =============================================================================
# Contract 7: Payload Integrity
# =============================================================================


class TestPayloadIntegrity:
    """Contract: Callback payloads contain correct data."""

    def test_segment_has_required_fields(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Finalized segment has all required fields."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(5.0, 7.5, "test content", "speaker_1"))
            session.end_of_stream()

        assert len(callbacks.finalized_segments) == 1
        seg = callbacks.finalized_segments[0]

        # Required fields per StreamCallbacks.on_segment_finalized docstring
        assert seg.start == 5.0
        assert seg.end == 7.5
        assert seg.text == "test content"
        assert seg.speaker_id == "speaker_1"

    def test_segment_speaker_id_preserved(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Speaker ID from chunk is preserved in callback segment."""
        callbacks = ContractCallbacks()
        base_config.base_config.max_gap_sec = 0.3

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 0.2, "one", "alice"))
            session.ingest_chunk(_chunk(0.6, 0.8, "two", "bob"))
            session.end_of_stream()

        speakers = [seg.speaker_id for seg in callbacks.finalized_segments]
        assert speakers == ["alice", "bob"]

    def test_turn_text_aggregates_segments(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Turn text contains all segment texts."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # Two segments, same speaker
            session.ingest_chunk(_chunk(0.0, 0.5, "hello", "spk_A"))
            session.ingest_chunk(_chunk(0.6, 1.0, "world", "spk_A"))
            session.end_of_stream()

        assert len(callbacks.speaker_turns) == 1
        turn_text = callbacks.speaker_turns[0]["text"]
        assert "hello" in turn_text
        assert "world" in turn_text

    def test_turn_time_bounds_span_all_segments(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Turn start/end span from first to last segment."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(1.0, 1.5, "one", "spk_A"))
            session.ingest_chunk(_chunk(1.6, 2.0, "two", "spk_A"))
            session.ingest_chunk(_chunk(2.1, 3.0, "three", "spk_A"))
            session.end_of_stream()

        assert len(callbacks.speaker_turns) == 1
        turn = callbacks.speaker_turns[0]
        assert turn["start"] == 1.0  # First segment start
        assert turn["end"] == 3.0  # Last segment end


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_text_segments_handled(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Segments with empty text don't cause errors."""
        callbacks = ContractCallbacks()
        base_config.base_config.max_gap_sec = 0.3

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 0.2, "", "spk_0"))
            session.ingest_chunk(_chunk(0.6, 0.8, "real text", "spk_0"))
            session.end_of_stream()

        # Both segments processed
        assert len(callbacks.finalized_segments) == 2

    def test_none_speaker_handled(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Segments with None speaker are handled gracefully."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 1.0, "no speaker", None))
            session.end_of_stream()

        assert len(callbacks.finalized_segments) == 1
        assert callbacks.finalized_segments[0].speaker_id is None

        assert len(callbacks.speaker_turns) == 1
        # Turn should have "unknown" or handle None speaker
        assert callbacks.speaker_turns[0]["speaker_id"] in ("unknown", None)

    def test_rapid_speaker_changes(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Rapid A→B→A→B→A changes all trigger correct turns."""
        callbacks = ContractCallbacks()

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            # Rapid alternation
            session.ingest_chunk(_chunk(0.0, 0.3, "A1", "spk_A"))
            session.ingest_chunk(_chunk(0.4, 0.7, "B1", "spk_B"))
            session.ingest_chunk(_chunk(0.8, 1.1, "A2", "spk_A"))
            session.ingest_chunk(_chunk(1.2, 1.5, "B2", "spk_B"))
            session.ingest_chunk(_chunk(1.6, 1.9, "A3", "spk_A"))
            session.end_of_stream()

        # 5 turns: A, B, A, B, A
        assert len(callbacks.speaker_turns) == 5
        speakers = [t["speaker_id"] for t in callbacks.speaker_turns]
        assert speakers == ["spk_A", "spk_B", "spk_A", "spk_B", "spk_A"]

    def test_very_long_segment_text(
        self, mock_extractor: MagicMock, base_config: StreamingEnrichmentConfig
    ) -> None:
        """Long segment text is preserved in callbacks."""
        callbacks = ContractCallbacks()
        long_text = "word " * 1000  # 5000 character text

        with patch(
            "transcription.streaming_enrich.AudioSegmentExtractor",
            return_value=mock_extractor,
        ):
            session = StreamingEnrichmentSession(
                wav_path=Path("/tmp/fake.wav"),
                config=base_config,
                callbacks=callbacks,
            )

            session.ingest_chunk(_chunk(0.0, 60.0, long_text, "spk_0"))
            session.end_of_stream()

        assert len(callbacks.finalized_segments) == 1
        assert callbacks.finalized_segments[0].text == long_text
