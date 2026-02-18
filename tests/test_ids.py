"""Tests for ID generators in transcription/ids.py.

This module tests the ID generation functions and validation for:
- run_id: Format `run-YYYYMMDD-HHMMSS-XXXXXX`
- stream_id: Format `str-{uuid4}`
- event_id: Monotonically increasing positive integer
- segment_id: Format `seg-{seq}`

Tests cover:
- Format correctness
- Uniqueness guarantees
- Monotonicity (for event_id)
- Stability rules (for segment_id)
- Validation functions
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from slower_whisper.pipeline.ids import (
    RUN_ID_PREFIX,
    RUN_ID_RANDOM_LENGTH,
    STREAM_ID_PREFIX,
    EventIdCounter,
    SegmentIdCounter,
    generate_run_id,
    generate_segment_id,
    generate_stream_id,
    is_valid_run_id,
    is_valid_segment_id,
    is_valid_stream_id,
    parse_segment_sequence,
)


class TestRunId:
    """Tests for run_id generation and validation."""

    def test_format(self) -> None:
        """run_id follows format: run-YYYYMMDD-HHMMSS-XXXXXX."""
        run_id = generate_run_id()

        # Check prefix
        assert run_id.startswith(f"{RUN_ID_PREFIX}-")

        # Check overall format with regex
        pattern = rf"^{RUN_ID_PREFIX}-\d{{8}}-\d{{6}}-[a-z0-9]{{{RUN_ID_RANDOM_LENGTH}}}$"
        assert re.match(pattern, run_id), f"run_id '{run_id}' doesn't match expected format"

    def test_date_component(self) -> None:
        """run_id date component matches provided timestamp."""
        timestamp = datetime(2026, 1, 28, 14, 30, 52, tzinfo=UTC)
        run_id = generate_run_id(timestamp=timestamp)

        # Extract date portion
        parts = run_id.split("-")
        assert parts[1] == "20260128", f"Date part should be 20260128, got {parts[1]}"
        assert parts[2] == "143052", f"Time part should be 143052, got {parts[2]}"

    def test_uniqueness(self) -> None:
        """Generated run_ids should be unique."""
        run_ids = [generate_run_id() for _ in range(100)]
        assert len(set(run_ids)) == 100, "All generated run_ids should be unique"

    def test_validation_valid(self) -> None:
        """is_valid_run_id returns True for valid run_ids."""
        assert is_valid_run_id("run-20260128-143052-x7k9p2")
        assert is_valid_run_id("run-20250101-000000-aaaaaa")
        assert is_valid_run_id("run-20301231-235959-z9z9z9")

        # Generated IDs should always be valid
        for _ in range(10):
            assert is_valid_run_id(generate_run_id())

    def test_validation_invalid(self) -> None:
        """is_valid_run_id returns False for invalid formats."""
        # Wrong prefix
        assert not is_valid_run_id("str-20260128-143052-x7k9p2")
        assert not is_valid_run_id("RUN-20260128-143052-x7k9p2")

        # Wrong date format
        assert not is_valid_run_id("run-2026012-143052-x7k9p2")  # 7 digit date
        assert not is_valid_run_id("run-202601281-143052-x7k9p2")  # 9 digit date

        # Wrong time format
        assert not is_valid_run_id("run-20260128-14305-x7k9p2")  # 5 digit time
        assert not is_valid_run_id("run-20260128-1430522-x7k9p2")  # 7 digit time

        # Wrong random suffix
        assert not is_valid_run_id("run-20260128-143052-x7k9p")  # 5 chars
        assert not is_valid_run_id("run-20260128-143052-x7k9p22")  # 7 chars
        assert not is_valid_run_id("run-20260128-143052-X7K9P2")  # uppercase

        # Missing components
        assert not is_valid_run_id("run-20260128-143052")
        assert not is_valid_run_id("run-20260128")
        assert not is_valid_run_id("run")

        # Not a run_id at all
        assert not is_valid_run_id("")
        assert not is_valid_run_id("invalid")
        assert not is_valid_run_id("550e8400-e29b-41d4-a716-446655440000")


class TestStreamId:
    """Tests for stream_id generation and validation."""

    def test_format(self) -> None:
        """stream_id follows format: str-{uuid4}."""
        stream_id = generate_stream_id()

        # Check prefix
        assert stream_id.startswith(f"{STREAM_ID_PREFIX}-")

        # Check UUID portion
        uuid_pattern = r"^str-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        assert re.match(uuid_pattern, stream_id), (
            f"stream_id '{stream_id}' doesn't match expected format"
        )

    def test_uniqueness(self) -> None:
        """Generated stream_ids should be unique."""
        stream_ids = [generate_stream_id() for _ in range(100)]
        assert len(set(stream_ids)) == 100, "All generated stream_ids should be unique"

    def test_validation_valid(self) -> None:
        """is_valid_stream_id returns True for valid stream_ids."""
        assert is_valid_stream_id("str-550e8400-e29b-41d4-a716-446655440000")
        assert is_valid_stream_id("str-00000000-0000-0000-0000-000000000000")
        assert is_valid_stream_id("str-ffffffff-ffff-ffff-ffff-ffffffffffff")

        # Generated IDs should always be valid
        for _ in range(10):
            assert is_valid_stream_id(generate_stream_id())

    def test_validation_invalid(self) -> None:
        """is_valid_stream_id returns False for invalid formats."""
        # Wrong prefix
        assert not is_valid_stream_id("run-550e8400-e29b-41d4-a716-446655440000")
        assert not is_valid_stream_id("STR-550e8400-e29b-41d4-a716-446655440000")

        # Invalid UUID
        assert not is_valid_stream_id("str-invalid-uuid")
        assert not is_valid_stream_id("str-550e8400-e29b-41d4-a716")  # truncated

        # Not a stream_id at all
        assert not is_valid_stream_id("")
        assert not is_valid_stream_id("invalid")
        assert not is_valid_stream_id("550e8400-e29b-41d4-a716-446655440000")


class TestSegmentId:
    """Tests for segment_id generation and validation."""

    def test_format(self) -> None:
        """segment_id follows format: seg-{seq}."""
        assert generate_segment_id(0) == "seg-0"
        assert generate_segment_id(1) == "seg-1"
        assert generate_segment_id(42) == "seg-42"
        assert generate_segment_id(999) == "seg-999"

    def test_validation_valid(self) -> None:
        """is_valid_segment_id returns True for valid segment_ids."""
        assert is_valid_segment_id("seg-0")
        assert is_valid_segment_id("seg-1")
        assert is_valid_segment_id("seg-42")
        assert is_valid_segment_id("seg-999999")

    def test_validation_invalid(self) -> None:
        """is_valid_segment_id returns False for invalid formats."""
        # Wrong prefix
        assert not is_valid_segment_id("segment-0")
        assert not is_valid_segment_id("SEG-0")

        # Invalid sequence
        assert not is_valid_segment_id("seg-")
        assert not is_valid_segment_id("seg-abc")
        assert not is_valid_segment_id("seg--1")  # negative
        assert not is_valid_segment_id("seg-1.5")  # float

        # Not a segment_id at all
        assert not is_valid_segment_id("")
        assert not is_valid_segment_id("invalid")
        assert not is_valid_segment_id("0")

    def test_parse_sequence(self) -> None:
        """parse_segment_sequence extracts the sequence number."""
        assert parse_segment_sequence("seg-0") == 0
        assert parse_segment_sequence("seg-1") == 1
        assert parse_segment_sequence("seg-42") == 42
        assert parse_segment_sequence("seg-999") == 999

    def test_parse_sequence_invalid(self) -> None:
        """parse_segment_sequence raises ValueError for invalid segment_ids."""
        with pytest.raises(ValueError):
            parse_segment_sequence("invalid")

        with pytest.raises(ValueError):
            parse_segment_sequence("seg-abc")


class TestEventIdCounter:
    """Tests for event_id counter monotonicity."""

    def test_starts_at_one(self) -> None:
        """First event_id should be 1."""
        counter = EventIdCounter()
        assert counter.next() == 1

    def test_monotonically_increasing(self) -> None:
        """event_ids should strictly increase."""
        counter = EventIdCounter()

        prev_id = 0
        for _ in range(100):
            event_id = counter.next()
            assert event_id > prev_id, f"event_id {event_id} should be > {prev_id}"
            prev_id = event_id

    def test_increments_by_one(self) -> None:
        """event_ids should increment by exactly 1."""
        counter = EventIdCounter()

        for expected in range(1, 101):
            assert counter.next() == expected

    def test_current_property(self) -> None:
        """current property returns the last generated event_id."""
        counter = EventIdCounter()

        assert counter.current == 0  # Before any generation

        counter.next()
        assert counter.current == 1

        counter.next()
        counter.next()
        assert counter.current == 3

    def test_reset(self) -> None:
        """reset() resets the counter to 0."""
        counter = EventIdCounter()

        counter.next()
        counter.next()
        assert counter.current == 2

        counter.reset()
        assert counter.current == 0
        assert counter.next() == 1


class TestSegmentIdCounter:
    """Tests for segment_id counter."""

    def test_starts_at_seg_0(self) -> None:
        """First segment_id should be seg-0."""
        counter = SegmentIdCounter()
        assert counter.next() == "seg-0"

    def test_sequential_ids(self) -> None:
        """segment_ids should be sequential."""
        counter = SegmentIdCounter()

        assert counter.next() == "seg-0"
        assert counter.next() == "seg-1"
        assert counter.next() == "seg-2"

    def test_current_property(self) -> None:
        """current property returns the last generated segment_id."""
        counter = SegmentIdCounter()

        assert counter.current is None  # Before any generation

        counter.next()
        assert counter.current == "seg-0"

        counter.next()
        counter.next()
        assert counter.current == "seg-2"

    def test_next_sequence_property(self) -> None:
        """next_sequence property returns the next sequence number."""
        counter = SegmentIdCounter()

        assert counter.next_sequence == 0

        counter.next()
        assert counter.next_sequence == 1

        counter.next()
        counter.next()
        assert counter.next_sequence == 3

    def test_reset(self) -> None:
        """reset() resets the counter."""
        counter = SegmentIdCounter()

        counter.next()
        counter.next()
        assert counter.current == "seg-1"

        counter.reset()
        assert counter.current is None
        assert counter.next_sequence == 0
        assert counter.next() == "seg-0"

    def test_generated_ids_are_valid(self) -> None:
        """All generated segment_ids should pass validation."""
        counter = SegmentIdCounter()

        for _ in range(100):
            segment_id = counter.next()
            assert is_valid_segment_id(segment_id)


class TestIdIntegration:
    """Integration tests for ID usage patterns."""

    def test_streaming_session_id_pattern(self) -> None:
        """Simulate streaming session ID usage pattern."""
        # Create session IDs
        stream_id = generate_stream_id()
        event_counter = EventIdCounter()
        segment_counter = SegmentIdCounter()

        # Simulate streaming events
        events = []
        for _ in range(5):
            # PARTIAL events
            segment_id = segment_counter.next()
            for _ in range(3):  # Multiple partials per segment
                event_id = event_counter.next()
                events.append(
                    {
                        "event_id": event_id,
                        "stream_id": stream_id,
                        "segment_id": segment_id,
                        "type": "PARTIAL",
                    }
                )

            # FINALIZED event
            event_id = event_counter.next()
            events.append(
                {
                    "event_id": event_id,
                    "stream_id": stream_id,
                    "segment_id": segment_id,
                    "type": "FINALIZED",
                }
            )

        # Verify all event_ids are monotonically increasing
        event_ids = [e["event_id"] for e in events]
        assert event_ids == sorted(event_ids)
        assert len(set(event_ids)) == len(event_ids)  # All unique

        # Verify all segment_ids are valid
        for event in events:
            assert is_valid_segment_id(event["segment_id"])

        # Verify stream_id consistency
        stream_ids = {e["stream_id"] for e in events}
        assert len(stream_ids) == 1
        assert is_valid_stream_id(stream_ids.pop())

    def test_batch_run_id_pattern(self) -> None:
        """Simulate batch transcription run_id usage."""
        run_id = generate_run_id()

        # Simulate processing multiple files in a batch
        receipts = []
        for i in range(10):
            receipts.append(
                {
                    "run_id": run_id,
                    "file_index": i,
                    "status": "success",
                }
            )

        # All receipts should have the same run_id
        run_ids = {r["run_id"] for r in receipts}
        assert len(run_ids) == 1
        assert is_valid_run_id(run_ids.pop())
