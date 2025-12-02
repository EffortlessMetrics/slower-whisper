"""Streaming state machine tests for post-ASR chunks."""

from __future__ import annotations

import pytest

from transcription.streaming import (
    StreamChunk,
    StreamConfig,
    StreamEventType,
    StreamingSession,
)


def _chunk(start: float, end: float, text: str, speaker: str | None = None) -> StreamChunk:
    return {"start": start, "end": end, "text": text, "speaker_id": speaker}


def test_straight_line_stream() -> None:
    session = StreamingSession()

    first = _chunk(0.0, 0.5, "hello", "spk_0")
    second = _chunk(0.6, 1.2, "world", "spk_0")

    events1 = session.ingest_chunk(first)
    assert [e.type for e in events1] == [StreamEventType.PARTIAL_SEGMENT]
    assert events1[0].segment.text == "hello"

    events2 = session.ingest_chunk(second)
    assert [e.type for e in events2] == [StreamEventType.PARTIAL_SEGMENT]
    assert events2[0].segment.text == "hello world"
    assert events2[0].segment.end == pytest.approx(1.2)

    # Original snapshot should remain unchanged after extension.
    assert events1[0].segment.text == "hello"

    finals = session.end_of_stream()
    assert len(finals) == 1
    final_segment = finals[0].segment
    assert finals[0].type == StreamEventType.FINAL_SEGMENT
    assert final_segment.start == pytest.approx(0.0)
    assert final_segment.end == pytest.approx(1.2)
    assert final_segment.text == "hello world"
    assert final_segment.speaker_id == "spk_0"


def test_gap_triggers_new_segment() -> None:
    session = StreamingSession(StreamConfig(max_gap_sec=0.5))

    first = _chunk(0.0, 0.4, "hi", "spk_0")
    second = _chunk(2.0, 2.5, "there", "spk_0")  # gap of 1.6 > max_gap_sec

    session.ingest_chunk(first)
    events = session.ingest_chunk(second)

    assert [e.type for e in events] == [
        StreamEventType.FINAL_SEGMENT,
        StreamEventType.PARTIAL_SEGMENT,
    ]
    assert events[0].segment.text == "hi"
    assert events[1].segment.text == "there"
    assert events[1].segment.start == pytest.approx(2.0)


def test_speaker_change_finalizes_segment() -> None:
    session = StreamingSession()

    first = _chunk(0.0, 0.7, "hey", "spk_a")
    second = _chunk(0.8, 1.3, "you", "spk_b")

    session.ingest_chunk(first)
    events = session.ingest_chunk(second)

    assert [e.type for e in events] == [
        StreamEventType.FINAL_SEGMENT,
        StreamEventType.PARTIAL_SEGMENT,
    ]
    assert events[0].segment.speaker_id == "spk_a"
    assert events[1].segment.speaker_id == "spk_b"
    assert events[1].segment.text == "you"


def test_end_of_stream_without_chunks() -> None:
    session = StreamingSession()
    assert session.end_of_stream() == []


def test_non_monotonic_input_is_rejected() -> None:
    session = StreamingSession()

    session.ingest_chunk(_chunk(0.0, 0.5, "start"))
    with pytest.raises(ValueError, match="non-decreasing time order"):
        session.ingest_chunk(_chunk(0.4, 0.6, "backtrack"))
