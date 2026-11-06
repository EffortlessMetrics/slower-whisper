"""Stable identity across core streaming, enrichment, and persisted events."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from transcription.streaming import (
    StreamChunk,
    StreamConfig,
    StreamEvent,
    StreamEventType,
    StreamingSession,
    StreamSegment,
)
from transcription.streaming_enrich import StreamingEnrichmentConfig, StreamingEnrichmentSession


def _chunk(start: float, text: str, speaker: str = "speaker-a") -> StreamChunk:
    """Build one short chunk without an audio model."""
    return {"start": start, "end": start + 0.25, "text": text, "speaker_id": speaker}


def _lifecycle(session: StreamingSession | StreamingEnrichmentSession) -> list[StreamEvent]:
    """Exercise extension, gap, speaker boundary, and repeated EOF."""
    events = session.ingest_chunk(_chunk(0.0, "hello"))
    events += session.ingest_chunk(_chunk(0.5, "world"))
    events += session.ingest_chunk(_chunk(3.0, "next"))
    events += session.ingest_chunk(_chunk(3.5, "reply", "speaker-b"))
    events += session.end_of_stream()
    assert session.end_of_stream() == []
    assert [event.segment.text for event in events] == [
        "hello",
        "hello world",
        "hello world",
        "next",
        "next",
        "reply",
        "reply",
    ]
    return events


def _assert_identity(events: list[StreamEvent], *, emit_ids: bool) -> None:
    """Check both snapshots and JSON records retain the finalization key."""
    assert [event.type for event in events] == [
        StreamEventType.PARTIAL_SEGMENT,
        StreamEventType.PARTIAL_SEGMENT,
        StreamEventType.FINAL_SEGMENT,
        StreamEventType.PARTIAL_SEGMENT,
        StreamEventType.FINAL_SEGMENT,
        StreamEventType.PARTIAL_SEGMENT,
        StreamEventType.FINAL_SEGMENT,
    ]
    if not emit_ids:
        assert all(event.event_id is None and event.stream_id is None for event in events)
        assert all(event.segment.segment_id is None for event in events)
        return

    assert [event.event_id for event in events] == list(range(1, 8))
    stream_id = events[0].stream_id
    assert stream_id is not None
    assert {event.stream_id for event in events} == {stream_id}
    assert [event.segment.segment_id for event in events] == [
        "seg-0",
        "seg-0",
        "seg-0",
        "seg-1",
        "seg-1",
        "seg-2",
        "seg-2",
    ]

    # Persist the public dataclass fields, then match finals to their partials.
    records = [{**asdict(event), "type": event.type.value} for event in events]
    restored = json.loads(json.dumps(records))
    pending: dict[tuple[str, str], str] = {}
    finalized: dict[tuple[str, str], str] = {}
    for record in restored:
        key = (record["stream_id"], record["segment"]["segment_id"])
        text = record["segment"]["text"]
        if record["type"] == StreamEventType.PARTIAL_SEGMENT.value:
            assert key not in finalized
            pending[key] = text
        else:
            assert key not in finalized
            assert pending.pop(key) == text
            finalized[key] = text
    assert not pending
    assert list(finalized.values()) == ["hello world", "next", "reply"]


@pytest.mark.parametrize("emit_ids", [False, True])
def test_core_finalization_identity(emit_ids: bool) -> None:
    """Each core segment retains one identity from first partial through final."""
    session = StreamingSession(StreamConfig(emit_ids=emit_ids))
    _assert_identity(_lifecycle(session), emit_ids=emit_ids)


@pytest.mark.parametrize("emit_ids", [False, True])
@pytest.mark.parametrize("enrichment", ["disabled", "success", "failure"])
def test_enrichment_finalization_identity(
    monkeypatch: pytest.MonkeyPatch, emit_ids: bool, enrichment: str
) -> None:
    """Enrichment and graceful failures preserve IDs in events and callbacks."""
    monkeypatch.setattr(
        "transcription.streaming_enrich.AudioSegmentExtractor",
        lambda _: SimpleNamespace(duration_seconds=10.0, sample_rate=16000),
    )

    def enrich(**kwargs: object) -> dict[str, object]:
        """Provide deterministic enrichment or exercise its recovery path."""
        if enrichment == "failure":
            raise RuntimeError("synthetic enrichment failure")
        return {"rendering": "synthetic enrichment"}

    monkeypatch.setattr("transcription.streaming_enrich._enrich_segment_with_extractor", enrich)
    callbacks: list[StreamSegment] = []
    session = StreamingEnrichmentSession(
        Path("unused.wav"),
        StreamingEnrichmentConfig(
            base_config=StreamConfig(emit_ids=emit_ids),
            enable_prosody=enrichment != "disabled",
        ),
        callbacks=SimpleNamespace(on_segment_finalized=callbacks.append),
    )
    events = _lifecycle(session)
    _assert_identity(events, emit_ids=emit_ids)
    finals = [event.segment for event in events if event.type == StreamEventType.FINAL_SEGMENT]
    assert callbacks == finals
    for segment in finals:
        if enrichment == "disabled":
            assert segment.audio_state is None
        elif enrichment == "success":
            assert segment.audio_state == {"rendering": "synthetic enrichment"}
        else:
            assert segment.audio_state is not None
            assert segment.audio_state["extraction_status"]["prosody"] == "failed"

    session.reset()
    callbacks.clear()
    restarted = _lifecycle(session)
    _assert_identity(restarted, emit_ids=emit_ids)
    assert len(callbacks) == 3
    if emit_ids:
        assert restarted[0].stream_id != events[0].stream_id
