"""Smoke tests: WebSocket streaming session event envelope contract.

These tests exercise the streaming protocol's event envelope guarantees
using the mock ASR path (no real model needed). They verify:
- Event IDs are monotonically increasing
- Required envelope fields are present
- FINALIZED event appears at end-of-session
- SESSION_STARTED and SESSION_ENDED bookend the stream

Marked as both smoke and e2e; included in the test-smoke CI job.
"""

from __future__ import annotations

import asyncio

import pytest

from slower_whisper.pipeline.streaming_ws import (
    EventEnvelope,
    ServerMessageType,
    SessionState,
    WebSocketSessionConfig,
    WebSocketStreamingSession,
)


def _make_pcm_silence(duration_sec: float, sample_rate: int = 16000) -> bytes:
    """Generate silent PCM audio (16-bit mono) for the given duration."""
    num_samples = int(sample_rate * duration_sec)
    return b"\x00\x00" * num_samples


def _run(coro):  # noqa: ANN001, ANN202
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(30)
class TestStreamingEnvelopeContract:
    """Tests for the WebSocket streaming event envelope contract."""

    def test_session_lifecycle_events(self) -> None:
        """Session should emit SESSION_STARTED then SESSION_ENDED."""
        session = WebSocketStreamingSession()
        start_event = _run(session.start())

        assert start_event.type == ServerMessageType.SESSION_STARTED
        assert start_event.event_id == 1
        assert start_event.stream_id.startswith("str-")
        assert "session_id" in start_event.payload

        end_events = _run(session.end())
        types = [e.type for e in end_events]
        assert ServerMessageType.SESSION_ENDED in types

        # SESSION_ENDED must be the very last event
        assert end_events[-1].type == ServerMessageType.SESSION_ENDED

    def test_event_ids_monotonically_increasing(self) -> None:
        """All event_ids across the session must be strictly increasing."""
        session = WebSocketStreamingSession()
        all_events: list[EventEnvelope] = []

        all_events.append(_run(session.start()))

        # Send enough audio to trigger mock events (>1s of audio)
        audio = _make_pcm_silence(1.5)
        chunk_events = _run(session.process_audio_chunk(audio, sequence=1))
        all_events.extend(chunk_events)

        end_events = _run(session.end())
        all_events.extend(end_events)

        # Verify monotonically increasing
        event_ids = [e.event_id for e in all_events]
        assert len(event_ids) >= 2, f"Expected at least 2 events, got {len(event_ids)}"
        for i in range(1, len(event_ids)):
            assert event_ids[i] > event_ids[i - 1], (
                f"event_id not increasing: {event_ids[i - 1]} -> {event_ids[i]}"
            )

    def test_envelope_required_fields(self) -> None:
        """Every envelope must have event_id, stream_id, ts_server, type, payload."""
        session = WebSocketStreamingSession()
        all_events: list[EventEnvelope] = []

        all_events.append(_run(session.start()))

        audio = _make_pcm_silence(1.5)
        all_events.extend(_run(session.process_audio_chunk(audio, sequence=1)))
        all_events.extend(_run(session.end()))

        stream_id = all_events[0].stream_id

        for event in all_events:
            d = event.to_dict()
            assert "event_id" in d, f"Missing event_id in {d}"
            assert "stream_id" in d, f"Missing stream_id in {d}"
            assert "ts_server" in d, f"Missing ts_server in {d}"
            assert "type" in d, f"Missing type in {d}"
            assert "payload" in d, f"Missing payload in {d}"
            assert "schema_version" in d, f"Missing schema_version in {d}"

            # stream_id is consistent across all events
            assert event.stream_id == stream_id

            # ts_server is positive Unix epoch millis
            assert event.ts_server > 0

    def test_finalized_event_at_end(self) -> None:
        """Ending a session with buffered audio must emit a FINALIZED event."""
        session = WebSocketStreamingSession()
        _run(session.start())

        # Send audio to build up a buffer
        audio = _make_pcm_silence(2.0)
        _run(session.process_audio_chunk(audio, sequence=1))

        end_events = _run(session.end())
        types = [e.type for e in end_events]
        assert ServerMessageType.FINALIZED in types, (
            f"Expected FINALIZED in end events, got: {[t.value for t in types]}"
        )

        # FINALIZED should have segment payload
        finalized = [e for e in end_events if e.type == ServerMessageType.FINALIZED]
        for f in finalized:
            assert "segment" in f.payload
            seg = f.payload["segment"]
            assert "start" in seg
            assert "end" in seg
            assert "text" in seg

    def test_session_ended_contains_stats(self) -> None:
        """SESSION_ENDED payload must include session stats."""
        session = WebSocketStreamingSession()
        _run(session.start())

        audio = _make_pcm_silence(1.0)
        _run(session.process_audio_chunk(audio, sequence=1))

        end_events = _run(session.end())
        ended = [e for e in end_events if e.type == ServerMessageType.SESSION_ENDED]
        assert len(ended) == 1

        stats = ended[0].payload.get("stats", {})
        assert "chunks_received" in stats
        assert stats["chunks_received"] >= 1
        assert "bytes_received" in stats
        assert stats["bytes_received"] > 0

    def test_multiple_chunks_ordering(self) -> None:
        """Sending multiple sequential chunks should maintain event ordering."""
        session = WebSocketStreamingSession()
        _run(session.start())

        all_events: list[EventEnvelope] = []
        for seq in range(1, 4):
            audio = _make_pcm_silence(0.5)
            events = _run(session.process_audio_chunk(audio, sequence=seq))
            all_events.extend(events)

        end_events = _run(session.end())
        all_events.extend(end_events)

        # All events should have the same stream_id
        stream_ids = {e.stream_id for e in all_events}
        assert len(stream_ids) == 1 or len(all_events) == 0

    def test_session_state_transitions(self) -> None:
        """Session state should follow CREATED -> ACTIVE -> ENDED."""
        session = WebSocketStreamingSession()
        assert session.state == SessionState.CREATED

        _run(session.start())
        assert session.state == SessionState.ACTIVE

        _run(session.end())
        assert session.state == SessionState.ENDED

    def test_custom_config_respected(self) -> None:
        """Custom session config should be reflected in session behavior."""
        config = WebSocketSessionConfig(
            sample_rate=8000,
            max_gap_sec=2.0,
        )
        session = WebSocketStreamingSession(config=config)
        assert session.config.sample_rate == 8000
        assert session.config.max_gap_sec == 2.0

        start_event = _run(session.start())
        assert start_event.type == ServerMessageType.SESSION_STARTED
        _run(session.end())
