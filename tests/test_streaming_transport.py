"""Contracts for the bounded one-writer WebSocket transport."""

from __future__ import annotations

import asyncio
import ast
import base64
from pathlib import Path
from typing import Any

import pytest

from transcription.exceptions import (
    StreamingInboundLimitError,
    StreamingSessionLimitError,
    StreamingTransportCapacityError,
    StreamingTransportDeliveryError,
)
from transcription.streaming_transport import (
    EventDurability,
    OutboundEventController,
    StreamingAdmissionController,
    StreamingTransportLimits,
    decode_audio_chunk_bounded,
    disable_legacy_replay,
    event_durability,
)
from transcription.streaming_ws import (
    EventEnvelope,
    ServerMessageType,
    WebSocketStreamingSession,
)


def _event(
    event_id: int,
    event_type: ServerMessageType,
    *,
    segment_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id,
        stream_id="str-00000000-0000-4000-8000-000000000001",
        type=event_type,
        ts_server=1_700_000_000_000 + event_id,
        payload=payload or {},
        segment_id=segment_id,
    )


def _limits(**overrides: Any) -> StreamingTransportLimits:
    values: dict[str, Any] = {
        "max_queue_events": 8,
        "max_queue_bytes": 16_384,
        "pressure_events": 6,
        "pressure_bytes": 12_288,
        "terminal_event_reserve": 1,
        "terminal_byte_reserve": 1_024,
        "max_durable_replay_events": 8,
        "max_durable_replay_bytes": 16_384,
        "max_transient_replay_events": 4,
        "max_transient_replay_bytes": 8_192,
        "max_decoded_audio_message_bytes": 32,
        "max_session_duration_sec": 60.0,
        "max_active_sessions": 2,
        "send_timeout_sec": 2.0,
        "drain_timeout_sec": 2.0,
    }
    values.update(overrides)
    return StreamingTransportLimits(**values)


@pytest.mark.asyncio
async def test_one_writer_coalesces_pending_partial_by_segment() -> None:
    release = asyncio.Event()
    entered = asyncio.Event()
    sent: list[dict[str, Any]] = []

    async def send(encoded: dict[str, Any]) -> None:
        entered.set()
        await release.wait()
        sent.append(encoded)

    transport = OutboundEventController(send, limits=_limits())
    transport.bind_stream(_event(1, ServerMessageType.SESSION_STARTED).stream_id)

    await transport.publish(_event(1, ServerMessageType.SESSION_STARTED))
    await entered.wait()
    await transport.publish(
        _event(
            2,
            ServerMessageType.PARTIAL,
            segment_id="seg-1",
            payload={"text": "old"},
        )
    )
    await transport.publish(
        _event(
            3,
            ServerMessageType.PARTIAL,
            segment_id="seg-1",
            payload={"text": "new"},
        )
    )

    release.set()
    await transport.drain()
    await transport.close()

    assert [item["event_id"] for item in sent] == [1, 3]
    assert sent[-1]["payload"]["text"] == "new"
    assert transport.metrics.coalesced_partials == 1


@pytest.mark.asyncio
async def test_replay_records_only_successfully_delivered_events() -> None:
    release = asyncio.Event()
    entered = asyncio.Event()

    async def send(_encoded: dict[str, Any]) -> None:
        entered.set()
        await release.wait()

    event = _event(1, ServerMessageType.SESSION_STARTED)
    transport = OutboundEventController(send, limits=_limits())
    transport.bind_stream(event.stream_id)

    await transport.publish(event)
    await entered.wait()
    assert transport.replay_since(0) == ([], False)

    release.set()
    await transport.drain()
    replay, gap = transport.replay_since(0)
    await transport.close()

    assert not gap
    assert replay == [event]


@pytest.mark.asyncio
async def test_partial_replay_churn_cannot_evict_durable_history() -> None:
    sent: list[dict[str, Any]] = []

    async def send(encoded: dict[str, Any]) -> None:
        sent.append(encoded)

    limits = _limits(
        max_transient_replay_events=1,
        max_transient_replay_bytes=256,
    )
    transport = OutboundEventController(send, limits=limits)
    start = _event(1, ServerMessageType.SESSION_STARTED)
    transport.bind_stream(start.stream_id)
    await transport.publish(start)
    for event_id in range(2, 7):
        await transport.publish(
            _event(
                event_id,
                ServerMessageType.PARTIAL,
                segment_id=f"seg-{event_id}",
                payload={"text": "x" * 80},
            )
        )
    await transport.drain()

    replay, gap = transport.replay_since(0)
    await transport.close()

    assert not gap
    assert start in replay
    assert transport.replay.transient_evictions > 0
    assert transport.replay.durable_evictions == 0


@pytest.mark.asyncio
async def test_durable_replay_eviction_reports_gap() -> None:
    async def send(_encoded: dict[str, Any]) -> None:
        return

    transport = OutboundEventController(
        send,
        limits=_limits(max_durable_replay_events=2),
    )
    stream_id = _event(1, ServerMessageType.FINALIZED).stream_id
    transport.bind_stream(stream_id)
    for event_id in range(1, 4):
        await transport.publish(
            _event(
                event_id,
                ServerMessageType.FINALIZED,
                segment_id=f"seg-{event_id}",
            )
        )
    await transport.drain()

    replay, gap = transport.replay_since(0)
    await transport.close()

    assert gap
    assert [event.event_id for event in replay] == [2, 3]


@pytest.mark.asyncio
async def test_terminal_slot_is_reserved_from_ordinary_durable_events() -> None:
    release = asyncio.Event()
    entered = asyncio.Event()

    async def send(_encoded: dict[str, Any]) -> None:
        entered.set()
        await release.wait()

    limits = _limits(
        max_queue_events=3,
        pressure_events=2,
        terminal_event_reserve=1,
    )
    transport = OutboundEventController(send, limits=limits)
    transport.bind_stream(_event(1, ServerMessageType.SESSION_STARTED).stream_id)
    await transport.publish(_event(1, ServerMessageType.SESSION_STARTED))
    await entered.wait()
    await transport.publish(_event(2, ServerMessageType.FINALIZED))
    await transport.publish(_event(3, ServerMessageType.FINALIZED))

    with pytest.raises(StreamingTransportCapacityError):
        await transport.publish(_event(4, ServerMessageType.FINALIZED))

    await transport.publish_terminal(_event(5, ServerMessageType.ERROR))
    release.set()
    await transport.drain()
    await transport.close()


@pytest.mark.asyncio
async def test_pressure_gate_waits_for_writer_progress() -> None:
    release = asyncio.Event()
    entered = asyncio.Event()

    async def send(_encoded: dict[str, Any]) -> None:
        entered.set()
        await release.wait()

    transport = OutboundEventController(
        send,
        limits=_limits(pressure_events=2),
    )
    await transport.publish(_event(1, ServerMessageType.SESSION_STARTED))
    await entered.wait()
    await transport.publish(_event(2, ServerMessageType.FINALIZED))
    await transport.publish(_event(3, ServerMessageType.FINALIZED))

    waiter = asyncio.create_task(transport.wait_below_pressure())
    await asyncio.sleep(0)
    assert not waiter.done()

    release.set()
    await waiter
    await transport.drain()
    await transport.close()
    assert transport.metrics.pressure_waits == 1


@pytest.mark.asyncio
async def test_writer_failure_is_explicit_and_releases_pending_memory() -> None:
    async def send(_encoded: dict[str, Any]) -> None:
        raise ConnectionError("closed")

    transport = OutboundEventController(send, limits=_limits())
    await transport.publish(_event(1, ServerMessageType.SESSION_STARTED))

    with pytest.raises(StreamingTransportDeliveryError) as captured:
        await transport.drain()

    assert isinstance(captured.value.__cause__, ConnectionError)
    assert transport.queued_events == 0
    assert transport.queued_bytes == 0
    assert transport.metrics.delivery_failures == 1
    await transport.abort()


@pytest.mark.asyncio
async def test_admission_controller_is_bounded_and_idempotent() -> None:
    admission = StreamingAdmissionController(1)
    await admission.acquire("one")
    await admission.acquire("one")
    assert admission.active_count == 1

    with pytest.raises(StreamingSessionLimitError):
        await admission.acquire("two")

    await admission.release("one")
    await admission.acquire("two")
    assert admission.active_count == 1


def test_decode_rejects_oversize_before_base64_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden_decode(*_args: Any, **_kwargs: Any) -> bytes:
        nonlocal called
        called = True
        raise AssertionError("decoder should not run")

    monkeypatch.setattr(
        "transcription.streaming_transport.base64.b64decode",
        forbidden_decode,
    )
    with pytest.raises(StreamingInboundLimitError):
        decode_audio_chunk_bounded(
            {"data": "A" * 48, "sequence": 1},
            max_decoded_bytes=16,
        )
    assert not called


def test_decode_is_strict_and_enforces_exact_decoded_bound() -> None:
    with pytest.raises(ValueError, match="Invalid base64"):
        decode_audio_chunk_bounded(
            {"data": "not base64!", "sequence": 1},
            max_decoded_bytes=32,
        )

    encoded = base64.b64encode(b"x" * 33).decode("ascii")
    with pytest.raises(StreamingInboundLimitError):
        decode_audio_chunk_bounded(
            {"data": encoded, "sequence": 2},
            max_decoded_bytes=32,
        )


def test_event_durability_keeps_finality_and_errors_durable() -> None:
    assert (
        event_durability(_event(1, ServerMessageType.PARTIAL))
        is EventDurability.COALESCIBLE
    )
    assert (
        event_durability(_event(2, ServerMessageType.PONG))
        is EventDurability.TRANSIENT
    )
    assert (
        event_durability(_event(3, ServerMessageType.FINALIZED))
        is EventDurability.DURABLE
    )
    assert (
        event_durability(_event(4, ServerMessageType.ERROR))
        is EventDurability.DURABLE
    )


def test_route_has_no_direct_websocket_send_authority() -> None:
    route_path = Path("transcription/service_runtime_streaming.py")
    source = route_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden = {
        "send_json",
        "send_text",
        "send_bytes",
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden
    ]
    assert calls == []


def test_legacy_session_replay_is_disabled() -> None:
    session = WebSocketStreamingSession()
    disable_legacy_replay(session)
    event = session.create_pong_event(1)

    assert session.get_events_for_resume(0) == ([], False)
    assert event.event_id == 1
