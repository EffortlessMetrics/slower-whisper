"""Bounded one-writer transport authority for public WebSocket streaming."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .exceptions import (
    StreamingInboundLimitError,
    StreamingSessionLimitError,
    StreamingTransportCapacityError,
    StreamingTransportDeliveryError,
)
from .streaming_ws import EventEnvelope, ServerMessageType, WebSocketStreamingSession

SendCallable = Callable[[dict[str, Any]], Awaitable[None]]


class EventDurability(str, Enum):
    """Transport and replay treatment for one outbound event."""

    COALESCIBLE = "coalescible"
    TRANSIENT = "transient"
    DURABLE = "durable"


@dataclass(frozen=True, slots=True)
class StreamingTransportLimits:
    """Server-owned bounds for one process-local streaming connection."""

    max_queue_events: int = 128
    max_queue_bytes: int = 2 * 1024 * 1024
    pressure_events: int = 96
    pressure_bytes: int = 1536 * 1024
    terminal_event_reserve: int = 1
    terminal_byte_reserve: int = 64 * 1024
    max_durable_replay_events: int = 256
    max_durable_replay_bytes: int = 4 * 1024 * 1024
    max_transient_replay_events: int = 64
    max_transient_replay_bytes: int = 512 * 1024
    max_decoded_audio_message_bytes: int = 512 * 1024
    max_session_duration_sec: float = 60.0 * 60.0
    max_active_sessions: int = 32
    send_timeout_sec: float = 30.0
    drain_timeout_sec: float = 30.0

    def __post_init__(self) -> None:
        positive = {
            "max_queue_events": self.max_queue_events,
            "max_queue_bytes": self.max_queue_bytes,
            "pressure_events": self.pressure_events,
            "pressure_bytes": self.pressure_bytes,
            "terminal_event_reserve": self.terminal_event_reserve,
            "terminal_byte_reserve": self.terminal_byte_reserve,
            "max_durable_replay_events": self.max_durable_replay_events,
            "max_durable_replay_bytes": self.max_durable_replay_bytes,
            "max_transient_replay_events": self.max_transient_replay_events,
            "max_transient_replay_bytes": self.max_transient_replay_bytes,
            "max_decoded_audio_message_bytes": self.max_decoded_audio_message_bytes,
            "max_session_duration_sec": self.max_session_duration_sec,
            "max_active_sessions": self.max_active_sessions,
            "send_timeout_sec": self.send_timeout_sec,
            "drain_timeout_sec": self.drain_timeout_sec,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Streaming transport limits must be positive: {invalid}")
        if self.pressure_events >= self.max_queue_events:
            raise ValueError("pressure_events must be below max_queue_events")
        if self.pressure_bytes >= self.max_queue_bytes:
            raise ValueError("pressure_bytes must be below max_queue_bytes")
        if self.terminal_event_reserve >= self.max_queue_events:
            raise ValueError("terminal_event_reserve must leave ordinary queue capacity")
        if self.terminal_byte_reserve >= self.max_queue_bytes:
            raise ValueError("terminal_byte_reserve must leave ordinary byte capacity")


DEFAULT_STREAMING_TRANSPORT_LIMITS = StreamingTransportLimits()


@dataclass(slots=True)
class StreamingTransportMetrics:
    """Evidence counters for bounded delivery and replay behavior."""

    enqueued_events: int = 0
    enqueued_bytes: int = 0
    delivered_events: int = 0
    delivered_bytes: int = 0
    coalesced_partials: int = 0
    dropped_partials: int = 0
    pressure_waits: int = 0
    delivery_failures: int = 0
    durable_replay_evictions: int = 0
    transient_replay_evictions: int = 0
    max_queued_events: int = 0
    max_queued_bytes: int = 0


@dataclass(slots=True)
class _ReplayEntry:
    event: EventEnvelope
    size: int


class DeliveredReplayLedger:
    """Replay history containing only events successfully written to the socket."""

    def __init__(self, limits: StreamingTransportLimits) -> None:
        self._limits = limits
        self._stream_id: str | None = None
        self._durable: deque[_ReplayEntry] = deque()
        self._transient: deque[_ReplayEntry] = deque()
        self._durable_bytes = 0
        self._transient_bytes = 0
        self._durable_evicted_through = 0
        self.durable_evictions = 0
        self.transient_evictions = 0

    @property
    def stream_id(self) -> str | None:
        return self._stream_id

    def bind_stream(self, stream_id: str) -> None:
        if self._stream_id == stream_id:
            return
        self._stream_id = stream_id
        self._durable.clear()
        self._transient.clear()
        self._durable_bytes = 0
        self._transient_bytes = 0
        self._durable_evicted_through = 0

    def record(
        self,
        event: EventEnvelope,
        size: int,
        durability: EventDurability,
    ) -> None:
        if self._stream_id is None or event.stream_id != self._stream_id:
            return
        entry = _ReplayEntry(event=event, size=size)
        if durability is EventDurability.DURABLE:
            self._durable.append(entry)
            self._durable_bytes += size
            self._trim_durable()
            return
        self._transient.append(entry)
        self._transient_bytes += size
        self._trim_transient()

    def events_since(self, last_event_id: int) -> tuple[list[EventEnvelope], bool]:
        gap_detected = last_event_id < self._durable_evicted_through
        entries = [
            entry
            for entry in (*self._durable, *self._transient)
            if entry.event.event_id > last_event_id
        ]
        entries.sort(key=lambda entry: entry.event.event_id)
        return [entry.event for entry in entries], gap_detected

    def _trim_durable(self) -> None:
        while (
            len(self._durable) > self._limits.max_durable_replay_events
            or self._durable_bytes > self._limits.max_durable_replay_bytes
        ):
            removed = self._durable.popleft()
            self._durable_bytes -= removed.size
            self._durable_evicted_through = max(
                self._durable_evicted_through,
                removed.event.event_id,
            )
            self.durable_evictions += 1

    def _trim_transient(self) -> None:
        while (
            len(self._transient) > self._limits.max_transient_replay_events
            or self._transient_bytes > self._limits.max_transient_replay_bytes
        ):
            removed = self._transient.popleft()
            self._transient_bytes -= removed.size
            self.transient_evictions += 1


@dataclass(slots=True)
class _PendingEvent:
    event: EventEnvelope
    encoded: dict[str, Any]
    size: int
    durability: EventDurability
    replay_eligible: bool


def event_durability(event: EventEnvelope) -> EventDurability:
    """Return server-owned delivery/replay policy for an event."""

    if event.type is ServerMessageType.PARTIAL:
        return EventDurability.COALESCIBLE
    if event.type in {
        ServerMessageType.PONG,
        ServerMessageType.PHYSICS_UPDATE,
        ServerMessageType.AUDIO_HEALTH,
        ServerMessageType.VAD_ACTIVITY,
        ServerMessageType.BARGE_IN,
        ServerMessageType.END_OF_TURN_HINT,
    }:
        return EventDurability.TRANSIENT
    return EventDurability.DURABLE


def serialized_event_size(encoded: dict[str, Any]) -> int:
    """Measure one event using the compact JSON representation sent on the wire."""

    return len(
        json.dumps(
            encoded,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


class OutboundEventController:
    """Single bounded writer and delivered-event replay authority."""

    def __init__(
        self,
        send: SendCallable,
        *,
        limits: StreamingTransportLimits = DEFAULT_STREAMING_TRANSPORT_LIMITS,
    ) -> None:
        self._send = send
        self.limits = limits
        self.metrics = StreamingTransportMetrics()
        self.replay = DeliveredReplayLedger(limits)
        self._queue: deque[_PendingEvent] = deque()
        self._queued_bytes = 0
        self._condition = asyncio.Condition()
        self._writer_task: asyncio.Task[None] | None = None
        self._closing = False
        self._aborted = False
        self._in_flight = False
        self._failure: BaseException | None = None

    @property
    def queued_events(self) -> int:
        return len(self._queue)

    @property
    def queued_bytes(self) -> int:
        return self._queued_bytes

    @property
    def failed(self) -> bool:
        return self._failure is not None

    async def start(self) -> None:
        if self._writer_task is not None:
            return
        self._writer_task = asyncio.create_task(
            self._writer_loop(),
            name="slower-whisper-websocket-writer",
        )

    def bind_stream(self, stream_id: str) -> None:
        self.replay.bind_stream(stream_id)

    async def publish(
        self,
        event: EventEnvelope,
        *,
        replay_eligible: bool = True,
        terminal: bool = False,
    ) -> bool:
        await self.start()
        encoded = event.to_dict()
        item = _PendingEvent(
            event=event,
            encoded=encoded,
            size=serialized_event_size(encoded),
            durability=event_durability(event),
            replay_eligible=replay_eligible,
        )
        async with self._condition:
            self._raise_if_unusable()
            if self._coalesce_pending_partial(item):
                self._condition.notify_all()
                return True
            event_limit = self.limits.max_queue_events
            if not terminal:
                event_limit -= self.limits.terminal_event_reserve
            byte_limit = self.limits.max_queue_bytes
            if not terminal:
                byte_limit -= self.limits.terminal_byte_reserve
            over_events = len(self._queue) >= event_limit
            over_bytes = self._queued_bytes + item.size > byte_limit
            if over_events or over_bytes:
                if item.durability is EventDurability.COALESCIBLE:
                    self.metrics.dropped_partials += 1
                    return False
                raise StreamingTransportCapacityError(
                    "Outbound WebSocket queue capacity exceeded",
                    context={
                        "queued_events": len(self._queue),
                        "queued_bytes": self._queued_bytes,
                        "event_size": item.size,
                        "terminal": terminal,
                    },
                )
            self._queue.append(item)
            self._queued_bytes += item.size
            self.metrics.enqueued_events += 1
            self.metrics.enqueued_bytes += item.size
            self.metrics.max_queued_events = max(
                self.metrics.max_queued_events,
                len(self._queue),
            )
            self.metrics.max_queued_bytes = max(
                self.metrics.max_queued_bytes,
                self._queued_bytes,
            )
            self._condition.notify_all()
        return True

    async def publish_many(
        self,
        events: list[EventEnvelope],
        *,
        replay_eligible: bool = True,
    ) -> None:
        for event in events:
            await self.publish(event, replay_eligible=replay_eligible)

    async def publish_replay(self, events: list[EventEnvelope]) -> None:
        for event in events:
            await self.publish(event, replay_eligible=False)

    async def publish_terminal(self, event: EventEnvelope) -> None:
        await self.publish(event, terminal=True)

    async def wait_below_pressure(self) -> None:
        async with self._condition:
            waited = False
            while (
                len(self._queue) >= self.limits.pressure_events
                or self._queued_bytes >= self.limits.pressure_bytes
            ):
                self._raise_if_unusable()
                waited = True
                await self._condition.wait()
            if waited:
                self.metrics.pressure_waits += 1
            self._raise_if_unusable()

    async def drain(self) -> None:
        await self.start()

        async def _wait() -> None:
            async with self._condition:
                while self._queue or self._in_flight:
                    self._raise_if_unusable()
                    await self._condition.wait()
                self._raise_if_unusable()

        try:
            await asyncio.wait_for(_wait(), timeout=self.limits.drain_timeout_sec)
        except TimeoutError as error:
            raise StreamingTransportDeliveryError(
                "Timed out draining WebSocket delivery",
                context={
                    "queued_events": len(self._queue),
                    "queued_bytes": self._queued_bytes,
                },
            ) from error

    async def close(self, *, drain: bool = True) -> None:
        if self._writer_task is None:
            self._closing = True
            return
        if drain:
            await self.drain()
        async with self._condition:
            self._closing = True
            self._condition.notify_all()
        try:
            await self._writer_task
        finally:
            self._writer_task = None

    async def abort(self) -> None:
        async with self._condition:
            self._aborted = True
            self._queue.clear()
            self._queued_bytes = 0
            self._condition.notify_all()
        task = self._writer_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._writer_task = None
        self._in_flight = False

    def replay_since(self, last_event_id: int) -> tuple[list[EventEnvelope], bool]:
        return self.replay.events_since(last_event_id)

    def metrics_receipt(self) -> dict[str, Any]:
        receipt = asdict(self.metrics)
        receipt.update(
            {
                "queued_events": len(self._queue),
                "queued_bytes": self._queued_bytes,
                "failed": self.failed,
                "durable_replay_evictions": self.replay.durable_evictions,
                "transient_replay_evictions": self.replay.transient_evictions,
            }
        )
        return receipt

    def _coalesce_pending_partial(self, item: _PendingEvent) -> bool:
        if (
            item.durability is not EventDurability.COALESCIBLE
            or item.event.segment_id is None
        ):
            return False
        for index in range(len(self._queue) - 1, -1, -1):
            existing = self._queue[index]
            if (
                existing.durability is EventDurability.COALESCIBLE
                and existing.event.segment_id == item.event.segment_id
            ):
                replacement_bytes = self._queued_bytes - existing.size + item.size
                ordinary_byte_limit = (
                    self.limits.max_queue_bytes
                    - self.limits.terminal_byte_reserve
                )
                if replacement_bytes > ordinary_byte_limit:
                    self.metrics.dropped_partials += 1
                    return True
                self._queued_bytes = replacement_bytes
                self._queue[index] = item
                self.metrics.coalesced_partials += 1
                self.metrics.enqueued_events += 1
                self.metrics.enqueued_bytes += item.size
                self.metrics.max_queued_bytes = max(
                    self.metrics.max_queued_bytes,
                    self._queued_bytes,
                )
                return True
        return False

    def _raise_if_unusable(self) -> None:
        if self._failure is not None:
            raise StreamingTransportDeliveryError(
                "WebSocket writer failed",
                context={"failure_type": type(self._failure).__name__},
            ) from self._failure
        if self._aborted:
            raise StreamingTransportDeliveryError("WebSocket writer was aborted")
        if self._closing:
            raise StreamingTransportDeliveryError("WebSocket writer is closed")

    async def _writer_loop(self) -> None:
        while True:
            async with self._condition:
                while not self._queue and not self._closing and not self._aborted:
                    await self._condition.wait()
                if self._aborted:
                    return
                if self._closing and not self._queue:
                    return
                item = self._queue.popleft()
                self._queued_bytes -= item.size
                self._in_flight = True
                self._condition.notify_all()
            try:
                await asyncio.wait_for(
                    self._send(item.encoded),
                    timeout=self.limits.send_timeout_sec,
                )
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError):
                    raise
                async with self._condition:
                    self._failure = error
                    self.metrics.delivery_failures += 1
                    self._queue.clear()
                    self._queued_bytes = 0
                    self._in_flight = False
                    self._condition.notify_all()
                return
            if item.replay_eligible:
                self.replay.record(item.event, item.size, item.durability)
            async with self._condition:
                self.metrics.delivered_events += 1
                self.metrics.delivered_bytes += item.size
                self._in_flight = False
                self._condition.notify_all()


class StreamingAdmissionController:
    """Process-local active-session admission bound."""

    def __init__(self, max_active_sessions: int) -> None:
        if max_active_sessions <= 0:
            raise ValueError("max_active_sessions must be positive")
        self._max_active_sessions = max_active_sessions
        self._active: set[str] = set()
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def acquire(self, stream_id: str) -> None:
        async with self._lock:
            if stream_id in self._active:
                return
            if len(self._active) >= self._max_active_sessions:
                raise StreamingSessionLimitError(
                    "Streaming session admission limit reached",
                    context={
                        "active_sessions": len(self._active),
                        "max_active_sessions": self._max_active_sessions,
                    },
                )
            self._active.add(stream_id)

    async def release(self, stream_id: str) -> None:
        async with self._lock:
            self._active.discard(stream_id)


def encoded_audio_upper_bound(encoded_length: int) -> int:
    """Maximum decoded bytes represented by a base64 string length."""

    if encoded_length < 0:
        raise ValueError("encoded_length cannot be negative")
    return ((encoded_length + 3) // 4) * 3


def decode_audio_chunk_bounded(
    payload: dict[str, Any],
    *,
    max_decoded_bytes: int,
) -> tuple[bytes, int]:
    """Decode an audio chunk after enforcing encoded and decoded size bounds."""

    data_b64 = payload.get("data")
    if not isinstance(data_b64, str) or not data_b64:
        raise ValueError("Missing 'data' field in AUDIO_CHUNK")
    sequence = payload.get("sequence")
    if sequence is None:
        raise ValueError("Missing 'sequence' field in AUDIO_CHUNK")
    max_encoded_length = ((max_decoded_bytes + 2) // 3) * 4
    if len(data_b64) > max_encoded_length:
        raise StreamingInboundLimitError(
            "Encoded WebSocket audio message exceeds the server limit",
            context={
                "encoded_length": len(data_b64),
                "max_encoded_length": max_encoded_length,
                "max_decoded_bytes": max_decoded_bytes,
            },
        )
    if encoded_audio_upper_bound(len(data_b64)) > max_decoded_bytes + 2:
        raise StreamingInboundLimitError(
            "WebSocket audio message exceeds the decoded server limit",
            context={"max_decoded_bytes": max_decoded_bytes},
        )
    try:
        audio_bytes = base64.b64decode(data_b64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Invalid base64 audio data") from error
    if len(audio_bytes) > max_decoded_bytes:
        raise StreamingInboundLimitError(
            "Decoded WebSocket audio message exceeds the server limit",
            context={
                "decoded_bytes": len(audio_bytes),
                "max_decoded_bytes": max_decoded_bytes,
            },
        )
    return audio_bytes, int(sequence)


class _DisabledReplayBuffer:
    oldest_event_id = 0

    def add(self, _event: EventEnvelope) -> None:
        return

    def get_events_since(self, _last_event_id: int) -> tuple[list[EventEnvelope], bool]:
        return [], False

    def clear(self) -> None:
        return


def disable_legacy_replay(session: WebSocketStreamingSession) -> None:
    """Make delivered transport state the only replay authority."""

    session._replay_buffer = _DisabledReplayBuffer()  # type: ignore[assignment]


def session_deadline(
    started_at: float,
    *,
    limits: StreamingTransportLimits = DEFAULT_STREAMING_TRANSPORT_LIMITS,
) -> float:
    """Return remaining allowed duration or raise a typed limit error."""

    remaining = limits.max_session_duration_sec - (time.monotonic() - started_at)
    if remaining <= 0:
        raise StreamingSessionLimitError(
            "Streaming session duration limit reached",
            context={"max_session_duration_sec": limits.max_session_duration_sec},
        )
    return remaining
