"""Public-route controller for revision-aware streaming ASR.

The legacy WebSocket session remains the protocol parser and VAD owner. This
controller replaces its ASR adapter and event projection on the public route so
one evolving utterance produces replacement revisions, not appended prefixes or
placeholder text. Typed inference failure is terminal and externally bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any

from .exceptions import RuntimeNotReadyError, TranscriptionError
from .incremental_asr import ASRRevision, IncrementalASRConfig, IncrementalASRSession
from .service_runtime import ASRRuntime
from .streaming_runtime_asr import RuntimeIncrementalASRBackend
from .streaming_ws import EventType, SessionState, WebSocketStreamingSession

_PLACEHOLDER_TEXT = frozenset({"[processing...]", "[final segment]"})


@dataclass(frozen=True, slots=True)
class _LegacyChunkProjection:
    """Duck-typed return object consumed by the legacy protocol session."""

    text: str
    start_time: float
    end_time: float
    is_final: bool
    confidence: float = 1.0
    language: str | None = None


class RevisionStreamingASRAdapter:
    """Translate legacy adapter calls into one incremental revision session."""

    def __init__(self, incremental: IncrementalASRSession) -> None:
        self.incremental = incremental
        self.pending_revisions: list[ASRRevision] = []
        self.pending_error: TranscriptionError | None = None
        self._pending_silence = bytearray()

    async def process_audio_chunk(
        self,
        audio_data: bytes,
        *,
        is_speech: bool = True,
        **_kwargs: Any,
    ) -> _LegacyChunkProjection | None:
        try:
            if is_speech:
                if self._pending_silence:
                    self.pending_revisions.extend(
                        await self.incremental.push_pcm(
                            bytes(self._pending_silence),
                            speech=True,
                        )
                    )
                    self._pending_silence.clear()
                self.pending_revisions.extend(
                    await self.incremental.push_pcm(audio_data, speech=True)
                )
            else:
                self._pending_silence.extend(audio_data)
        except TranscriptionError as error:
            self.pending_error = error
        return self._legacy_projection()

    async def finalize_segment(self, **_kwargs: Any) -> _LegacyChunkProjection | None:
        try:
            if self._pending_silence:
                self.pending_revisions.extend(
                    await self.incremental.push_pcm(
                        bytes(self._pending_silence),
                        speech=False,
                    )
                )
                self._pending_silence.clear()
            else:
                self.pending_revisions.extend(await self.incremental.end())
        except TranscriptionError as error:
            self.pending_error = error
        return self._legacy_projection()

    async def end(self) -> _LegacyChunkProjection | None:
        """Finalize the stream without inventing trailing-silence time."""
        self._pending_silence.clear()
        try:
            self.pending_revisions.extend(await self.incremental.end())
        except TranscriptionError as error:
            self.pending_error = error
        return self._legacy_projection()

    def reset(self) -> None:
        self._pending_silence.clear()
        self.pending_revisions.clear()
        self.pending_error = None

    def _legacy_projection(self) -> _LegacyChunkProjection | None:
        if not self.pending_revisions:
            return None
        revision = self.pending_revisions[-1]
        start, end = revision.to_seconds(self.incremental.config.sample_rate)
        return _LegacyChunkProjection(
            text=revision.text,
            start_time=start,
            end_time=end,
            is_final=revision.final,
        )


class RevisionStreamingController:
    """Install the stable incremental ASR path on one public WebSocket session."""

    def __init__(
        self,
        session: WebSocketStreamingSession,
        runtime: ASRRuntime,
        *,
        config: IncrementalASRConfig | None = None,
    ) -> None:
        if not runtime.ready:
            raise RuntimeNotReadyError(
                "The process-owned ASR runtime is not ready",
                context={"state": runtime.state.value},
            )
        self.session = session
        self.incremental = IncrementalASRSession(
            RuntimeIncrementalASRBackend(runtime),
            config=config,
        )
        self.adapter = RevisionStreamingASRAdapter(self.incremental)
        self._original_send_event = session.send_event
        session.asr_adapter = self.adapter
        session.send_event = MethodType(_send_revision_event, session)
        setattr(session, "_revision_streaming_controller", self)

    async def handle_start_session(self, config_data: dict[str, Any]) -> None:
        """Reject unsupported audio negotiation before session/model work."""
        sample_rate = config_data.get("sample_rate", 16_000)
        channels = config_data.get("channels", 1)
        audio_format = config_data.get("audio_format", "pcm_s16le")
        normalized_format = (
            audio_format.value if hasattr(audio_format, "value") else str(audio_format)
        )
        mismatches: dict[str, object] = {}
        if sample_rate != self.incremental.config.sample_rate:
            mismatches["sample_rate"] = sample_rate
        if channels != self.incremental.config.channels:
            mismatches["channels"] = channels
        if normalized_format.lower() != self.incremental.config.encoding:
            mismatches["audio_format"] = normalized_format
        if mismatches:
            raise RuntimeNotReadyError(
                "Streaming audio configuration is unsupported by stable incremental ASR",
                context={"mismatches": mismatches},
            )
        await self.session.handle_start_session(config_data)

    async def handle_end_session(self) -> None:
        """Finalize incremental state before the protocol terminal event."""
        await self.adapter.end()
        if self.adapter.pending_error is not None:
            await self._emit_pending_error()
        if self.adapter.pending_revisions:
            await self._emit_pending_revisions()
        await self.session.handle_end_session()

    async def _emit_pending_revisions(self) -> None:
        revisions = tuple(self.adapter.pending_revisions)
        self.adapter.pending_revisions.clear()
        for revision in revisions:
            event_type = EventType.FINALIZED if revision.final else EventType.PARTIAL
            start, end = revision.to_seconds(self.incremental.config.sample_rate)
            data = {
                "segment_id": revision.segment_id,
                "revision": revision.revision,
                "text": revision.text,
                "start_sample": revision.start_sample,
                "end_sample": revision.end_sample,
                "sample_rate": self.incremental.config.sample_rate,
                "start": start,
                "end": end,
                "start_time": start,
                "end_time": end,
                "final": revision.final,
                "final_reason": revision.final_reason,
            }
            await self._original_send_event(
                event_type,
                data,
                segment_id=revision.segment_id,
            )

    async def _emit_pending_error(self) -> None:
        error = self.adapter.pending_error
        self.adapter.pending_error = None
        if error is None:
            return
        self.adapter.pending_revisions.clear()
        self.adapter._pending_silence.clear()
        self.session.state = SessionState.ERROR
        await self._original_send_event(
            EventType.ERROR,
            {
                "code": error.reason_code,
                "message": "Streaming ASR failed",
                "recoverable": False,
                "context": dict(error.context),
            },
        )
        raise error


async def _send_revision_event(
    session: WebSocketStreamingSession,
    event_type: EventType,
    data: dict[str, Any] | None = None,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Replace append-shaped ASR events while passing protocol events through."""
    controller = getattr(session, "_revision_streaming_controller", None)
    if not isinstance(controller, RevisionStreamingController):
        raise RuntimeError("revision streaming controller is not installed")

    normalized_type = event_type.value if hasattr(event_type, "value") else str(event_type)
    asr_event_types = {EventType.PARTIAL.value, EventType.FINALIZED.value}
    if normalized_type in asr_event_types:
        if controller.adapter.pending_error is not None:
            await controller._emit_pending_error()
        if controller.adapter.pending_revisions:
            await controller._emit_pending_revisions()
            return
        text = (data or {}).get("text")
        if text in _PLACEHOLDER_TEXT:
            return

    await controller._original_send_event(event_type, data or {}, *args, **kwargs)
