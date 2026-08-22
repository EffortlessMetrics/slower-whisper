"""Public-route controller for revision-aware streaming ASR.

The legacy WebSocket session remains the protocol-envelope and statistics owner.
This controller replaces only its append-shaped ASR projection: one evolving
utterance produces replacement revisions with stable sample identity. Typed
inference failure is terminal and externally bounded.
"""

from __future__ import annotations

import inspect
import math
import struct
import time
from collections.abc import Awaitable
from typing import Any, Protocol

from .exceptions import (
    ASRInferenceError,
    ASROutputError,
    RuntimeNotReadyError,
    StreamingNegotiationError,
    TranscriptionError,
)
from .incremental_asr import (
    ASRRevision,
    IncrementalASRConfig,
    IncrementalASRSession,
)
from .service_runtime import ASRRuntime
from .streaming_runtime_asr import RuntimeIncrementalASRBackend
from .streaming_ws import (
    EventEnvelope,
    ServerMessageType,
    SessionState,
    WebSocketStreamingSession,
)


class SpeechClassifier(Protocol):
    """Classify one boundary-homogeneous PCM chunk."""

    def is_speech(
        self,
        pcm_s16le: bytes,
        *,
        sample_rate: int,
    ) -> bool | Awaitable[bool]: ...


class PCMChunkEnergyClassifier:
    """Bounded dependency-free RMS classifier for stable PCM input."""

    def __init__(self, *, threshold: float = 0.01) -> None:
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("speech energy threshold must be finite and within [0, 1]")
        self.threshold = threshold

    def is_speech(
        self,
        pcm_s16le: bytes,
        *,
        sample_rate: int,
    ) -> bool:
        if sample_rate != 16_000:
            raise ValueError("speech classification requires a 16 kHz sample rate")
        if len(pcm_s16le) % 2 != 0:
            raise ValueError("speech classification requires complete 16-bit samples")
        sample_count = len(pcm_s16le) // 2
        if sample_count == 0:
            return False
        sum_squares = sum(sample * sample for (sample,) in struct.iter_unpack("<h", pcm_s16le))
        normalized_rms = math.sqrt(sum_squares / sample_count) / 32_768.0
        return normalized_rms > self.threshold


_UNSUPPORTED_FEATURE_FLAGS = (
    "enable_prosody",
    "enable_emotion",
    "enable_categorical_emotion",
    "enable_diarization",
    "enable_conversation_physics",
    "enable_audio_health",
    "enable_reflex_events",
    "enable_tts_style",
    "enable_correction_detection",
    "enable_commitment_tracking",
)


class RevisionStreamingController:
    """Run real incremental ASR through one existing WebSocket session."""

    def __init__(
        self,
        session: WebSocketStreamingSession,
        runtime: ASRRuntime,
        *,
        config: IncrementalASRConfig | None = None,
        classifier: SpeechClassifier | None = None,
    ) -> None:
        if not runtime.ready:
            raise RuntimeNotReadyError(
                "The process-owned ASR runtime is not ready",
                context={"state": runtime.state.value},
            )
        self.session = session
        self.runtime = runtime
        self.incremental = IncrementalASRSession(
            RuntimeIncrementalASRBackend(runtime),
            config=config,
        )
        self.classifier = classifier or PCMChunkEnergyClassifier()
        self._pending_silence = bytearray()
        self._terminal_error: TranscriptionError | None = None

        frame_bytes = self.incremental.config.bytes_per_sample_frame
        requested_gap_bytes = int(
            max(0.0, self.session.config.max_gap_sec)
            * self.incremental.config.sample_rate
            * frame_bytes
        )
        bounded_gap_bytes = min(
            max(frame_bytes, requested_gap_bytes),
            self.incremental.config.max_chunk_bytes,
        )
        self._silence_limit_bytes = bounded_gap_bytes - (bounded_gap_bytes % frame_bytes)
        self._silence_limit_bytes = max(frame_bytes, self._silence_limit_bytes)

    @property
    def pending_silence_bytes(self) -> int:
        return len(self._pending_silence)

    async def start(self, config_data: dict[str, Any]) -> EventEnvelope:
        """Validate the stable surface before starting protocol state."""
        mismatches = self._negotiation_mismatches(config_data)
        if mismatches:
            raise StreamingNegotiationError(
                "Streaming audio configuration is unsupported",
                context={"mismatches": mismatches},
            )
        return await self.session.start()

    async def process_audio_chunk(
        self,
        audio_data: bytes,
        sequence: int,
    ) -> list[EventEnvelope]:
        """Process one PCM chunk and return canonical revision envelopes."""
        self._require_active()
        self._validate_audio(audio_data)
        self._validate_sequence(sequence)

        self.session._last_chunk_sequence = sequence
        self.session.stats.chunks_received += 1
        self.session.stats.bytes_received += len(audio_data)

        try:
            raw_decision = self.classifier.is_speech(
                audio_data,
                sample_rate=self.incremental.config.sample_rate,
            )
            decision = await raw_decision if inspect.isawaitable(raw_decision) else raw_decision
            if not isinstance(decision, bool):
                raise ASROutputError(
                    "Streaming speech classifier returned invalid output",
                    context={"violation": "speech_decision_not_boolean"},
                )

            revisions: list[ASRRevision] = []
            if decision:
                bridge_silence = self.incremental.active_segment_id is not None
                revisions.extend(await self._flush_pending_silence(as_speech=bridge_silence))
                revisions.extend(await self.incremental.push_pcm(audio_data, speech=True))
            else:
                revisions.extend(await self._consume_silence(audio_data))
        except Exception as error:  # noqa: BLE001 - converted to typed terminal state
            raise self._record_failure(error, phase="process_audio") from error

        events = self._revision_envelopes(revisions)
        self.session.stats.events_sent += len(events)
        return events

    async def end(self) -> list[EventEnvelope]:
        """Finalize real ASR before emitting the protocol terminal event."""
        self._require_active()
        self.session.state = SessionState.ENDING
        try:
            revisions = list(await self._flush_pending_silence(as_speech=False))
            revisions.extend(await self.incremental.end())
        except Exception as error:  # noqa: BLE001 - converted to typed terminal state
            raise self._record_failure(error, phase="end_session") from error

        events = self._revision_envelopes(revisions)
        self.session.stats.end_time = time.time()
        self.session.stats.events_sent += len(events) + 1
        events.append(
            self.session._create_envelope(
                ServerMessageType.SESSION_ENDED,
                {"stats": self.session.stats.to_dict()},
            )
        )
        self.session.state = SessionState.ENDED
        self.session._audio_buffer.clear()
        return events

    def terminal_error_event(self, error: Exception) -> EventEnvelope:
        """Create one sanitized terminal event through session envelope authority."""
        typed = (
            error
            if isinstance(error, TranscriptionError)
            else self._record_failure(error, phase="route")
        )
        self._terminal_error = typed
        self._pending_silence.clear()
        self.session.state = SessionState.ERROR
        self.session.stats.errors += 1
        self.session.stats.events_sent += 1

        if isinstance(typed, StreamingNegotiationError):
            message = "Unsupported streaming audio configuration"
        elif isinstance(typed, RuntimeNotReadyError):
            message = "Streaming ASR runtime is not ready"
        else:
            message = "Streaming ASR failed"

        return self.session._create_envelope(
            ServerMessageType.ERROR,
            {
                "code": typed.reason_code,
                "message": message,
                "recoverable": False,
                "context": dict(typed.context),
            },
        )

    def abort(self) -> None:
        """Release bounded per-connection buffers after transport loss."""
        self._pending_silence.clear()
        self.session._audio_buffer.clear()
        if self.session.state not in {SessionState.ENDED, SessionState.ERROR}:
            self.session.state = SessionState.ENDED

    async def _consume_silence(self, audio_data: bytes) -> list[ASRRevision]:
        revisions: list[ASRRevision] = []
        offset = 0
        while offset < len(audio_data):
            capacity = self._silence_limit_bytes - len(self._pending_silence)
            take = min(capacity, len(audio_data) - offset)
            self._pending_silence.extend(audio_data[offset : offset + take])
            offset += take
            if len(self._pending_silence) == self._silence_limit_bytes:
                revisions.extend(await self._flush_pending_silence(as_speech=False))
        return revisions

    async def _flush_pending_silence(
        self,
        *,
        as_speech: bool,
    ) -> tuple[ASRRevision, ...]:
        if not self._pending_silence:
            return ()
        buffered = bytes(self._pending_silence)
        self._pending_silence.clear()
        return await self.incremental.push_pcm(buffered, speech=as_speech)

    def _revision_envelopes(
        self,
        revisions: list[ASRRevision],
    ) -> list[EventEnvelope]:
        events: list[EventEnvelope] = []
        for revision in revisions:
            start, end = revision.to_seconds(self.incremental.config.sample_rate)
            segment = {
                "start": start,
                "end": end,
                "text": revision.text,
                "speaker_id": None,
            }
            if revision.final:
                segment["audio_state"] = None
                self.session.stats.segments_finalized += 1
                event_type = ServerMessageType.FINALIZED
            else:
                self.session.stats.segments_partial += 1
                event_type = ServerMessageType.PARTIAL

            events.append(
                self.session._create_envelope(
                    event_type,
                    {
                        "segment": segment,
                        "revision": revision.revision,
                        "text": revision.text,
                        "start_sample": revision.start_sample,
                        "end_sample": revision.end_sample,
                        "sample_rate": self.incremental.config.sample_rate,
                        "final": revision.final,
                        "final_reason": revision.final_reason,
                    },
                    segment_id=revision.segment_id,
                    ts_audio_start=start,
                    ts_audio_end=end,
                )
            )
        return events

    def _negotiation_mismatches(
        self,
        config_data: dict[str, Any],
    ) -> dict[str, object]:
        mismatches: dict[str, object] = {}
        sample_rate = config_data.get("sample_rate", 16_000)
        channels = config_data.get("channels", 1)
        audio_format = config_data.get("audio_format", "pcm_s16le")
        normalized_format = (
            audio_format.value if hasattr(audio_format, "value") else str(audio_format)
        )
        max_gap_sec = config_data.get("max_gap_sec", self.session.config.max_gap_sec)

        if type(sample_rate) is not int or sample_rate != 16_000:
            mismatches["sample_rate"] = sample_rate
        if type(channels) is not int or channels != 1:
            mismatches["channels"] = channels
        if normalized_format.lower() != "pcm_s16le":
            mismatches["audio_format"] = normalized_format
        if (
            isinstance(max_gap_sec, bool)
            or not isinstance(max_gap_sec, int | float)
            or not math.isfinite(float(max_gap_sec))
            or float(max_gap_sec) <= 0.0
        ):
            mismatches["max_gap_sec"] = max_gap_sec
        for field_name in _UNSUPPORTED_FEATURE_FLAGS:
            if field_name in config_data and config_data[field_name] is not False:
                mismatches[field_name] = config_data[field_name]
        return mismatches

    def _validate_audio(self, audio_data: bytes) -> None:
        if not isinstance(audio_data, bytes):
            raise TypeError("audio_data must be bytes")
        if len(audio_data) > self.incremental.config.max_chunk_bytes:
            raise ValueError("audio chunk exceeds max_chunk_bytes")
        if len(audio_data) % self.incremental.config.bytes_per_sample_frame != 0:
            raise ValueError("audio chunk must contain complete sample frames")

    def _validate_sequence(self, sequence: int) -> None:
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise TypeError("audio sequence must be an integer")
        previous = self.session._last_chunk_sequence
        if previous is not None and sequence <= previous:
            raise ValueError(f"audio sequence {sequence} must be greater than {previous}")

    def _require_active(self) -> None:
        if self._terminal_error is not None:
            raise RuntimeNotReadyError(
                "Streaming ASR session is terminal",
                context={
                    "state": "error",
                    "failure_reason_code": self._terminal_error.reason_code,
                },
            )
        if self.session.state is not SessionState.ACTIVE:
            raise RuntimeNotReadyError(
                "Streaming session is not active",
                context={"state": self.session.state.value},
            )

    def _record_failure(
        self,
        error: Exception,
        *,
        phase: str,
    ) -> TranscriptionError:
        if isinstance(error, TranscriptionError):
            typed = error
        else:
            typed = ASRInferenceError(
                "Streaming ASR failed",
                context={
                    "phase": phase,
                    "violation": "unexpected_streaming_error",
                },
            )
            typed.__cause__ = error
        self._terminal_error = typed
        self._pending_silence.clear()
        self.session.state = SessionState.ERROR
        return typed
