"""Revision-aware incremental ASR state for raw PCM streaming.

This module owns evolving utterance identity, absolute sample time, bounded
hypothesis cadence, explicit finality, and typed inference failure. Transport,
replay, WebSocket framing, and VAD detection remain separate ports.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from .exceptions import (
    ASRInferenceError,
    ASROutputError,
    RuntimeNotReadyError,
    TranscriptionError,
)

FinalReason = Literal["vad_boundary", "max_utterance", "end_of_stream"]
_FINAL_REASONS = frozenset({"vad_boundary", "max_utterance", "end_of_stream"})


class IncrementalASRBackend(Protocol):
    """Backend port used by :class:`IncrementalASRSession`."""

    def transcribe(
        self,
        pcm_s16le: bytes,
        *,
        sample_rate: int,
        start_sample: int,
        end_sample: int,
    ) -> object | Awaitable[object]: ...


SegmentIdFactory = Callable[[int], str]


class IncrementalASRState(StrEnum):
    """Lifecycle state for one incremental session."""

    ACTIVE = "active"
    FAILED = "failed"
    ENDED = "ended"


@dataclass(frozen=True, slots=True)
class IncrementalASRConfig:
    """Server-owned bounds for the stable PCM streaming contract."""

    sample_rate: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2
    encoding: str = "pcm_s16le"
    min_hypothesis_samples: int = 16_000
    hypothesis_interval_samples: int = 8_000
    hypothesis_backoff_factor: int = 2
    max_utterance_samples: int = 480_000
    max_chunk_bytes: int = 128 * 1024

    def __post_init__(self) -> None:
        if self.sample_rate != 16_000:
            raise ValueError("stable incremental ASR requires a 16 kHz sample rate")
        if self.channels != 1:
            raise ValueError("stable incremental ASR requires mono audio")
        if self.sample_width_bytes != 2:
            raise ValueError("stable incremental ASR requires 16-bit samples")
        if self.encoding != "pcm_s16le":
            raise ValueError("stable incremental ASR requires pcm_s16le encoding")
        if self.min_hypothesis_samples <= 0:
            raise ValueError("min_hypothesis_samples must be positive")
        if self.hypothesis_interval_samples <= 0:
            raise ValueError("hypothesis_interval_samples must be positive")
        if self.hypothesis_backoff_factor < 1:
            raise ValueError("hypothesis_backoff_factor must be at least 1")
        if self.max_utterance_samples < self.min_hypothesis_samples:
            raise ValueError(
                "max_utterance_samples must be at least min_hypothesis_samples"
            )
        if self.max_chunk_bytes <= 0:
            raise ValueError("max_chunk_bytes must be positive")
        if self.max_chunk_bytes % self.bytes_per_sample_frame != 0:
            raise ValueError("max_chunk_bytes must align to complete sample frames")

    @property
    def bytes_per_sample_frame(self) -> int:
        return self.channels * self.sample_width_bytes


@dataclass(frozen=True, slots=True)
class ASRRevision:
    """One replacement hypothesis or final form of a public utterance."""

    segment_id: str
    revision: int
    start_sample: int
    end_sample: int
    text: str
    final: bool
    final_reason: FinalReason | None = None

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id must not be empty")
        if self.revision <= 0:
            raise ValueError("revision must be positive")
        if self.start_sample < 0:
            raise ValueError("start_sample must not be negative")
        if self.end_sample < self.start_sample:
            raise ValueError("end_sample must not precede start_sample")
        if self.final != (self.final_reason is not None):
            raise ValueError("final and final_reason must agree")
        if self.final_reason is not None and self.final_reason not in _FINAL_REASONS:
            raise ValueError("final_reason is not a supported finalization reason")

    def to_seconds(self, sample_rate: int) -> tuple[float, float]:
        """Derive seconds without changing sample-clock authority."""
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        return self.start_sample / sample_rate, self.end_sample / sample_rate


@dataclass(frozen=True, slots=True)
class IncrementalASRMetrics:
    """Bounded-work receipt for an incremental session."""

    model_calls: int
    submitted_audio_samples: int
    peak_active_audio_bytes: int
    revisions_emitted: int
    segments_finalized: int
    absolute_samples_received: int


class IncrementalASRSession:
    """Packetization-invariant revision state for one PCM stream.

    ``speech`` is an authoritative upstream VAD decision for the complete
    supplied chunk. The caller must split a chunk at speech/silence boundaries.
    A non-speech chunk finalizes an active utterance before advancing the
    absolute sample clock through the silence.
    """

    def __init__(
        self,
        backend: IncrementalASRBackend,
        *,
        config: IncrementalASRConfig | None = None,
        segment_id_factory: SegmentIdFactory | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or IncrementalASRConfig()
        self._segment_id_factory = segment_id_factory or (
            lambda number: f"seg-{number:08d}"
        )

        self._state = IncrementalASRState.ACTIVE
        self._absolute_sample = 0
        self._active_start_sample: int | None = None
        self._active_segment_id: str | None = None
        self._active_audio = bytearray()
        self._active_revision = 0
        self._last_hypothesis_samples = 0
        self._last_hypothesis_text: str | None = None
        self._segment_number = 0
        self._seen_segment_ids: set[str] = set()

        self._model_calls = 0
        self._submitted_audio_samples = 0
        self._peak_active_audio_bytes = 0
        self._revisions_emitted = 0
        self._segments_finalized = 0
        self._failure: TranscriptionError | None = None

    @property
    def state(self) -> IncrementalASRState:
        return self._state

    @property
    def absolute_sample(self) -> int:
        return self._absolute_sample

    @property
    def active_segment_id(self) -> str | None:
        return self._active_segment_id

    @property
    def active_audio_bytes(self) -> int:
        return len(self._active_audio)

    @property
    def metrics(self) -> IncrementalASRMetrics:
        return IncrementalASRMetrics(
            model_calls=self._model_calls,
            submitted_audio_samples=self._submitted_audio_samples,
            peak_active_audio_bytes=self._peak_active_audio_bytes,
            revisions_emitted=self._revisions_emitted,
            segments_finalized=self._segments_finalized,
            absolute_samples_received=self._absolute_sample,
        )

    async def push_pcm(
        self,
        pcm_s16le: bytes,
        *,
        speech: bool,
    ) -> tuple[ASRRevision, ...]:
        """Consume one boundary-homogeneous PCM chunk."""
        self._require_active()
        pcm = self._validate_chunk(pcm_s16le)
        sample_count = len(pcm) // self.config.bytes_per_sample_frame
        if sample_count == 0:
            return ()

        if not speech:
            revisions: list[ASRRevision] = []
            if self._active_segment_id is not None:
                revisions.append(await self._emit(final_reason="vad_boundary"))
            self._absolute_sample += sample_count
            return tuple(revisions)

        return await self._consume_speech(pcm)

    async def end(self) -> tuple[ASRRevision, ...]:
        """Finalize an active utterance and end the session idempotently."""
        if self._state is IncrementalASRState.ENDED:
            return ()
        self._require_active()

        revisions: list[ASRRevision] = []
        if self._active_segment_id is not None:
            revisions.append(await self._emit(final_reason="end_of_stream"))
        self._state = IncrementalASRState.ENDED
        return tuple(revisions)

    async def _consume_speech(self, pcm: bytes) -> tuple[ASRRevision, ...]:
        frame_bytes = self.config.bytes_per_sample_frame
        remaining_samples = len(pcm) // frame_bytes
        byte_offset = 0
        revisions: list[ASRRevision] = []

        while remaining_samples > 0:
            self._ensure_active_utterance()
            active_samples = self._active_sample_count
            next_hypothesis = self._next_hypothesis_threshold()
            next_boundary = min(
                next_hypothesis,
                self.config.max_utterance_samples,
            )
            samples_to_boundary = next_boundary - active_samples
            if samples_to_boundary <= 0:
                raise RuntimeError("incremental ASR boundary did not advance")
            take_samples = min(remaining_samples, samples_to_boundary)
            take_bytes = take_samples * frame_bytes

            self._active_audio.extend(pcm[byte_offset : byte_offset + take_bytes])
            self._absolute_sample += take_samples
            remaining_samples -= take_samples
            byte_offset += take_bytes
            self._peak_active_audio_bytes = max(
                self._peak_active_audio_bytes,
                len(self._active_audio),
            )

            active_samples = self._active_sample_count
            if active_samples == self.config.max_utterance_samples:
                revisions.append(await self._emit(final_reason="max_utterance"))
            elif active_samples == next_hypothesis:
                revisions.append(await self._emit(final_reason=None))

        return tuple(revisions)

    def _ensure_active_utterance(self) -> None:
        if self._active_segment_id is not None:
            return
        self._segment_number += 1
        segment_id = self._segment_id_factory(self._segment_number)
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise ValueError("segment_id_factory returned an invalid identifier")
        if segment_id in self._seen_segment_ids:
            raise ValueError("segment_id_factory returned a duplicate identifier")
        self._seen_segment_ids.add(segment_id)
        self._active_segment_id = segment_id
        self._active_start_sample = self._absolute_sample
        self._active_revision = 0
        self._last_hypothesis_samples = 0
        self._last_hypothesis_text = None

    @property
    def _active_sample_count(self) -> int:
        return len(self._active_audio) // self.config.bytes_per_sample_frame

    def _next_hypothesis_threshold(self) -> int:
        if self._last_hypothesis_samples == 0:
            return self.config.min_hypothesis_samples
        linear_threshold = (
            self._last_hypothesis_samples + self.config.hypothesis_interval_samples
        )
        geometric_threshold = (
            self._last_hypothesis_samples * self.config.hypothesis_backoff_factor
        )
        return max(linear_threshold, geometric_threshold)

    async def _emit(self, *, final_reason: FinalReason | None) -> ASRRevision:
        segment_id = self._active_segment_id
        start_sample = self._active_start_sample
        if segment_id is None or start_sample is None:
            raise RuntimeError("cannot emit without an active utterance")

        end_sample = self._absolute_sample
        active_samples = self._active_sample_count
        cached_text = self._last_hypothesis_text
        reuse_cached_text = (
            final_reason is not None
            and active_samples == self._last_hypothesis_samples
            and cached_text is not None
        )

        if reuse_cached_text:
            text = cached_text
        else:
            pcm = bytes(self._active_audio)
            self._model_calls += 1
            self._submitted_audio_samples += active_samples
            try:
                result = self.backend.transcribe(
                    pcm,
                    sample_rate=self.config.sample_rate,
                    start_sample=start_sample,
                    end_sample=end_sample,
                )
                raw_text = await result if inspect.isawaitable(result) else result
            except (ASRInferenceError, ASROutputError) as error:
                self._fail(error)
                raise
            except Exception as error:  # noqa: BLE001 - preserve the local cause
                inference_error = ASRInferenceError(
                    "Incremental ASR inference failed",
                    context={
                        "phase": "incremental_hypothesis",
                        "segment_id": segment_id,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                    },
                )
                self._fail(inference_error)
                raise inference_error from error

            if not isinstance(raw_text, str):
                output_error = ASROutputError(
                    "Incremental ASR backend returned non-text output",
                    context={
                        "violation": "text_not_string",
                        "segment_id": segment_id,
                    },
                )
                self._fail(output_error)
                raise output_error
            text = raw_text.strip()
            self._last_hypothesis_text = text

        self._active_revision += 1
        revision = ASRRevision(
            segment_id=segment_id,
            revision=self._active_revision,
            start_sample=start_sample,
            end_sample=end_sample,
            text=text,
            final=final_reason is not None,
            final_reason=final_reason,
        )
        self._revisions_emitted += 1
        self._last_hypothesis_samples = active_samples

        if final_reason is not None:
            self._segments_finalized += 1
            self._reset_active_utterance()

        return revision

    def _reset_active_utterance(self) -> None:
        self._active_start_sample = None
        self._active_segment_id = None
        self._active_audio.clear()
        self._active_revision = 0
        self._last_hypothesis_samples = 0
        self._last_hypothesis_text = None

    def _fail(self, error: TranscriptionError) -> None:
        self._state = IncrementalASRState.FAILED
        self._failure = error
        self._reset_active_utterance()

    def _require_active(self) -> None:
        if self._state is IncrementalASRState.ACTIVE:
            return
        context: dict[str, object] = {"state": self._state.value}
        if self._failure is not None:
            context["failure_reason_code"] = self._failure.reason_code
        raise RuntimeNotReadyError(
            "Incremental ASR session is not active",
            context=context,
        )

    def _validate_chunk(self, pcm_s16le: bytes) -> bytes:
        if not isinstance(pcm_s16le, bytes):
            raise TypeError("pcm_s16le must be bytes")
        if len(pcm_s16le) > self.config.max_chunk_bytes:
            raise ValueError("PCM chunk exceeds max_chunk_bytes")
        if len(pcm_s16le) % self.config.bytes_per_sample_frame != 0:
            raise ValueError("PCM chunk must contain complete sample frames")
        return pcm_s16le
