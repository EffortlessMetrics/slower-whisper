"""Bounded PCM framing independent of network message boundaries.

The stable input is 16 kHz mono PCM S16LE. Normal classifier frames contain
320 samples (20 ms). At EOF, a short complete-sample tail is emitted once,
unpadded; an incomplete final sample is a typed terminal input error.
"""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import RuntimeNotReadyError, StreamingNegotiationError


@dataclass(frozen=True, slots=True)
class PCMFramingMetrics:
    """Sample accounting, not model-quality or transport-delivery evidence."""

    frame_samples: int
    accepted_bytes: int
    emitted_samples: int
    frames_emitted: int
    residual_bytes: int


class PCMFrameBuffer:
    """Retain less than one frame between bounded input messages.

    Frame size and message limits are server-owned. Returned frames contain at
    most one input message plus one carried frame, never accumulated session
    audio. No sample is padded, discarded, or classified twice on success.
    """

    def __init__(
        self,
        *,
        frame_samples: int = 320,
        max_chunk_bytes: int = 128 * 1024,
    ) -> None:
        for name, value in (
            ("frame_samples", frame_samples),
            ("max_chunk_bytes", max_chunk_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if frame_samples * 2 > max_chunk_bytes:
            raise ValueError("PCM frame must fit within max_chunk_bytes")
        self._frame_bytes = frame_samples * 2
        self._max_chunk_bytes = max_chunk_bytes
        self._residual = bytearray()
        self._ended = False
        self._failure: StreamingNegotiationError | None = None
        self._accepted_bytes = 0
        self._emitted_samples = 0
        self._frames_emitted = 0

    @property
    def metrics(self) -> PCMFramingMetrics:
        return PCMFramingMetrics(
            frame_samples=self._frame_bytes // 2,
            accepted_bytes=self._accepted_bytes,
            emitted_samples=self._emitted_samples,
            frames_emitted=self._frames_emitted,
            residual_bytes=len(self._residual),
        )

    def feed(self, pcm: bytes) -> tuple[bytes, ...]:
        """Frame one message; odd byte splits are valid until final EOF."""
        self._require_active()
        if not isinstance(pcm, bytes):
            raise TypeError("PCM input must be bytes")
        if len(pcm) > self._max_chunk_bytes:
            raise ValueError("PCM chunk exceeds max_chunk_bytes")

        self._accepted_bytes += len(pcm)
        frames: list[bytes] = []
        offset = 0
        if self._residual:
            take = min(self._frame_bytes - len(self._residual), len(pcm))
            self._residual.extend(pcm[:take])
            offset = take
            if len(self._residual) == self._frame_bytes:
                frames.append(bytes(self._residual))
                self._residual.clear()
        while offset + self._frame_bytes <= len(pcm):
            frames.append(pcm[offset : offset + self._frame_bytes])
            offset += self._frame_bytes
        self._residual.extend(pcm[offset:])
        self._frames_emitted += len(frames)
        self._emitted_samples += len(frames) * (self._frame_bytes // 2)
        return tuple(frames)

    def end(self) -> tuple[bytes, ...]:
        """Flush complete residual samples once; never invent a padding byte."""
        if self._failure is not None:
            raise RuntimeNotReadyError(
                "PCM framer failed",
                context={
                    "state": "failed",
                    "failure_reason_code": self._failure.reason_code,
                },
            )
        if self._ended:
            return ()
        self._ended = True
        if len(self._residual) % 2:
            residual_bytes = len(self._residual)
            self._residual.clear()
            self._failure = StreamingNegotiationError(
                "PCM stream ended with an incomplete 16-bit sample",
                reason_code="streaming_pcm_incomplete",
                context={
                    "violation": "incomplete_pcm_sample",
                    "residual_bytes": residual_bytes,
                },
            )
            raise self._failure
        if not self._residual:
            return ()
        tail = bytes(self._residual)
        self._residual.clear()
        self._frames_emitted += 1
        self._emitted_samples += len(tail) // 2
        return (tail,)

    def abort(self) -> None:
        """Release residual input without pretending it was processed."""
        self._residual.clear()
        self._ended = True

    def _require_active(self) -> None:
        if self._ended:
            raise RuntimeNotReadyError(
                "PCM framer is terminal",
                context={"state": "ended"},
            )
