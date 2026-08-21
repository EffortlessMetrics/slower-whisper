"""Service-owned ASR runtime lifecycle and capacity boundary.

Direct Python and CLI calls continue to construct their own engines. The FastAPI
service owns one configured runtime per process so lifecycle and readiness have
one authority. REST reuse is a later integration seam tracked in issue #623.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from .asr_engine import TranscriptionEngine
from .config import AsrConfig, TranscriptionConfig
from .exceptions import (
    ASRModelLoadError,
    RuntimeNotReadyError,
    RuntimeProfileMismatchError,
    TranscriptionError,
)
from .models import Transcript

logger = logging.getLogger(__name__)

EngineFactory = Callable[[AsrConfig], Any]


class TranscribeFileFn(Protocol):
    def __call__(
        self,
        audio_path: str | Path,
        root: str | Path,
        config: TranscriptionConfig,
        *,
        _engine: Any | None = None,
    ) -> Transcript: ...


class RuntimeState(StrEnum):
    """Observable service runtime lifecycle."""

    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Resolved ASR settings fixed for one service process."""

    model: str
    device: str
    compute_type: str
    language: str | None
    task: str
    beam_size: int
    vad_min_silence_ms: int
    word_timestamps: bool

    @classmethod
    def from_config(cls, config: TranscriptionConfig) -> RuntimeProfile:
        if config.device not in {"cpu", "cuda"}:
            raise ValueError(
                "Service runtime device must be resolved to 'cpu' or 'cuda', "
                f"got {config.device!r}"
            )
        compute_type = config.compute_type
        if compute_type is None:
            raise ValueError("TranscriptionConfig must resolve compute_type")
        return cls(
            model=config.model,
            device=config.device,
            compute_type=compute_type,
            language=config.language,
            task=config.task,
            beam_size=config.beam_size,
            vad_min_silence_ms=config.vad_min_silence_ms,
            word_timestamps=config.word_timestamps,
        )

    def to_asr_config(self) -> AsrConfig:
        return AsrConfig(
            model_name=self.model,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
            task=self.task,
            beam_size=self.beam_size,
            vad_min_silence_ms=self.vad_min_silence_ms,
            word_timestamps=self.word_timestamps,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "task": self.task,
            "beam_size": self.beam_size,
            "vad_min_silence_ms": self.vad_min_silence_ms,
            "word_timestamps": self.word_timestamps,
        }

    def mismatches(self, config: TranscriptionConfig) -> dict[str, dict[str, Any]]:
        requested = RuntimeProfile.from_config(config).to_dict()
        configured = self.to_dict()
        return {
            field: {"configured": configured[field], "requested": value}
            for field, value in requested.items()
            if value != configured[field]
        }


class ASRRuntime:
    """One bounded ASR engine owned by one service process."""

    def __init__(
        self,
        profile: RuntimeProfile,
        *,
        engine_factory: EngineFactory = TranscriptionEngine,
        max_concurrency: int = 1,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self.profile = profile
        self.max_concurrency = max_concurrency
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self._state = RuntimeState.STOPPED
        self._startup_error: TranscriptionError | None = None
        self._startup_attempts: list[dict[str, Any]] = []
        self._state_lock = asyncio.Lock()
        self._inference_limit = asyncio.Semaphore(max_concurrency)
        self._active_inferences = 0
        self._max_observed_inferences = 0

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def ready(self) -> bool:
        return self._state == RuntimeState.READY and self._engine is not None

    @property
    def max_observed_inferences(self) -> int:
        return self._max_observed_inferences

    @property
    def engine(self) -> Any:
        if not self.ready:
            raise self._not_ready_error()
        return self._engine

    def _not_ready_error(self) -> RuntimeNotReadyError:
        context: dict[str, Any] = {"state": self._state.value}
        if self._startup_error is not None:
            context["startup_error"] = self._startup_error.public_details()
        return RuntimeNotReadyError(
            "The configured ASR runtime is not ready",
            context=context,
        )

    async def start(self) -> None:
        """Initialize the configured engine once without taking down the app."""
        async with self._state_lock:
            if self.ready:
                return
            self._state = RuntimeState.STARTING
            self._startup_error = None
            self._startup_attempts = []
            try:
                engine = await asyncio.to_thread(
                    self._engine_factory,
                    self.profile.to_asr_config(),
                )
            except TranscriptionError as exc:
                self._engine = None
                self._startup_error = exc
                self._startup_attempts = self._attempts_from_error(exc)
                self._state = RuntimeState.FAILED
                logger.error("ASR runtime initialization failed", exc_info=exc)
                return
            except Exception as exc:  # noqa: BLE001 - preserve local cause
                load_error = ASRModelLoadError(
                    "The configured ASR runtime could not be initialized",
                    context={
                        "model": self.profile.model,
                        "device": self.profile.device,
                        "compute_type": self.profile.compute_type,
                    },
                )
                load_error.__cause__ = exc
                self._engine = None
                self._startup_error = load_error
                self._startup_attempts = self._attempts_from_error(load_error)
                self._state = RuntimeState.FAILED
                logger.exception("Unexpected ASR runtime initialization failure")
                return

            self._engine = engine
            self._startup_attempts = [
                dict(attempt)
                for attempt in getattr(engine, "model_load_attempts", [])
                if isinstance(attempt, Mapping)
            ]
            self._state = RuntimeState.READY

    async def close(self) -> None:
        """Close the owned engine at most once and always reach stopped state."""
        async with self._state_lock:
            if self._state == RuntimeState.STOPPED and self._engine is None:
                return
            self._state = RuntimeState.STOPPING
            engine = self._engine
            self._engine = None
            try:
                if engine is not None:
                    close = getattr(engine, "close", None)
                    if callable(close):
                        if inspect.iscoroutinefunction(close):
                            await close()
                        else:
                            await asyncio.to_thread(close)
            finally:
                self._state = RuntimeState.STOPPED

    def selected_profile(self) -> dict[str, Any] | None:
        if self._engine is None:
            return None
        cfg = getattr(self._engine, "cfg", None)
        return {
            "model": getattr(cfg, "model_name", self.profile.model),
            "device": getattr(cfg, "device", self.profile.device),
            "compute_type": getattr(cfg, "compute_type", self.profile.compute_type),
            "language": getattr(cfg, "language", self.profile.language),
            "task": getattr(cfg, "task", self.profile.task),
            "beam_size": getattr(cfg, "beam_size", self.profile.beam_size),
            "vad_min_silence_ms": getattr(
                cfg,
                "vad_min_silence_ms",
                self.profile.vad_min_silence_ms,
            ),
            "word_timestamps": getattr(
                cfg,
                "word_timestamps",
                self.profile.word_timestamps,
            ),
        }

    def status(self) -> dict[str, Any]:
        error = self._startup_error.public_details() if self._startup_error else None
        return {
            "state": self._state.value,
            "ready": self.ready,
            "profile": self.profile.to_dict(),
            "selected": self.selected_profile(),
            "attempts": [dict(attempt) for attempt in self._startup_attempts],
            "max_concurrency": self.max_concurrency,
            "active_inferences": self._active_inferences,
            "max_observed_inferences": self._max_observed_inferences,
            "error": error,
        }

    def assert_profile(self, config: TranscriptionConfig) -> None:
        mismatches = self.profile.mismatches(config)
        if mismatches:
            raise RuntimeProfileMismatchError(
                "The request does not match the configured service ASR profile",
                context={"mismatches": mismatches},
            )

    async def transcribe_file(
        self,
        transcribe: TranscribeFileFn,
        *,
        audio_path: str | Path,
        root: str | Path,
        config: TranscriptionConfig,
    ) -> Transcript:
        """Run a compatible file orchestrator through the owned engine."""
        engine = self.engine
        self.assert_profile(config)
        async with self._inference_limit:
            self._active_inferences += 1
            self._max_observed_inferences = max(
                self._max_observed_inferences,
                self._active_inferences,
            )
            try:
                return await asyncio.to_thread(
                    transcribe,
                    audio_path,
                    root,
                    config,
                    _engine=engine,
                )
            finally:
                self._active_inferences -= 1

    def _attempts_from_error(self, error: TranscriptionError) -> list[dict[str, Any]]:
        attempts = error.context.get("attempts")
        if isinstance(attempts, list):
            copied = [dict(attempt) for attempt in attempts if isinstance(attempt, Mapping)]
            if copied:
                return copied
        return [
            {
                "device": self.profile.device,
                "compute_type": self.profile.compute_type,
                "outcome": "failed",
                "reason_code": error.reason_code,
            }
        ]
