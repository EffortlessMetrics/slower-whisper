"""Tests for the process-owned ASR runtime boundary."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from transcription.config import AsrConfig, TranscriptionConfig
from transcription.exceptions import (
    ASRModelLoadError,
    RuntimeNotReadyError,
    RuntimeProfileMismatchError,
)
from transcription.models import Transcript
from transcription.service_runtime import ASRRuntime, RuntimeProfile, RuntimeState


class FakeEngine:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.model_load_attempts = [
            {
                "device": cfg.device,
                "compute_type": cfg.compute_type or "unknown",
                "outcome": "selected",
                "reason_code": "ok",
            }
        ]
        self.close_count = 0

    def transcribe_file(self, path: Path) -> Transcript:
        return Transcript(file_name=path.name, language="en", segments=[])

    def close(self) -> None:
        self.close_count += 1


def profile() -> RuntimeProfile:
    return RuntimeProfile.from_config(
        TranscriptionConfig(
            model="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            task="transcribe",
            beam_size=3,
            vad_min_silence_ms=400,
            word_timestamps=False,
        )
    )


@pytest.mark.asyncio
async def test_runtime_starts_once_and_closes_once() -> None:
    created: list[FakeEngine] = []

    def factory(cfg: AsrConfig) -> FakeEngine:
        engine = FakeEngine(cfg)
        created.append(engine)
        return engine

    runtime = ASRRuntime(profile(), engine_factory=factory)
    assert runtime.state == RuntimeState.STOPPED

    await runtime.start()
    await runtime.start()

    assert runtime.ready is True
    assert len(created) == 1
    assert runtime.status()["selected"]["device"] == "cpu"
    assert runtime.status()["attempts"] == created[0].model_load_attempts

    await runtime.close()
    await runtime.close()

    assert runtime.state == RuntimeState.STOPPED
    assert created[0].close_count == 1


@pytest.mark.asyncio
async def test_runtime_start_failure_preserves_ordered_attempts() -> None:
    attempts = [
        {
            "device": "cuda",
            "compute_type": "float16",
            "outcome": "failed",
            "reason_code": "asr_model_load_failed",
        },
        {
            "device": "cpu",
            "compute_type": "int8",
            "outcome": "failed",
            "reason_code": "asr_model_load_failed",
        },
    ]

    def factory(_cfg: AsrConfig) -> FakeEngine:
        raise ASRModelLoadError(
            "model unavailable",
            context={"model": "tiny", "attempts": attempts},
        )

    runtime = ASRRuntime(profile(), engine_factory=factory)
    await runtime.start()

    status = runtime.status()
    assert runtime.state == RuntimeState.FAILED
    assert runtime.ready is False
    assert status["error"]["reason_code"] == "asr_model_load_failed"
    assert status["attempts"] == attempts
    with pytest.raises(RuntimeNotReadyError):
        _ = runtime.engine


@pytest.mark.asyncio
async def test_unexpected_start_failure_records_attempted_profile() -> None:
    def factory(_cfg: AsrConfig) -> FakeEngine:
        raise RuntimeError("provider detail")

    runtime = ASRRuntime(profile(), engine_factory=factory)
    await runtime.start()

    assert runtime.status()["attempts"] == [
        {
            "device": "cpu",
            "compute_type": "int8",
            "outcome": "failed",
            "reason_code": "asr_model_load_failed",
        }
    ]
    assert "provider detail" not in str(runtime.status())


def test_runtime_profile_rejects_unresolved_device() -> None:
    config = TranscriptionConfig(
        model="tiny",
        device="auto",
        compute_type=None,
    )

    with pytest.raises(ValueError, match="resolved"):
        RuntimeProfile.from_config(config)


@pytest.mark.asyncio
async def test_runtime_reports_backend_selected_profile_after_fallback() -> None:
    def factory(cfg: AsrConfig) -> FakeEngine:
        cfg.device = "cpu"
        cfg.compute_type = "int8"
        engine = FakeEngine(cfg)
        engine.model_load_attempts = [
            {
                "device": "cuda",
                "compute_type": "float16",
                "outcome": "failed",
                "reason_code": "asr_model_load_failed",
            },
            {
                "device": "cpu",
                "compute_type": "int8",
                "outcome": "selected",
                "reason_code": "ok",
            },
        ]
        return engine

    requested = RuntimeProfile.from_config(
        TranscriptionConfig(
            model="tiny",
            device="cuda",
            compute_type="float16",
        )
    )
    runtime = ASRRuntime(requested, engine_factory=factory)
    await runtime.start()

    status = runtime.status()
    assert status["profile"]["device"] == "cuda"
    assert status["selected"]["device"] == "cpu"
    assert status["selected"]["compute_type"] == "int8"
    assert [attempt["outcome"] for attempt in status["attempts"]] == [
        "failed",
        "selected",
    ]


@pytest.mark.asyncio
async def test_runtime_rejects_profile_mismatch_before_transcription(tmp_path: Path) -> None:
    runtime = ASRRuntime(profile(), engine_factory=FakeEngine)
    await runtime.start()
    mismatched = TranscriptionConfig(
        model="base",
        device="cpu",
        compute_type="int8",
        language="en",
        task="transcribe",
        beam_size=3,
        vad_min_silence_ms=400,
        word_timestamps=False,
    )
    called = False

    def transcribe(*_args: Any, **_kwargs: Any) -> Transcript:
        nonlocal called
        called = True
        return Transcript(file_name="x.wav", language="en", segments=[])

    with pytest.raises(RuntimeProfileMismatchError) as raised:
        await runtime.transcribe_file(
            transcribe,
            audio_path=tmp_path / "x.wav",
            root=tmp_path,
            config=mismatched,
        )

    assert called is False
    assert "model" in raised.value.context["mismatches"]


@pytest.mark.asyncio
async def test_runtime_bounds_concurrent_inference(tmp_path: Path) -> None:
    runtime = ASRRuntime(profile(), engine_factory=FakeEngine, max_concurrency=1)
    await runtime.start()
    lock = threading.Lock()
    active = 0
    max_active = 0

    def transcribe(
        _audio_path: Path,
        _root: Path,
        _config: TranscriptionConfig,
        *,
        _engine: Any,
    ) -> Transcript:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return Transcript(file_name="x.wav", language="en", segments=[])

    config = TranscriptionConfig(
        model="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        task="transcribe",
        beam_size=3,
        vad_min_silence_ms=400,
        word_timestamps=False,
    )
    await asyncio.gather(
        *[
            runtime.transcribe_file(
                transcribe,
                audio_path=tmp_path / f"{index}.wav",
                root=tmp_path,
                config=config,
            )
            for index in range(3)
        ]
    )

    assert max_active == 1
    assert runtime.max_observed_inferences == 1
    assert runtime.status()["active_inferences"] == 0


@pytest.mark.asyncio
async def test_close_failure_still_leaves_runtime_stopped() -> None:
    class FailingCloseEngine(FakeEngine):
        def close(self) -> None:
            raise RuntimeError("close failed")

    runtime = ASRRuntime(profile(), engine_factory=FailingCloseEngine)
    await runtime.start()

    with pytest.raises(RuntimeError, match="close failed"):
        await runtime.close()

    assert runtime.state == RuntimeState.STOPPED
    assert runtime.ready is False
