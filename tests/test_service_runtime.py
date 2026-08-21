"""Tests for the process-owned ASR runtime boundary."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace
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

    await runtime.close()
    await runtime.close()

    assert runtime.state == RuntimeState.STOPPED
    assert created[0].close_count == 1


@pytest.mark.asyncio
async def test_runtime_start_failure_is_observable_without_raising() -> None:
    def factory(_cfg: AsrConfig) -> FakeEngine:
        raise ASRModelLoadError("model unavailable", context={"model": "tiny"})

    runtime = ASRRuntime(profile(), engine_factory=factory)
    await runtime.start()

    assert runtime.state == RuntimeState.FAILED
    assert runtime.ready is False
    assert runtime.status()["error"]["reason_code"] == "asr_model_load_failed"
    with pytest.raises(RuntimeNotReadyError):
        _ = runtime.engine


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


@pytest.mark.asyncio
async def test_deep_probe_distinguishes_valid_silence(tmp_path: Path) -> None:
    runtime = ASRRuntime(profile(), engine_factory=FakeEngine)
    await runtime.start()
    audio = tmp_path / "silence.wav"
    audio.write_bytes(b"not read by fake engine")

    transcript = await runtime.deep_probe(audio)

    assert transcript.segments == []
    assert transcript.file_name == "silence.wav"
