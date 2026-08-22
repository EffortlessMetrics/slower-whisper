"""Process-runtime adapter contract for incremental PCM ASR."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from transcription.config import AsrConfig, TranscriptionConfig
from transcription.exceptions import RuntimeNotReadyError
from transcription.service_runtime import ASRRuntime, RuntimeProfile
from transcription.streaming_runtime_asr import RuntimeIncrementalASRBackend


def pcm(samples: int, value: int = 7) -> bytes:
    return value.to_bytes(2, "little", signed=True) * samples


class RecordingEngine:
    instances = 0

    def __init__(self, cfg: AsrConfig) -> None:
        type(self).instances += 1
        self.cfg = cfg
        self.calls: list[tuple[int, int, int]] = []
        self.close_count = 0
        self.model_load_attempts = [
            {
                "device": cfg.device,
                "compute_type": cfg.compute_type or "unknown",
                "outcome": "selected",
                "reason_code": "ok",
            }
        ]

    def transcribe_file(self, path: Path):
        with wave.open(str(path), "rb") as wav_file:
            self.calls.append(
                (
                    wav_file.getframerate(),
                    wav_file.getnchannels(),
                    wav_file.getnframes(),
                )
            )
        return SimpleNamespace(
            segments=[
                SimpleNamespace(text=" first "),
                SimpleNamespace(text=""),
                SimpleNamespace(text="second"),
            ]
        )

    def close(self) -> None:
        self.close_count += 1


def runtime_profile() -> RuntimeProfile:
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
async def test_adapter_reuses_process_engine_and_writes_exact_pcm_wav() -> None:
    RecordingEngine.instances = 0
    engines: list[RecordingEngine] = []

    def factory(cfg: AsrConfig) -> RecordingEngine:
        engine = RecordingEngine(cfg)
        engines.append(engine)
        return engine

    runtime = ASRRuntime(runtime_profile(), engine_factory=factory)
    await runtime.start()
    adapter = RuntimeIncrementalASRBackend(runtime)

    first = await adapter.transcribe(
        pcm(160),
        sample_rate=16_000,
        start_sample=0,
        end_sample=160,
    )
    second = await adapter.transcribe(
        pcm(320),
        sample_rate=16_000,
        start_sample=0,
        end_sample=320,
    )

    assert first == second == "first second"
    assert RecordingEngine.instances == 1
    assert engines[0].calls == [(16_000, 1, 160), (16_000, 1, 320)]
    assert runtime.max_observed_inferences == 1

    await runtime.close()
    assert engines[0].close_count == 1


@pytest.mark.asyncio
async def test_adapter_rejects_invalid_pcm_before_runtime_work() -> None:
    engines: list[RecordingEngine] = []

    def factory(cfg: AsrConfig) -> RecordingEngine:
        engine = RecordingEngine(cfg)
        engines.append(engine)
        return engine

    runtime = ASRRuntime(runtime_profile(), engine_factory=factory)
    await runtime.start()
    adapter = RuntimeIncrementalASRBackend(runtime)

    with pytest.raises(TypeError, match="must be bytes"):
        await adapter.transcribe(
            bytearray(pcm(1)),  # type: ignore[arg-type]
            sample_rate=16_000,
            start_sample=0,
            end_sample=1,
        )
    with pytest.raises(ValueError, match="16 kHz"):
        await adapter.transcribe(
            pcm(1),
            sample_rate=8_000,
            start_sample=0,
            end_sample=1,
        )
    with pytest.raises(ValueError, match="complete"):
        await adapter.transcribe(
            b"\x00",
            sample_rate=16_000,
            start_sample=0,
            end_sample=1,
        )
    with pytest.raises(ValueError, match="bounds"):
        await adapter.transcribe(
            pcm(1),
            sample_rate=16_000,
            start_sample=2,
            end_sample=1,
        )
    with pytest.raises(ValueError, match="span"):
        await adapter.transcribe(
            pcm(2),
            sample_rate=16_000,
            start_sample=0,
            end_sample=1,
        )

    assert engines[0].calls == []
    await runtime.close()


@pytest.mark.asyncio
async def test_adapter_fails_typed_when_runtime_is_not_ready() -> None:
    runtime = ASRRuntime(runtime_profile(), engine_factory=RecordingEngine)
    adapter = RuntimeIncrementalASRBackend(runtime)

    with pytest.raises(RuntimeNotReadyError) as exc_info:
        await adapter.transcribe(
            pcm(1),
            sample_rate=16_000,
            start_sample=0,
            end_sample=1,
        )

    assert exc_info.value.context == {"state": "stopped"}
