"""Edge contracts for public streaming revision boundaries."""

from __future__ import annotations

import wave
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from transcription.config import AsrConfig, TranscriptionConfig
from transcription.exceptions import StreamingNegotiationError
from transcription.incremental_asr import IncrementalASRConfig
from transcription.service_runtime import ASRRuntime, RuntimeProfile
from transcription.streaming_revision_controller import RevisionStreamingController
from transcription.streaming_ws import WebSocketSessionConfig, WebSocketStreamingSession


def pcm(samples: int, value: int = 3) -> bytes:
    return value.to_bytes(2, "little", signed=True) * samples


class FrameEngine:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.calls: list[int] = []
        self.model_load_attempts = []

    def transcribe_file(self, path: Path):
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
        self.calls.append(frames)
        return SimpleNamespace(segments=[SimpleNamespace(text=f"samples:{frames}")])


class SequenceClassifier:
    def __init__(self, decisions: list[bool]) -> None:
        self.decisions = deque(decisions)

    def is_speech(self, _audio: bytes, *, sample_rate: int) -> bool:
        assert sample_rate == 16_000
        return self.decisions.popleft()


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


async def ready_runtime() -> tuple[ASRRuntime, list[FrameEngine]]:
    engines: list[FrameEngine] = []

    def factory(cfg: AsrConfig) -> FrameEngine:
        engine = FrameEngine(cfg)
        engines.append(engine)
        return engine

    runtime = ASRRuntime(profile(), engine_factory=factory)
    await runtime.start()
    return runtime, engines


def core_config() -> IncrementalASRConfig:
    return IncrementalASRConfig(
        min_hypothesis_samples=4,
        hypothesis_interval_samples=4,
        hypothesis_backoff_factor=1,
        max_utterance_samples=20,
        max_chunk_bytes=64,
    )


@pytest.mark.asyncio
async def test_short_leading_silence_advances_clock_without_joining_utterance() -> None:
    runtime, engines = await ready_runtime()
    protocol = WebSocketStreamingSession(config=WebSocketSessionConfig(max_gap_sec=4 / 16_000))
    controller = RevisionStreamingController(
        protocol,
        runtime,
        config=core_config(),
        classifier=SequenceClassifier([False, True]),
    )
    try:
        await controller.start({"max_gap_sec": 4 / 16_000})
        assert await controller.process_audio_chunk(pcm(2, 0), 1) == []
        revisions = await controller.process_audio_chunk(pcm(4), 2)

        assert len(revisions) == 1
        payload = revisions[0].payload
        assert payload["start_sample"] == 2
        assert payload["end_sample"] == 6
        assert payload["text"] == "samples:4"
        assert engines[0].calls == [4]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_present_unsupported_feature_requires_exact_false() -> None:
    runtime, _engines = await ready_runtime()
    try:
        rejected = RevisionStreamingController(
            WebSocketStreamingSession(),
            runtime,
            config=core_config(),
        )
        with pytest.raises(StreamingNegotiationError) as exc_info:
            await rejected.start({"enable_prosody": "false"})
        assert exc_info.value.context["mismatches"] == {"enable_prosody": "false"}

        accepted = RevisionStreamingController(
            WebSocketStreamingSession(),
            runtime,
            config=core_config(),
        )
        started = await accepted.start({"enable_prosody": False})
        assert started.type.value == "SESSION_STARTED"
    finally:
        await runtime.close()
