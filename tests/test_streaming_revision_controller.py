"""Revision projection, envelope identity, and terminal-failure contract."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from transcription.config import AsrConfig, TranscriptionConfig
from transcription.exceptions import (
    ASRInferenceError,
    RuntimeNotReadyError,
    StreamingNegotiationError,
)
from transcription.incremental_asr import IncrementalASRConfig
from transcription.service_runtime import ASRRuntime, RuntimeProfile
from transcription.streaming_revision_controller import RevisionStreamingController
from transcription.streaming_ws import (
    ServerMessageType,
    SessionState,
    WebSocketSessionConfig,
    WebSocketStreamingSession,
)


def pcm(samples: int, value: int = 5) -> bytes:
    return value.to_bytes(2, "little", signed=True) * samples


class TextEngine:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.calls: list[int] = []
        self.model_load_attempts = []

    def transcribe_file(self, path: Path):
        self.calls.append(path.stat().st_size)
        return SimpleNamespace(segments=[SimpleNamespace(text=f"call-{len(self.calls)}")])


class FailingEngine(TextEngine):
    def transcribe_file(self, path: Path):
        del path
        raise RuntimeError("private provider path /srv/models/tiny")


class SequenceClassifier:
    def __init__(self, decisions: list[bool | BaseException]) -> None:
        self.decisions = deque(decisions)

    def is_speech(self, _pcm: bytes, *, sample_rate: int) -> bool:
        assert sample_rate == 16_000
        decision = self.decisions.popleft()
        if isinstance(decision, BaseException):
            raise decision
        return decision


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


async def ready_runtime(engine_factory=TextEngine) -> ASRRuntime:
    runtime = ASRRuntime(profile(), engine_factory=engine_factory)
    await runtime.start()
    return runtime


def incremental_config(
    *,
    minimum: int = 4,
    interval: int = 4,
    maximum: int = 20,
    max_chunk_bytes: int = 64,
) -> IncrementalASRConfig:
    return IncrementalASRConfig(
        min_hypothesis_samples=minimum,
        hypothesis_interval_samples=interval,
        hypothesis_backoff_factor=1,
        max_utterance_samples=maximum,
        max_chunk_bytes=max_chunk_bytes,
    )


def session(*, silence_samples: int = 2) -> WebSocketStreamingSession:
    return WebSocketStreamingSession(
        config=WebSocketSessionConfig(
            max_gap_sec=silence_samples / 16_000,
            sample_rate=16_000,
            audio_format="pcm_s16le",
        )
    )


def payload(event) -> dict[str, Any]:
    return event.payload


@pytest.mark.asyncio
async def test_controller_rejects_unsupported_negotiation_before_start() -> None:
    runtime = await ready_runtime()
    protocol = session()
    controller = RevisionStreamingController(protocol, runtime)
    try:
        with pytest.raises(StreamingNegotiationError) as exc_info:
            await controller.start(
                {
                    "sample_rate": 8_000,
                    "channels": 2,
                    "audio_format": "float32",
                    "enable_diarization": True,
                }
            )

        assert exc_info.value.reason_code == "streaming_audio_unsupported"
        assert set(exc_info.value.context["mismatches"]) == {
            "sample_rate",
            "channels",
            "audio_format",
            "enable_diarization",
        }
        assert protocol.state is SessionState.CREATED
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_controller_emits_replacement_revisions_through_session_envelopes() -> None:
    runtime = await ready_runtime()
    protocol = session(silence_samples=2)
    controller = RevisionStreamingController(
        protocol,
        runtime,
        config=incremental_config(),
        classifier=SequenceClassifier([True, True, False]),
    )
    try:
        started = await controller.start(
            {
                "sample_rate": 16_000,
                "channels": 1,
                "audio_format": "pcm_s16le",
                "max_gap_sec": 2 / 16_000,
            }
        )
        first = await controller.process_audio_chunk(pcm(4), 1)
        second = await controller.process_audio_chunk(pcm(4), 2)
        final = await controller.process_audio_chunk(pcm(2, 0), 3)

        events = [started, *first, *second, *final]
        assert [event.type for event in events] == [
            ServerMessageType.SESSION_STARTED,
            ServerMessageType.PARTIAL,
            ServerMessageType.PARTIAL,
            ServerMessageType.FINALIZED,
        ]
        assert [event.event_id for event in events] == [1, 2, 3, 4]
        revisions = [payload(event) for event in events[1:]]
        assert [item["revision"] for item in revisions] == [1, 2, 3]
        assert len({event.segment_id for event in events[1:]}) == 1
        assert [item["text"] for item in revisions] == ["call-1", "call-2", "call-2"]
        assert [(item["start_sample"], item["end_sample"]) for item in revisions] == [
            (0, 4),
            (0, 8),
            (0, 8),
        ]
        assert revisions[-1]["final"] is True
        assert revisions[-1]["final_reason"] == "vad_boundary"
        assert all("[processing...]" not in str(event.to_dict()) for event in events)
        assert all("[final segment]" not in str(event.to_dict()) for event in events)

        replay, gap = protocol.get_events_for_resume(1)
        assert not gap
        assert [event.event_id for event in replay] == [2, 3, 4]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_continuous_silence_is_bounded_and_never_invokes_asr() -> None:
    engines: list[TextEngine] = []

    def factory(cfg: AsrConfig) -> TextEngine:
        engine = TextEngine(cfg)
        engines.append(engine)
        return engine

    runtime = await ready_runtime(factory)
    protocol = WebSocketStreamingSession(config=WebSocketSessionConfig(max_gap_sec=10.0))
    controller = RevisionStreamingController(
        protocol,
        runtime,
        config=incremental_config(max_chunk_bytes=64),
        classifier=SequenceClassifier([False] * 100),
    )
    try:
        await controller.start({})
        for sequence in range(1, 101):
            assert await controller.process_audio_chunk(pcm(16, 0), sequence) == []
            assert controller.pending_silence_bytes <= 64
        assert engines[0].calls == []
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_finalized_revision_remains_before_later_terminal_error() -> None:
    runtime = await ready_runtime()
    protocol = session()
    controller = RevisionStreamingController(
        protocol,
        runtime,
        config=incremental_config(minimum=4, interval=4, maximum=4),
        classifier=SequenceClassifier([True, RuntimeError("classifier secret")]),
    )
    try:
        await controller.start({})
        finalized = await controller.process_audio_chunk(pcm(4), 1)
        assert len(finalized) == 1
        assert finalized[0].type is ServerMessageType.FINALIZED

        with pytest.raises(ASRInferenceError) as exc_info:
            await controller.process_audio_chunk(pcm(4), 2)
        error_event = controller.terminal_error_event(exc_info.value)

        assert finalized[0].event_id < error_event.event_id
        assert error_event.type is ServerMessageType.ERROR
        assert error_event.payload["code"] == "asr_inference_failed"
        assert "classifier secret" not in str(error_event.to_dict())
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_inference_failure_is_one_sanitized_terminal_error() -> None:
    runtime = await ready_runtime(FailingEngine)
    protocol = session()
    controller = RevisionStreamingController(
        protocol,
        runtime,
        config=incremental_config(),
        classifier=SequenceClassifier([True]),
    )
    try:
        await controller.start({})
        with pytest.raises(ASRInferenceError) as exc_info:
            await controller.process_audio_chunk(pcm(4), 1)
        error_event = controller.terminal_error_event(exc_info.value)

        assert protocol.state is SessionState.ERROR
        assert error_event.type is ServerMessageType.ERROR
        assert error_event.payload["code"] == "asr_inference_failed"
        assert error_event.payload["message"] == "Streaming ASR failed"
        assert error_event.payload["recoverable"] is False
        assert "private provider path" not in str(error_event.to_dict())
        assert controller.incremental.active_audio_bytes == 0

        with pytest.raises(RuntimeNotReadyError):
            await controller.process_audio_chunk(pcm(1), 2)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_controller_end_emits_final_before_session_ended() -> None:
    runtime = await ready_runtime()
    protocol = session()
    controller = RevisionStreamingController(
        protocol,
        runtime,
        config=incremental_config(),
        classifier=SequenceClassifier([True]),
    )
    try:
        await controller.start({})
        assert await controller.process_audio_chunk(pcm(2), 1) == []
        events = await controller.end()

        assert [event.type for event in events] == [
            ServerMessageType.FINALIZED,
            ServerMessageType.SESSION_ENDED,
        ]
        assert events[0].payload["final_reason"] == "end_of_stream"
        assert events[0].event_id < events[1].event_id
        assert protocol.state is SessionState.ENDED
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_controller_requires_a_ready_process_runtime() -> None:
    runtime = ASRRuntime(profile(), engine_factory=TextEngine)
    with pytest.raises(RuntimeNotReadyError) as exc_info:
        RevisionStreamingController(session(), runtime)
    assert exc_info.value.context == {"state": "stopped"}
