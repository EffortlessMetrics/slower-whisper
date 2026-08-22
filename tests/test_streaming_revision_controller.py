"""Public-route revision projection and placeholder-removal contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from transcription.config import AsrConfig, TranscriptionConfig
from transcription.exceptions import ASRInferenceError, RuntimeNotReadyError
from transcription.incremental_asr import IncrementalASRConfig
from transcription.service_runtime import ASRRuntime, RuntimeProfile
from transcription.streaming_revision_controller import RevisionStreamingController
from transcription.streaming_ws import EventType, SessionState


def pcm(samples: int, value: int = 5) -> bytes:
    return value.to_bytes(2, "little", signed=True) * samples


class TextEngine:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.calls: list[int] = []
        self.model_load_attempts = []

    def transcribe_file(self, path: Path):
        self.calls.append(path.stat().st_size)
        return SimpleNamespace(
            segments=[SimpleNamespace(text=f"call-{len(self.calls)}")]
        )


class FailingEngine(TextEngine):
    def transcribe_file(self, path: Path):
        del path
        raise RuntimeError("private provider path /srv/models/tiny")


class FakeProtocolSession:
    def __init__(self) -> None:
        self.events: list[tuple[Any, dict[str, Any], tuple[Any, ...], dict[str, Any]]] = []
        self.started: list[dict[str, Any]] = []
        self.ended = 0
        self.state = SessionState.CONNECTED
        self.asr_adapter = None

    async def send_event(
        self,
        event_type,
        data=None,
        *args,
        **kwargs,
    ) -> None:
        self.events.append((event_type, dict(data or {}), args, dict(kwargs)))

    async def handle_start_session(self, config_data: dict[str, Any]) -> None:
        self.started.append(dict(config_data))
        self.state = SessionState.ACTIVE
        await self.send_event(EventType.SESSION_STARTED, {"accepted": True})

    async def handle_end_session(self) -> None:
        self.ended += 1
        await self.send_event(EventType.SESSION_ENDED, {})


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


async def ready_runtime(engine_factory=TextEngine):
    runtime = ASRRuntime(profile(), engine_factory=engine_factory)
    await runtime.start()
    return runtime


def event_name(event_type: Any) -> str:
    value = getattr(event_type, "value", event_type)
    return str(value).upper()


@pytest.mark.asyncio
async def test_controller_suppresses_placeholder_before_first_revision() -> None:
    runtime = await ready_runtime()
    session = FakeProtocolSession()
    controller = RevisionStreamingController(
        session,  # type: ignore[arg-type]
        runtime,
        config=IncrementalASRConfig(
            min_hypothesis_samples=4,
            hypothesis_interval_samples=4,
            hypothesis_backoff_factor=1,
            max_utterance_samples=20,
            max_chunk_bytes=64,
        ),
    )

    await controller.adapter.process_audio_chunk(pcm(3), is_speech=True)
    await session.send_event(EventType.PARTIAL, {"text": "[processing...]"})

    assert session.events == []
    await runtime.close()


@pytest.mark.asyncio
async def test_controller_emits_replacement_revisions_with_one_identity() -> None:
    runtime = await ready_runtime()
    session = FakeProtocolSession()
    controller = RevisionStreamingController(
        session,  # type: ignore[arg-type]
        runtime,
        config=IncrementalASRConfig(
            min_hypothesis_samples=4,
            hypothesis_interval_samples=4,
            hypothesis_backoff_factor=1,
            max_utterance_samples=20,
            max_chunk_bytes=64,
        ),
    )

    await controller.adapter.process_audio_chunk(pcm(8), is_speech=True)
    await session.send_event(EventType.PARTIAL, {"text": "legacy"})
    await controller.adapter.process_audio_chunk(pcm(2, 0), is_speech=False)
    await controller.adapter.finalize_segment()
    await session.send_event(EventType.FINALIZED, {"text": "legacy-final"})

    assert [event_name(event[0]) for event in session.events] == [
        event_name(EventType.PARTIAL),
        event_name(EventType.PARTIAL),
        event_name(EventType.FINALIZED),
    ]
    data = [event[1] for event in session.events]
    assert [item["revision"] for item in data] == [1, 2, 3]
    assert len({item["segment_id"] for item in data}) == 1
    assert [item["text"] for item in data] == ["call-1", "call-2", "call-2"]
    assert [(item["start_sample"], item["end_sample"]) for item in data] == [
        (0, 4),
        (0, 8),
        (0, 8),
    ]
    assert data[-1]["final"] is True
    assert data[-1]["final_reason"] == "vad_boundary"
    assert all(item["text"] not in {"[processing...]", "[final segment]"} for item in data)
    await runtime.close()


@pytest.mark.asyncio
async def test_controller_rejects_unsupported_negotiation_before_base_start() -> None:
    runtime = await ready_runtime()
    session = FakeProtocolSession()
    controller = RevisionStreamingController(session, runtime)  # type: ignore[arg-type]

    with pytest.raises(RuntimeNotReadyError) as exc_info:
        await controller.handle_start_session(
            {
                "sample_rate": 8_000,
                "channels": 2,
                "audio_format": "float32",
            }
        )

    assert set(exc_info.value.context["mismatches"]) == {
        "sample_rate",
        "channels",
        "audio_format",
    }
    assert session.started == []
    assert session.events == []
    await runtime.close()


@pytest.mark.asyncio
async def test_controller_turns_inference_failure_into_one_terminal_public_error() -> None:
    runtime = await ready_runtime(FailingEngine)
    session = FakeProtocolSession()
    controller = RevisionStreamingController(
        session,  # type: ignore[arg-type]
        runtime,
        config=IncrementalASRConfig(
            min_hypothesis_samples=4,
            hypothesis_interval_samples=4,
            hypothesis_backoff_factor=1,
            max_utterance_samples=20,
            max_chunk_bytes=64,
        ),
    )

    await controller.adapter.process_audio_chunk(pcm(4), is_speech=True)
    with pytest.raises(ASRInferenceError) as exc_info:
        await session.send_event(EventType.PARTIAL, {"text": "[processing...]"})

    assert exc_info.value.reason_code == "asr_inference_failed"
    assert session.state is SessionState.ERROR
    assert len(session.events) == 1
    event_type, data, _args, _kwargs = session.events[0]
    assert event_name(event_type) == event_name(EventType.ERROR)
    assert data["code"] == "asr_inference_failed"
    assert data["message"] == "Streaming ASR failed"
    assert data["recoverable"] is False
    assert "private provider path" not in str(data)
    assert controller.incremental.active_audio_bytes == 0
    await runtime.close()


@pytest.mark.asyncio
async def test_controller_end_emits_final_before_session_ended() -> None:
    runtime = await ready_runtime()
    session = FakeProtocolSession()
    controller = RevisionStreamingController(
        session,  # type: ignore[arg-type]
        runtime,
        config=IncrementalASRConfig(
            min_hypothesis_samples=4,
            hypothesis_interval_samples=4,
            hypothesis_backoff_factor=1,
            max_utterance_samples=20,
            max_chunk_bytes=64,
        ),
    )

    await controller.adapter.process_audio_chunk(pcm(2), is_speech=True)
    await controller.handle_end_session()

    assert [event_name(event[0]) for event in session.events] == [
        event_name(EventType.FINALIZED),
        event_name(EventType.SESSION_ENDED),
    ]
    assert session.events[0][1]["final_reason"] == "end_of_stream"
    assert session.ended == 1
    await runtime.close()
