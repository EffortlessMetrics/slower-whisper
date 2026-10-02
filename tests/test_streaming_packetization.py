"""Default classifier/framing/core contract with a deterministic ASR backend.

The backend is intentionally a test double. These tests establish sample and
revision invariance, not recognition quality or live transport acceptance.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import pytest

from transcription import streaming_revision_controller as controller_module
from transcription.exceptions import (
    ASRInferenceError,
    ASROutputError,
    RuntimeNotReadyError,
    StreamingNegotiationError,
)
from transcription.incremental_asr import IncrementalASRConfig
from transcription.streaming_revision_controller import RevisionStreamingController
from transcription.streaming_ws import (
    SessionState,
    WebSocketSessionConfig,
    WebSocketStreamingSession,
)


def pcm(samples: int, value: int) -> bytes:
    return value.to_bytes(2, "little", signed=True) * samples


def recording() -> bytes:
    return (
        pcm(320, 0)
        + pcm(960, 4000)
        + pcm(320, 0)  # bridged short gap
        + pcm(640, 4000)
        + pcm(960, 0)  # finalizes preceding speech
        + pcm(1120, 4000)  # includes a short complete-sample EOF frame
    )


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, str]] = []

    def transcribe(
        self, data: bytes, *, sample_rate: int, start_sample: int, end_sample: int
    ) -> str:
        assert sample_rate == 16_000
        assert end_sample - start_sample == len(data) // 2
        digest = hashlib.sha256(data).hexdigest()
        self.calls.append((start_sample, end_sample, digest))
        return f"fixture:{start_sample}:{end_sample}:{digest}"


def make_controller(monkeypatch: pytest.MonkeyPatch, *, classifier=None):
    backend = RecordingBackend()
    monkeypatch.setattr(controller_module, "RuntimeIncrementalASRBackend", lambda _runtime: backend)
    protocol = WebSocketStreamingSession(config=WebSocketSessionConfig(max_gap_sec=0.04))
    controller = RevisionStreamingController(
        protocol,
        cast(Any, SimpleNamespace(ready=True)),
        config=IncrementalASRConfig(
            min_hypothesis_samples=320,
            hypothesis_interval_samples=320,
            max_utterance_samples=1600,
            max_chunk_bytes=16384,
        ),
        classifier=classifier,
    )
    return controller, backend, protocol


async def run_partition(monkeypatch: pytest.MonkeyPatch, sizes: list[int]):
    controller, backend, _protocol = make_controller(monkeypatch)
    await controller.start({})
    audio = recording()
    events = []
    offset = 0
    sequence = 0
    while offset < len(audio):
        size = sizes[sequence % len(sizes)]
        packet = audio[offset : offset + size]
        events.extend(await controller.process_audio_chunk(packet, sequence))
        offset += len(packet)
        sequence += 1
    events.extend(await controller.end())
    projection = [
        (event.segment_id, dict(event.payload)) for event in events if "revision" in event.payload
    ]
    return projection, backend.calls, asdict(controller.incremental.metrics), controller


@pytest.mark.asyncio
@pytest.mark.parametrize("sizes", [[16384], [640], [1], [639, 2, 641, 17], [2049, 7, 3]])
async def test_default_public_controller_is_packetization_invariant(monkeypatch, sizes) -> None:
    reference = await run_partition(monkeypatch, [640])
    actual = await run_partition(monkeypatch, sizes)
    assert actual[:3] == reference[:3]
    final = [payload for _segment, payload in actual[0] if payload["final"]]
    assert [(item["start_sample"], item["end_sample"], item["final_reason"]) for item in final] == [
        (320, 1920, "max_utterance"),
        (1920, 2240, "vad_boundary"),
        (3200, 4320, "end_of_stream"),
    ]
    assert actual[2]["absolute_samples_received"] == 4320
    metrics = actual[3].framing_metrics
    assert metrics.accepted_bytes == 8640
    assert metrics.emitted_samples == 4320
    assert metrics.frames_emitted == 14
    assert metrics.residual_bytes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(10))
async def test_random_network_partitions_preserve_revisions_and_work(monkeypatch, seed) -> None:
    rng = random.Random(seed)
    sizes = [rng.randint(1, 2500) for _ in range(30)]
    reference = await run_partition(monkeypatch, [640])
    actual = await run_partition(monkeypatch, sizes)
    assert actual[:3] == reference[:3]


@pytest.mark.asyncio
async def test_all_silence_never_invokes_model_and_preserves_sample_count(monkeypatch) -> None:
    controller, backend, _protocol = make_controller(monkeypatch)
    await controller.start({})
    for sequence in range(100):
        assert await controller.process_audio_chunk(b"\0" * 333, sequence) == []
        assert controller.framing_metrics.residual_bytes < 640
        assert controller.pending_silence_bytes < 1280
    ended = await controller.end()
    assert len(ended) == 1
    assert backend.calls == []
    assert controller.incremental.metrics.absolute_samples_received == 16650


@pytest.mark.asyncio
@pytest.mark.parametrize("audio", [b"\x01", pcm(319, 4000) + b"\x01"])
async def test_incomplete_final_sample_cannot_turn_into_success(monkeypatch, audio) -> None:
    controller, backend, protocol = make_controller(monkeypatch)
    await controller.start({})
    assert await controller.process_audio_chunk(audio, 0) == []
    with pytest.raises(StreamingNegotiationError) as exc:
        await controller.end()
    assert exc.value.reason_code == "streaming_pcm_incomplete"
    event = controller.terminal_error_event(exc.value)
    assert event.payload == {
        "code": "streaming_pcm_incomplete",
        "message": "Streaming PCM ended with an incomplete sample",
        "recoverable": False,
        "context": {"violation": "incomplete_pcm_sample", "residual_bytes": len(audio)},
    }
    with pytest.raises(RuntimeNotReadyError) as retry:
        await controller.end()
    assert retry.value.context["failure_reason_code"] == "streaming_pcm_incomplete"
    with pytest.raises(RuntimeNotReadyError):
        await controller.process_audio_chunk(b"\x00", 1)
    assert backend.calls == []
    assert protocol.state is SessionState.ERROR
    assert controller.framing_metrics.residual_bytes == 0


@pytest.mark.asyncio
async def test_classifier_sees_frames_not_packets_and_tail_is_unpadded(monkeypatch) -> None:
    seen: list[bytes] = []

    class Classifier:
        async def is_speech(self, data: bytes, *, sample_rate: int) -> bool:
            assert sample_rate == 16_000
            seen.append(data)
            return True

    controller, _backend, _protocol = make_controller(monkeypatch, classifier=Classifier())
    await controller.start({})
    audio = pcm(641, 5000)
    for sequence, packet in enumerate([audio[:1], audio[1:777], audio[777:]]):
        await controller.process_audio_chunk(packet, sequence)
    assert [len(frame) for frame in seen] == [640, 640]
    await controller.end()
    assert [len(frame) for frame in seen] == [640, 640, 2]
    assert b"".join(seen) == audio


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [None, 1, "false"])
async def test_nonboolean_classifier_output_is_terminal_not_silence(monkeypatch, decision) -> None:
    class Classifier:
        def is_speech(self, data: bytes, *, sample_rate: int):
            return decision

    controller, backend, protocol = make_controller(monkeypatch, classifier=Classifier())
    await controller.start({})
    with pytest.raises(ASROutputError):
        await controller.process_audio_chunk(pcm(320, 5000), 0)
    assert backend.calls == []
    assert protocol.state is SessionState.ERROR


@pytest.mark.asyncio
async def test_classifier_exception_retains_sanitized_terminal_contract(monkeypatch) -> None:
    class Classifier:
        def is_speech(self, data: bytes, *, sample_rate: int) -> bool:
            raise RuntimeError("private classifier details")

    controller, backend, protocol = make_controller(monkeypatch, classifier=Classifier())
    await controller.start({})
    with pytest.raises(ASRInferenceError) as exc:
        await controller.process_audio_chunk(pcm(320, 5000), 0)
    event = controller.terminal_error_event(exc.value)
    assert "private classifier details" not in str(event.payload)
    assert backend.calls == []
    assert protocol.state is SessionState.ERROR


@pytest.mark.asyncio
async def test_invalid_sequence_does_not_consume_pending_sample_byte(monkeypatch) -> None:
    controller, _backend, _protocol = make_controller(monkeypatch)
    await controller.start({})
    await controller.process_audio_chunk(b"\0", 0)
    before = controller.framing_metrics
    with pytest.raises(ValueError, match="greater"):
        await controller.process_audio_chunk(b"\0", 0)
    assert controller.framing_metrics == before
    await controller.process_audio_chunk(b"\0", 1)
    await controller.end()
    assert controller.incremental.metrics.absolute_samples_received == 1
