"""Behavioral contract for revision-aware incremental ASR."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from transcription.exceptions import (
    ASRInferenceError,
    ASROutputError,
    RuntimeNotReadyError,
)
from transcription.incremental_asr import (
    ASRRevision,
    IncrementalASRConfig,
    IncrementalASRSession,
    IncrementalASRState,
)


def pcm(samples: int, value: int = 1) -> bytes:
    """Return ``samples`` mono signed-16 PCM frames."""
    frame = value.to_bytes(2, byteorder="little", signed=True)
    return frame * samples


@dataclass
class RecordingBackend:
    """Deterministic backend whose text identifies the complete active prefix."""

    calls: list[tuple[int, int, int]] = field(default_factory=list)

    def transcribe(
        self,
        pcm_s16le: bytes,
        *,
        sample_rate: int,
        start_sample: int,
        end_sample: int,
    ) -> str:
        assert sample_rate == 16_000
        samples = len(pcm_s16le) // 2
        self.calls.append((samples, start_sample, end_sample))
        return f"samples:{samples}"


class AsyncBackend(RecordingBackend):
    async def transcribe(
        self,
        pcm_s16le: bytes,
        *,
        sample_rate: int,
        start_sample: int,
        end_sample: int,
    ) -> str:
        return super().transcribe(
            pcm_s16le,
            sample_rate=sample_rate,
            start_sample=start_sample,
            end_sample=end_sample,
        )


class FailingBackend:
    def transcribe(
        self,
        pcm_s16le: bytes,
        *,
        sample_rate: int,
        start_sample: int,
        end_sample: int,
    ) -> str:
        del pcm_s16le, sample_rate, start_sample, end_sample
        raise RuntimeError("provider detail must remain chained locally")


class NonTextBackend:
    def transcribe(
        self,
        pcm_s16le: bytes,
        *,
        sample_rate: int,
        start_sample: int,
        end_sample: int,
    ) -> object:
        del pcm_s16le, sample_rate, start_sample, end_sample
        return object()


def config(
    *,
    minimum: int = 4,
    interval: int = 4,
    maximum: int = 20,
    max_chunk_bytes: int = 256,
) -> IncrementalASRConfig:
    return IncrementalASRConfig(
        min_hypothesis_samples=minimum,
        hypothesis_interval_samples=interval,
        max_utterance_samples=maximum,
        max_chunk_bytes=max_chunk_bytes,
    )


def projection(revision: ASRRevision) -> tuple[object, ...]:
    return (
        revision.segment_id,
        revision.revision,
        revision.start_sample,
        revision.end_sample,
        revision.text,
        revision.final,
        revision.final_reason,
    )


async def run_fragmented(
    speech_chunks: list[int],
    *,
    silence_samples: int = 4,
) -> tuple[list[ASRRevision], RecordingBackend, IncrementalASRSession]:
    backend = RecordingBackend()
    session = IncrementalASRSession(backend, config=config())
    revisions: list[ASRRevision] = []
    for samples in speech_chunks:
        revisions.extend(await session.push_pcm(pcm(samples), speech=True))
    revisions.extend(await session.push_pcm(pcm(silence_samples, 0), speech=False))
    return revisions, backend, session


@pytest.mark.asyncio
async def test_revisions_replace_text_and_finalize_on_vad_without_reinference() -> None:
    backend = RecordingBackend()
    session = IncrementalASRSession(backend, config=config())

    first = await session.push_pcm(pcm(4), speech=True)
    second = await session.push_pcm(pcm(4), speech=True)
    final = await session.push_pcm(pcm(2, 0), speech=False)

    assert len(first) == len(second) == len(final) == 1
    first_revision = first[0]
    second_revision = second[0]
    final_revision = final[0]

    assert first_revision.segment_id == second_revision.segment_id == final_revision.segment_id
    assert [first_revision.revision, second_revision.revision, final_revision.revision] == [1, 2, 3]
    assert [first_revision.text, second_revision.text, final_revision.text] == [
        "samples:4",
        "samples:8",
        "samples:8",
    ]
    assert first_revision.start_sample == second_revision.start_sample == 0
    assert first_revision.end_sample == 4
    assert second_revision.end_sample == final_revision.end_sample == 8
    assert not first_revision.final
    assert not second_revision.final
    assert final_revision.final_reason == "vad_boundary"
    assert session.absolute_sample == 10
    assert session.active_segment_id is None
    assert backend.calls == [(4, 0, 4), (8, 0, 8)]
    assert session.metrics.model_calls == 2
    assert session.metrics.decoded_audio_samples == 12

    next_revision = await session.push_pcm(pcm(4), speech=True)
    assert len(next_revision) == 1
    assert next_revision[0].segment_id != first_revision.segment_id
    assert next_revision[0].start_sample == 10
    assert next_revision[0].end_sample == 14


@pytest.mark.asyncio
async def test_packet_fragmentation_does_not_change_revision_sequence_or_work() -> None:
    whole, whole_backend, whole_session = await run_fragmented([12])
    fragmented, fragmented_backend, fragmented_session = await run_fragmented([1] * 12)
    uneven, uneven_backend, uneven_session = await run_fragmented([3, 2, 5, 2])

    expected = [projection(revision) for revision in whole]
    assert [projection(revision) for revision in fragmented] == expected
    assert [projection(revision) for revision in uneven] == expected
    assert whole_backend.calls == fragmented_backend.calls == uneven_backend.calls
    assert whole_session.metrics == fragmented_session.metrics == uneven_session.metrics


@pytest.mark.asyncio
async def test_max_utterance_rollover_keeps_absolute_half_open_time() -> None:
    backend = RecordingBackend()
    session = IncrementalASRSession(
        backend,
        config=config(minimum=5, interval=5, maximum=10),
    )

    revisions = list(await session.push_pcm(pcm(25), speech=True))
    revisions.extend(await session.end())

    finals = [revision for revision in revisions if revision.final]
    assert [
        (revision.start_sample, revision.end_sample, revision.final_reason)
        for revision in finals
    ] == [
        (0, 10, "max_utterance"),
        (10, 20, "max_utterance"),
        (20, 25, "end_of_stream"),
    ]
    assert [revision.segment_id for revision in finals] == [
        "seg-00000001",
        "seg-00000002",
        "seg-00000003",
    ]
    assert all(
        previous.end_sample <= following.start_sample
        for previous, following in zip(finals, finals[1:], strict=False)
    )
    assert session.absolute_sample == 25
    assert session.state is IncrementalASRState.ENDED


@pytest.mark.asyncio
async def test_model_work_is_bounded_by_audio_not_network_packets() -> None:
    async def run(chunks: list[int]) -> tuple[int, int, list[tuple[int, int, int]]]:
        backend = RecordingBackend()
        session = IncrementalASRSession(
            backend,
            config=config(minimum=10, interval=10, maximum=30),
        )
        for samples in chunks:
            await session.push_pcm(pcm(samples), speech=True)
        await session.end()
        return (
            session.metrics.model_calls,
            session.metrics.decoded_audio_samples,
            backend.calls,
        )

    one_packet = await run([30])
    thirty_packets = await run([1] * 30)
    uneven_packets = await run([7, 2, 11, 1, 9])

    assert one_packet == thirty_packets == uneven_packets
    model_calls, decoded_samples, calls = one_packet
    assert model_calls == 3
    assert decoded_samples == 60
    assert calls == [(10, 0, 10), (20, 0, 20), (30, 0, 30)]


@pytest.mark.asyncio
async def test_active_memory_never_exceeds_max_utterance() -> None:
    backend = RecordingBackend()
    session = IncrementalASRSession(
        backend,
        config=config(minimum=5, interval=5, maximum=10),
    )

    for _ in range(100):
        await session.push_pcm(pcm(1), speech=True)

    assert session.metrics.peak_active_audio_bytes <= 10 * 2
    assert session.active_audio_bytes == 0
    assert session.metrics.segments_finalized == 10


@pytest.mark.asyncio
async def test_backend_failure_is_distinct_from_no_revision_yet() -> None:
    session = IncrementalASRSession(FailingBackend(), config=config())

    assert await session.push_pcm(pcm(3), speech=True) == ()
    assert session.state is IncrementalASRState.ACTIVE

    with pytest.raises(ASRInferenceError) as exc_info:
        await session.push_pcm(pcm(1), speech=True)

    assert exc_info.value.reason_code == "asr_inference_failed"
    assert "provider detail" not in str(exc_info.value.public_details())
    assert session.state is IncrementalASRState.FAILED

    with pytest.raises(RuntimeNotReadyError) as not_ready:
        await session.push_pcm(pcm(1), speech=True)
    assert not_ready.value.context["state"] == "failed"
    assert not_ready.value.context["failure_reason_code"] == "asr_inference_failed"


@pytest.mark.asyncio
async def test_non_text_backend_output_is_typed_invalid_output() -> None:
    session = IncrementalASRSession(NonTextBackend(), config=config())

    with pytest.raises(ASROutputError) as exc_info:
        await session.push_pcm(pcm(4), speech=True)

    assert exc_info.value.reason_code == "asr_output_invalid"
    assert exc_info.value.context["violation"] == "text_not_string"
    assert session.state is IncrementalASRState.FAILED


@pytest.mark.asyncio
async def test_async_backend_is_supported() -> None:
    backend = AsyncBackend()
    session = IncrementalASRSession(backend, config=config())

    revisions = await session.push_pcm(pcm(4), speech=True)

    assert len(revisions) == 1
    assert revisions[0].text == "samples:4"
    assert backend.calls == [(4, 0, 4)]


@pytest.mark.asyncio
async def test_input_contract_rejects_unsupported_or_unbounded_audio_before_model_work() -> None:
    with pytest.raises(ValueError, match="16 kHz"):
        IncrementalASRConfig(sample_rate=8_000)
    with pytest.raises(ValueError, match="mono"):
        IncrementalASRConfig(channels=2)
    with pytest.raises(ValueError, match="16-bit"):
        IncrementalASRConfig(sample_width_bytes=4)
    with pytest.raises(ValueError, match="pcm_s16le"):
        IncrementalASRConfig(encoding="float32")

    backend = RecordingBackend()
    session = IncrementalASRSession(
        backend,
        config=config(max_chunk_bytes=8),
    )

    with pytest.raises(ValueError, match="max_chunk_bytes"):
        await session.push_pcm(pcm(5), speech=True)
    with pytest.raises(ValueError, match="complete sample frames"):
        await session.push_pcm(b"\x00", speech=True)
    with pytest.raises(TypeError, match="must be bytes"):
        await session.push_pcm(bytearray(pcm(1)), speech=True)  # type: ignore[arg-type]

    assert backend.calls == []
    assert session.metrics.model_calls == 0
    assert session.absolute_sample == 0


@pytest.mark.asyncio
async def test_empty_input_and_end_are_idempotent() -> None:
    backend = RecordingBackend()
    session = IncrementalASRSession(backend, config=config())

    assert await session.push_pcm(b"", speech=True) == ()
    assert await session.push_pcm(pcm(2), speech=True) == ()

    ended = await session.end()
    assert len(ended) == 1
    assert ended[0].text == "samples:2"
    assert ended[0].final_reason == "end_of_stream"
    assert session.state is IncrementalASRState.ENDED
    assert await session.end() == ()

    with pytest.raises(RuntimeNotReadyError):
        await session.push_pcm(pcm(1), speech=True)


def test_revision_validates_identity_time_and_finality() -> None:
    with pytest.raises(ValueError, match="segment_id"):
        ASRRevision("", 1, 0, 1, "text", False)
    with pytest.raises(ValueError, match="revision"):
        ASRRevision("seg", 0, 0, 1, "text", False)
    with pytest.raises(ValueError, match="precede"):
        ASRRevision("seg", 1, 2, 1, "text", False)
    with pytest.raises(ValueError, match="agree"):
        ASRRevision("seg", 1, 0, 1, "text", True)

    revision = ASRRevision(
        "seg",
        1,
        8_000,
        16_000,
        "text",
        True,
        "vad_boundary",
    )
    assert revision.to_seconds(16_000) == (0.5, 1.0)
