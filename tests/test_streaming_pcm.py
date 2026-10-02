"""PCM framing without ML, transport, or synthetic linguistic output."""

from __future__ import annotations

import random

import pytest

from transcription.exceptions import RuntimeNotReadyError, StreamingNegotiationError
from transcription.streaming_pcm import PCMFrameBuffer


def partitions(data: bytes, sizes: list[int]) -> list[bytes]:
    packets = []
    offset = 0
    index = 0
    while offset < len(data):
        size = sizes[index % len(sizes)]
        packets.append(data[offset : offset + size])
        offset += size
        index += 1
    return packets


@pytest.mark.parametrize("sizes", [[8192], [640], [1], [639, 2, 641, 17], [1025, 3, 7]])
def test_frames_and_tail_are_independent_of_packetization(sizes: list[int]) -> None:
    audio = bytes(range(256)) * 21 + b"\x01\x02"
    framer = PCMFrameBuffer(max_chunk_bytes=8192)
    frames = []
    for packet in partitions(audio, sizes):
        frames.extend(framer.feed(packet))
        metrics = framer.metrics
        assert metrics.residual_bytes < 640
        assert metrics.accepted_bytes == 2 * metrics.emitted_samples + metrics.residual_bytes
    frames.extend(framer.end())
    expected = [audio[i : i + 640] for i in range(0, len(audio), 640)]
    assert frames == expected
    assert b"".join(frames) == audio
    assert framer.metrics.emitted_samples == len(audio) // 2
    assert framer.metrics.frames_emitted == len(expected)
    assert framer.metrics.residual_bytes == 0
    assert framer.end() == ()


@pytest.mark.parametrize("seed", range(20))
def test_random_partitions_preserve_every_sample(seed: int) -> None:
    rng = random.Random(seed)
    audio = rng.randbytes(640 * 6 + 238)
    sizes = [rng.randint(1, 2048) for _ in range(25)]
    framer = PCMFrameBuffer(max_chunk_bytes=2048)
    frames = [frame for packet in partitions(audio, sizes) for frame in framer.feed(packet)]
    frames.extend(framer.end())
    assert frames == [audio[i : i + 640] for i in range(0, len(audio), 640)]
    assert framer.metrics.emitted_samples == len(audio) // 2


def test_short_tail_is_not_padded_or_emitted_early() -> None:
    framer = PCMFrameBuffer()
    assert framer.feed(b"\x01") == ()
    assert framer.feed(b"\x02\x03") == ()
    assert framer.feed(b"\x04") == ()
    assert framer.end() == (b"\x01\x02\x03\x04",)
    assert framer.metrics.emitted_samples == 2
    with pytest.raises(RuntimeNotReadyError):
        framer.feed(b"\x05\x06")


@pytest.mark.parametrize("tail_bytes", [1, 3, 639])
def test_incomplete_final_sample_is_terminal_and_never_padded(tail_bytes: int) -> None:
    framer = PCMFrameBuffer()
    assert framer.feed(b"\0" * tail_bytes) == ()
    with pytest.raises(StreamingNegotiationError) as exc:
        framer.end()
    assert exc.value.reason_code == "streaming_pcm_incomplete"
    assert exc.value.context == {
        "violation": "incomplete_pcm_sample",
        "residual_bytes": tail_bytes,
    }
    assert framer.metrics.emitted_samples == 0
    assert framer.metrics.residual_bytes == 0
    with pytest.raises(RuntimeNotReadyError):
        framer.end()
    with pytest.raises(RuntimeNotReadyError):
        framer.feed(b"\0")


def test_empty_packets_do_not_create_frames_or_samples() -> None:
    framer = PCMFrameBuffer()
    for _ in range(10):
        assert framer.feed(b"") == ()
    assert framer.end() == ()
    assert framer.metrics.accepted_bytes == 0
    assert framer.metrics.frames_emitted == 0


def test_invalid_packet_does_not_mutate_residual_or_counters() -> None:
    framer = PCMFrameBuffer(frame_samples=2, max_chunk_bytes=4)
    framer.feed(b"\x01")
    before = framer.metrics
    with pytest.raises(ValueError, match="max_chunk_bytes"):
        framer.feed(b"\0" * 5)
    assert framer.metrics == before
    with pytest.raises(TypeError, match="bytes"):
        framer.feed(bytearray(b"\0"))  # type: ignore[arg-type]
    assert framer.metrics == before
    assert framer.feed(b"\x02\x03\x04") == (b"\x01\x02\x03\x04",)


def test_abort_releases_residual_without_claiming_processing() -> None:
    framer = PCMFrameBuffer()
    framer.feed(b"\0" * 17)
    framer.abort()
    framer.abort()
    assert framer.metrics.accepted_bytes == 17
    assert framer.metrics.emitted_samples == 0
    assert framer.metrics.residual_bytes == 0
    assert framer.end() == ()
    with pytest.raises(RuntimeNotReadyError):
        framer.feed(b"\0")


@pytest.mark.parametrize("field", ["frame_samples", "max_chunk_bytes"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "320"])
def test_invalid_server_framing_configuration(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        PCMFrameBuffer(**{field: value})  # type: ignore[arg-type]


def test_frame_cannot_exceed_the_core_input_bound() -> None:
    with pytest.raises(ValueError, match="fit"):
        PCMFrameBuffer(frame_samples=320, max_chunk_bytes=638)
