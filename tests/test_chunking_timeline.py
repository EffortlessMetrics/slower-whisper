"""Cross-stage interval coverage for start-ordered chunk inputs."""

import json
import shutil
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from transcription.asr_engine import TranscriptionEngine
from transcription.audio_io import normalize_single
from transcription.chunking import ChunkingConfig, build_chunks
from transcription.config import AsrConfig
from transcription.models import Chunk, Segment, Transcript, Turn
from transcription.service_serialization import _transcript_to_dict
from transcription.validation import validate_transcript_json
from transcription.writers import load_transcript_from_json, write_json


def _transcript(intervals: list[tuple[float, float]], representation: str) -> Transcript:
    """Represent the same ordered intervals as segments, typed turns, or dict turns."""
    segments = [
        Segment(id=i, start=start, end=end, text=f"unit_{i}", speaker={"id": "spk_0"})
        for i, (start, end) in enumerate(intervals)
    ]
    turns: list[Turn | dict[str, Any]] = []
    if representation != "segments":
        for seg in segments:
            turn = Turn(
                id=f"turn_{seg.id}",
                speaker_id="spk_0",
                segment_ids=[seg.id],
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
            turns.append(turn.to_dict() if representation == "dicts" else turn)
    return Transcript(file_name="timeline.wav", language="en", segments=segments, turns=turns)


@pytest.mark.parametrize("representation", ["segments", "turns", "dicts"])
@pytest.mark.parametrize(
    "intervals",
    [
        [(0.0, 10.0), (1.0, 2.0)],
        [(0.0, 10.0), (0.0, 2.0)],
        [(0.0, 10.0), (1.0, 1.0)],
        [(0.0, 2.0), (3.0, 4.0)],
    ],
    ids=["nested", "equal-start", "zero-duration", "disjoint-control"],
)
def test_chunk_covers_all_members(intervals, representation):
    """A nested last member cannot truncate a chunk's exported interval."""
    transcript = _transcript(intervals, representation)
    chunks = build_chunks(transcript, ChunkingConfig(target_duration_s=100, max_duration_s=100))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.start == intervals[0][0]
    assert chunk.end == max(end for _, end in intervals)
    assert chunk.segment_ids == [0, 1]
    assert chunk.text == "unit_0 unit_1"
    assert chunk.has_overlapping_speech == (intervals[1][0] < intervals[0][1])


@pytest.mark.parametrize("representation", ["segments", "turns", "dicts"])
@pytest.mark.parametrize("next_start", [9.0, 10.5, 11.0])
def test_pause_is_measured_from_covered_end(next_start, representation):
    """Only a full pause after all current members end may split these chunks."""
    transcript = _transcript([(0, 10), (1, 8), (next_start, next_start + 1)], representation)
    config = ChunkingConfig(
        target_duration_s=8,
        max_duration_s=20,
        pause_split_threshold_s=1,
        turn_affinity=1,
        cross_turn_penalty=2,
    )
    chunks = build_chunks(transcript, config)

    if next_start < 11:
        assert [chunk.segment_ids for chunk in chunks] == [[0, 1, 2]]
        assert chunks[0].end == max(10, next_start + 1)
    else:
        assert [chunk.segment_ids for chunk in chunks] == [[0, 1], [2]]
        assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 10), (11, 12)]


def test_oversized_unit_stays_alone_and_coverage_resets():
    """An indivisible oversized unit must not admit nested units under the hard limit."""
    transcript = _transcript([(0, 10), (1, 2), (3, 4)], "segments")
    chunks = build_chunks(transcript, ChunkingConfig(target_duration_s=100, max_duration_s=8))

    assert [chunk.segment_ids for chunk in chunks] == [[0], [1, 2]]
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 10), (1, 4)]


def test_coverage_resets_after_token_limit():
    """A token split must not carry a previous chunk's long end into a nested chunk."""
    transcript = _transcript([(0, 10), (1, 2), (1.5, 1.75)], "segments")
    chunks = build_chunks(
        transcript, ChunkingConfig(target_duration_s=100, max_duration_s=100, target_tokens=2)
    )

    assert [chunk.segment_ids for chunk in chunks] == [[0, 1], [2]]
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 10), (1.5, 1.75)]


@pytest.mark.parametrize("outer_end", [4.4, 10.0], ids=["before-unit", "look-ahead"])
def test_turn_split_uses_covered_duration_and_resets(outer_end):
    """Nested units cannot hide the soft duration target at a speaker boundary."""
    transcript = _transcript([(0, outer_end), (0.1, 0.2), (0.3, 0.4)], "segments")
    transcript.segments[-1].speaker = {"id": "spk_1"}
    chunks = build_chunks(
        transcript,
        ChunkingConfig(target_duration_s=8, max_duration_s=20, turn_affinity=0.7),
    )

    assert [chunk.segment_ids for chunk in chunks] == [[0, 1], [2]]
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, outer_end), (0.3, 0.4)]


def test_soft_split_does_not_leak_coverage_into_next_split_decision():
    """A prior long chunk must not prematurely split later short speaker turns."""
    transcript = _transcript([(0, 10), (1, 2), (1.5, 3)], "segments")
    for seg in transcript.segments:
        seg.speaker = {"id": f"spk_{seg.id}"}
    chunks = build_chunks(
        transcript,
        ChunkingConfig(target_duration_s=10, max_duration_s=20, turn_affinity=1),
    )

    assert [chunk.segment_ids for chunk in chunks] == [[0], [1, 2]]
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 10), (1, 3)]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required for normalization")
def test_normalized_nested_timeline_survives_json_round_trip(tmp_path):
    """Normalize tiny audio and backend metadata, then preserve coverage across JSON paths."""
    source = tmp_path / "source.wav"
    normalized = tmp_path / "normalized.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0" * (8000 * 2 * 2))
    normalize_single(source, normalized)
    with wave.open(str(normalized), "rb") as normalized_audio:
        assert (
            normalized_audio.getnchannels(),
            normalized_audio.getframerate(),
            normalized_audio.getnframes(),
        ) == (1, 16000, 16000)

    class MetadataBackend:
        """Return out-of-order, overlapping metadata without loading a model."""

        def transcribe(self, audio_path: str, *_args: Any, **_kwargs: Any):
            """Check the normalized input and supply a deterministic interval counterexample."""
            assert Path(audio_path) == normalized
            return iter(
                [
                    SimpleNamespace(start=0.1, end=0.2, text="nested"),
                    SimpleNamespace(start=0.0, end=1.0, text="outer"),
                ]
            ), SimpleNamespace(language="en")

    engine = TranscriptionEngine(
        AsrConfig(device="cpu", compute_type="int8"),
        backend_factory=lambda *_args: MetadataBackend(),
    )
    transcript = engine.transcribe_file(normalized)
    assert [(seg.id, seg.start, seg.end) for seg in transcript.segments] == [
        (0, 0.0, 1.0),
        (1, 0.1, 0.2),
    ]
    # Separate speakers keep each turn correct independently of turn aggregation.
    for seg in transcript.segments:
        seg.speaker = {"id": f"spk_{seg.id}", "confidence": 1.0}
    chunks = build_chunks(transcript, ChunkingConfig())
    assert len(chunks) == 1
    assert chunks[0].segment_ids == [0, 1]
    assert chunks[0].turn_ids == ["turn_0", "turn_1"]
    assert chunks[0].text == "outer nested"
    assert chunks[0].has_overlapping_speech

    written = tmp_path / "written.json"
    write_json(transcript, written)
    served = tmp_path / "served.json"
    served.write_text(json.dumps(_transcript_to_dict(transcript)), encoding="utf-8")
    for path in (written, served):
        assert validate_transcript_json(path) == (True, [])
        restored = load_transcript_from_json(path)
        assert restored.chunks is not None
        assert isinstance(restored.chunks[0], Chunk)
        assert restored.chunks[0].to_dict() == chunks[0].to_dict()
        assert (restored.chunks[0].start, restored.chunks[0].end) == (0.0, 1.0)
        assert [seg.text for seg in restored.segments] == ["outer", "nested"]
