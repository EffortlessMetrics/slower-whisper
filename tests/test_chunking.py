from typing import Any

from transcription.chunking import ChunkingConfig, build_chunks
from transcription.models import Segment, Transcript, Turn


def test_chunks_respect_turn_boundaries_when_hitting_max_duration():
    segments = [
        Segment(id=0, start=0.0, end=4.0, text="hello there", speaker={"id": "spk_0"}),
        Segment(id=1, start=4.0, end=8.0, text="more words", speaker={"id": "spk_0"}),
        Segment(id=2, start=8.0, end=12.0, text="reply", speaker={"id": "spk_1"}),
    ]
    turns: list[Turn | dict[str, Any]] = [
        Turn(
            id="turn_0",
            speaker_id="spk_0",
            segment_ids=[0, 1],
            start=0.0,
            end=8.0,
            text="hello there more words",
            metadata={},
        ),
        Turn(
            id="turn_1",
            speaker_id="spk_1",
            segment_ids=[2],
            start=8.0,
            end=12.0,
            text="reply",
            metadata={},
        ),
    ]
    transcript = Transcript(file_name="demo.wav", language="en", segments=segments, turns=turns)

    cfg = ChunkingConfig(
        target_duration_s=6.0, max_duration_s=8.0, target_tokens=400, pause_split_threshold_s=0.5
    )
    chunks = build_chunks(transcript, cfg)

    assert len(chunks) == 2
    assert chunks[0].turn_ids == ["turn_0"]
    assert chunks[1].turn_ids == ["turn_1"]
    assert set(chunks[0].segment_ids) == {0, 1}
    assert set(chunks[1].segment_ids) == {2}


def test_pause_triggers_split_at_target():
    segments = [
        Segment(id=0, start=0.0, end=3.0, text="first", speaker={"id": "spk_0"}),
        Segment(id=1, start=5.5, end=8.5, text="second", speaker={"id": "spk_0"}),
        Segment(id=2, start=9.0, end=11.0, text="third", speaker={"id": "spk_0"}),
    ]
    turns: list[Turn | dict[str, Any]] = [
        Turn(
            id="turn_0",
            speaker_id="spk_0",
            segment_ids=[0],
            start=0.0,
            end=3.0,
            text="first",
            metadata={},
        ),
        Turn(
            id="turn_1",
            speaker_id="spk_0",
            segment_ids=[1],
            start=5.5,
            end=8.5,
            text="second",
            metadata={},
        ),
        Turn(
            id="turn_2",
            speaker_id="spk_0",
            segment_ids=[2],
            start=9.0,
            end=11.0,
            text="third",
            metadata={},
        ),
    ]
    transcript = Transcript(file_name="demo.wav", language="en", segments=segments, turns=turns)

    cfg = ChunkingConfig(
        target_duration_s=5.0, max_duration_s=9.0, target_tokens=400, pause_split_threshold_s=1.5
    )
    chunks = build_chunks(transcript, cfg)

    assert len(chunks) == 2
    assert chunks[0].turn_ids == ["turn_0", "turn_1"]
    assert chunks[1].turn_ids == ["turn_2"]


def test_segment_fallback_when_no_turns():
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="hello"),
        Segment(id=1, start=2.1, end=4.0, text="world"),
    ]
    transcript = Transcript(file_name="demo.wav", language="en", segments=segments)

    cfg = ChunkingConfig(
        target_duration_s=10.0, max_duration_s=12.0, target_tokens=5, pause_split_threshold_s=0.5
    )
    chunks = build_chunks(transcript, cfg)

    assert len(chunks) == 1
    assert chunks[0].segment_ids == [0, 1]
    assert chunks[0].turn_ids == ["seg_0", "seg_1"]
