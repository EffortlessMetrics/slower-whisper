from slower_whisper.pipeline.models import Segment, Transcript
from slower_whisper.pipeline.turns_enrich import _estimate_disfluency_ratio, enrich_turns_metadata


def test_disfluency_ratio_handles_punctuation_and_empty():
    assert _estimate_disfluency_ratio("") == 0.0
    assert _estimate_disfluency_ratio("Um, uh, like... you know?") > 0.5


def test_interruption_flag_when_overlap_exceeds_threshold():
    transcript = Transcript(
        file_name="sample.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=1.0, text="First turn"),
            Segment(id=1, start=0.8, end=2.0, text="Second turn"),
        ],
        turns=[
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "segment_ids": [0],
                "start": 0.0,
                "end": 1.0,
                "text": "First turn",
            },
            {
                "id": "turn_1",
                "speaker_id": "spk_1",
                "segment_ids": [1],
                "start": 0.8,
                "end": 2.0,
                "text": "Second turn",
            },
        ],
    )

    enriched = enrich_turns_metadata(transcript)

    assert enriched[1]["metadata"]["interruption_started_here"] is True


def test_interruption_flag_false_when_gap_exceeds_threshold():
    transcript = Transcript(
        file_name="sample.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=1.0, text="First turn"),
            Segment(id=1, start=1.1, end=2.2, text="Second turn"),
        ],
        turns=[
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "segment_ids": [0],
                "start": 0.0,
                "end": 1.0,
                "text": "First turn",
            },
            {
                "id": "turn_1",
                "speaker_id": "spk_1",
                "segment_ids": [1],
                "start": 1.1,
                "end": 2.2,
                "text": "Second turn",
            },
        ],
    )

    enriched = enrich_turns_metadata(transcript)

    assert enriched[1]["metadata"]["interruption_started_here"] is False


def test_enrich_turns_metadata_builds_turns_when_none() -> None:
    """enrich_turns_metadata should auto-build turns when transcript.turns is None."""
    segments = [
        Segment(
            id=0,
            start=0.0,
            end=1.0,
            text="Hello there.",
            speaker={"id": "spk_0"},
        ),
        Segment(
            id=1,
            start=1.1,
            end=2.0,
            text="Hi!",
            speaker={"id": "spk_1"},
        ),
    ]

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        turns=None,  # Explicitly None - should be auto-built
    )

    enriched = enrich_turns_metadata(transcript)

    # Should have created turns
    assert enriched is not None
    assert len(enriched) == 2
    assert transcript.turns is enriched

    # Each turn should have metadata
    for turn in enriched:
        assert isinstance(turn, dict)
        assert "metadata" in turn
        meta = turn["metadata"]
        assert "question_count" in meta
        assert "interruption_started_here" in meta
