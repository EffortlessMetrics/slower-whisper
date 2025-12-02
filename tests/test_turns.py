"""
Unit tests for conversational turn structure.

Tests the turn building logic that groups contiguous segments by speaker.
"""

import pytest

from transcription.models import Segment, Transcript
from transcription.turns import build_turns


def make_segment(
    seg_id: int, start: float, end: float, text: str, speaker_id: str | None = None
) -> Segment:
    """Helper to create Segment for testing."""
    seg = Segment(id=seg_id, start=start, end=end, text=text)
    if speaker_id is not None:
        seg.speaker = {"id": speaker_id, "confidence": 0.9}
    return seg


def make_transcript(segments: list[Segment]) -> Transcript:
    """Helper to create Transcript for testing."""
    # Build speakers list from segments
    speaker_ids = {seg.speaker["id"] for seg in segments if seg.speaker is not None}
    speakers = [
        {"id": sid, "label": None, "total_speech_time": 0.0, "num_segments": 0}
        for sid in sorted(speaker_ids)
    ]
    return Transcript(file_name="test.wav", language="en", segments=segments, speakers=speakers)


class TestBuildTurns:
    """Tests for build_turns() function."""

    def test_single_speaker_one_turn(self):
        """Single speaker across multiple segments → 1 turn."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", "spk_0"),
            make_segment(1, 2.1, 4.0, "world", "spk_0"),
            make_segment(2, 4.2, 6.0, "today", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 1
        turn = result.turns[0]
        assert turn["id"] == "turn_0"
        assert turn["speaker_id"] == "spk_0"
        assert turn["start"] == 0.0
        assert turn["end"] == 6.0
        assert turn["segment_ids"] == [0, 1, 2]
        assert turn["text"] == "Hello world today"

    def test_alternating_speakers(self):
        """A-B-A-B pattern → 4 turns."""
        segments = [
            make_segment(0, 0.0, 2.0, "A1", "spk_0"),
            make_segment(1, 3.0, 5.0, "B1", "spk_1"),
            make_segment(2, 6.0, 8.0, "A2", "spk_0"),
            make_segment(3, 9.0, 11.0, "B2", "spk_1"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 4
        assert result.turns[0]["speaker_id"] == "spk_0"
        assert result.turns[1]["speaker_id"] == "spk_1"
        assert result.turns[2]["speaker_id"] == "spk_0"
        assert result.turns[3]["speaker_id"] == "spk_1"

        assert result.turns[0]["segment_ids"] == [0]
        assert result.turns[1]["segment_ids"] == [1]
        assert result.turns[2]["segment_ids"] == [2]
        assert result.turns[3]["segment_ids"] == [3]

    def test_multi_segment_turns(self):
        """Multiple consecutive segments per speaker → grouped into turns."""
        segments = [
            make_segment(0, 0.0, 2.0, "A1", "spk_0"),
            make_segment(1, 2.1, 4.0, "A2", "spk_0"),
            make_segment(2, 5.0, 7.0, "B1", "spk_1"),
            make_segment(3, 7.1, 9.0, "B2", "spk_1"),
            make_segment(4, 10.0, 12.0, "A3", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 3
        assert result.turns[0]["speaker_id"] == "spk_0"
        assert result.turns[0]["segment_ids"] == [0, 1]
        assert result.turns[0]["text"] == "A1 A2"

        assert result.turns[1]["speaker_id"] == "spk_1"
        assert result.turns[1]["segment_ids"] == [2, 3]
        assert result.turns[1]["text"] == "B1 B2"

        assert result.turns[2]["speaker_id"] == "spk_0"
        assert result.turns[2]["segment_ids"] == [4]
        assert result.turns[2]["text"] == "A3"

    def test_unknown_speaker_segments_skipped(self):
        """Segments with speaker=None → excluded from turns."""
        segments = [
            make_segment(0, 0.0, 2.0, "A1", "spk_0"),
            make_segment(1, 3.0, 5.0, "Unknown", None),  # No speaker
            make_segment(2, 6.0, 8.0, "B1", "spk_1"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        # Should have 2 turns (unknown segment excluded)
        assert len(result.turns) == 2
        assert result.turns[0]["segment_ids"] == [0]
        assert result.turns[1]["segment_ids"] == [2]

    def test_turn_boundaries_from_first_last_segment(self):
        """Turn start/end derived from first/last segment in turn."""
        segments = [
            make_segment(0, 1.5, 3.0, "A1", "spk_0"),
            make_segment(1, 3.2, 5.7, "A2", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 1
        assert result.turns[0]["start"] == 1.5
        assert result.turns[0]["end"] == 5.7

    def test_text_concatenation_with_spaces(self):
        """Turn text = segments joined with spaces."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", "spk_0"),
            make_segment(1, 2.1, 4.0, "world!", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert result.turns[0]["text"] == "Hello world!"

    def test_text_concatenation_strips_whitespace(self):
        """Segment text is stripped before concatenation."""
        segments = [
            make_segment(0, 0.0, 2.0, "  Hello  ", "spk_0"),
            make_segment(1, 2.1, 4.0, "  world  ", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert result.turns[0]["text"] == "Hello world"

    def test_empty_transcript(self):
        """Transcript with no segments → no turns."""
        transcript = Transcript(file_name="test.wav", language="en", segments=[], speakers=[])

        result = build_turns(transcript)

        assert result.turns == []

    def test_all_unknown_speakers(self):
        """All segments have speaker=None → no turns."""
        segments = [
            make_segment(0, 0.0, 2.0, "Unknown1", None),
            make_segment(1, 3.0, 5.0, "Unknown2", None),
        ]
        transcript = Transcript(file_name="test.wav", language="en", segments=segments, speakers=[])

        result = build_turns(transcript)

        assert result.turns == []

    def test_turn_ids_sequential(self):
        """Turn IDs are turn_0, turn_1, turn_2, ..."""
        segments = [
            make_segment(0, 0.0, 2.0, "A", "spk_0"),
            make_segment(1, 3.0, 5.0, "B", "spk_1"),
            make_segment(2, 6.0, 8.0, "C", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert result.turns[0]["id"] == "turn_0"
        assert result.turns[1]["id"] == "turn_1"
        assert result.turns[2]["id"] == "turn_2"

    def test_single_segment_per_turn(self):
        """Each segment is its own turn (all different speakers)."""
        segments = [
            make_segment(0, 0.0, 2.0, "A", "spk_0"),
            make_segment(1, 3.0, 5.0, "B", "spk_1"),
            make_segment(2, 6.0, 8.0, "C", "spk_2"),
        ]
        # Need to build speakers list manually for this test
        speakers = [
            {"id": "spk_0", "label": None, "total_speech_time": 0.0, "num_segments": 0},
            {"id": "spk_1", "label": None, "total_speech_time": 0.0, "num_segments": 0},
            {"id": "spk_2", "label": None, "total_speech_time": 0.0, "num_segments": 0},
        ]
        transcript = Transcript(
            file_name="test.wav", language="en", segments=segments, speakers=speakers
        )

        result = build_turns(transcript)

        assert len(result.turns) == 3
        assert [t["segment_ids"] for t in result.turns] == [[0], [1], [2]]

    def test_turn_preserves_segment_order(self):
        """Turns built in segment order (should already be sorted by time)."""
        segments = [
            make_segment(0, 0.0, 2.0, "A1", "spk_0"),
            make_segment(1, 2.1, 4.0, "A2", "spk_0"),
            make_segment(2, 5.0, 7.0, "B", "spk_1"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert result.turns[0]["segment_ids"] == [0, 1]
        assert result.turns[1]["segment_ids"] == [2]

    def test_unknown_then_known_speaker(self):
        """Unknown segments followed by known → turns start after unknown."""
        segments = [
            make_segment(0, 0.0, 2.0, "Unknown1", None),
            make_segment(1, 3.0, 5.0, "A", "spk_0"),
            make_segment(2, 5.1, 7.0, "B", "spk_1"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 2
        assert result.turns[0]["speaker_id"] == "spk_0"
        assert result.turns[1]["speaker_id"] == "spk_1"

    def test_known_then_unknown_speaker(self):
        """Known speaker followed by unknown → turn ends, unknown skipped."""
        segments = [
            make_segment(0, 0.0, 2.0, "A", "spk_0"),
            make_segment(1, 3.0, 5.0, "Unknown", None),
            make_segment(2, 6.0, 8.0, "B", "spk_1"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 2
        assert result.turns[0]["segment_ids"] == [0]
        assert result.turns[1]["segment_ids"] == [2]


class TestEdgeCases:
    """Edge case handling for turn building."""

    def test_empty_text_segments(self):
        """Segments with empty text → included in turn but don't break concatenation."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", "spk_0"),
            make_segment(1, 2.1, 4.0, "", "spk_0"),  # Empty text
            make_segment(2, 4.2, 6.0, "world", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 1
        assert result.turns[0]["segment_ids"] == [0, 1, 2]
        # Empty text should not create extra spaces
        assert result.turns[0]["text"] == "Hello world"

    def test_long_turn_many_segments(self):
        """Single speaker monologue with many segments → 1 long turn."""
        segments = [
            make_segment(i, float(i), float(i + 1), f"Segment {i}", "spk_0") for i in range(100)
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 1
        assert len(result.turns[0]["segment_ids"]) == 100
        assert result.turns[0]["start"] == 0.0
        assert result.turns[0]["end"] == 100.0

    def test_string_speaker_ids_supported(self):
        """String-valued speaker fields should not crash and still group turns."""
        segments = [
            Segment(id=0, start=0.0, end=1.0, text="Hi", speaker="agent"),
            Segment(id=1, start=1.1, end=2.0, text="Hello", speaker="user"),
            Segment(id=2, start=2.1, end=3.0, text="Again", speaker="user"),
        ]
        transcript = Transcript(file_name="test.wav", language="en", segments=segments, speakers=[])

        result = build_turns(transcript)

        assert [t["speaker_id"] for t in result.turns] == ["agent", "user"]
        assert result.turns[0]["segment_ids"] == [0]
        assert result.turns[1]["segment_ids"] == [1, 2]

    def test_missing_speaker_id_dicts_skipped(self):
        """Dict speakers without an id should be ignored instead of crashing."""
        segments = [
            Segment(id=0, start=0.0, end=1.0, text="Unknown", speaker={"label": "Agent"}),
            Segment(id=1, start=1.1, end=2.0, text="Known", speaker={"id": "spk_0"}),
        ]
        transcript = Transcript(file_name="test.wav", language="en", segments=segments, speakers=[])

        result = build_turns(transcript)

        assert len(result.turns) == 1
        assert result.turns[0]["speaker_id"] == "spk_0"
        assert result.turns[0]["segment_ids"] == [1]

    def test_whitespace_only_text_segments(self):
        """Segments with only whitespace → treated as empty in concatenation."""
        segments = [
            make_segment(0, 0.0, 2.0, "Hello", "spk_0"),
            make_segment(1, 2.1, 4.0, "   ", "spk_0"),  # Whitespace only
            make_segment(2, 4.2, 6.0, "world", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 1
        assert result.turns[0]["segment_ids"] == [0, 1, 2]
        # Whitespace-only segment excluded from text
        assert result.turns[0]["text"] == "Hello world"

    def test_speaker_confidence_variations(self):
        """Segments can have different confidence levels (stored, but doesn't affect turns)."""
        segments = [
            Segment(
                id=0,
                start=0.0,
                end=2.0,
                text="Confident",
                speaker={"id": "spk_0", "confidence": 0.95},
            ),
            Segment(
                id=1,
                start=2.1,
                end=4.0,
                text="Less sure",
                speaker={"id": "spk_0", "confidence": 0.65},
            ),
            Segment(
                id=2,
                start=5.0,
                end=7.0,
                text="Another",
                speaker={"id": "spk_1", "confidence": 0.50},
            ),
        ]
        speakers = [
            {"id": "spk_0", "label": None, "total_speech_time": 0.0, "num_segments": 0},
            {"id": "spk_1", "label": None, "total_speech_time": 0.0, "num_segments": 0},
        ]
        transcript = Transcript(
            file_name="test.wav", language="en", segments=segments, speakers=speakers
        )

        result = build_turns(transcript)

        # Turns should be grouped by speaker ID regardless of confidence
        assert len(result.turns) == 2
        assert result.turns[0]["speaker_id"] == "spk_0"
        assert result.turns[0]["segment_ids"] == [0, 1]
        assert result.turns[1]["speaker_id"] == "spk_1"
        assert result.turns[1]["segment_ids"] == [2]

    def test_in_place_mutation_returns_same_object(self):
        """build_turns mutates transcript in-place and returns it."""
        segments = [make_segment(0, 0.0, 2.0, "Test", "spk_0")]
        transcript = make_transcript(segments)
        original_id = id(transcript)

        result = build_turns(transcript)

        # Should return same object (in-place mutation)
        assert id(result) == original_id
        assert result.turns is not None
        assert len(result.turns) == 1

    def test_build_turns_idempotent(self):
        """Calling build_turns twice produces same result."""
        segments = [
            make_segment(0, 0.0, 2.0, "A", "spk_0"),
            make_segment(1, 3.0, 5.0, "B", "spk_1"),
        ]
        transcript = make_transcript(segments)

        result1 = build_turns(transcript)
        result2 = build_turns(transcript)

        # Both should have same turns
        assert len(result1.turns) == len(result2.turns)
        assert result1.turns[0] == result2.turns[0]
        assert result1.turns[1] == result2.turns[1]

    def test_exact_timestamp_boundaries(self):
        """Speaker change at exact timestamp boundary."""
        segments = [
            make_segment(0, 0.0, 5.0, "Speaker A", "spk_0"),
            make_segment(1, 5.0, 10.0, "Speaker B", "spk_1"),  # Starts exactly when A ends
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 2
        assert result.turns[0]["end"] == 5.0
        assert result.turns[1]["start"] == 5.0
        assert result.turns[0]["speaker_id"] == "spk_0"
        assert result.turns[1]["speaker_id"] == "spk_1"

    def test_complex_speaker_ids(self):
        """Speaker IDs with special characters or long strings."""
        segments = [
            make_segment(0, 0.0, 2.0, "First", "speaker_001_participant"),
            make_segment(1, 3.0, 5.0, "Second", "speaker_001_participant"),
            make_segment(2, 6.0, 8.0, "Third", "speaker-002-guest"),
        ]
        speakers = [
            {
                "id": "speaker_001_participant",
                "label": None,
                "total_speech_time": 0.0,
                "num_segments": 0,
            },
            {"id": "speaker-002-guest", "label": None, "total_speech_time": 0.0, "num_segments": 0},
        ]
        transcript = Transcript(
            file_name="test.wav", language="en", segments=segments, speakers=speakers
        )

        result = build_turns(transcript)

        assert len(result.turns) == 2
        assert result.turns[0]["speaker_id"] == "speaker_001_participant"
        assert result.turns[0]["segment_ids"] == [0, 1]
        assert result.turns[1]["speaker_id"] == "speaker-002-guest"
        assert result.turns[1]["segment_ids"] == [2]

    def test_very_short_segments(self):
        """Very short segments (< 100ms) still grouped correctly."""
        segments = [
            make_segment(0, 0.0, 0.05, "Ah", "spk_0"),
            make_segment(1, 0.05, 0.1, "um", "spk_0"),
            make_segment(2, 0.1, 0.15, "yeah", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 1
        assert result.turns[0]["segment_ids"] == [0, 1, 2]
        assert result.turns[0]["start"] == 0.0
        assert result.turns[0]["end"] == 0.15

    def test_mixed_empty_and_content_segments(self):
        """Mix of empty, whitespace-only, and content-filled segments."""
        segments = [
            make_segment(0, 0.0, 1.0, "Start", "spk_0"),
            make_segment(1, 1.1, 2.0, "", "spk_0"),  # Empty
            make_segment(2, 2.1, 3.0, "   ", "spk_0"),  # Whitespace only
            make_segment(3, 3.1, 4.0, "\t\n", "spk_0"),  # Whitespace variants
            make_segment(4, 4.1, 5.0, "End", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 1
        assert result.turns[0]["segment_ids"] == [0, 1, 2, 3, 4]
        # Only Start and End should appear in text
        assert result.turns[0]["text"] == "Start End"

    def test_three_way_alternation(self):
        """Three speakers alternating (A-B-C-A-B-C pattern)."""
        segments = [
            make_segment(0, 0.0, 1.0, "A1", "spk_0"),
            make_segment(1, 2.0, 3.0, "B1", "spk_1"),
            make_segment(2, 4.0, 5.0, "C1", "spk_2"),
            make_segment(3, 6.0, 7.0, "A2", "spk_0"),
            make_segment(4, 8.0, 9.0, "B2", "spk_1"),
            make_segment(5, 10.0, 11.0, "C2", "spk_2"),
        ]
        speakers = [
            {"id": "spk_0", "label": None, "total_speech_time": 0.0, "num_segments": 0},
            {"id": "spk_1", "label": None, "total_speech_time": 0.0, "num_segments": 0},
            {"id": "spk_2", "label": None, "total_speech_time": 0.0, "num_segments": 0},
        ]
        transcript = Transcript(
            file_name="test.wav", language="en", segments=segments, speakers=speakers
        )

        result = build_turns(transcript)

        assert len(result.turns) == 6
        expected_speakers = ["spk_0", "spk_1", "spk_2", "spk_0", "spk_1", "spk_2"]
        for i, expected_spk in enumerate(expected_speakers):
            assert result.turns[i]["speaker_id"] == expected_spk
            assert result.turns[i]["segment_ids"] == [i]

    def test_consecutive_speaker_changes(self):
        """Rapid speaker changes (multiple turns in quick succession)."""
        segments = [
            make_segment(0, 0.0, 1.0, "A", "spk_0"),
            make_segment(1, 1.0, 2.0, "B", "spk_1"),
            make_segment(2, 2.0, 3.0, "C", "spk_0"),
            make_segment(3, 3.0, 4.0, "D", "spk_1"),
            make_segment(4, 4.0, 5.0, "E", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 5
        for i, _seg in enumerate(segments):
            assert result.turns[i]["segment_ids"] == [i]

    def test_turn_id_format(self):
        """Turn IDs follow turn_N format."""
        segments = [
            make_segment(0, 0.0, 2.0, "A", "spk_0"),
            make_segment(1, 3.0, 5.0, "B", "spk_1"),
            make_segment(2, 6.0, 8.0, "C", "spk_0"),
            make_segment(3, 9.0, 11.0, "D", "spk_1"),
            make_segment(4, 12.0, 14.0, "E", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        for i in range(5):
            assert result.turns[i]["id"] == f"turn_{i}"

    def test_float_precision_timestamps(self):
        """Float timestamps with high precision preserved correctly."""
        segments = [
            make_segment(0, 0.123456, 2.654321, "A", "spk_0"),
            make_segment(1, 2.654322, 5.987654, "B", "spk_0"),
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        assert len(result.turns) == 1
        # Timestamps should be preserved exactly (or close with floating point)
        assert result.turns[0]["start"] == pytest.approx(0.123456)
        assert result.turns[0]["end"] == pytest.approx(5.987654)


class TestPauseThreshold:
    """Tests for pause_threshold parameter in build_turns()."""

    def test_pause_threshold_disabled_by_default(self):
        """When pause_threshold=None (default), only speaker changes split turns."""
        segments = [
            make_segment(0, 0.0, 2.0, "First", "spk_0"),
            make_segment(1, 10.0, 12.0, "Second after long pause", "spk_0"),  # 8s gap
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript)

        # Should be 1 turn despite 8s gap (pause_threshold disabled)
        assert len(result.turns) == 1
        assert result.turns[0]["segment_ids"] == [0, 1]

    def test_pause_threshold_splits_same_speaker(self):
        """Long pause >= pause_threshold splits turns for same speaker."""
        segments = [
            make_segment(0, 0.0, 2.0, "First", "spk_0"),
            make_segment(1, 10.0, 12.0, "Second after long pause", "spk_0"),  # 8s gap
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=5.0)

        # Should be 2 turns (gap of 8s >= 5s threshold)
        assert len(result.turns) == 2
        assert result.turns[0]["segment_ids"] == [0]
        assert result.turns[1]["segment_ids"] == [1]
        assert result.turns[0]["speaker_id"] == "spk_0"
        assert result.turns[1]["speaker_id"] == "spk_0"

    def test_pause_threshold_exact_boundary(self):
        """Gap exactly equal to pause_threshold triggers split."""
        segments = [
            make_segment(0, 0.0, 2.0, "First", "spk_0"),
            make_segment(1, 5.0, 7.0, "Second", "spk_0"),  # Gap = 3.0s
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=3.0)

        # Gap of 3.0s == 3.0s threshold → should split
        assert len(result.turns) == 2

    def test_pause_threshold_just_below_boundary(self):
        """Gap slightly below pause_threshold does not split."""
        segments = [
            make_segment(0, 0.0, 2.0, "First", "spk_0"),
            make_segment(1, 4.9, 7.0, "Second", "spk_0"),  # Gap = 2.9s
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=3.0)

        # Gap of 2.9s < 3.0s threshold → no split
        assert len(result.turns) == 1
        assert result.turns[0]["segment_ids"] == [0, 1]

    def test_pause_threshold_multiple_splits_same_speaker(self):
        """Multiple long pauses split into multiple turns."""
        segments = [
            make_segment(0, 0.0, 1.0, "First", "spk_0"),
            make_segment(1, 4.0, 5.0, "Second", "spk_0"),  # Gap = 3.0s
            make_segment(2, 5.5, 6.5, "Third", "spk_0"),  # Gap = 0.5s
            make_segment(3, 10.0, 11.0, "Fourth", "spk_0"),  # Gap = 3.5s
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=2.0)

        # Should have 3 turns: [0], [1,2], [3]
        assert len(result.turns) == 3
        assert result.turns[0]["segment_ids"] == [0]
        assert result.turns[1]["segment_ids"] == [1, 2]
        assert result.turns[2]["segment_ids"] == [3]

    def test_pause_threshold_with_speaker_changes(self):
        """Pause threshold works together with speaker changes."""
        segments = [
            make_segment(0, 0.0, 1.0, "A1", "spk_0"),
            make_segment(1, 5.0, 6.0, "A2", "spk_0"),  # Gap = 4.0s (long pause)
            make_segment(2, 6.5, 7.5, "B", "spk_1"),  # Speaker change
            make_segment(3, 8.0, 9.0, "C", "spk_0"),  # Back to spk_0
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=3.0)

        # Should have 4 turns: pause splits A, speaker change, back to A
        assert len(result.turns) == 4
        assert result.turns[0]["speaker_id"] == "spk_0"
        assert result.turns[1]["speaker_id"] == "spk_0"
        assert result.turns[2]["speaker_id"] == "spk_1"
        assert result.turns[3]["speaker_id"] == "spk_0"

    def test_pause_threshold_zero(self):
        """pause_threshold=0.0 splits on any gap."""
        segments = [
            make_segment(0, 0.0, 1.0, "First", "spk_0"),
            make_segment(1, 1.1, 2.0, "Second", "spk_0"),  # Gap = 0.1s
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=0.0)

        # Even tiny gap splits turns
        assert len(result.turns) == 2

    def test_pause_threshold_very_small(self):
        """Very small pause_threshold (0.01s) splits on tiny gaps."""
        segments = [
            make_segment(0, 0.0, 1.0, "First", "spk_0"),
            make_segment(1, 1.02, 2.0, "Second", "spk_0"),  # Gap = 0.02s
            make_segment(2, 2.005, 3.0, "Third", "spk_0"),  # Gap = 0.005s
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=0.01)

        # Gap 0.02s >= 0.01s → split
        # Gap 0.005s < 0.01s → no split
        assert len(result.turns) == 2
        assert result.turns[0]["segment_ids"] == [0]
        assert result.turns[1]["segment_ids"] == [1, 2]

    def test_pause_threshold_negative_raises_error(self):
        """Negative pause_threshold raises ValueError."""
        segments = [make_segment(0, 0.0, 1.0, "Test", "spk_0")]
        transcript = make_transcript(segments)

        with pytest.raises(ValueError, match="pause_threshold must be >= 0.0"):
            build_turns(transcript, pause_threshold=-1.0)

    def test_pause_threshold_no_gaps(self):
        """Segments with no gaps (overlapping or continuous) don't split."""
        segments = [
            make_segment(0, 0.0, 2.0, "First", "spk_0"),
            make_segment(1, 2.0, 4.0, "Second", "spk_0"),  # Gap = 0.0s
            make_segment(2, 3.5, 5.0, "Third", "spk_0"),  # Overlap (gap = -0.5s)
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=1.0)

        # No gaps >= 1.0s, so all in one turn
        assert len(result.turns) == 1
        assert result.turns[0]["segment_ids"] == [0, 1, 2]

    def test_pause_threshold_preserves_turn_ids(self):
        """Turn IDs remain sequential when pause splits occur."""
        segments = [
            make_segment(0, 0.0, 1.0, "A1", "spk_0"),
            make_segment(1, 5.0, 6.0, "A2", "spk_0"),  # Gap = 4.0s
            make_segment(2, 6.5, 7.5, "B", "spk_1"),  # Speaker change
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=3.0)

        # Should have turn_0, turn_1, turn_2
        assert result.turns[0]["id"] == "turn_0"
        assert result.turns[1]["id"] == "turn_1"
        assert result.turns[2]["id"] == "turn_2"

    def test_pause_threshold_with_unknown_speakers(self):
        """Pause threshold only applies to segments with speakers."""
        segments = [
            make_segment(0, 0.0, 1.0, "Known", "spk_0"),
            make_segment(1, 10.0, 11.0, "Unknown", None),  # Gap = 9s, but no speaker
            make_segment(2, 20.0, 21.0, "Known again", "spk_0"),  # Gap = 9s
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=5.0)

        # Unknown segment skipped; known segments treated separately
        assert len(result.turns) == 2
        assert result.turns[0]["segment_ids"] == [0]
        assert result.turns[1]["segment_ids"] == [2]

    def test_pause_threshold_text_concatenation(self):
        """Pause-split turns correctly concatenate text."""
        segments = [
            make_segment(0, 0.0, 1.0, "Hello", "spk_0"),
            make_segment(1, 1.5, 2.5, "world", "spk_0"),  # Gap = 0.5s
            make_segment(2, 10.0, 11.0, "After pause", "spk_0"),  # Gap = 7.5s
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=5.0)

        assert len(result.turns) == 2
        assert result.turns[0]["text"] == "Hello world"
        assert result.turns[1]["text"] == "After pause"

    def test_pause_threshold_large_value(self):
        """Very large pause_threshold effectively disables splitting."""
        segments = [
            make_segment(0, 0.0, 1.0, "First", "spk_0"),
            make_segment(1, 100.0, 101.0, "Second", "spk_0"),  # Gap = 99s
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=1000.0)

        # Gap of 99s < 1000s → no split
        assert len(result.turns) == 1
        assert result.turns[0]["segment_ids"] == [0, 1]

    def test_pause_threshold_realistic_conversation(self):
        """Realistic multi-speaker conversation with natural pauses."""
        segments = [
            make_segment(0, 0.0, 3.0, "Hi, how are you?", "spk_0"),
            make_segment(1, 3.5, 5.0, "I'm good, thanks!", "spk_1"),
            make_segment(2, 5.2, 8.0, "And you?", "spk_1"),  # Same speaker, short gap
            make_segment(3, 15.0, 18.0, "I'm doing well", "spk_0"),  # Long pause (7s)
            make_segment(4, 18.2, 20.0, "Had a busy day", "spk_0"),  # Short gap
        ]
        transcript = make_transcript(segments)

        result = build_turns(transcript, pause_threshold=5.0)

        # Expected turns:
        # turn_0: [0] (spk_0)
        # turn_1: [1, 2] (spk_1, short gap)
        # turn_2: [3, 4] (spk_0, long pause splits, then short gap keeps together)
        assert len(result.turns) == 3
        assert result.turns[0]["speaker_id"] == "spk_0"
        assert result.turns[0]["segment_ids"] == [0]
        assert result.turns[1]["speaker_id"] == "spk_1"
        assert result.turns[1]["segment_ids"] == [1, 2]
        assert result.turns[2]["speaker_id"] == "spk_0"
        assert result.turns[2]["segment_ids"] == [3, 4]
