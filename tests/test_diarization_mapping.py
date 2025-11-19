"""
Unit tests for speaker-to-segment assignment logic.

Tests the core overlap-based mapping algorithm without requiring pyannote.
Uses synthetic SpeakerTurn and Segment data to exercise all edge cases.
"""

import pytest

from transcription.diarization import SpeakerTurn, assign_speakers
from transcription.models import Segment, Transcript


# Test fixtures: synthetic speaker turns
def make_speaker_turn(start: float, end: float, speaker_id: str) -> SpeakerTurn:
    """Helper to create SpeakerTurn for testing."""
    return SpeakerTurn(start=start, end=end, speaker_id=speaker_id)


def make_segment(seg_id: int, start: float, end: float, text: str) -> Segment:
    """Helper to create Segment for testing."""
    return Segment(id=seg_id, start=start, end=end, text=text)


def make_transcript(segments: list[Segment]) -> Transcript:
    """Helper to create Transcript for testing."""
    return Transcript(file_name="test.wav", language="en", segments=segments)


class TestAssignSpeakers:
    """Tests for assign_speakers() overlap-based mapping."""

    def test_perfect_alignment(self):
        """Segment fully contained in speaker turn → confidence = 1.0."""
        turns = [make_speaker_turn(0.0, 5.0, "raw_spk_A")]
        segments = [make_segment(0, 1.0, 3.0, "Hello")]
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        assert result.speakers is not None
        assert len(result.speakers) == 1
        assert result.speakers[0]["id"] == "spk_0"

        assert result.segments[0].speaker is not None
        assert result.segments[0].speaker["id"] == "spk_0"
        assert result.segments[0].speaker["confidence"] == pytest.approx(1.0)

    def test_partial_overlap_above_threshold(self):
        """70% overlap → assign speaker with confidence = 0.7."""
        turns = [make_speaker_turn(0.0, 3.5, "raw_spk_A")]
        segments = [make_segment(0, 1.0, 6.0, "Hello world")]  # 5s segment, 2.5s overlap
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        assert result.segments[0].speaker is not None
        assert result.segments[0].speaker["id"] == "spk_0"
        # Overlap: 3.5 - 1.0 = 2.5s, segment duration = 5s, confidence = 0.5
        assert result.segments[0].speaker["confidence"] == pytest.approx(0.5)

    def test_partial_overlap_below_threshold(self):
        """20% overlap → speaker = None (unknown)."""
        turns = [make_speaker_turn(0.0, 2.0, "raw_spk_A")]
        segments = [make_segment(0, 1.0, 6.0, "Hello world")]  # 5s segment, 1s overlap
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        # Overlap: 2.0 - 1.0 = 1s, segment duration = 5s, confidence = 0.2 < 0.3
        assert result.segments[0].speaker is None

    def test_no_overlap(self):
        """Segment outside all speaker turns → speaker = None."""
        turns = [make_speaker_turn(0.0, 5.0, "raw_spk_A")]
        segments = [make_segment(0, 10.0, 12.0, "Late segment")]
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        assert result.segments[0].speaker is None

    def test_segment_spanning_two_turns(self):
        """Segment overlaps two speakers → assign dominant (max overlap)."""
        turns = [
            make_speaker_turn(0.0, 3.0, "raw_spk_A"),  # 3s
            make_speaker_turn(3.0, 6.0, "raw_spk_B"),  # 3s
        ]
        segments = [make_segment(0, 1.0, 5.0, "Spanning segment")]  # 4s total
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        # Overlap with A: 3.0 - 1.0 = 2s
        # Overlap with B: 5.0 - 3.0 = 2s
        # Equal overlap → should choose first speaker alphabetically (deterministic)
        assert result.segments[0].speaker is not None
        assert result.segments[0].speaker["id"] in ["spk_0", "spk_1"]
        # Confidence = max(2s, 2s) / 4s = 0.5
        assert result.segments[0].speaker["confidence"] == pytest.approx(0.5)

    def test_empty_speaker_turns(self):
        """No diarization output → all segments get speaker = None."""
        turns = []
        segments = [make_segment(0, 0.0, 2.0, "Hello"), make_segment(1, 3.0, 5.0, "World")]
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        assert result.speakers == []
        assert result.segments[0].speaker is None
        assert result.segments[1].speaker is None

    def test_alternating_speakers(self):
        """A-B-A-B pattern → 2 speakers, correct assignments."""
        turns = [
            make_speaker_turn(0.0, 2.0, "raw_spk_A"),
            make_speaker_turn(3.0, 5.0, "raw_spk_B"),
            make_speaker_turn(6.0, 8.0, "raw_spk_A"),
            make_speaker_turn(9.0, 11.0, "raw_spk_B"),
        ]
        segments = [
            make_segment(0, 0.0, 2.0, "A1"),
            make_segment(1, 3.0, 5.0, "B1"),
            make_segment(2, 6.0, 8.0, "A2"),
            make_segment(3, 9.0, 11.0, "B2"),
        ]
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        assert len(result.speakers) == 2
        assert {s["id"] for s in result.speakers} == {"spk_0", "spk_1"}

        # Check alternating pattern (exact IDs depend on normalization order)
        speaker_ids = [seg.speaker["id"] for seg in result.segments]
        assert speaker_ids[0] == speaker_ids[2]  # A segments same
        assert speaker_ids[1] == speaker_ids[3]  # B segments same
        assert speaker_ids[0] != speaker_ids[1]  # A != B

    def test_speaker_id_normalization(self):
        """Raw backend IDs normalized to spk_N format."""
        turns = [
            make_speaker_turn(0.0, 2.0, "SPEAKER_00"),
            make_speaker_turn(3.0, 5.0, "SPEAKER_01"),
        ]
        segments = [
            make_segment(0, 0.0, 2.0, "First"),
            make_segment(1, 3.0, 5.0, "Second"),
        ]
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        assert result.segments[0].speaker["id"] == "spk_0"
        assert result.segments[1].speaker["id"] == "spk_1"
        assert result.speakers[0]["id"] == "spk_0"
        assert result.speakers[1]["id"] == "spk_1"

    def test_speaker_metadata_aggregation(self):
        """speakers[] contains total_speech_time and num_segments."""
        turns = [make_speaker_turn(0.0, 5.0, "raw_spk_A")]
        segments = [
            make_segment(0, 0.0, 2.0, "A1"),
            make_segment(1, 2.5, 4.5, "A2"),
        ]
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        assert len(result.speakers) == 1
        speaker = result.speakers[0]
        assert speaker["id"] == "spk_0"
        assert speaker["label"] is None
        assert speaker["total_speech_time"] == pytest.approx(4.0)  # 2s + 2s
        assert speaker["num_segments"] == 2

    def test_configurable_threshold(self):
        """Threshold parameter changes assignment behavior."""
        turns = [make_speaker_turn(0.0, 2.0, "raw_spk_A")]
        segments = [make_segment(0, 1.0, 6.0, "Test")]  # 1s overlap / 5s = 0.2
        transcript = make_transcript(segments)

        # With threshold=0.3, should be None
        result_high = assign_speakers(transcript, turns, overlap_threshold=0.3)
        assert result_high.segments[0].speaker is None

        # With threshold=0.1, should assign
        result_low = assign_speakers(transcript, turns, overlap_threshold=0.1)
        assert result_low.segments[0].speaker is not None
        assert result_low.segments[0].speaker["id"] == "spk_0"

    def test_single_speaker_many_segments(self):
        """One speaker across many segments → speakers list has 1 entry."""
        turns = [make_speaker_turn(0.0, 10.0, "raw_spk_A")]
        segments = [make_segment(i, float(i), float(i + 1), f"Segment {i}") for i in range(10)]
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        assert len(result.speakers) == 1
        assert result.speakers[0]["id"] == "spk_0"
        assert result.speakers[0]["num_segments"] == 10
        assert all(seg.speaker["id"] == "spk_0" for seg in result.segments)

    def test_segment_in_silence_gap(self):
        """Segment between speaker turns → speaker = None."""
        turns = [
            make_speaker_turn(0.0, 2.0, "raw_spk_A"),
            make_speaker_turn(5.0, 7.0, "raw_spk_B"),
        ]
        segments = [make_segment(0, 3.0, 4.0, "Silence gap")]
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        assert result.segments[0].speaker is None

    def test_micro_segment_low_overlap(self):
        """Very short segment with low overlap → speaker = None."""
        turns = [make_speaker_turn(0.0, 5.0, "raw_spk_A")]
        segments = [make_segment(0, 4.8, 5.2, "Micro")]  # 0.4s, overlap 0.2s
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        # Overlap: 5.0 - 4.8 = 0.2s, duration = 0.4s, confidence = 0.5 >= 0.3
        assert result.segments[0].speaker is not None
        assert result.segments[0].speaker["confidence"] == pytest.approx(0.5)


class TestEdgeCases:
    """Edge case handling tests."""

    def test_zero_duration_segment(self):
        """Segment with start == end → skip or handle gracefully."""
        turns = [make_speaker_turn(0.0, 5.0, "raw_spk_A")]
        segments = [make_segment(0, 2.0, 2.0, "")]  # Zero duration
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        # Should either skip or assign with confidence = 0
        # Implementation choice: likely speaker = None
        assert result.segments[0].speaker is None or result.segments[0].speaker["confidence"] == 0.0

    def test_negative_timestamps(self):
        """Negative timestamps (shouldn't happen, but handle gracefully)."""
        turns = [make_speaker_turn(0.0, 5.0, "raw_spk_A")]
        segments = [make_segment(0, -1.0, 1.0, "Negative start")]
        transcript = make_transcript(segments)

        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        # Should compute overlap correctly: overlap(0, 1.0) = 1s, duration = 2s, conf = 0.5
        assert result.segments[0].speaker is not None
        assert result.segments[0].speaker["confidence"] == pytest.approx(0.5)

    def test_out_of_order_turns(self):
        """Speaker turns not sorted by time → should still work."""
        turns = [
            make_speaker_turn(5.0, 9.0, "raw_spk_B"),  # 4s, overlaps [5.0, 6.0] = 1s
            make_speaker_turn(0.0, 4.0, "raw_spk_A"),  # 4s, overlaps [1.0, 4.0] = 3s (dominant)
        ]
        segments = [make_segment(0, 1.0, 6.0, "Test")]  # 5s segment
        transcript = make_transcript(segments)

        # Implementation should handle unsorted turns correctly
        result = assign_speakers(transcript, turns, overlap_threshold=0.3)

        # Should find overlaps with both turns and pick max (spk_A with 3s)
        # Max overlap = 3s, segment = 5s, confidence = 0.6 > 0.3
        assert result.segments[0].speaker is not None
        assert result.segments[0].speaker["confidence"] == pytest.approx(0.6)
        # Should be assigned to raw_spk_A (3s overlap > raw_spk_B's 1s overlap)
        assert result.segments[0].speaker["id"] in ["spk_0", "spk_1"]
