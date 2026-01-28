"""Tests for conversation physics tracking (v2.1.0).

This test suite validates the ConversationPhysicsTracker:
1. Recording segments from single speaker
2. Recording segments from multiple speakers (transitions)
3. Interruption detection (overlap)
4. Response latency calculation
5. get_snapshot() returns correct values
6. reset() clears state
7. Edge cases (empty, single segment)
"""

from __future__ import annotations

import pytest

from transcription.conversation_physics import (
    ConversationPhysicsTracker,
)

# =============================================================================
# 1. Test Recording Segments from Single Speaker
# =============================================================================


class TestSingleSpeaker:
    """Tests for recording segments from a single speaker."""

    def test_single_segment_talk_time(self) -> None:
        """Single segment correctly records talk time."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 2.5)

        snapshot = tracker.get_snapshot()

        assert snapshot.speaker_talk_times == {"alice": 2.5}
        assert snapshot.total_duration_sec == 2.5
        assert snapshot.speaker_transitions == 0
        assert snapshot.interruption_count == 0

    def test_multiple_segments_same_speaker(self) -> None:
        """Multiple segments from same speaker accumulate talk time."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)
        tracker.record_segment("alice", 2.0, 3.5)
        tracker.record_segment("alice", 5.0, 6.0)

        snapshot = tracker.get_snapshot()

        # Talk time: 1.0 + 1.5 + 1.0 = 3.5
        assert snapshot.speaker_talk_times == {"alice": 3.5}
        # Duration: 6.0 - 0.0 = 6.0
        assert snapshot.total_duration_sec == 6.0
        # No speaker transitions (same speaker)
        assert snapshot.speaker_transitions == 0

    def test_no_response_latencies_for_single_speaker(self) -> None:
        """No response latencies computed for consecutive same-speaker segments."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)
        tracker.record_segment("alice", 1.5, 2.5)

        snapshot = tracker.get_snapshot()

        # No latencies because no speaker change
        assert snapshot.response_latencies == []
        assert snapshot.mean_response_latency_sec is None


# =============================================================================
# 2. Test Recording Segments from Multiple Speakers (Transitions)
# =============================================================================


class TestMultipleSpeakers:
    """Tests for recording segments from multiple speakers."""

    def test_two_speakers_talk_times(self) -> None:
        """Two speakers' talk times are tracked separately."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 2.0)
        tracker.record_segment("bob", 3.0, 5.0)

        snapshot = tracker.get_snapshot()

        assert snapshot.speaker_talk_times == {"alice": 2.0, "bob": 2.0}
        assert snapshot.total_duration_sec == 5.0

    def test_speaker_transition_count(self) -> None:
        """Speaker transitions are counted correctly."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)
        tracker.record_segment("bob", 1.5, 2.5)
        tracker.record_segment("alice", 3.0, 4.0)
        tracker.record_segment("bob", 4.5, 5.5)

        snapshot = tracker.get_snapshot()

        # Transitions: alice->bob, bob->alice, alice->bob = 3
        assert snapshot.speaker_transitions == 3

    def test_three_speakers(self) -> None:
        """Three speakers are all tracked correctly."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)
        tracker.record_segment("bob", 1.5, 2.5)
        tracker.record_segment("charlie", 3.0, 4.5)

        snapshot = tracker.get_snapshot()

        assert snapshot.speaker_talk_times == {
            "alice": 1.0,
            "bob": 1.0,
            "charlie": 1.5,
        }
        assert snapshot.speaker_transitions == 2


# =============================================================================
# 3. Test Interruption Detection (Overlap)
# =============================================================================


class TestInterruptionDetection:
    """Tests for interruption/overlap detection."""

    def test_no_interruption_with_gap(self) -> None:
        """No interruption when there's a gap between speakers."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)
        tracker.record_segment("bob", 1.5, 2.5)  # Gap of 0.5s

        snapshot = tracker.get_snapshot()

        assert snapshot.interruption_count == 0
        assert snapshot.overlap_duration_sec == 0.0

    def test_interruption_detected_on_overlap(self) -> None:
        """Interruption detected when new speaker starts before previous ends."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 2.0)
        tracker.record_segment("bob", 1.5, 3.0)  # Starts 0.5s before alice ends

        snapshot = tracker.get_snapshot()

        assert snapshot.interruption_count == 1
        # Overlap: alice ends at 2.0, bob starts at 1.5 -> 0.5s overlap
        assert snapshot.overlap_duration_sec == 0.5

    def test_multiple_interruptions(self) -> None:
        """Multiple interruptions are counted correctly."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 2.0)
        tracker.record_segment("bob", 1.5, 3.0)  # Interrupts alice
        tracker.record_segment("alice", 2.5, 4.0)  # Interrupts bob

        snapshot = tracker.get_snapshot()

        assert snapshot.interruption_count == 2

    def test_interruption_rate_calculation(self) -> None:
        """Interruption rate (per minute) is calculated correctly."""
        tracker = ConversationPhysicsTracker()
        # 30 second conversation with 2 interruptions
        tracker.record_segment("alice", 0.0, 10.0)
        tracker.record_segment("bob", 9.0, 20.0)  # Interrupts
        tracker.record_segment("alice", 19.0, 30.0)  # Interrupts

        snapshot = tracker.get_snapshot()

        # 2 interruptions in 30 seconds = 4 per minute
        assert snapshot.interruption_count == 2
        assert snapshot.interruption_rate == pytest.approx(4.0, rel=0.01)

    def test_same_speaker_no_interruption(self) -> None:
        """Overlapping segments from same speaker don't count as interruptions."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 2.0)
        tracker.record_segment("alice", 1.5, 3.0)  # Same speaker, overlapping

        snapshot = tracker.get_snapshot()

        # No interruption (same speaker)
        assert snapshot.interruption_count == 0


# =============================================================================
# 4. Test Response Latency Calculation
# =============================================================================


class TestResponseLatency:
    """Tests for response latency calculation."""

    def test_response_latency_simple(self) -> None:
        """Response latency is calculated for speaker transitions."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)
        tracker.record_segment("bob", 1.5, 2.5)  # 0.5s latency

        snapshot = tracker.get_snapshot()

        assert snapshot.response_latencies == [0.5]
        assert snapshot.mean_response_latency_sec == pytest.approx(0.5, rel=0.01)

    def test_multiple_response_latencies(self) -> None:
        """Multiple latencies are tracked and averaged."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)
        tracker.record_segment("bob", 1.2, 2.0)  # 0.2s latency
        tracker.record_segment("alice", 2.5, 3.5)  # 0.5s latency
        tracker.record_segment("bob", 4.0, 5.0)  # 0.5s latency

        snapshot = tracker.get_snapshot()

        assert len(snapshot.response_latencies) == 3
        # Use pytest.approx for floating point comparisons
        assert snapshot.response_latencies[0] == pytest.approx(0.2, rel=0.01)
        assert snapshot.response_latencies[1] == pytest.approx(0.5, rel=0.01)
        assert snapshot.response_latencies[2] == pytest.approx(0.5, rel=0.01)
        # Mean: (0.2 + 0.5 + 0.5) / 3 = 0.4
        assert snapshot.mean_response_latency_sec == pytest.approx(0.4, rel=0.01)

    def test_negative_latency_not_recorded(self) -> None:
        """Negative latency (overlap) is not recorded as latency."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 2.0)
        tracker.record_segment("bob", 1.5, 3.0)  # Negative latency (overlap)

        snapshot = tracker.get_snapshot()

        # No latencies recorded (overlap case)
        assert snapshot.response_latencies == []
        assert snapshot.mean_response_latency_sec is None

    def test_zero_latency_recorded(self) -> None:
        """Zero latency (immediate response) is recorded."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)
        tracker.record_segment("bob", 1.0, 2.0)  # Starts exactly when alice ends

        snapshot = tracker.get_snapshot()

        assert snapshot.response_latencies == [0.0]
        assert snapshot.mean_response_latency_sec == 0.0

    def test_latency_window_limit(self) -> None:
        """Latency window respects max_latency_window parameter."""
        tracker = ConversationPhysicsTracker(max_latency_window=3)

        # Record more latencies than the window can hold
        tracker.record_segment("alice", 0.0, 1.0)
        tracker.record_segment("bob", 1.1, 2.0)  # latency 0.1
        tracker.record_segment("alice", 2.2, 3.0)  # latency 0.2
        tracker.record_segment("bob", 3.3, 4.0)  # latency 0.3
        tracker.record_segment("alice", 4.4, 5.0)  # latency 0.4 (pushes out 0.1)
        tracker.record_segment("bob", 5.5, 6.0)  # latency 0.5 (pushes out 0.2)

        snapshot = tracker.get_snapshot()

        # Only last 3 latencies: 0.3, 0.4, 0.5
        assert len(snapshot.response_latencies) == 3


# =============================================================================
# 5. Test get_snapshot() Returns Correct Values
# =============================================================================


class TestGetSnapshot:
    """Tests for get_snapshot() method."""

    def test_snapshot_is_immutable(self) -> None:
        """ConversationPhysicsSnapshot is frozen dataclass."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)

        snapshot = tracker.get_snapshot()

        # Attempting to modify should raise
        with pytest.raises(AttributeError):
            snapshot.total_duration_sec = 999.0  # type: ignore[misc]

    def test_snapshot_contains_all_fields(self) -> None:
        """Snapshot contains all expected fields."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 2.0)
        tracker.record_segment("bob", 2.5, 4.0)

        snapshot = tracker.get_snapshot()

        # Verify all fields exist and have correct types
        assert isinstance(snapshot.speaker_talk_times, dict)
        assert isinstance(snapshot.total_duration_sec, float)
        assert isinstance(snapshot.interruption_count, int)
        assert isinstance(snapshot.interruption_rate, float)
        assert isinstance(snapshot.response_latencies, list)
        assert isinstance(snapshot.speaker_transitions, int)
        assert isinstance(snapshot.overlap_duration_sec, float)

    def test_multiple_snapshots_independent(self) -> None:
        """Multiple snapshots are independent of each other."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)

        snapshot1 = tracker.get_snapshot()

        tracker.record_segment("bob", 1.5, 2.5)

        snapshot2 = tracker.get_snapshot()

        # Snapshots should be different
        assert snapshot1.total_duration_sec == 1.0
        assert snapshot2.total_duration_sec == 2.5


# =============================================================================
# 6. Test reset() Clears State
# =============================================================================


class TestReset:
    """Tests for reset() method."""

    def test_reset_clears_segments(self) -> None:
        """reset() clears all recorded segments."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)
        tracker.record_segment("bob", 1.5, 2.5)

        tracker.reset()

        snapshot = tracker.get_snapshot()
        assert snapshot.speaker_talk_times == {}
        assert snapshot.total_duration_sec == 0.0

    def test_reset_clears_latencies(self) -> None:
        """reset() clears recorded latencies."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)
        tracker.record_segment("bob", 1.5, 2.5)

        tracker.reset()

        snapshot = tracker.get_snapshot()
        assert snapshot.response_latencies == []
        assert snapshot.mean_response_latency_sec is None

    def test_can_record_after_reset(self) -> None:
        """Can record new segments after reset."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 1.0)

        tracker.reset()

        tracker.record_segment("bob", 0.0, 2.0)

        snapshot = tracker.get_snapshot()
        assert snapshot.speaker_talk_times == {"bob": 2.0}


# =============================================================================
# 7. Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_tracker(self) -> None:
        """Empty tracker returns zero/empty snapshot."""
        tracker = ConversationPhysicsTracker()

        snapshot = tracker.get_snapshot()

        assert snapshot.speaker_talk_times == {}
        assert snapshot.total_duration_sec == 0.0
        assert snapshot.interruption_count == 0
        assert snapshot.interruption_rate == 0.0
        assert snapshot.response_latencies == []
        assert snapshot.mean_response_latency_sec is None
        assert snapshot.speaker_transitions == 0
        assert snapshot.overlap_duration_sec == 0.0

    def test_invalid_segment_end_before_start(self) -> None:
        """Segment with end < start is silently ignored."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 5.0, 3.0)  # Invalid: end < start

        snapshot = tracker.get_snapshot()

        # Should be empty as invalid segment was skipped
        assert snapshot.speaker_talk_times == {}

    def test_zero_duration_segment(self) -> None:
        """Zero duration segment (start == end) is accepted."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 1.0, 1.0)

        snapshot = tracker.get_snapshot()

        assert snapshot.speaker_talk_times == {"alice": 0.0}
        assert snapshot.total_duration_sec == 0.0

    def test_very_long_conversation(self) -> None:
        """Handles conversation with many segments."""
        tracker = ConversationPhysicsTracker()

        # Simulate 1000 segments alternating speakers
        for i in range(1000):
            speaker = "alice" if i % 2 == 0 else "bob"
            tracker.record_segment(speaker, float(i), float(i + 0.5))

        snapshot = tracker.get_snapshot()

        # 500 segments each at 0.5s = 250s each
        assert snapshot.speaker_talk_times["alice"] == pytest.approx(250.0, rel=0.01)
        assert snapshot.speaker_talk_times["bob"] == pytest.approx(250.0, rel=0.01)
        assert snapshot.speaker_transitions == 999  # n-1 transitions

    def test_floating_point_precision(self) -> None:
        """Floating point values are handled correctly."""
        tracker = ConversationPhysicsTracker()
        tracker.record_segment("alice", 0.0, 0.333)
        tracker.record_segment("bob", 0.5, 0.833)

        snapshot = tracker.get_snapshot()

        assert snapshot.speaker_talk_times["alice"] == pytest.approx(0.333, rel=0.01)
        assert snapshot.speaker_talk_times["bob"] == pytest.approx(0.333, rel=0.01)
