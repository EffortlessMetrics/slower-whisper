"""Conversation physics tracking for real-time conversation analysis.

This module provides tools for tracking and analyzing the physical dynamics
of conversations, including talk time distribution, interruptions, response
latencies, and speaker transitions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

__all__ = [
    "ConversationPhysicsSnapshot",
    "ConversationPhysicsTracker",
]


class Segment(NamedTuple):
    """A recorded speech segment."""

    speaker_id: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class ConversationPhysicsSnapshot:
    """Immutable snapshot of conversation physics metrics.

    Attributes:
        speaker_talk_times: Seconds of talk time per speaker.
        total_duration_sec: Total conversation duration from first to last segment.
        interruption_count: Total number of interruptions detected.
        interruption_rate: Interruptions per minute of conversation.
        response_latencies: List of response latencies in seconds.
        mean_response_latency_sec: Mean response latency, or None if no latencies.
        speaker_transitions: Number of speaker changes.
        overlap_duration_sec: Total duration of overlapping speech.
    """

    speaker_talk_times: dict[str, float]
    total_duration_sec: float
    interruption_count: int
    interruption_rate: float
    response_latencies: list[float]
    mean_response_latency_sec: float | None
    speaker_transitions: int
    overlap_duration_sec: float


@dataclass
class ConversationPhysicsTracker:
    """Tracks conversation physics from speech segments.

    This tracker accumulates speech segments and computes various metrics
    about the conversation dynamics, including talk time distribution,
    interruption patterns, and response latencies.

    Example:
        >>> tracker = ConversationPhysicsTracker()
        >>> tracker.record_segment("alice", 0.0, 2.5)
        >>> tracker.record_segment("bob", 2.8, 5.0)
        >>> snapshot = tracker.get_snapshot()
        >>> print(f"Total duration: {snapshot.total_duration_sec}s")
        Total duration: 5.0s

    """

    _segments: list[Segment] = field(default_factory=list)
    _response_latencies: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    _max_latency_window: int = field(default=1000, repr=False)

    def __init__(self, max_latency_window: int = 1000) -> None:
        """Initialize the conversation physics tracker.

        Args:
            max_latency_window: Maximum number of response latencies to retain
                in the rolling window. Defaults to 1000.
        """
        self._segments: list[Segment] = []
        self._max_latency_window = max_latency_window
        self._response_latencies: deque[float] = deque(maxlen=max_latency_window)

    def record_segment(self, speaker_id: str, start: float, end: float) -> None:
        """Record a finalized speech segment.

        Args:
            speaker_id: Identifier for the speaker.
            start: Start time of the segment in seconds.
            end: End time of the segment in seconds.

        Note:
            Segments should be recorded in chronological order by start time
            for accurate interruption and latency detection.
        """
        if end < start:
            # Invalid segment, skip silently per invariant #3
            return

        segment = Segment(speaker_id=speaker_id, start=start, end=end)
        self._segments.append(segment)

        # Compute response latency if there's a previous segment from different speaker
        if len(self._segments) >= 2:
            prev = self._segments[-2]
            if prev.speaker_id != speaker_id:
                # Response latency is gap between previous end and current start
                # Negative values indicate overlap (interruption)
                latency = start - prev.end
                if latency >= 0:
                    self._response_latencies.append(latency)

    def get_snapshot(self) -> ConversationPhysicsSnapshot:
        """Compute and return current conversation physics state.

        Returns:
            A snapshot containing all computed metrics.
        """
        if not self._segments:
            return ConversationPhysicsSnapshot(
                speaker_talk_times={},
                total_duration_sec=0.0,
                interruption_count=0,
                interruption_rate=0.0,
                response_latencies=[],
                mean_response_latency_sec=None,
                speaker_transitions=0,
                overlap_duration_sec=0.0,
            )

        # Compute talk times per speaker
        speaker_talk_times: dict[str, float] = {}
        for seg in self._segments:
            duration = seg.end - seg.start
            speaker_talk_times[seg.speaker_id] = (
                speaker_talk_times.get(seg.speaker_id, 0.0) + duration
            )

        # Compute total duration (first start to last end)
        min_start = min(seg.start for seg in self._segments)
        max_end = max(seg.end for seg in self._segments)
        total_duration_sec = max_end - min_start

        # Count interruptions and speaker transitions
        # Also compute total overlap duration
        interruption_count = 0
        speaker_transitions = 0
        overlap_duration_sec = 0.0

        for i in range(1, len(self._segments)):
            prev = self._segments[i - 1]
            curr = self._segments[i]

            # Speaker transition
            if prev.speaker_id != curr.speaker_id:
                speaker_transitions += 1

                # Interruption: new speaker starts before previous ends
                if curr.start < prev.end:
                    interruption_count += 1
                    # Overlap is from current start to previous end
                    overlap = min(prev.end, curr.end) - curr.start
                    if overlap > 0:
                        overlap_duration_sec += overlap

        # Compute interruption rate (per minute)
        if total_duration_sec > 0:
            interruption_rate = (interruption_count / total_duration_sec) * 60.0
        else:
            interruption_rate = 0.0

        # Compute mean response latency
        response_latencies = list(self._response_latencies)
        if response_latencies:
            mean_response_latency_sec = sum(response_latencies) / len(response_latencies)
        else:
            mean_response_latency_sec = None

        return ConversationPhysicsSnapshot(
            speaker_talk_times=speaker_talk_times,
            total_duration_sec=total_duration_sec,
            interruption_count=interruption_count,
            interruption_rate=interruption_rate,
            response_latencies=response_latencies,
            mean_response_latency_sec=mean_response_latency_sec,
            speaker_transitions=speaker_transitions,
            overlap_duration_sec=overlap_duration_sec,
        )

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._segments.clear()
        self._response_latencies.clear()
