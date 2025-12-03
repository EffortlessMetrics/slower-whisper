"""Comprehensive tests for LiveSemanticSession (streaming + semantic annotation).

This test suite validates the integration of streaming transcription with real-time
semantic annotation, following the test plan:

1. State machine tests: single speaker, speaker changes, pause splits
2. Semantic annotation tests: keywords, risk_tags, actions detected
3. Multi-speaker scenarios: alternating speakers, rapid switches
4. Edge cases: empty text, no speaker IDs, annotation failures
5. Turn metadata: question counting, interruptions
6. Context window: eviction by size and time

Design:
- Uses helper functions for chunk creation and assertion
- Clear test names describing scenario and expected behavior
- Mirrors patterns from test_streaming.py for consistency
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pytest

from transcription.models import Segment, Transcript
from transcription.semantic import KeywordSemanticAnnotator
from transcription.streaming import StreamChunk, StreamConfig, StreamEventType, StreamingSession

# =============================================================================
# Helper Classes
# =============================================================================


@dataclass
class SemanticConfig:
    """Configuration for semantic annotation in streaming."""

    enable_keywords: bool = True
    enable_risk_tags: bool = True
    enable_actions: bool = True
    annotator: KeywordSemanticAnnotator | None = None

    def __post_init__(self) -> None:
        if self.annotator is None:
            object.__setattr__(self, "annotator", KeywordSemanticAnnotator())


@dataclass
class ContextWindow:
    """Rolling context window for finalized segments with semantic annotations."""

    max_segments: int = 10
    max_time_sec: float = 120.0
    segments: deque[dict[str, Any]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if not isinstance(self.segments, deque):
            object.__setattr__(self, "segments", deque(maxlen=self.max_segments))

    def add(self, segment: dict[str, Any]) -> None:
        """Add segment and prune old segments by time."""
        self.segments.append(segment)

        # Prune segments older than max_time_sec
        if len(self.segments) > 1:
            latest_end = self.segments[-1]["end"]
            while self.segments and (latest_end - self.segments[0]["end"]) > self.max_time_sec:
                self.segments.popleft()

    def get_keywords(self) -> list[str]:
        """Extract unique keywords from context window."""
        all_keywords = []
        for seg in self.segments:
            all_keywords.extend(seg.get("keywords", []))
        return sorted(set(all_keywords))

    def get_risk_tags(self) -> list[str]:
        """Extract unique risk tags from context window."""
        all_tags = []
        for seg in self.segments:
            all_tags.extend(seg.get("risk_tags", []))
        return sorted(set(all_tags))


@dataclass
class TurnMetadata:
    """Metadata for a turn (speaker turn)."""

    question_count: int = 0
    interruption_started_here: bool = False
    avg_pause_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_count": self.question_count,
            "interruption_started_here": self.interruption_started_here,
            "avg_pause_ms": self.avg_pause_ms,
        }


class LiveSemanticSession:
    """
    Streaming session with integrated semantic annotation.

    Combines StreamingSession for segment aggregation with KeywordSemanticAnnotator
    for real-time keyword/risk detection. Maintains a context window of finalized
    segments and tracks turn-level metadata.

    Usage:
        session = LiveSemanticSession(
            stream_config=StreamConfig(max_gap_sec=1.0),
            semantic_config=SemanticConfig()
        )

        for chunk in chunks:
            events = session.ingest_chunk(chunk)
            for event in events:
                if event.type == StreamEventType.FINAL_SEGMENT:
                    # Access semantic annotations
                    keywords = event.segment.keywords
                    risk_tags = event.segment.risk_tags
    """

    def __init__(
        self,
        stream_config: StreamConfig | None = None,
        semantic_config: SemanticConfig | None = None,
        context_window_config: dict[str, Any] | None = None,
    ) -> None:
        self.stream_session = StreamingSession(stream_config or StreamConfig())
        self.semantic_config = semantic_config or SemanticConfig()
        self.context_window = ContextWindow(
            **(context_window_config or {"max_segments": 10, "max_time_sec": 120.0})
        )

        # Track turn metadata (keyed by speaker_id)
        self.turn_metadata: dict[str, TurnMetadata] = {}
        self.previous_speaker: str | None = None

    def ingest_chunk(self, chunk: StreamChunk) -> list[SemanticStreamEvent]:
        """Ingest chunk and return semantically annotated events."""
        base_events = self.stream_session.ingest_chunk(chunk)
        semantic_events = []

        for event in base_events:
            if event.type == StreamEventType.FINAL_SEGMENT:
                # Annotate finalized segment
                annotated_segment = self._annotate_segment(event.segment)
                self.context_window.add(annotated_segment)

                # Update turn metadata
                self._update_turn_metadata(annotated_segment)

                semantic_events.append(
                    SemanticStreamEvent(
                        type=StreamEventType.FINAL_SEGMENT, segment=annotated_segment
                    )
                )
            else:
                # Partial segment - no annotation yet
                partial = self._segment_to_dict(event.segment)
                semantic_events.append(
                    SemanticStreamEvent(type=StreamEventType.PARTIAL_SEGMENT, segment=partial)
                )

        return semantic_events

    def end_of_stream(self) -> list[SemanticStreamEvent]:
        """Finalize stream and annotate remaining segments."""
        base_events = self.stream_session.end_of_stream()
        semantic_events = []

        for event in base_events:
            annotated_segment = self._annotate_segment(event.segment)
            self.context_window.add(annotated_segment)
            self._update_turn_metadata(annotated_segment)

            semantic_events.append(
                SemanticStreamEvent(type=StreamEventType.FINAL_SEGMENT, segment=annotated_segment)
            )

        return semantic_events

    def _annotate_segment(self, segment: Any) -> dict[str, Any]:
        """Annotate a segment with semantic tags."""
        # Convert StreamSegment to Segment for annotation
        temp_segment = Segment(
            id=0,
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker={"id": segment.speaker_id} if segment.speaker_id else None,
        )

        # Create minimal transcript for annotation
        temp_transcript = Transcript(
            file_name="stream", language="en", segments=[temp_segment], meta={}
        )

        # Run semantic annotation
        if (
            self.semantic_config.annotator
            and self.semantic_config.enable_keywords
            or self.semantic_config.enable_risk_tags
            or self.semantic_config.enable_actions
        ):
            annotated_transcript = self.semantic_config.annotator.annotate(temp_transcript)
            semantic_data = (
                annotated_transcript.annotations.get("semantic", {})
                if annotated_transcript.annotations
                else {}
            )
        else:
            semantic_data = {}

        # Build annotated segment dict
        return {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "speaker_id": segment.speaker_id,
            "keywords": semantic_data.get("keywords", []),
            "risk_tags": semantic_data.get("risk_tags", []),
            "actions": semantic_data.get("actions", []),
        }

    def _segment_to_dict(self, segment: Any) -> dict[str, Any]:
        """Convert StreamSegment to dict without annotation."""
        return {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "speaker_id": segment.speaker_id,
            "keywords": [],
            "risk_tags": [],
            "actions": [],
        }

    def _update_turn_metadata(self, segment: dict[str, Any]) -> None:
        """Update turn metadata based on segment content."""
        speaker_id = segment.get("speaker_id") or "unknown"

        # Initialize metadata if needed
        if speaker_id not in self.turn_metadata:
            self.turn_metadata[speaker_id] = TurnMetadata()

        metadata = self.turn_metadata[speaker_id]

        # Count questions (simple heuristic: ends with '?')
        text = segment.get("text", "")
        if text.strip().endswith("?"):
            metadata.question_count += 1

        # Detect interruptions (speaker change with small gap)
        if self.previous_speaker and self.previous_speaker != speaker_id:
            # Check if this is a rapid speaker change (gap < 0.2s)
            if len(self.context_window.segments) > 0:
                prev_seg = self.context_window.segments[-1]
                gap = segment["start"] - prev_seg["end"]
                if gap < 0.2:
                    metadata.interruption_started_here = True

        self.previous_speaker = speaker_id


@dataclass
class SemanticStreamEvent:
    """Streaming event with semantic annotations."""

    type: StreamEventType
    segment: dict[str, Any]


# =============================================================================
# Helper Functions
# =============================================================================


def _chunk(start: float, end: float, text: str, speaker: str | None = None) -> StreamChunk:
    """Create a StreamChunk for testing."""
    return {"start": start, "end": end, "text": text, "speaker_id": speaker}


def assert_keywords(segment: dict[str, Any], expected: list[str]) -> None:
    """Assert segment contains expected keywords."""
    assert sorted(segment.get("keywords", [])) == sorted(expected)


def assert_risk_tags(segment: dict[str, Any], expected: list[str]) -> None:
    """Assert segment contains expected risk tags."""
    assert sorted(segment.get("risk_tags", [])) == sorted(expected)


def assert_actions(segment: dict[str, Any], min_count: int = 0) -> None:
    """Assert segment has at least min_count actions."""
    assert len(segment.get("actions", [])) >= min_count


# =============================================================================
# 1. State Machine Tests
# =============================================================================


def test_single_speaker_straight_line() -> None:
    """Single speaker, continuous speech -> aggregates into one segment."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 0.5, "hello there", "spk_0")
    chunk2 = _chunk(0.6, 1.2, "how are you", "spk_0")

    events1 = session.ingest_chunk(chunk1)
    assert len(events1) == 1
    assert events1[0].type == StreamEventType.PARTIAL_SEGMENT
    assert events1[0].segment["text"] == "hello there"

    events2 = session.ingest_chunk(chunk2)
    assert len(events2) == 1
    assert events2[0].type == StreamEventType.PARTIAL_SEGMENT
    assert events2[0].segment["text"] == "hello there how are you"

    finals = session.end_of_stream()
    assert len(finals) == 1
    assert finals[0].type == StreamEventType.FINAL_SEGMENT
    assert finals[0].segment["text"] == "hello there how are you"
    assert finals[0].segment["speaker_id"] == "spk_0"


def test_speaker_change_finalizes_and_annotates() -> None:
    """Speaker change -> finalizes previous segment with annotations."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 0.7, "I need to cancel my account", "spk_0")
    chunk2 = _chunk(0.8, 1.5, "Let me help you with that", "spk_1")

    session.ingest_chunk(chunk1)
    events = session.ingest_chunk(chunk2)

    # Should get: FINAL (spk_0) + PARTIAL (spk_1)
    assert [e.type for e in events] == [
        StreamEventType.FINAL_SEGMENT,
        StreamEventType.PARTIAL_SEGMENT,
    ]

    final_seg = events[0].segment
    assert final_seg["speaker_id"] == "spk_0"
    assert final_seg["text"] == "I need to cancel my account"
    # Should detect "cancel" keyword and churn_risk tag
    assert "cancel" in final_seg["keywords"]
    assert "churn_risk" in final_seg["risk_tags"]

    partial_seg = events[1].segment
    assert partial_seg["speaker_id"] == "spk_1"
    assert partial_seg["text"] == "Let me help you with that"


def test_pause_split_finalizes_segment() -> None:
    """Large gap (pause) -> finalizes segment even with same speaker."""
    session = LiveSemanticSession(stream_config=StreamConfig(max_gap_sec=0.5))

    chunk1 = _chunk(0.0, 0.4, "hello", "spk_0")
    chunk2 = _chunk(2.0, 2.5, "there", "spk_0")  # gap of 1.6s > 0.5s

    session.ingest_chunk(chunk1)
    events = session.ingest_chunk(chunk2)

    assert [e.type for e in events] == [
        StreamEventType.FINAL_SEGMENT,
        StreamEventType.PARTIAL_SEGMENT,
    ]
    assert events[0].segment["text"] == "hello"
    assert events[1].segment["text"] == "there"
    assert events[1].segment["start"] == pytest.approx(2.0)


# =============================================================================
# 2. Semantic Annotation Tests
# =============================================================================


def test_escalation_keyword_detection() -> None:
    """Detect escalation keywords and risk tags."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 1.0, "This is unacceptable, I want to speak to your manager", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    final_seg = events[0].segment
    assert "unacceptable" in final_seg["keywords"]
    assert "manager" in final_seg["keywords"]
    assert "escalation" in final_seg["risk_tags"]


def test_churn_keyword_detection() -> None:
    """Detect churn keywords and risk tags."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 1.0, "I'm going to cancel and switch to your competitor", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    final_seg = events[0].segment
    assert "cancel" in final_seg["keywords"]
    assert "switch" in final_seg["keywords"]
    assert "competitor" in final_seg["keywords"]
    assert "churn_risk" in final_seg["risk_tags"]


def test_pricing_keyword_detection() -> None:
    """Detect pricing keywords and risk tags."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 1.0, "The price is too expensive for my budget", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    final_seg = events[0].segment
    assert "price" in final_seg["keywords"]
    assert "expensive" in final_seg["keywords"]
    assert "budget" in final_seg["keywords"]
    assert "pricing" in final_seg["risk_tags"]


def test_action_item_detection() -> None:
    """Detect action items from commitment phrases."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 1.0, "I'll send you the details by email tomorrow", "spk_1")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    final_seg = events[0].segment
    assert len(final_seg["actions"]) > 0
    action = final_seg["actions"][0]
    assert action["text"] == "I'll send you the details by email tomorrow"
    assert action["speaker_id"] == "spk_1"


def test_multiple_risk_tags_in_one_segment() -> None:
    """Multiple risk types in single segment -> all detected."""
    session = LiveSemanticSession()

    chunk = _chunk(
        0.0, 2.0, "The price is too high, I want to cancel and escalate this complaint", "spk_0"
    )
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    final_seg = events[0].segment
    # Should detect pricing, churn, and escalation
    risk_tags = final_seg["risk_tags"]
    assert "pricing" in risk_tags
    assert "churn_risk" in risk_tags
    assert "escalation" in risk_tags


# =============================================================================
# 3. Multi-Speaker Scenarios
# =============================================================================


def test_alternating_speakers() -> None:
    """Alternating speakers -> each speaker change finalizes segment."""
    session = LiveSemanticSession()

    chunks = [
        _chunk(0.0, 1.0, "I need help", "spk_0"),
        _chunk(1.1, 2.0, "How can I assist you", "spk_1"),
        _chunk(2.1, 3.0, "I want to cancel", "spk_0"),
        _chunk(3.1, 4.0, "I understand", "spk_1"),
    ]

    all_finals = []
    for chunk in chunks:
        events = session.ingest_chunk(chunk)
        all_finals.extend([e for e in events if e.type == StreamEventType.FINAL_SEGMENT])

    # Should have 3 finals (last one is still partial)
    assert len(all_finals) == 3

    # Check speakers alternate
    assert all_finals[0].segment["speaker_id"] == "spk_0"
    assert all_finals[1].segment["speaker_id"] == "spk_1"
    assert all_finals[2].segment["speaker_id"] == "spk_0"

    # Check semantic annotation on spk_0's second turn
    assert "cancel" in all_finals[2].segment["keywords"]


def test_rapid_speaker_switches() -> None:
    """Rapid speaker switches (< 0.2s gap) -> detected as interruptions."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 1.0, "Let me explain", "spk_0")
    chunk2 = _chunk(1.05, 2.0, "No wait", "spk_1")  # gap = 0.05s < 0.2s

    session.ingest_chunk(chunk1)
    session.ingest_chunk(chunk2)

    # Check turn metadata for interruption
    assert session.turn_metadata["spk_1"].interruption_started_here is True


def test_same_speaker_multiple_segments() -> None:
    """Same speaker with pauses -> multiple segments for same speaker."""
    session = LiveSemanticSession(stream_config=StreamConfig(max_gap_sec=0.5))

    chunks = [
        _chunk(0.0, 0.5, "First thought", "spk_0"),
        _chunk(2.0, 2.5, "Second thought", "spk_0"),  # gap = 1.5s
        _chunk(4.0, 4.5, "Third thought", "spk_0"),  # gap = 1.5s
    ]

    all_events = []
    for chunk in chunks:
        all_events.extend(session.ingest_chunk(chunk))

    finals = [e for e in all_events if e.type == StreamEventType.FINAL_SEGMENT]
    assert len(finals) == 2  # Two gaps created 2 finals (third is still partial)
    assert all(f.segment["speaker_id"] == "spk_0" for f in finals)


# =============================================================================
# 4. Edge Cases
# =============================================================================


def test_empty_text_chunk() -> None:
    """Empty text chunk -> handled gracefully, no keywords detected."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 0.5, "", "spk_0")
    events = session.ingest_chunk(chunk)

    assert len(events) == 1
    assert events[0].segment["text"] == ""
    assert events[0].segment["keywords"] == []
    assert events[0].segment["risk_tags"] == []


def test_whitespace_only_text() -> None:
    """Whitespace-only text -> no keywords detected."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 0.5, "   \n\t  ", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    final_seg = events[0].segment
    assert final_seg["keywords"] == []
    assert final_seg["risk_tags"] == []


def test_no_speaker_id() -> None:
    """Chunks without speaker_id -> handled gracefully."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 0.5, "hello", None)
    chunk2 = _chunk(0.6, 1.0, "world", None)

    session.ingest_chunk(chunk1)
    events = session.ingest_chunk(chunk2)

    # Should aggregate since both have None speaker_id
    assert len(events) == 1
    assert events[0].type == StreamEventType.PARTIAL_SEGMENT
    assert events[0].segment["text"] == "hello world"
    assert events[0].segment["speaker_id"] is None


def test_mixed_speaker_and_none() -> None:
    """Mix of speaker_id and None -> treated as different speakers."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 0.5, "hello", "spk_0")
    chunk2 = _chunk(0.6, 1.0, "world", None)

    session.ingest_chunk(chunk1)
    events = session.ingest_chunk(chunk2)

    # Speaker change from "spk_0" to None
    assert [e.type for e in events] == [
        StreamEventType.FINAL_SEGMENT,
        StreamEventType.PARTIAL_SEGMENT,
    ]
    assert events[0].segment["speaker_id"] == "spk_0"
    assert events[1].segment["speaker_id"] is None


def test_annotation_with_no_config() -> None:
    """No semantic config -> annotation disabled, no keywords/tags."""
    session = LiveSemanticSession(
        semantic_config=SemanticConfig(
            enable_keywords=False, enable_risk_tags=False, enable_actions=False
        )
    )

    chunk = _chunk(0.0, 1.0, "I want to cancel my expensive account", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    final_seg = events[0].segment
    # Should still have empty lists (not None)
    assert final_seg["keywords"] == []
    assert final_seg["risk_tags"] == []
    assert final_seg["actions"] == []


def test_end_of_stream_with_no_chunks() -> None:
    """end_of_stream() with no chunks -> returns empty list."""
    session = LiveSemanticSession()
    assert session.end_of_stream() == []


# =============================================================================
# 5. Turn Metadata Tests
# =============================================================================


def test_question_counting() -> None:
    """Questions (ending with '?') -> counted in turn metadata."""
    session = LiveSemanticSession()

    chunks = [
        _chunk(0.0, 1.0, "What is your name?", "spk_0"),
        _chunk(1.5, 2.5, "How can I help?", "spk_0"),
        _chunk(3.0, 4.0, "Is this correct?", "spk_0"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    session.end_of_stream()

    # All three questions from spk_0
    assert session.turn_metadata["spk_0"].question_count == 3


def test_question_count_per_speaker() -> None:
    """Questions counted separately per speaker."""
    session = LiveSemanticSession()

    chunks = [
        _chunk(0.0, 1.0, "What is your name?", "spk_0"),
        _chunk(1.5, 2.5, "Can you help me?", "spk_1"),
        _chunk(3.0, 4.0, "Is this right?", "spk_0"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    session.end_of_stream()

    assert session.turn_metadata["spk_0"].question_count == 2
    assert session.turn_metadata["spk_1"].question_count == 1


def test_non_question_text() -> None:
    """Non-question text -> question_count stays 0."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 1.0, "This is a statement.", "spk_0")
    session.ingest_chunk(chunk)
    session.end_of_stream()

    assert session.turn_metadata["spk_0"].question_count == 0


def test_interruption_detection() -> None:
    """Rapid speaker change (< 0.2s) -> interruption flagged."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 1.0, "Let me finish", "spk_0")
    chunk2 = _chunk(1.05, 2.0, "But I disagree", "spk_1")  # 0.05s gap

    session.ingest_chunk(chunk1)
    session.ingest_chunk(chunk2)

    assert session.turn_metadata["spk_1"].interruption_started_here is True
    assert session.turn_metadata["spk_0"].interruption_started_here is False


def test_no_interruption_with_normal_gap() -> None:
    """Normal gap (>= 0.2s) -> no interruption flagged."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 1.0, "First speaker", "spk_0")
    chunk2 = _chunk(1.3, 2.0, "Second speaker", "spk_1")  # 0.3s gap

    session.ingest_chunk(chunk1)
    session.ingest_chunk(chunk2)

    assert session.turn_metadata["spk_1"].interruption_started_here is False


# =============================================================================
# 6. Context Window Tests
# =============================================================================


def test_context_window_eviction_by_size() -> None:
    """Context window evicts old segments when max_segments reached."""
    session = LiveSemanticSession(context_window_config={"max_segments": 3, "max_time_sec": 1000})

    chunks = [
        _chunk(0.0, 1.0, "one", "spk_0"),
        _chunk(1.5, 2.0, "two", "spk_1"),
        _chunk(2.5, 3.0, "three", "spk_0"),
        _chunk(3.5, 4.0, "four", "spk_1"),
        _chunk(4.5, 5.0, "five", "spk_0"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    session.end_of_stream()

    # Window should only contain last 3 segments
    assert len(session.context_window.segments) == 3
    texts = [seg["text"] for seg in session.context_window.segments]
    assert texts == ["three", "four", "five"]


def test_context_window_eviction_by_time() -> None:
    """Context window evicts segments older than max_time_sec."""
    session = LiveSemanticSession(context_window_config={"max_segments": 100, "max_time_sec": 5.0})

    chunks = [
        _chunk(0.0, 1.0, "old", "spk_0"),
        _chunk(2.0, 3.0, "also old", "spk_1"),
        _chunk(10.0, 11.0, "recent", "spk_0"),  # 7s gap from previous
        _chunk(11.5, 12.0, "current", "spk_1"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    session.end_of_stream()

    # Only "recent" and "current" should remain (within 5s of latest)
    texts = [seg["text"] for seg in session.context_window.segments]
    assert "old" not in texts
    assert "also old" not in texts
    assert "recent" in texts
    assert "current" in texts


def test_context_window_keyword_aggregation() -> None:
    """Context window aggregates keywords from all segments."""
    session = LiveSemanticSession(context_window_config={"max_segments": 10, "max_time_sec": 120})

    chunks = [
        _chunk(0.0, 1.0, "I want to cancel", "spk_0"),
        _chunk(2.0, 3.0, "The price is too high", "spk_1"),
        _chunk(4.0, 5.0, "I need to escalate", "spk_0"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    session.end_of_stream()

    # Should aggregate keywords from all segments
    keywords = session.context_window.get_keywords()
    assert "cancel" in keywords
    assert "price" in keywords
    assert "escalate" in keywords


def test_context_window_risk_tag_aggregation() -> None:
    """Context window aggregates risk tags from all segments."""
    session = LiveSemanticSession(context_window_config={"max_segments": 10, "max_time_sec": 120})

    chunks = [
        _chunk(0.0, 1.0, "I want to cancel", "spk_0"),
        _chunk(2.0, 3.0, "The price is expensive", "spk_1"),
        _chunk(4.0, 5.0, "This is unacceptable", "spk_0"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    session.end_of_stream()

    # Should aggregate risk tags from all segments
    risk_tags = session.context_window.get_risk_tags()
    assert "churn_risk" in risk_tags
    assert "pricing" in risk_tags
    assert "escalation" in risk_tags


def test_context_window_deduplicates_keywords() -> None:
    """Context window deduplicates repeated keywords."""
    session = LiveSemanticSession(context_window_config={"max_segments": 10, "max_time_sec": 120})

    chunks = [
        _chunk(0.0, 1.0, "cancel cancel cancel", "spk_0"),
        _chunk(2.0, 3.0, "I need to cancel again", "spk_1"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    session.end_of_stream()

    # "cancel" should appear only once despite multiple occurrences
    keywords = session.context_window.get_keywords()
    assert keywords.count("cancel") == 1


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_conversation_flow() -> None:
    """Full conversation with multiple speakers and semantic events."""
    session = LiveSemanticSession(
        stream_config=StreamConfig(max_gap_sec=1.0),
        semantic_config=SemanticConfig(),
        context_window_config={"max_segments": 20, "max_time_sec": 300},
    )

    # Simulate a customer service call
    chunks = [
        _chunk(0.0, 2.0, "Hello, I need help with my account", "customer"),
        _chunk(2.5, 4.0, "Of course, how can I assist you?", "agent"),
        _chunk(4.5, 7.0, "The price is too expensive and I want to cancel", "customer"),
        _chunk(7.5, 10.0, "I understand. Let me see what options we have", "agent"),
        _chunk(10.5, 12.0, "This is unacceptable, I want to speak to a manager", "customer"),
        _chunk(12.5, 14.0, "I'll escalate this right away", "agent"),
    ]

    all_finals = []
    for chunk in chunks:
        events = session.ingest_chunk(chunk)
        all_finals.extend([e for e in events if e.type == StreamEventType.FINAL_SEGMENT])

    # Finalize stream
    all_finals.extend(session.end_of_stream())

    # Should have 6 final segments (6 speaker changes)
    assert len(all_finals) >= 5

    # Check customer's complaint segment
    complaint_seg = next(
        seg for seg in all_finals if "expensive" in seg.segment.get("text", "").lower()
    )
    assert "price" in complaint_seg.segment["keywords"]
    assert "expensive" in complaint_seg.segment["keywords"]
    assert "cancel" in complaint_seg.segment["keywords"]
    assert "pricing" in complaint_seg.segment["risk_tags"]
    assert "churn_risk" in complaint_seg.segment["risk_tags"]

    # Check escalation segment
    escalation_seg = next(
        seg for seg in all_finals if "unacceptable" in seg.segment.get("text", "").lower()
    )
    assert "escalation" in escalation_seg.segment["risk_tags"]
    assert "unacceptable" in escalation_seg.segment["keywords"]
    assert "manager" in escalation_seg.segment["keywords"]

    # Check agent's action commitment
    action_seg = next(seg for seg in all_finals if "I'll" in seg.segment.get("text", ""))
    assert len(action_seg.segment["actions"]) > 0

    # Verify context window contains aggregated insights
    keywords = session.context_window.get_keywords()
    risk_tags = session.context_window.get_risk_tags()

    assert "cancel" in keywords
    assert "expensive" in keywords
    assert "escalate" in keywords
    assert "churn_risk" in risk_tags
    assert "escalation" in risk_tags
    assert "pricing" in risk_tags

    # Check turn metadata
    assert session.turn_metadata["customer"].question_count == 0  # No questions from customer
    assert session.turn_metadata["agent"].question_count == 1  # "how can I assist you?"


def test_streaming_partial_then_final_annotations() -> None:
    """Partial segments have no annotations, final segments are annotated."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 1.0, "I want to cancel", "spk_0")
    events1 = session.ingest_chunk(chunk1)

    # Partial should have empty annotations
    partial = events1[0]
    assert partial.type == StreamEventType.PARTIAL_SEGMENT
    assert partial.segment["keywords"] == []
    assert partial.segment["risk_tags"] == []

    # Trigger finalization with speaker change
    chunk2 = _chunk(1.5, 2.0, "I understand", "spk_1")
    events2 = session.ingest_chunk(chunk2)

    # Final should have annotations
    final = events2[0]
    assert final.type == StreamEventType.FINAL_SEGMENT
    assert "cancel" in final.segment["keywords"]
    assert "churn_risk" in final.segment["risk_tags"]
