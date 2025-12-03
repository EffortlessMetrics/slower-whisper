"""Live semantic annotation for streaming transcription (v0.2).

This module provides turn-aware semantic enrichment for streaming conversations.
It buffers chunks into speaker turns, annotates finalized turns with semantic
tags (keywords, risk flags, action items), and emits annotated updates.

Architecture:
- Ingests StreamChunk objects from StreamingSession
- Buffers chunks into turns using speaker and time gap boundaries
- Annotates finalized turns with KeywordSemanticAnnotator
- Emits SEMANTIC_UPDATE events with turn metadata
- Manages context window for conversation coherence

See docs/STREAMING_ARCHITECTURE.md for integration patterns.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .models import Turn, TurnMeta
from .semantic import KeywordSemanticAnnotator
from .streaming import StreamChunk, StreamEvent, StreamEventType, StreamSegment

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LiveSemanticsConfig:
    """Configuration for live semantic annotation.

    Attributes:
        turn_gap_sec: Minimum gap between chunks to finalize a turn (seconds).
                     When chunks from the same speaker have a gap >= this threshold,
                     the current turn is finalized. Default: 2.0 seconds.
        context_window_turns: Maximum number of recent turns to keep in context.
                             Used for conversation coherence and LLM prompting.
                             Default: 10 turns.
        context_window_sec: Maximum time span of context window (seconds).
                           Older turns beyond this time are pruned.
                           Default: 120.0 seconds (2 minutes).
        enable_question_detection: Whether to count questions in turn metadata.
                                  Default: True.
        enable_action_detection: Whether to detect action items in turns.
                                Default: True.
    """

    turn_gap_sec: float = 2.0
    context_window_turns: int = 10
    context_window_sec: float = 120.0
    enable_question_detection: bool = True
    enable_action_detection: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.turn_gap_sec < 0.0:
            raise ValueError(f"turn_gap_sec must be >= 0.0, got {self.turn_gap_sec}")
        if self.context_window_turns <= 0:
            raise ValueError(f"context_window_turns must be > 0, got {self.context_window_turns}")
        if self.context_window_sec <= 0.0:
            raise ValueError(f"context_window_sec must be > 0.0, got {self.context_window_sec}")


@dataclass(slots=True)
class SemanticUpdatePayload:
    """Payload for SEMANTIC_UPDATE events.

    Contains the finalized turn with semantic annotations and computed metadata.

    Attributes:
        turn: The finalized Turn object with all metadata.
        keywords: List of semantic keywords extracted from the turn text.
        risk_tags: List of risk tags (escalation, churn_risk, pricing).
        actions: List of detected action items (each a dict with text, pattern).
        question_count: Number of questions detected in the turn.
        context_size: Current number of turns in the context window.
    """

    turn: Turn
    keywords: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    question_count: int = 0
    context_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize payload to a JSON-serializable dict."""
        return {
            "turn": self.turn.to_dict(),
            "keywords": list(self.keywords),
            "risk_tags": list(self.risk_tags),
            "actions": list(self.actions),
            "question_count": self.question_count,
            "context_size": self.context_size,
        }


class LiveSemanticSession:
    """
    Streaming semantic annotator with turn buffering and context management.

    This session tracks streaming chunks, groups them into speaker turns,
    annotates finalized turns with semantic tags, and maintains a rolling
    context window for conversation coherence.

    Usage:
        >>> config = LiveSemanticsConfig(turn_gap_sec=2.0)
        >>> session = LiveSemanticSession(config)
        >>> for chunk in stream_of_chunks:
        ...     events = session.ingest_chunk(chunk)
        ...     for event in events:
        ...         if event.type == StreamEventType.SEMANTIC_UPDATE:
        ...             payload = event.semantic
        ...             print(f"Turn {payload.turn.id}: {payload.keywords}")
        >>> final_events = session.end_of_stream()

    State Management:
        - _turn_buffer: Current in-flight turn segments (not yet finalized)
        - _current_speaker: Speaker ID of the current turn buffer
        - _context_window: Deque of recently finalized annotated turns
        - _turn_counter: Monotonic turn ID counter
        - _annotator: KeywordSemanticAnnotator instance for semantic tagging

    Finalization Rules:
        A turn is finalized when:
        1. Speaker changes (different speaker_id in new chunk)
        2. Time gap >= turn_gap_sec between chunks from same speaker
        3. end_of_stream() is called (flushes remaining buffer)
    """

    def __init__(
        self,
        config: LiveSemanticsConfig | None = None,
        annotator: KeywordSemanticAnnotator | None = None,
    ) -> None:
        """Initialize live semantic session.

        Args:
            config: Configuration for turn boundaries and context window.
                   If None, uses default LiveSemanticsConfig().
            annotator: Semantic annotator instance. If None, creates a
                      default KeywordSemanticAnnotator().
        """
        self.config = config or LiveSemanticsConfig()
        self._annotator = annotator or KeywordSemanticAnnotator()

        # Turn buffer state
        self._turn_buffer: list[StreamSegment] = []
        self._current_speaker: str | None = None
        self._last_chunk_end: float = 0.0

        # Context window (bounded deque)
        self._context_window: deque[Turn] = deque(maxlen=self.config.context_window_turns)

        # Turn ID tracking
        self._turn_counter: int = 0

    def ingest_chunk(self, chunk: StreamChunk) -> list[StreamEvent]:
        """Consume a chunk and return any semantic update events.

        Args:
            chunk: Post-ASR chunk with start, end, text, speaker_id.

        Returns:
            List of StreamEvent objects. May include:
            - SEMANTIC_UPDATE event if a turn was finalized
            - Empty list if chunk was buffered without finalization

        Raises:
            ValueError: If chunk violates monotonic time ordering.
        """
        self._validate_monotonic(chunk)

        events: list[StreamEvent] = []
        chunk_speaker = chunk.get("speaker_id")

        # Determine if we should finalize the current turn
        if self._should_finalize_turn(chunk, chunk_speaker):
            if self._turn_buffer:
                # Finalize current turn and emit semantic update
                semantic_event = self._finalize_turn()
                if semantic_event:
                    events.append(semantic_event)

            # Start new turn with this chunk
            self._start_new_turn(chunk, chunk_speaker)
        else:
            # Extend current turn buffer
            self._extend_turn_buffer(chunk, chunk_speaker)

        return events

    def end_of_stream(self) -> list[StreamEvent]:
        """Finalize the stream, emitting semantic update for remaining turn.

        Returns:
            List containing SEMANTIC_UPDATE event for the buffered turn,
            or empty list if no buffered chunks.
        """
        if not self._turn_buffer:
            return []

        semantic_event = self._finalize_turn()
        return [semantic_event] if semantic_event else []

    def get_context_window(self) -> list[Turn]:
        """Return the current context window as a list of turns.

        Returns:
            List of Turn objects in chronological order (oldest to newest).
        """
        return list(self._context_window)

    def render_context_for_llm(self) -> str:
        """Render the context window as LLM-ready text.

        Returns:
            Multi-line string with speaker, timestamp, and text for each turn.
            Format: "[timestamp] speaker_id: text"
        """
        if not self._context_window:
            return ""

        lines = ["Recent conversation context:"]
        for turn in self._context_window:
            timestamp = f"{turn.start:.1f}s"
            lines.append(f"[{timestamp}] {turn.speaker_id}: {turn.text}")

        return "\n".join(lines)

    def _validate_monotonic(self, chunk: StreamChunk) -> None:
        """Ensure chunk arrives in non-decreasing time order.

        Args:
            chunk: The chunk to validate.

        Raises:
            ValueError: If chunk violates monotonic ordering.
        """
        if chunk["end"] < chunk["start"]:
            raise ValueError(f"Chunk end ({chunk['end']}) < start ({chunk['start']})")
        if self._turn_buffer and chunk["start"] < self._last_chunk_end:
            raise ValueError(
                f"Chunk start ({chunk['start']}) < last chunk end ({self._last_chunk_end})"
            )

    def _should_finalize_turn(self, chunk: StreamChunk, chunk_speaker: str | None) -> bool:
        """Determine if the current turn should be finalized.

        A turn is finalized when:
        1. Turn buffer is empty (first chunk ever)
        2. Speaker changes
        3. Gap between chunks >= turn_gap_sec (same speaker, long pause)

        Args:
            chunk: The incoming chunk.
            chunk_speaker: Speaker ID from the chunk.

        Returns:
            True if current turn should be finalized, False otherwise.
        """
        if not self._turn_buffer:
            return False

        # Speaker change
        if chunk_speaker != self._current_speaker:
            return True

        # Long pause (same speaker)
        gap = chunk["start"] - self._last_chunk_end
        if gap >= self.config.turn_gap_sec:
            return True

        return False

    def _start_new_turn(self, chunk: StreamChunk, chunk_speaker: str | None) -> None:
        """Start a new turn with the given chunk.

        Args:
            chunk: The first chunk of the new turn.
            chunk_speaker: Speaker ID for the new turn.
        """
        self._turn_buffer = [self._segment_from_chunk(chunk)]
        self._current_speaker = chunk_speaker
        self._last_chunk_end = chunk["end"]

    def _extend_turn_buffer(self, chunk: StreamChunk, chunk_speaker: str | None) -> None:
        """Add chunk to the current turn buffer.

        Args:
            chunk: The chunk to add to the buffer.
            chunk_speaker: Speaker ID (should match _current_speaker).
        """
        # Initialize buffer if empty (first chunk ever)
        if not self._turn_buffer:
            self._start_new_turn(chunk, chunk_speaker)
            return

        self._turn_buffer.append(self._segment_from_chunk(chunk))
        self._last_chunk_end = chunk["end"]

    def _segment_from_chunk(self, chunk: StreamChunk) -> StreamSegment:
        """Convert a StreamChunk to a StreamSegment.

        Args:
            chunk: The chunk to convert.

        Returns:
            StreamSegment with start, end, text, speaker_id.
        """
        return StreamSegment(
            start=float(chunk["start"]),
            end=float(chunk["end"]),
            text=chunk["text"],
            speaker_id=chunk.get("speaker_id"),
        )

    def _finalize_turn(self) -> StreamEvent | None:
        """Finalize the current turn buffer and emit semantic update event.

        Returns:
            StreamEvent with SEMANTIC_UPDATE type and SemanticUpdatePayload,
            or None if buffer is empty.
        """
        if not self._turn_buffer:
            return None

        # Build turn from buffer
        turn = self._build_turn_from_buffer()

        # Annotate turn with semantic tags
        annotated_turn = self._annotate_turn(turn)

        # Prune context window before adding new turn
        self._prune_context(annotated_turn)

        # Add to context window
        self._context_window.append(annotated_turn)

        # Build semantic payload
        payload = self._build_semantic_payload(annotated_turn)

        # Clear turn buffer
        self._turn_buffer = []
        self._current_speaker = None

        # Create semantic update event
        event = StreamEvent(
            type=StreamEventType.SEMANTIC_UPDATE,
            segment=self._turn_to_segment(annotated_turn),
            semantic=payload,
        )

        return event

    def _build_turn_from_buffer(self) -> Turn:
        """Build a Turn object from the current buffer.

        Returns:
            Turn with aggregated text, time bounds, and basic metadata.
        """
        if not self._turn_buffer:
            raise ValueError("Cannot build turn from empty buffer")

        turn_id = f"turn_{self._turn_counter}"
        self._turn_counter += 1

        # Aggregate text (join non-empty segments with spaces)
        texts = [seg.text.strip() for seg in self._turn_buffer if seg.text.strip()]
        full_text = " ".join(texts)

        # Time bounds
        start = self._turn_buffer[0].start
        end = self._turn_buffer[-1].end

        # Speaker ID (use current speaker or default)
        speaker_id = self._current_speaker or "unknown"

        # Segment IDs (placeholder: use indices)
        segment_ids = list(range(len(self._turn_buffer)))

        return Turn(
            id=turn_id,
            speaker_id=speaker_id,
            segment_ids=segment_ids,
            start=start,
            end=end,
            text=full_text,
            metadata={},
        )

    def _annotate_turn(self, turn: Turn) -> Turn:
        """Annotate a turn with semantic tags and metadata.

        Uses KeywordSemanticAnnotator to extract keywords, risk tags, and
        action items. Computes turn-level metadata like question counts.

        Args:
            turn: The turn to annotate.

        Returns:
            The same Turn object with updated metadata field.
        """
        # Create temporary transcript for annotator
        from .models import Segment, Transcript

        temp_segment = Segment(
            id=0,
            start=turn.start,
            end=turn.end,
            text=turn.text,
            speaker={"id": turn.speaker_id},
        )
        temp_transcript = Transcript(
            file_name="stream",
            language="en",
            segments=[temp_segment],
            annotations={},
        )

        # Annotate with KeywordSemanticAnnotator
        annotated = self._annotator.annotate(temp_transcript)
        semantic = annotated.annotations.get("semantic", {}) if annotated.annotations else {}

        # Extract semantic data
        keywords = semantic.get("keywords", [])
        risk_tags = semantic.get("risk_tags", [])
        actions = semantic.get("actions", [])

        # Compute turn metadata
        turn_meta = self._compute_turn_metadata(turn, keywords, risk_tags, actions)

        # Update turn metadata
        turn.metadata = {
            "keywords": keywords,
            "risk_tags": risk_tags,
            "actions": actions,
            "turn_meta": turn_meta.to_dict(),
        }

        return turn

    def _compute_turn_metadata(
        self,
        turn: Turn,
        keywords: list[str],
        risk_tags: list[str],
        actions: list[dict[str, Any]],
    ) -> TurnMeta:
        """Compute turn-level metadata.

        Args:
            turn: The turn to analyze.
            keywords: Semantic keywords extracted from the turn.
            risk_tags: Risk tags extracted from the turn.
            actions: Action items detected in the turn.

        Returns:
            TurnMeta with question_count and other metadata fields.
        """
        question_count = 0
        if self.config.enable_question_detection:
            question_count = self._count_questions(turn.text)

        # Future: detect interruptions, disfluency, pauses
        return TurnMeta(
            question_count=question_count,
            interruption_started_here=False,  # Future: detect interruptions
            avg_pause_ms=None,  # Future: compute from segment gaps
            disfluency_ratio=None,  # Future: detect filler words
        )

    def _count_questions(self, text: str) -> int:
        """Count questions in text by looking for question marks.

        Args:
            text: The text to analyze.

        Returns:
            Number of question marks found.
        """
        # Simple heuristic: count '?' characters
        # Future: use more sophisticated question detection (NLP, patterns)
        return text.count("?")

    def _prune_context(self, new_turn: Turn) -> None:
        """Prune old turns from context window based on time threshold.

        Removes turns older than context_window_sec from the oldest end
        of the deque. The deque's maxlen handles the turn count limit.

        Args:
            new_turn: The new turn about to be added (used for time reference).
        """
        if not self._context_window:
            return

        cutoff_time = new_turn.end - self.config.context_window_sec

        # Remove turns older than cutoff_time
        while self._context_window and self._context_window[0].end < cutoff_time:
            self._context_window.popleft()

    def _build_semantic_payload(self, turn: Turn) -> SemanticUpdatePayload:
        """Build a SemanticUpdatePayload from an annotated turn.

        Args:
            turn: The annotated turn.

        Returns:
            SemanticUpdatePayload with extracted semantic data.
        """
        metadata = turn.metadata or {}
        keywords = metadata.get("keywords", [])
        risk_tags = metadata.get("risk_tags", [])
        actions = metadata.get("actions", [])
        turn_meta_dict = metadata.get("turn_meta", {})
        question_count = turn_meta_dict.get("question_count", 0)

        return SemanticUpdatePayload(
            turn=turn,
            keywords=keywords,
            risk_tags=risk_tags,
            actions=actions,
            question_count=question_count,
            context_size=len(self._context_window),
        )

    def _turn_to_segment(self, turn: Turn) -> StreamSegment:
        """Convert a Turn to a StreamSegment for event compatibility.

        Args:
            turn: The turn to convert.

        Returns:
            StreamSegment with turn's time bounds, text, and speaker.
        """
        return StreamSegment(
            start=turn.start,
            end=turn.end,
            text=turn.text,
            speaker_id=turn.speaker_id,
        )
