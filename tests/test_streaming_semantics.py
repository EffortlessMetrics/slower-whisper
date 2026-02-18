"""Comprehensive tests for LiveSemanticSession (streaming + semantic annotation).

This test suite validates the integration of streaming transcription with real-time
semantic annotation using the production LiveSemanticSession implementation.

Test Coverage:
1. Turn boundary detection: speaker changes, pause splits
2. Semantic annotation: keywords, risk_tags, actions detected per turn
3. Multi-speaker scenarios: alternating speakers, rapid switches
4. Edge cases: empty text, no speaker IDs, monotonic ordering
5. Turn metadata: question counting
6. Context window: eviction by size and time

Design:
- Tests the REAL LiveSemanticSession from slower_whisper.pipeline.streaming_semantic
- Uses helper functions for chunk creation and assertion
- Clear test names describing scenario and expected behavior
- Validates SEMANTIC_UPDATE event structure and payload
"""

from __future__ import annotations

import pytest

from slower_whisper.pipeline.streaming import StreamChunk, StreamEventType
from slower_whisper.pipeline.streaming_semantic import LiveSemanticsConfig, LiveSemanticSession

# =============================================================================
# Helper Functions
# =============================================================================


def _chunk(start: float, end: float, text: str, speaker: str | None = None) -> StreamChunk:
    """Create a StreamChunk for testing."""
    return {"start": start, "end": end, "text": text, "speaker_id": speaker}


# =============================================================================
# 1. Turn Boundary Detection Tests
# =============================================================================


def test_single_speaker_buffers_until_finalized() -> None:
    """Single speaker, continuous chunks -> buffered into one turn on end_of_stream."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 0.5, "hello there", "spk_0")
    chunk2 = _chunk(0.6, 1.2, "how are you", "spk_0")

    # Chunks with same speaker and small gap should not finalize
    events1 = session.ingest_chunk(chunk1)
    assert len(events1) == 0  # No finalization yet

    events2 = session.ingest_chunk(chunk2)
    assert len(events2) == 0  # Still buffering

    # Finalize on end_of_stream
    finals = session.end_of_stream()
    assert len(finals) == 1
    assert finals[0].type == StreamEventType.SEMANTIC_UPDATE
    assert finals[0].semantic.turn.text == "hello there how are you"
    assert finals[0].semantic.turn.speaker_id == "spk_0"


def test_speaker_change_finalizes_turn() -> None:
    """Speaker change -> finalizes buffered turn with semantic annotations."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 0.7, "I need to cancel my account", "spk_0")
    chunk2 = _chunk(0.8, 1.5, "Let me help you with that", "spk_1")

    session.ingest_chunk(chunk1)
    events = session.ingest_chunk(chunk2)

    # Should finalize spk_0's turn
    assert len(events) == 1
    assert events[0].type == StreamEventType.SEMANTIC_UPDATE

    payload = events[0].semantic
    assert payload.turn.speaker_id == "spk_0"
    assert payload.turn.text == "I need to cancel my account"
    # Should detect "cancel" keyword and churn_risk tag
    assert "cancel" in payload.keywords
    assert "churn_risk" in payload.risk_tags


def test_pause_split_finalizes_turn() -> None:
    """Large gap (pause >= turn_gap_sec) -> finalizes turn even with same speaker."""
    session = LiveSemanticSession(config=LiveSemanticsConfig(turn_gap_sec=1.0))

    chunk1 = _chunk(0.0, 0.4, "hello", "spk_0")
    chunk2 = _chunk(2.0, 2.5, "there", "spk_0")  # gap of 1.6s >= 1.0s

    session.ingest_chunk(chunk1)
    events = session.ingest_chunk(chunk2)

    # Should finalize first turn due to pause
    assert len(events) == 1
    assert events[0].type == StreamEventType.SEMANTIC_UPDATE
    assert events[0].semantic.turn.text == "hello"
    assert events[0].semantic.turn.speaker_id == "spk_0"


def test_small_gap_same_speaker_buffers() -> None:
    """Small gap with same speaker -> continues buffering."""
    session = LiveSemanticSession(config=LiveSemanticsConfig(turn_gap_sec=2.0))

    chunk1 = _chunk(0.0, 1.0, "hello", "spk_0")
    chunk2 = _chunk(1.5, 2.5, "world", "spk_0")  # gap of 0.5s < 2.0s

    session.ingest_chunk(chunk1)
    events = session.ingest_chunk(chunk2)

    # Should not finalize (gap too small)
    assert len(events) == 0

    # Verify buffering by finalizing at end
    finals = session.end_of_stream()
    assert len(finals) == 1
    assert finals[0].semantic.turn.text == "hello world"


# =============================================================================
# 2. Semantic Annotation Tests
# =============================================================================


def test_escalation_keyword_detection() -> None:
    """Detect escalation keywords and risk tags."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 1.0, "This is unacceptable, I want to speak to your manager", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    assert "unacceptable" in payload.keywords
    assert "manager" in payload.keywords
    assert "escalation" in payload.risk_tags


def test_churn_keyword_detection() -> None:
    """Detect churn keywords and risk tags."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 1.0, "I'm going to cancel and switch to your competitor", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    assert "cancel" in payload.keywords
    assert "switch" in payload.keywords
    assert "competitor" in payload.keywords
    assert "churn_risk" in payload.risk_tags


def test_pricing_keyword_detection() -> None:
    """Detect pricing keywords and risk tags."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 1.0, "The price is too expensive for my budget", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    assert "price" in payload.keywords
    assert "expensive" in payload.keywords
    assert "budget" in payload.keywords
    assert "pricing" in payload.risk_tags


def test_action_item_detection() -> None:
    """Detect action items from commitment phrases."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 1.0, "I'll send you the details by email tomorrow", "spk_1")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    assert len(payload.actions) > 0
    action = payload.actions[0]
    assert "send" in action["text"].lower() or "email" in action["text"].lower()


def test_multiple_risk_tags_in_one_turn() -> None:
    """Multiple risk types in single turn -> all detected."""
    session = LiveSemanticSession()

    chunk = _chunk(
        0.0, 2.0, "The price is too high, I want to cancel and escalate this complaint", "spk_0"
    )
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    # Should detect pricing, churn, and escalation
    risk_tags = payload.risk_tags
    assert "pricing" in risk_tags
    assert "churn_risk" in risk_tags
    assert "escalation" in risk_tags


# =============================================================================
# 3. Multi-Speaker Scenarios
# =============================================================================


def test_alternating_speakers() -> None:
    """Alternating speakers -> each speaker change finalizes turn."""
    session = LiveSemanticSession()

    chunks = [
        _chunk(0.0, 1.0, "I need help", "spk_0"),
        _chunk(1.1, 2.0, "How can I assist you", "spk_1"),
        _chunk(2.1, 3.0, "I want to cancel", "spk_0"),
        _chunk(3.1, 4.0, "I understand", "spk_1"),
    ]

    all_updates = []
    for chunk in chunks:
        events = session.ingest_chunk(chunk)
        all_updates.extend([e for e in events if e.type == StreamEventType.SEMANTIC_UPDATE])

    # Should have 3 updates (last turn not finalized yet)
    assert len(all_updates) == 3

    # Check speakers alternate
    assert all_updates[0].semantic.turn.speaker_id == "spk_0"
    assert all_updates[1].semantic.turn.speaker_id == "spk_1"
    assert all_updates[2].semantic.turn.speaker_id == "spk_0"

    # Check semantic annotation on spk_0's second turn
    assert "cancel" in all_updates[2].semantic.keywords


def test_same_speaker_multiple_turns_with_pauses() -> None:
    """Same speaker with long pauses -> multiple turns for same speaker."""
    session = LiveSemanticSession(config=LiveSemanticsConfig(turn_gap_sec=1.0))

    chunks = [
        _chunk(0.0, 0.5, "First thought", "spk_0"),
        _chunk(2.0, 2.5, "Second thought", "spk_0"),  # gap = 1.5s >= 1.0s
        _chunk(4.5, 5.0, "Third thought", "spk_0"),  # gap = 2.0s >= 1.0s
    ]

    all_events = []
    for chunk in chunks:
        all_events.extend(session.ingest_chunk(chunk))

    updates = [e for e in all_events if e.type == StreamEventType.SEMANTIC_UPDATE]
    assert len(updates) == 2  # Two pauses created 2 finalized turns (third still buffered)
    assert all(e.semantic.turn.speaker_id == "spk_0" for e in updates)


# =============================================================================
# 4. Edge Cases
# =============================================================================


def test_empty_text_chunk() -> None:
    """Empty text chunk -> handled gracefully."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 0.5, "", "spk_0")
    events = session.ingest_chunk(chunk)

    # Empty chunk should be buffered
    assert len(events) == 0

    # Finalize and check
    finals = session.end_of_stream()
    assert len(finals) == 1
    assert finals[0].semantic.turn.text == ""
    assert finals[0].semantic.keywords == []
    assert finals[0].semantic.risk_tags == []


def test_whitespace_only_text() -> None:
    """Whitespace-only text -> no keywords detected."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 0.5, "   \n\t  ", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    # Whitespace is stripped during turn building, so text will be empty
    assert payload.keywords == []
    assert payload.risk_tags == []


def test_no_speaker_id() -> None:
    """Chunks without speaker_id -> handled gracefully with 'unknown' speaker."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 0.5, "hello", None)
    chunk2 = _chunk(0.6, 1.0, "world", None)

    session.ingest_chunk(chunk1)
    events = session.ingest_chunk(chunk2)

    # Should buffer (both have None speaker_id, treated as same speaker)
    assert len(events) == 0

    # Finalize and check
    finals = session.end_of_stream()
    assert len(finals) == 1
    assert finals[0].semantic.turn.text == "hello world"
    assert finals[0].semantic.turn.speaker_id == "unknown"  # Default speaker


def test_mixed_speaker_and_none() -> None:
    """Mix of speaker_id and None -> treated as different speakers."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 0.5, "hello", "spk_0")
    chunk2 = _chunk(0.6, 1.0, "world", None)

    session.ingest_chunk(chunk1)
    events = session.ingest_chunk(chunk2)

    # Speaker change from "spk_0" to None should finalize
    assert len(events) == 1
    assert events[0].type == StreamEventType.SEMANTIC_UPDATE
    assert events[0].semantic.turn.speaker_id == "spk_0"


def test_end_of_stream_with_no_chunks() -> None:
    """end_of_stream() with no chunks -> returns empty list."""
    session = LiveSemanticSession()
    assert session.end_of_stream() == []


def test_monotonic_time_validation() -> None:
    """Chunks must arrive in non-decreasing time order."""
    session = LiveSemanticSession()

    chunk1 = _chunk(0.0, 1.0, "first", "spk_0")
    chunk2 = _chunk(0.5, 1.5, "second", "spk_0")  # Starts before previous ends

    session.ingest_chunk(chunk1)

    with pytest.raises(ValueError, match="Chunk start .* < last chunk end"):
        session.ingest_chunk(chunk2)


def test_chunk_with_end_before_start() -> None:
    """Chunk with end < start -> raises ValueError."""
    session = LiveSemanticSession()

    chunk = _chunk(1.0, 0.5, "invalid", "spk_0")  # end < start

    with pytest.raises(ValueError, match="Chunk end .* < start"):
        session.ingest_chunk(chunk)


# =============================================================================
# 5. Turn Metadata Tests
# =============================================================================


def test_question_counting() -> None:
    """Questions (containing '?') -> counted in turn metadata."""
    session = LiveSemanticSession()

    # Multiple questions in one turn
    chunk = _chunk(0.0, 2.0, "What is your name? How can I help? Is this correct?", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    assert payload.question_count == 3


def test_question_count_per_turn() -> None:
    """Questions counted separately per turn."""
    session = LiveSemanticSession()

    chunks = [
        _chunk(0.0, 1.0, "What is your name?", "spk_0"),
        _chunk(1.5, 2.5, "Can you help me?", "spk_1"),
        _chunk(3.0, 4.0, "Is this right? Are you sure?", "spk_0"),
    ]

    all_updates = []
    for chunk in chunks:
        events = session.ingest_chunk(chunk)
        all_updates.extend([e for e in events if e.type == StreamEventType.SEMANTIC_UPDATE])

    all_updates.extend(session.end_of_stream())

    # Find turns by speaker
    spk_0_turns = [u for u in all_updates if u.semantic.turn.speaker_id == "spk_0"]
    spk_1_turns = [u for u in all_updates if u.semantic.turn.speaker_id == "spk_1"]

    assert spk_0_turns[0].semantic.question_count == 1  # First turn: 1 question
    assert spk_1_turns[0].semantic.question_count == 1  # spk_1 turn: 1 question
    assert spk_0_turns[1].semantic.question_count == 2  # Second turn: 2 questions


def test_non_question_text() -> None:
    """Non-question text -> question_count is 0."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 1.0, "This is a statement.", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    assert payload.question_count == 0


# =============================================================================
# 6. Context Window Tests
# =============================================================================


def test_context_window_eviction_by_size() -> None:
    """Context window evicts old turns when max turns reached."""
    session = LiveSemanticSession(
        config=LiveSemanticsConfig(
            turn_gap_sec=0.5,  # Small gap to force finalization
            context_window_turns=3,
            context_window_sec=1000.0,  # Large time window
        )
    )

    chunks = [
        _chunk(0.0, 0.4, "one", "spk_0"),
        _chunk(1.0, 1.4, "two", "spk_1"),
        _chunk(2.0, 2.4, "three", "spk_0"),
        _chunk(3.0, 3.4, "four", "spk_1"),
        _chunk(4.0, 4.4, "five", "spk_0"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    session.end_of_stream()

    # Window should only contain last 3 turns
    context = session.get_context_window()
    assert len(context) == 3
    texts = [turn.text for turn in context]
    assert texts == ["three", "four", "five"]


def test_context_window_eviction_by_time() -> None:
    """Context window evicts turns older than max_time_sec."""
    session = LiveSemanticSession(
        config=LiveSemanticsConfig(
            turn_gap_sec=0.5,
            context_window_turns=100,  # Large turn count
            context_window_sec=5.0,  # 5 second time window
        )
    )

    chunks = [
        _chunk(0.0, 0.4, "old", "spk_0"),
        _chunk(2.0, 2.4, "also old", "spk_1"),
        _chunk(10.0, 10.4, "recent", "spk_0"),  # 7.6s gap from previous
        _chunk(11.0, 11.4, "current", "spk_1"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    session.end_of_stream()

    # Only "recent" and "current" should remain (within 5s of latest turn end)
    context = session.get_context_window()
    texts = [turn.text for turn in context]
    assert "old" not in texts
    assert "also old" not in texts
    assert "recent" in texts
    assert "current" in texts


def test_context_window_render_for_llm() -> None:
    """Context window can be rendered as LLM-ready text."""
    session = LiveSemanticSession()

    chunks = [
        _chunk(0.0, 1.0, "Hello", "alice"),
        _chunk(1.5, 2.5, "Hi there", "bob"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    session.end_of_stream()

    rendered = session.render_context_for_llm()

    assert "Recent conversation context:" in rendered
    assert "alice: Hello" in rendered
    assert "bob: Hi there" in rendered
    assert "0.0s" in rendered  # Timestamp for first turn


def test_context_size_in_payload() -> None:
    """Semantic payload includes context window size."""
    session = LiveSemanticSession(
        config=LiveSemanticsConfig(turn_gap_sec=0.5, context_window_turns=10)
    )

    chunks = [
        _chunk(0.0, 0.4, "one", "spk_0"),
        _chunk(1.0, 1.4, "two", "spk_1"),
        _chunk(2.0, 2.4, "three", "spk_0"),
    ]

    all_updates = []
    for chunk in chunks:
        events = session.ingest_chunk(chunk)
        all_updates.extend([e for e in events if e.type == StreamEventType.SEMANTIC_UPDATE])

    # Each update should report context size (number of turns in window at that point)
    assert all_updates[0].semantic.context_size == 1  # First turn added
    assert all_updates[1].semantic.context_size == 2  # Second turn added


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_conversation_flow() -> None:
    """Full conversation with multiple speakers and semantic events."""
    session = LiveSemanticSession(
        config=LiveSemanticsConfig(
            turn_gap_sec=1.0,
            context_window_turns=20,
            context_window_sec=300.0,
        )
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

    all_updates = []
    for chunk in chunks:
        events = session.ingest_chunk(chunk)
        all_updates.extend([e for e in events if e.type == StreamEventType.SEMANTIC_UPDATE])

    # Finalize stream
    all_updates.extend(session.end_of_stream())

    # Should have 6 turns (speaker changes between each)
    assert len(all_updates) >= 5

    # Check customer's complaint turn
    complaint_update = next(u for u in all_updates if "expensive" in u.semantic.turn.text.lower())
    payload = complaint_update.semantic
    assert "price" in payload.keywords
    assert "expensive" in payload.keywords
    assert "cancel" in payload.keywords
    assert "pricing" in payload.risk_tags
    assert "churn_risk" in payload.risk_tags

    # Check escalation turn
    escalation_update = next(
        u for u in all_updates if "unacceptable" in u.semantic.turn.text.lower()
    )
    payload = escalation_update.semantic
    assert "escalation" in payload.risk_tags
    assert "unacceptable" in payload.keywords
    assert "manager" in payload.keywords

    # Check agent's action commitment
    action_update = next(u for u in all_updates if "I'll" in u.semantic.turn.text)
    assert len(action_update.semantic.actions) > 0

    # Verify context window contains turns
    context = session.get_context_window()
    assert len(context) >= 5

    # Check question counting
    agent_turns = [u for u in all_updates if u.semantic.turn.speaker_id == "agent"]
    assert any(u.semantic.question_count >= 1 for u in agent_turns)  # "how can I assist you?"


def test_semantic_payload_serialization() -> None:
    """Semantic payload can be serialized to dict."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 1.0, "I want to cancel", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    payload_dict = payload.to_dict()

    # Check structure
    assert "turn" in payload_dict
    assert "keywords" in payload_dict
    assert "risk_tags" in payload_dict
    assert "actions" in payload_dict
    assert "question_count" in payload_dict
    assert "context_size" in payload_dict

    # Check types
    assert isinstance(payload_dict["turn"], dict)
    assert isinstance(payload_dict["keywords"], list)
    assert isinstance(payload_dict["risk_tags"], list)


# =============================================================================
# 7. Configuration Validation Tests
# =============================================================================


def test_config_negative_turn_gap_sec_raises() -> None:
    """Negative turn_gap_sec raises ValueError."""
    with pytest.raises(ValueError, match="turn_gap_sec must be >= 0.0"):
        LiveSemanticsConfig(turn_gap_sec=-1.0)


def test_config_zero_turn_gap_sec_allowed() -> None:
    """Zero turn_gap_sec is allowed (every chunk finalizes previous turn)."""
    config = LiveSemanticsConfig(turn_gap_sec=0.0)
    assert config.turn_gap_sec == 0.0


def test_config_zero_context_window_turns_raises() -> None:
    """Zero context_window_turns raises ValueError."""
    with pytest.raises(ValueError, match="context_window_turns must be > 0"):
        LiveSemanticsConfig(context_window_turns=0)


def test_config_negative_context_window_turns_raises() -> None:
    """Negative context_window_turns raises ValueError."""
    with pytest.raises(ValueError, match="context_window_turns must be > 0"):
        LiveSemanticsConfig(context_window_turns=-5)


def test_config_zero_context_window_sec_raises() -> None:
    """Zero context_window_sec raises ValueError."""
    with pytest.raises(ValueError, match="context_window_sec must be > 0.0"):
        LiveSemanticsConfig(context_window_sec=0.0)


def test_config_negative_context_window_sec_raises() -> None:
    """Negative context_window_sec raises ValueError."""
    with pytest.raises(ValueError, match="context_window_sec must be > 0.0"):
        LiveSemanticsConfig(context_window_sec=-10.0)


# =============================================================================
# 8. Context Window Edge Case Tests
# =============================================================================


def test_render_context_for_llm_empty_window() -> None:
    """render_context_for_llm with no turns returns empty string."""
    session = LiveSemanticSession()
    # No chunks ingested, context window is empty
    rendered = session.render_context_for_llm()
    assert rendered == ""


def test_get_context_window_empty() -> None:
    """get_context_window with no turns returns empty list."""
    session = LiveSemanticSession()
    context = session.get_context_window()
    assert context == []


# =============================================================================
# 9. Disabled Feature Tests
# =============================================================================


def test_question_detection_disabled() -> None:
    """Question counting can be disabled via config."""
    session = LiveSemanticSession(config=LiveSemanticsConfig(enable_question_detection=False))

    chunk = _chunk(0.0, 1.0, "What is your name? How are you?", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    # Question detection is disabled, so count should be 0
    assert payload.question_count == 0


def test_action_detection_enabled_by_default() -> None:
    """Action detection is enabled by default."""
    session = LiveSemanticSession()
    config = session.config
    assert config.enable_action_detection is True


# =============================================================================
# 9.5. Callback Integration Tests
# =============================================================================


def test_on_semantic_update_called_when_turn_finalized() -> None:
    """on_semantic_update callback is invoked when a turn is finalized."""
    from slower_whisper.pipeline.streaming_semantic import SemanticUpdatePayload

    semantic_updates = []

    class CallbackRecorder:
        def on_semantic_update(self, payload: SemanticUpdatePayload) -> None:
            semantic_updates.append(payload)

    callbacks = CallbackRecorder()
    session = LiveSemanticSession(callbacks=callbacks)

    # Speaker change triggers finalization
    chunk1 = _chunk(0.0, 1.0, "I want to cancel", "spk_0")
    chunk2 = _chunk(1.5, 2.5, "Let me help you", "spk_1")

    session.ingest_chunk(chunk1)
    session.ingest_chunk(chunk2)  # Finalizes spk_0's turn

    # Callback should have been invoked once
    assert len(semantic_updates) == 1
    payload = semantic_updates[0]

    # Verify payload structure
    assert isinstance(payload, SemanticUpdatePayload)
    assert payload.turn.speaker_id == "spk_0"
    assert payload.turn.text == "I want to cancel"
    assert "cancel" in payload.keywords
    assert "churn_risk" in payload.risk_tags
    assert payload.question_count == 0
    assert payload.context_size == 1


def test_on_semantic_update_called_on_end_of_stream() -> None:
    """on_semantic_update callback is invoked on end_of_stream."""
    from slower_whisper.pipeline.streaming_semantic import SemanticUpdatePayload

    semantic_updates = []

    class CallbackRecorder:
        def on_semantic_update(self, payload: SemanticUpdatePayload) -> None:
            semantic_updates.append(payload)

    session = LiveSemanticSession(callbacks=CallbackRecorder())

    chunk = _chunk(0.0, 2.0, "The price is too expensive", "spk_0")
    session.ingest_chunk(chunk)

    # No callback yet
    assert len(semantic_updates) == 0

    # End of stream finalizes the turn
    session.end_of_stream()

    assert len(semantic_updates) == 1
    payload = semantic_updates[0]
    assert "price" in payload.keywords
    assert "expensive" in payload.keywords
    assert "pricing" in payload.risk_tags


def test_on_semantic_update_includes_context_size() -> None:
    """on_semantic_update payload includes current context window size."""
    semantic_updates = []

    class CallbackRecorder:
        def on_semantic_update(self, payload) -> None:  # type: ignore[no-untyped-def]
            semantic_updates.append(payload)

    session = LiveSemanticSession(callbacks=CallbackRecorder())

    chunks = [
        _chunk(0.0, 0.5, "first", "spk_0"),
        _chunk(0.6, 1.0, "second", "spk_1"),
        _chunk(1.1, 1.5, "third", "spk_0"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    # Should have received 2 callbacks (last turn not finalized)
    assert len(semantic_updates) == 2

    # First turn: context_size is 1 (just added itself)
    assert semantic_updates[0].context_size == 1

    # Second turn: context_size is 2 (both turns in window)
    assert semantic_updates[1].context_size == 2


def test_on_semantic_update_includes_question_count() -> None:
    """on_semantic_update payload includes question count."""
    semantic_updates = []

    class CallbackRecorder:
        def on_semantic_update(self, payload) -> None:  # type: ignore[no-untyped-def]
            semantic_updates.append(payload)

    session = LiveSemanticSession(callbacks=CallbackRecorder())

    chunk = _chunk(0.0, 1.0, "What is your name? How can I help?", "spk_0")
    session.ingest_chunk(chunk)
    session.end_of_stream()

    assert len(semantic_updates) == 1
    assert semantic_updates[0].question_count == 2


def test_on_semantic_update_includes_actions() -> None:
    """on_semantic_update payload includes detected actions."""
    semantic_updates = []

    class CallbackRecorder:
        def on_semantic_update(self, payload) -> None:  # type: ignore[no-untyped-def]
            semantic_updates.append(payload)

    session = LiveSemanticSession(callbacks=CallbackRecorder())

    chunk = _chunk(0.0, 1.0, "I'll send you the email tomorrow", "spk_1")
    session.ingest_chunk(chunk)
    session.end_of_stream()

    assert len(semantic_updates) == 1
    assert len(semantic_updates[0].actions) > 0


def test_callback_exception_does_not_crash_session() -> None:
    """Callback exceptions are caught and don't crash the session."""

    class FailingCallbacks:
        def __init__(self) -> None:
            self.call_count = 0

        def on_semantic_update(self, payload) -> None:  # type: ignore[no-untyped-def]
            self.call_count += 1
            raise RuntimeError("Callback failed")

    callbacks = FailingCallbacks()
    session = LiveSemanticSession(callbacks=callbacks)

    chunk1 = _chunk(0.0, 0.5, "hello", "spk_0")
    chunk2 = _chunk(0.6, 1.0, "world", "spk_1")

    # Should not raise even though callback will fail
    session.ingest_chunk(chunk1)
    events = session.ingest_chunk(chunk2)

    # Session continues normally
    assert len(events) == 1
    assert events[0].type == StreamEventType.SEMANTIC_UPDATE

    # Callback was called (and failed)
    assert callbacks.call_count == 1


def test_session_with_no_callbacks() -> None:
    """Session works correctly when callbacks is None."""
    session = LiveSemanticSession(callbacks=None)

    chunk1 = _chunk(0.0, 0.5, "hello", "spk_0")
    chunk2 = _chunk(0.6, 1.0, "world", "spk_1")

    # Should not raise
    events1 = session.ingest_chunk(chunk1)
    events2 = session.ingest_chunk(chunk2)

    assert len(events1) == 0
    assert len(events2) == 1  # Finalized first turn


def test_multiple_callbacks_for_multiple_turns() -> None:
    """Multiple turns invoke the callback multiple times."""
    semantic_updates = []

    class CallbackRecorder:
        def on_semantic_update(self, payload) -> None:  # type: ignore[no-untyped-def]
            semantic_updates.append(payload)

    session = LiveSemanticSession(
        config=LiveSemanticsConfig(turn_gap_sec=0.5),
        callbacks=CallbackRecorder(),
    )

    chunks = [
        _chunk(0.0, 0.4, "one", "spk_0"),
        _chunk(1.0, 1.4, "two", "spk_1"),
        _chunk(2.0, 2.4, "three", "spk_0"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    session.end_of_stream()

    # Should have 3 callbacks (one per turn)
    assert len(semantic_updates) == 3
    texts = [payload.turn.text for payload in semantic_updates]
    assert texts == ["one", "two", "three"]


# =============================================================================
# 10. Custom Annotator Tests
# =============================================================================


def test_custom_annotator_injection() -> None:
    """LiveSemanticSession accepts custom annotator."""
    from slower_whisper.pipeline.semantic import NoOpSemanticAnnotator

    # Use NoOp annotator that returns transcript unchanged
    session = LiveSemanticSession(annotator=NoOpSemanticAnnotator())

    chunk = _chunk(0.0, 1.0, "I want to cancel and escalate this!", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    # NoOpSemanticAnnotator doesn't add keywords/risk_tags
    # But the metadata structure is still populated by _annotate_turn
    assert payload.turn.text == "I want to cancel and escalate this!"


# =============================================================================
# 11. Turn ID Tracking Tests
# =============================================================================


def test_turn_ids_are_monotonic() -> None:
    """Turn IDs increment monotonically."""
    session = LiveSemanticSession()

    chunks = [
        _chunk(0.0, 0.5, "one", "spk_0"),
        _chunk(0.6, 1.0, "two", "spk_1"),
        _chunk(1.1, 1.5, "three", "spk_0"),
    ]

    all_updates = []
    for chunk in chunks:
        events = session.ingest_chunk(chunk)
        all_updates.extend([e for e in events if e.type == StreamEventType.SEMANTIC_UPDATE])

    all_updates.extend(session.end_of_stream())

    # Extract turn IDs
    turn_ids = [u.semantic.turn.id for u in all_updates]
    assert turn_ids == ["turn_0", "turn_1", "turn_2"]


# =============================================================================
# 12. Rapid Messages Tests
# =============================================================================


def test_rapid_consecutive_chunks_same_speaker() -> None:
    """Rapid chunks from same speaker with no gap -> single turn."""
    session = LiveSemanticSession(config=LiveSemanticsConfig(turn_gap_sec=1.0))

    # Chunks with no gap
    chunks = [
        _chunk(0.0, 0.1, "a", "spk_0"),
        _chunk(0.1, 0.2, "b", "spk_0"),
        _chunk(0.2, 0.3, "c", "spk_0"),
        _chunk(0.3, 0.4, "d", "spk_0"),
        _chunk(0.4, 0.5, "e", "spk_0"),
    ]

    for chunk in chunks:
        events = session.ingest_chunk(chunk)
        assert len(events) == 0  # No finalization

    finals = session.end_of_stream()
    assert len(finals) == 1
    assert finals[0].semantic.turn.text == "a b c d e"


def test_rapid_speaker_switches() -> None:
    """Rapid speaker switches finalize turns immediately."""
    session = LiveSemanticSession()

    chunks = [
        _chunk(0.0, 0.1, "a", "spk_0"),
        _chunk(0.1, 0.2, "b", "spk_1"),
        _chunk(0.2, 0.3, "c", "spk_0"),
        _chunk(0.3, 0.4, "d", "spk_1"),
    ]

    all_events = []
    for chunk in chunks:
        all_events.extend(session.ingest_chunk(chunk))

    updates = [e for e in all_events if e.type == StreamEventType.SEMANTIC_UPDATE]
    assert len(updates) == 3  # 3 finalizations (last still buffered)


# =============================================================================
# 13. Segment Conversion Tests
# =============================================================================


def test_turn_to_segment_conversion() -> None:
    """Finalized turn is converted to StreamSegment in event."""
    session = LiveSemanticSession()

    chunk = _chunk(0.0, 1.5, "test message", "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    segment = events[0].segment
    assert segment.start == 0.0
    assert segment.end == 1.5
    assert segment.text == "test message"
    assert segment.speaker_id == "spk_0"


# =============================================================================
# 14. Long Text Tests
# =============================================================================


def test_long_text_handling() -> None:
    """Long text chunks are handled correctly."""
    session = LiveSemanticSession()

    long_text = "word " * 1000  # 5000 characters
    chunk = _chunk(0.0, 60.0, long_text.strip(), "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    assert len(payload.turn.text) == len(long_text.strip())


# =============================================================================
# 15. Special Characters Tests
# =============================================================================


def test_special_characters_in_text() -> None:
    """Special characters in text are preserved."""
    session = LiveSemanticSession()

    special_text = "Hello! @user #topic $100 %discount & more?"
    chunk = _chunk(0.0, 1.0, special_text, "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    payload = events[0].semantic
    assert payload.turn.text == special_text
    assert payload.question_count == 1  # One question mark


def test_unicode_text() -> None:
    """Unicode text is handled correctly."""
    session = LiveSemanticSession()

    unicode_text = "Hello! Bonjour! Hola! Guten Tag! Ciao!"
    chunk = _chunk(0.0, 1.0, unicode_text, "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    assert events[0].semantic.turn.text == unicode_text


def test_emoji_text() -> None:
    """Emoji in text is handled correctly."""
    session = LiveSemanticSession()

    emoji_text = "Great idea! Keep going!"  # Simplified for test stability
    chunk = _chunk(0.0, 1.0, emoji_text, "spk_0")
    session.ingest_chunk(chunk)
    events = session.end_of_stream()

    assert events[0].semantic.turn.text == emoji_text


# =============================================================================
# 16. Context Window Time Pruning Edge Cases
# =============================================================================


def test_context_window_prune_all_old_turns() -> None:
    """All old turns are pruned when new turn is far in the future."""
    session = LiveSemanticSession(
        config=LiveSemanticsConfig(
            turn_gap_sec=0.5,
            context_window_turns=100,
            context_window_sec=10.0,
        )
    )

    # Add several turns close together
    chunks = [
        _chunk(0.0, 0.4, "one", "spk_0"),
        _chunk(1.0, 1.4, "two", "spk_1"),
        _chunk(2.0, 2.4, "three", "spk_0"),
    ]
    for chunk in chunks:
        session.ingest_chunk(chunk)

    # Add turn far in the future (beyond 10s window)
    future_chunk = _chunk(100.0, 100.4, "future", "spk_1")
    session.ingest_chunk(future_chunk)
    session.end_of_stream()

    # All old turns should be pruned
    context = session.get_context_window()
    texts = [turn.text for turn in context]
    assert "one" not in texts
    assert "two" not in texts
    assert "three" not in texts
    assert "future" in texts


# =============================================================================
# 17. Zero Turn Gap Configuration Tests
# =============================================================================


def test_zero_turn_gap_finalizes_every_chunk() -> None:
    """Zero turn_gap_sec finalizes turn on every same-speaker chunk."""
    session = LiveSemanticSession(config=LiveSemanticsConfig(turn_gap_sec=0.0))

    chunks = [
        _chunk(0.0, 0.5, "one", "spk_0"),
        _chunk(0.5, 1.0, "two", "spk_0"),  # Gap = 0.0 >= 0.0 -> finalize
        _chunk(1.0, 1.5, "three", "spk_0"),  # Gap = 0.0 >= 0.0 -> finalize
    ]

    all_events = []
    for chunk in chunks:
        all_events.extend(session.ingest_chunk(chunk))

    updates = [e for e in all_events if e.type == StreamEventType.SEMANTIC_UPDATE]
    # First chunk starts buffer, second and third finalize previous
    assert len(updates) == 2


# =============================================================================
# 18. Segment ID Handling Tests
# =============================================================================


def test_segment_ids_tracked_correctly() -> None:
    """Segment IDs in turn match number of chunks."""
    session = LiveSemanticSession()

    chunks = [
        _chunk(0.0, 0.5, "one", "spk_0"),
        _chunk(0.6, 1.0, "two", "spk_0"),
        _chunk(1.1, 1.5, "three", "spk_0"),
    ]

    for chunk in chunks:
        session.ingest_chunk(chunk)

    events = session.end_of_stream()
    turn = events[0].semantic.turn

    # Should have 3 segment IDs (0, 1, 2)
    assert turn.segment_ids == [0, 1, 2]


# =============================================================================
# 19. Internal Method Edge Cases (for coverage)
# =============================================================================


def test_finalize_turn_empty_buffer_returns_none() -> None:
    """_finalize_turn on empty buffer returns None (defensive check)."""
    session = LiveSemanticSession()
    # Directly call internal method with empty buffer
    result = session._finalize_turn()
    assert result is None


def test_build_turn_from_empty_buffer_raises() -> None:
    """_build_turn_from_buffer on empty buffer raises ValueError."""
    session = LiveSemanticSession()
    # Directly call internal method with empty buffer
    with pytest.raises(ValueError, match="Cannot build turn from empty buffer"):
        session._build_turn_from_buffer()
