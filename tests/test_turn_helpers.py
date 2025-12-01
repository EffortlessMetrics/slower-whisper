"""
Tests for turn_helpers.py module.

Test coverage:
- turn_to_dict() with dicts (copy=False and copy=True)
- turn_to_dict() with Turn dataclass instances
- turn_to_dict() with objects implementing to_dict()
- turn_to_dict() error handling for unsupported types
"""

from typing import Any

import pytest

from transcription.models import Turn
from transcription.turn_helpers import turn_to_dict

# ============================================================================
# Dict handling tests
# ============================================================================


def test_turn_to_dict_returns_dict_unchanged() -> None:
    """turn_to_dict should return plain dicts unchanged when copy=False."""
    input_dict = {"speaker_id": "spk_0", "text": "Hello", "start": 0.0}
    result = turn_to_dict(input_dict, copy=False)

    assert result == input_dict
    assert result is input_dict  # Should be the same object (no copy)


def test_turn_to_dict_returns_dict_copy() -> None:
    """turn_to_dict should return a shallow copy of dicts when copy=True."""
    input_dict = {"speaker_id": "spk_1", "text": "World", "end": 5.0}
    result = turn_to_dict(input_dict, copy=True)

    assert result == input_dict  # Same content
    assert result is not input_dict  # Different object (copy made)
    assert id(result) != id(input_dict)  # Verify different object IDs


# ============================================================================
# Turn dataclass handling tests
# ============================================================================


def test_turn_to_dict_converts_turn_dataclass() -> None:
    """turn_to_dict should convert Turn dataclass instances to dicts."""
    turn = Turn(
        id="turn_0",
        speaker_id="spk_0",
        segment_ids=[0, 1, 2],
        start=0.0,
        end=5.5,
        text="Hello world",
        metadata={"question_count": 2, "interruption_started_here": False},
    )

    result = turn_to_dict(turn)

    assert isinstance(result, dict)
    assert result["id"] == "turn_0"
    assert result["speaker_id"] == "spk_0"
    assert result["segment_ids"] == [0, 1, 2]
    assert result["start"] == 0.0
    assert result["end"] == 5.5
    assert result["text"] == "Hello world"
    assert result["metadata"]["question_count"] == 2
    assert result["metadata"]["interruption_started_here"] is False


def test_turn_to_dict_converts_turn_with_empty_metadata() -> None:
    """turn_to_dict should handle Turn instances with empty metadata."""
    turn = Turn(
        id="turn_1",
        speaker_id="spk_1",
        segment_ids=[3],
        start=6.0,
        end=8.0,
        text="Simple turn",
        metadata={},
    )

    result = turn_to_dict(turn)

    assert isinstance(result, dict)
    assert result["id"] == "turn_1"
    assert result["metadata"] == {}


# ============================================================================
# to_dict protocol tests
# ============================================================================


class CustomObjectWithToDict:
    """Test object with a to_dict method."""

    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def test_turn_to_dict_calls_to_dict_method() -> None:
    """turn_to_dict should call to_dict() on objects that have the method."""
    custom = CustomObjectWithToDict(
        {
            "id": "custom_turn",
            "speaker_id": "spk_custom",
            "text": "Custom object",
            "start": 10.0,
            "end": 15.0,
        }
    )

    result = turn_to_dict(custom)

    assert isinstance(result, dict)
    assert result["id"] == "custom_turn"
    assert result["speaker_id"] == "spk_custom"
    assert result["text"] == "Custom object"
    assert result["start"] == 10.0


# ============================================================================
# Error handling tests
# ============================================================================


def test_turn_to_dict_raises_for_unsupported_type() -> None:
    """turn_to_dict should raise TypeError for unsupported types."""
    # String
    with pytest.raises(TypeError, match="Unsupported turn type"):
        turn_to_dict("a string")

    # Integer
    with pytest.raises(TypeError, match="Unsupported turn type"):
        turn_to_dict(42)

    # List
    with pytest.raises(TypeError, match="Unsupported turn type"):
        turn_to_dict([1, 2, 3])

    # None
    with pytest.raises(TypeError, match="Unsupported turn type"):
        turn_to_dict(None)


def test_turn_to_dict_raises_for_class_not_instance() -> None:
    """turn_to_dict should raise TypeError for dataclass classes (not instances)."""
    # Passing the Turn class itself, not an instance
    # Note: Turn has to_dict() method, so it tries to call it and fails with missing 'self'
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        turn_to_dict(Turn)


# ============================================================================
# Edge cases and additional scenarios
# ============================================================================


def test_turn_to_dict_with_plain_dataclass() -> None:
    """turn_to_dict should handle plain dataclasses without to_dict() method."""
    from dataclasses import dataclass

    @dataclass
    class SimpleTurn:
        """A simple dataclass without to_dict method."""

        id: str
        text: str
        start: float

    simple = SimpleTurn(id="simple_1", text="Plain dataclass", start=5.0)
    result = turn_to_dict(simple)

    assert isinstance(result, dict)
    assert result["id"] == "simple_1"
    assert result["text"] == "Plain dataclass"
    assert result["start"] == 5.0


def test_turn_to_dict_with_nested_structures() -> None:
    """turn_to_dict should handle Turn instances with nested metadata."""
    turn = Turn(
        id="turn_complex",
        speaker_id="spk_0",
        segment_ids=[0, 1],
        start=0.0,
        end=3.0,
        text="Complex turn",
        metadata={
            "question_count": 1,
            "nested": {
                "key": "value",
                "count": 42,
            },
        },
    )

    result = turn_to_dict(turn)

    assert result["metadata"]["nested"]["key"] == "value"
    assert result["metadata"]["nested"]["count"] == 42


def test_turn_to_dict_preserves_segment_ids_list() -> None:
    """turn_to_dict should preserve segment_ids as a list."""
    turn = Turn(
        id="turn_segments",
        speaker_id="spk_2",
        segment_ids=[5, 6, 7, 8],
        start=20.0,
        end=30.0,
        text="Multiple segments",
        metadata={},
    )

    result = turn_to_dict(turn)

    assert isinstance(result["segment_ids"], list)
    assert result["segment_ids"] == [5, 6, 7, 8]


def test_turn_to_dict_dict_copy_shallow() -> None:
    """Verify that copy=True creates a shallow copy (nested objects shared)."""
    nested_obj = {"shared": "value"}
    input_dict = {
        "id": "test",
        "nested": nested_obj,
    }

    result = turn_to_dict(input_dict, copy=True)

    # Top-level dict is different
    assert result is not input_dict

    # But nested objects are the same (shallow copy)
    assert result["nested"] is nested_obj


# ============================================================================
# Integration with enrich_turns_metadata
# ============================================================================


def test_enrich_turns_metadata_normalizes_turns_and_adds_metadata() -> None:
    """
    Integration: verify that enrichment uses the shared turn_to_dict helper.

    We start with Transcript.turns as Turn dataclass instances and confirm:
    - enrich_turns_metadata() returns a list of plain dicts
    - each turn dict includes a 'metadata' block with expected keys
    """
    from transcription.models import Segment, Transcript, Turn
    from transcription.turns_enrich import enrich_turns_metadata

    segments = [
        Segment(
            id=0,
            start=0.0,
            end=1.0,
            text="Do you have a moment?",
            speaker={"id": "spk_0"},
        ),
        Segment(
            id=1,
            start=1.1,
            end=2.0,
            text="Sure.",
            speaker={"id": "spk_1"},
        ),
    ]

    turns: list[Turn | dict[str, Any]] = [
        Turn(
            id="turn_0",
            speaker_id="spk_0",
            segment_ids=[0],
            start=0.0,
            end=1.0,
            text="Do you have a moment?",
            metadata={},
        ),
        Turn(
            id="turn_1",
            speaker_id="spk_1",
            segment_ids=[1],
            start=1.1,
            end=2.0,
            text="Sure.",
            metadata={},
        ),
    ]

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        turns=turns,
    )

    enriched = enrich_turns_metadata(transcript)

    # Enrichment returns dict-backed turns and mutates transcript.turns
    assert isinstance(enriched, list)
    assert all(isinstance(t, dict) for t in enriched)
    assert transcript.turns is enriched  # current contract in turns_enrich

    # Each turn should have a metadata dict with the expected keys
    for turn in enriched:
        assert "metadata" in turn
        meta = turn["metadata"]
        assert isinstance(meta, dict)
        assert "question_count" in meta
        assert "interruption_started_here" in meta
        assert "avg_pause_ms" in meta
        assert "disfluency_ratio" in meta


def test_enrich_turns_metadata_handles_question_detection() -> None:
    """
    Integration: verify question_count is correctly computed.
    """
    from transcription.models import Segment, Transcript, Turn
    from transcription.turns_enrich import enrich_turns_metadata

    segments = [
        Segment(
            id=0,
            start=0.0,
            end=1.0,
            text="What time is it?",  # Question (ends with ?)
            speaker={"id": "spk_0"},
        ),
        Segment(
            id=1,
            start=1.0,
            end=2.0,
            text="When can we meet?",  # Question (starts with When)
            speaker={"id": "spk_0"},
        ),
        Segment(
            id=2,
            start=2.0,
            end=3.0,
            text="I need your help.",  # Not a question
            speaker={"id": "spk_0"},
        ),
    ]

    turns: list[Turn | dict[str, Any]] = [
        Turn(
            id="turn_0",
            speaker_id="spk_0",
            segment_ids=[0, 1, 2],
            start=0.0,
            end=3.0,
            text="What time is it? When can we meet? I need your help.",
            metadata={},
        ),
    ]

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        turns=turns,
    )

    enriched = enrich_turns_metadata(transcript)

    # Should detect 2 questions in the turn
    assert enriched[0]["metadata"]["question_count"] == 2
