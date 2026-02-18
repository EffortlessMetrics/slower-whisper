"""
Tests for LLM rendering utilities.

Test coverage:
- render_segment() with and without audio cues
- render_conversation_for_llm() in turns/segments mode
- render_conversation_compact() with token limits
- _as_dict() dict conversion helper
"""

from typing import Any

import pytest

from slower_whisper.pipeline.llm_utils import (
    _as_dict,
    render_conversation_compact,
    render_conversation_for_llm,
    render_segment,
    to_speaker_summary,
    to_turn_view,
)
from slower_whisper.pipeline.models import Segment, SpeakerStats, Transcript, Turn


def test_render_segment_basic() -> None:
    """Test rendering a simple segment without enrichment."""
    segment = Segment(
        id=0,
        start=0.0,
        end=2.0,
        text="Hello, how can I help you?",
        speaker={"id": "spk_0"},
    )

    # Basic rendering
    result = render_segment(segment, include_audio_cues=False, include_timestamps=False)
    assert result == "[spk_0] Hello, how can I help you?"


def test_render_segment_with_audio_cues() -> None:
    """Test rendering a segment with audio enrichment."""
    segment = Segment(
        id=0,
        start=0.0,
        end=2.0,
        text="Hello, how can I help you?",
        speaker={"id": "spk_0"},
        audio_state={
            "rendering": "[audio: calm tone, low pitch, slow rate]",
            "prosody": {"pitch": {"level": "low"}},
        },
    )

    result = render_segment(segment, include_audio_cues=True, include_timestamps=False)
    assert result == "[spk_0 | calm tone, low pitch, slow rate] Hello, how can I help you?"


def test_render_segment_with_timestamps() -> None:
    """Test rendering with timestamp prefix."""
    segment = Segment(
        id=0,
        start=65.5,  # 00:01:05
        end=68.0,
        text="This is a test.",
        speaker={"id": "spk_1"},
    )

    result = render_segment(segment, include_audio_cues=False, include_timestamps=True)
    assert result == "[00:01:05] [spk_1] This is a test."


def test_render_segment_no_speaker() -> None:
    """Test rendering segment without speaker attribution."""
    segment = Segment(
        id=0,
        start=0.0,
        end=2.0,
        text="Unattributed text.",
        speaker=None,
    )

    result = render_segment(segment, include_audio_cues=False, include_timestamps=False)
    assert result == "Unattributed text."


def test_render_conversation_for_llm_turns_mode() -> None:
    """Test full conversation rendering in turns mode."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Hello.", speaker={"id": "spk_0"}),
        Segment(id=1, start=2.5, end=4.0, text="Hi there!", speaker={"id": "spk_1"}),
        Segment(id=2, start=4.5, end=6.0, text="How can I help?", speaker={"id": "spk_0"}),
    ]
    turns: list[Turn | dict[str, Any]] = [
        {"speaker_id": "spk_0", "start": 0.0, "end": 2.0, "segment_ids": [0], "text": "Hello."},
        {
            "speaker_id": "spk_1",
            "start": 2.5,
            "end": 4.0,
            "segment_ids": [1],
            "text": "Hi there!",
        },
        {
            "speaker_id": "spk_0",
            "start": 4.5,
            "end": 6.0,
            "segment_ids": [2],
            "text": "How can I help?",
        },
    ]
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        turns=turns,
        speakers=[{"id": "spk_0"}, {"id": "spk_1"}],
    )

    result = render_conversation_for_llm(
        transcript,
        mode="turns",
        include_audio_cues=False,
        include_timestamps=False,
        include_metadata=True,
    )

    # Check metadata header
    assert "Conversation: test.wav (en)" in result
    assert "Speakers: 2" in result
    assert "Turns: 3" in result

    # Check turn content
    assert "[spk_0] Hello." in result
    assert "[spk_1] Hi there!" in result
    assert "[spk_0] How can I help?" in result


def test_render_conversation_for_llm_segments_mode() -> None:
    """Test conversation rendering in segments mode (fallback)."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="First segment.", speaker={"id": "spk_0"}),
        Segment(id=1, start=2.5, end=4.0, text="Second segment.", speaker={"id": "spk_1"}),
    ]
    transcript = Transcript(
        file_name="test.wav", language="en", segments=segments, turns=None, speakers=None
    )

    result = render_conversation_for_llm(
        transcript,
        mode="segments",
        include_audio_cues=False,
        include_timestamps=False,
        include_metadata=False,
    )

    assert "[spk_0] First segment." in result
    assert "[spk_1] Second segment." in result


def test_render_conversation_for_llm_with_audio_cues() -> None:
    """Test conversation rendering with audio enrichment in turns mode."""
    segments = [
        Segment(
            id=0,
            start=0.0,
            end=2.0,
            text="I'm calm.",
            speaker={"id": "spk_0"},
            audio_state={"rendering": "[audio: calm tone, low pitch]"},
        ),
        Segment(
            id=1,
            start=2.5,
            end=4.0,
            text="I'm excited!",
            speaker={"id": "spk_1"},
            audio_state={"rendering": "[audio: happy tone, high pitch, fast rate]"},
        ),
    ]
    turns: list[Turn | dict[str, Any]] = [
        {"speaker_id": "spk_0", "start": 0.0, "end": 2.0, "segment_ids": [0], "text": "I'm calm."},
        {
            "speaker_id": "spk_1",
            "start": 2.5,
            "end": 4.0,
            "segment_ids": [1],
            "text": "I'm excited!",
        },
    ]
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        turns=turns,
        speakers=[{"id": "spk_0"}, {"id": "spk_1"}],
    )

    result = render_conversation_for_llm(
        transcript,
        mode="turns",
        include_audio_cues=True,
        include_timestamps=False,
        include_metadata=False,
    )

    assert "calm tone" in result or "low pitch" in result
    assert "I'm calm." in result
    assert "fast rate" in result or "happy tone" in result or "high pitch" in result
    assert "I'm excited!" in result


def test_render_conversation_compact() -> None:
    """Test compact rendering for token-constrained contexts."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Hello.", speaker={"id": "spk_0"}),
        Segment(id=1, start=2.5, end=4.0, text="Hi there!", speaker={"id": "spk_1"}),
        Segment(id=2, start=4.5, end=6.0, text="How can I help?", speaker={"id": "spk_0"}),
    ]
    turns: list[Turn | dict[str, Any]] = [
        {"speaker_id": "spk_0", "start": 0.0, "end": 2.0, "segment_ids": [0], "text": "Hello."},
        {
            "speaker_id": "spk_1",
            "start": 2.5,
            "end": 4.0,
            "segment_ids": [1],
            "text": "Hi there!",
        },
        {
            "speaker_id": "spk_0",
            "start": 4.5,
            "end": 6.0,
            "segment_ids": [2],
            "text": "How can I help?",
        },
    ]
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        turns=turns,
        speakers=[{"id": "spk_0"}, {"id": "spk_1"}],
    )

    result = render_conversation_compact(transcript)

    # Should be minimal format
    assert result == "spk_0: Hello.\nspk_1: Hi there!\nspk_0: How can I help?"


def test_render_conversation_compact_with_token_limit() -> None:
    """Test compact rendering with truncation."""
    segments = [
        Segment(
            id=0,
            start=0.0,
            end=2.0,
            text="This is a very long segment that should be truncated.",
            speaker={"id": "spk_0"},
        ),
        Segment(id=1, start=2.5, end=4.0, text="Another segment.", speaker={"id": "spk_1"}),
    ]
    transcript = Transcript(
        file_name="test.wav", language="en", segments=segments, turns=None, speakers=None
    )

    # Set very low token limit (e.g., 10 tokens ≈ 40 chars)
    result = render_conversation_compact(transcript, max_tokens=10)

    # Should be truncated (40 chars + "\n[...truncated]" = ~56 chars)
    assert len(result) <= 60
    assert "[...truncated]" in result


def test_render_conversation_compact_fallback_to_segments() -> None:
    """Test compact rendering falls back to segments if no turns."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="First.", speaker={"id": "spk_0"}),
        Segment(id=1, start=2.5, end=4.0, text="Second.", speaker={"id": "spk_1"}),
    ]
    transcript = Transcript(
        file_name="test.wav", language="en", segments=segments, turns=None, speakers=None
    )

    result = render_conversation_compact(transcript)

    assert result == "spk_0: First.\nspk_1: Second."


def test_render_segment_strips_whitespace() -> None:
    """Test that segment rendering strips leading/trailing whitespace from text."""
    segment = Segment(id=0, start=0.0, end=2.0, text="  Hello world.  ", speaker={"id": "spk_0"})

    result = render_segment(segment, include_audio_cues=False, include_timestamps=False)
    assert result == "[spk_0] Hello world."


def test_render_conversation_metadata_duration_formatting() -> None:
    """Test that conversation metadata formats duration correctly."""
    segments = [
        Segment(id=0, start=0.0, end=3665.0, text="Long conversation.", speaker={"id": "spk_0"})
    ]
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        turns=None,
        speakers=[{"id": "spk_0"}],
    )

    result = render_conversation_for_llm(transcript, mode="segments", include_metadata=True)

    # 3665 seconds = 1:01:05
    assert "Duration: 01:01:05" in result


def test_render_turn_with_audio_aggregation() -> None:
    """Test that turn rendering aggregates audio cues from multiple segments."""
    segments = [
        Segment(
            id=0,
            start=0.0,
            end=2.0,
            text="I'm frustrated.",
            speaker={"id": "spk_1"},
            audio_state={"rendering": "[audio: high pitch, fast rate]"},
        ),
        Segment(
            id=1,
            start=2.0,
            end=4.0,
            text="This is not working!",
            speaker={"id": "spk_1"},
            audio_state={"rendering": "[audio: loud volume, fast rate]"},
        ),
    ]
    turns: list[Turn | dict[str, Any]] = [
        {
            "speaker_id": "spk_1",
            "start": 0.0,
            "end": 4.0,
            "segment_ids": [0, 1],
            "text": "I'm frustrated. This is not working!",
        }
    ]
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        turns=turns,
        speakers=[{"id": "spk_1"}],
    )

    result = render_conversation_for_llm(
        transcript, mode="turns", include_audio_cues=True, include_metadata=False
    )

    # Should aggregate unique cues: fast rate, high pitch, loud volume
    assert "spk_1" in result
    assert "fast rate" in result
    assert "high pitch" in result
    assert "loud volume" in result
    assert "I'm frustrated. This is not working!" in result


def test_render_turn_with_timestamps() -> None:
    """Test turn rendering with timestamp prefix."""
    segments = [
        Segment(id=0, start=125.0, end=127.0, text="At two minutes.", speaker={"id": "spk_0"})
    ]
    turns: list[Turn | dict[str, Any]] = [
        {
            "speaker_id": "spk_0",
            "start": 125.0,
            "end": 127.0,
            "segment_ids": [0],
            "text": "At two minutes.",
        }
    ]
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        turns=turns,
        speakers=[{"id": "spk_0"}],
    )

    result = render_conversation_for_llm(
        transcript, mode="turns", include_timestamps=True, include_metadata=False
    )

    assert "[00:02:05]" in result
    assert "spk_0" in result
    assert "At two minutes." in result


def test_render_segment_with_speaker_labels() -> None:
    """Test that speaker_labels parameter maps speaker IDs to readable names."""
    segment = Segment(id=0, start=0.0, end=2.0, text="Hello there.", speaker={"id": "spk_0"})

    result = render_segment(
        segment,
        include_audio_cues=False,
        include_timestamps=False,
        speaker_labels={"spk_0": "Agent", "spk_1": "Customer"},
    )
    assert result == "[Agent] Hello there."


def test_render_conversation_with_speaker_labels() -> None:
    """Test that speaker_labels work in full conversation rendering."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Hello.", speaker={"id": "spk_0"}),
        Segment(id=1, start=2.5, end=4.0, text="Hi there!", speaker={"id": "spk_1"}),
    ]
    turns: list[Turn | dict[str, Any]] = [
        {"speaker_id": "spk_0", "start": 0.0, "end": 2.0, "segment_ids": [0], "text": "Hello."},
        {
            "speaker_id": "spk_1",
            "start": 2.5,
            "end": 4.0,
            "segment_ids": [1],
            "text": "Hi there!",
        },
    ]
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        turns=turns,
        speakers=[{"id": "spk_0"}, {"id": "spk_1"}],
    )

    result = render_conversation_for_llm(
        transcript,
        mode="turns",
        include_metadata=False,
        speaker_labels={"spk_0": "Agent", "spk_1": "Customer"},
    )

    assert "[Agent]" in result
    assert "[Customer]" in result
    assert "spk_0" not in result
    assert "spk_1" not in result


def test_render_compact_with_speaker_labels() -> None:
    """Test that speaker_labels work in compact rendering."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Hello.", speaker={"id": "spk_0"}),
        Segment(id=1, start=2.5, end=4.0, text="Hi!", speaker={"id": "spk_1"}),
    ]
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        speakers=[{"id": "spk_0"}, {"id": "spk_1"}],
    )

    result = render_conversation_compact(
        transcript, speaker_labels={"spk_0": "Agent", "spk_1": "Customer"}
    )

    assert "Agent: Hello." in result
    assert "Customer: Hi!" in result
    assert "spk_0" not in result
    assert "spk_1" not in result


def test_render_empty_transcript() -> None:
    """Test rendering a transcript with no segments."""
    transcript = Transcript(file_name="empty.wav", language="en", segments=[], speakers=[])

    result = render_conversation_for_llm(transcript, mode="segments", include_metadata=True)

    # Should still have metadata
    assert "empty.wav" in result
    assert "Duration: 00:00:00" in result


def test_render_no_turns_no_speakers() -> None:
    """Test rendering when transcript has no turns and no speaker info."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="First line.", speaker=None),
        Segment(id=1, start=2.0, end=4.0, text="Second line.", speaker=None),
    ]
    transcript = Transcript(file_name="test.wav", language="en", segments=segments)

    # Should still render in segments mode
    result = render_conversation_for_llm(transcript, mode="segments", include_metadata=False)

    assert "First line." in result
    assert "Second line." in result


def test_render_segment_none_speaker_with_audio() -> None:
    """Test that audio cues are preserved even when speaker is None."""
    segment = Segment(
        id=0,
        start=0.0,
        end=2.0,
        text="No speaker but has audio.",
        speaker=None,
        audio_state={"rendering": "[audio: loud volume, fast rate]"},
    )

    result = render_segment(segment, include_audio_cues=True, include_timestamps=False)

    # Audio cues should still appear even without speaker
    assert "loud volume" in result
    assert "fast rate" in result
    assert "No speaker but has audio." in result


def test_render_turns_mode_fallback_to_segments() -> None:
    """Test that turns mode gracefully falls back to segments when no turns exist."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Line one.", speaker={"id": "spk_0"}),
        Segment(id=1, start=2.0, end=4.0, text="Line two.", speaker={"id": "spk_0"}),
    ]
    transcript = Transcript(
        file_name="test.wav", language="en", segments=segments, speakers=[{"id": "spk_0"}]
    )

    # Request turns mode but transcript has no turns
    result = render_conversation_for_llm(transcript, mode="turns", include_metadata=False)

    # Should fall back to segment rendering
    assert "Line one." in result
    assert "Line two." in result


def test_to_turn_view_renders_metadata_and_audio() -> None:
    """Turn view should include timestamps, metadata, and aggregated audio cues."""
    segments = [
        Segment(
            id=0,
            start=0.0,
            end=1.0,
            text="Hi there.",
            speaker={"id": "spk_0"},
            audio_state={"rendering": "[audio: calm tone]"},
        ),
        Segment(
            id=1,
            start=1.0,
            end=2.0,
            text="Can we start now?",
            speaker={"id": "spk_1"},
            audio_state={"rendering": "[audio: tense, high pitch]"},
        ),
    ]
    turns: list[Turn | dict[str, Any]] = [
        {
            "speaker_id": "spk_0",
            "start": 0.0,
            "end": 1.0,
            "segment_ids": [0],
            "text": "Hi there.",
            "metadata": {
                "question_count": 0,
                "interruption_started_here": False,
                "avg_pause_ms": 25.0,
                "disfluency_ratio": 0.05,
            },
        },
        {
            "speaker_id": "spk_1",
            "start": 1.0,
            "end": 2.0,
            "segment_ids": [1],
            "text": "Can we start now?",
            "metadata": {
                "question_count": 1,
                "interruption_started_here": True,
                "avg_pause_ms": None,
                "disfluency_ratio": 0.35,
            },
        },
    ]
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        turns=turns,
        speakers=[{"id": "spk_0"}, {"id": "spk_1"}],
    )

    view = to_turn_view(
        transcript,
        include_audio_state=True,
        include_timestamps=True,
        speaker_labels={"spk_0": "Agent", "spk_1": "Customer"},
    )

    assert "Agent" in view
    assert "Customer" in view
    assert "question_count=1" in view
    assert "interruption_started_here=true" in view
    assert "disfluency=0.35" in view
    assert "audio=" in view
    assert "high pitch" in view
    assert "tense" in view
    assert view.count("\n") == 1  # two lines total


def test_to_speaker_summary_formats_stats() -> None:
    """Speaker summary should render core analytics with labels."""
    stats: list[SpeakerStats | dict[str, Any]] = [
        {
            "speaker_id": "spk_0",
            "total_talk_time": 12.3,
            "num_turns": 3,
            "avg_turn_duration": 4.1,
            "interruptions_initiated": 1,
            "interruptions_received": 2,
            "question_turns": 1,
            "prosody_summary": {"pitch_median_hz": 180.0, "energy_median_db": -10.5},
            "sentiment_summary": {"positive": 0.5, "neutral": 0.3, "negative": 0.2},
        }
    ]
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[],
        turns=[],
        speaker_stats=stats,
    )

    summary = to_speaker_summary(transcript, speaker_labels={"spk_0": "Agent", "spk_1": "Customer"})

    assert "Speaker stats summary" in summary
    assert "Agent" in summary
    assert "12.3s across 3 turns" in summary
    assert "1 interruptions started" in summary
    assert "2 received" in summary
    assert "1 question turns" in summary
    assert "pitch~=180Hz" in summary
    assert "energy~=-10.5dB" in summary


# ============================================================================
# Tests for _as_dict helper function
# ============================================================================


def test_as_dict_with_plain_dict() -> None:
    """_as_dict should return plain dicts unchanged."""
    input_dict = {"speaker_id": "spk_0", "text": "Hello"}
    result = _as_dict(input_dict)
    assert result == input_dict
    assert result is input_dict  # Should be the same object


def test_as_dict_with_to_dict_object() -> None:
    """_as_dict should call to_dict() on objects that have it."""
    turn = Turn(
        id="turn_0",
        speaker_id="spk_0",
        segment_ids=[0, 1],
        start=0.0,
        end=2.5,
        text="Hello world",
        metadata={"question_count": 1},
    )
    result = _as_dict(turn)

    assert isinstance(result, dict)
    assert result["id"] == "turn_0"
    assert result["speaker_id"] == "spk_0"
    assert result["segment_ids"] == [0, 1]
    assert result["text"] == "Hello world"
    assert result["metadata"] == {"question_count": 1}


def test_as_dict_with_dataclass_instance() -> None:
    """_as_dict should convert dataclass instances using asdict."""
    # Segment is a dataclass that does NOT have to_dict
    segment = Segment(
        id=0,
        start=0.0,
        end=1.5,
        text="Test segment",
        speaker={"id": "spk_0"},
    )
    result = _as_dict(segment)

    assert isinstance(result, dict)
    assert result["id"] == 0
    assert result["start"] == 0.0
    assert result["end"] == 1.5
    assert result["text"] == "Test segment"
    assert result["speaker"] == {"id": "spk_0"}


def test_as_dict_with_dataclass_class_raises() -> None:
    """_as_dict should raise TypeError for dataclass classes (not instances)."""
    with pytest.raises(TypeError, match="Unsupported object type"):
        _as_dict(Turn)


def test_as_dict_with_unsupported_type_raises() -> None:
    """_as_dict should raise TypeError for unsupported types."""
    with pytest.raises(TypeError, match="Unsupported object type"):
        _as_dict("a string")

    with pytest.raises(TypeError, match="Unsupported object type"):
        _as_dict(42)

    with pytest.raises(TypeError, match="Unsupported object type"):
        _as_dict([1, 2, 3])


def test_as_dict_with_nested_dataclass() -> None:
    """_as_dict should handle dataclasses with nested structures."""
    turn = Turn(
        id="turn_1",
        speaker_id="spk_1",
        segment_ids=[0],
        start=0.0,
        end=1.0,
        text="Nested test",
        metadata={"nested": {"key": "value"}},
    )
    result = _as_dict(turn)

    assert result["metadata"]["nested"]["key"] == "value"


class CustomObjectWithToDict:
    """Test object with a to_dict method."""

    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def test_as_dict_with_custom_to_dict() -> None:
    """_as_dict should work with any object that has to_dict()."""
    custom = CustomObjectWithToDict({"custom": "data", "count": 5})
    result = _as_dict(custom)

    assert result == {"custom": "data", "count": 5}
