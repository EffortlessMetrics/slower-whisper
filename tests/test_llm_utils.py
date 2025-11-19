"""
Tests for LLM rendering utilities.

Test coverage:
- render_segment() with and without audio cues
- render_conversation_for_llm() in turns/segments mode
- render_conversation_compact() with token limits
"""

from transcription.llm_utils import (
    render_conversation_compact,
    render_conversation_for_llm,
    render_segment,
)
from transcription.models import Segment, Transcript


def test_render_segment_basic():
    """Test rendering a simple segment without enrichment."""
    segment = Segment(
        id=0,
        start=0.0,
        end=2.0,
        text="Hello, how can I help you?",
        speaker="spk_0",
    )

    # Basic rendering
    result = render_segment(segment, include_audio_cues=False, include_timestamps=False)
    assert result == "[spk_0] Hello, how can I help you?"


def test_render_segment_with_audio_cues():
    """Test rendering a segment with audio enrichment."""
    segment = Segment(
        id=0,
        start=0.0,
        end=2.0,
        text="Hello, how can I help you?",
        speaker="spk_0",
        audio_state={
            "rendering": "[audio: calm tone, low pitch, slow rate]",
            "prosody": {"pitch": {"level": "low"}},
        },
    )

    result = render_segment(segment, include_audio_cues=True, include_timestamps=False)
    assert result == "[spk_0 | calm tone, low pitch, slow rate] Hello, how can I help you?"


def test_render_segment_with_timestamps():
    """Test rendering with timestamp prefix."""
    segment = Segment(
        id=0,
        start=65.5,  # 00:01:05
        end=68.0,
        text="This is a test.",
        speaker="spk_1",
    )

    result = render_segment(segment, include_audio_cues=False, include_timestamps=True)
    assert result == "[00:01:05] [spk_1] This is a test."


def test_render_segment_no_speaker():
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


def test_render_conversation_for_llm_turns_mode():
    """Test full conversation rendering in turns mode."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Hello.", speaker="spk_0"),
        Segment(id=1, start=2.5, end=4.0, text="Hi there!", speaker="spk_1"),
        Segment(id=2, start=4.5, end=6.0, text="How can I help?", speaker="spk_0"),
    ]
    turns = [
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
        speakers=["spk_0", "spk_1"],
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


def test_render_conversation_for_llm_segments_mode():
    """Test conversation rendering in segments mode (fallback)."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="First segment.", speaker="spk_0"),
        Segment(id=1, start=2.5, end=4.0, text="Second segment.", speaker="spk_1"),
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


def test_render_conversation_for_llm_with_audio_cues():
    """Test conversation rendering with audio enrichment in turns mode."""
    segments = [
        Segment(
            id=0,
            start=0.0,
            end=2.0,
            text="I'm calm.",
            speaker="spk_0",
            audio_state={"rendering": "[audio: calm tone, low pitch]"},
        ),
        Segment(
            id=1,
            start=2.5,
            end=4.0,
            text="I'm excited!",
            speaker="spk_1",
            audio_state={"rendering": "[audio: happy tone, high pitch, fast rate]"},
        ),
    ]
    turns = [
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
        speakers=["spk_0", "spk_1"],
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


def test_render_conversation_compact():
    """Test compact rendering for token-constrained contexts."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Hello.", speaker="spk_0"),
        Segment(id=1, start=2.5, end=4.0, text="Hi there!", speaker="spk_1"),
        Segment(id=2, start=4.5, end=6.0, text="How can I help?", speaker="spk_0"),
    ]
    turns = [
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
        speakers=["spk_0", "spk_1"],
    )

    result = render_conversation_compact(transcript)

    # Should be minimal format
    assert result == "spk_0: Hello.\nspk_1: Hi there!\nspk_0: How can I help?"


def test_render_conversation_compact_with_token_limit():
    """Test compact rendering with truncation."""
    segments = [
        Segment(
            id=0,
            start=0.0,
            end=2.0,
            text="This is a very long segment that should be truncated.",
            speaker="spk_0",
        ),
        Segment(id=1, start=2.5, end=4.0, text="Another segment.", speaker="spk_1"),
    ]
    transcript = Transcript(
        file_name="test.wav", language="en", segments=segments, turns=None, speakers=None
    )

    # Set very low token limit (e.g., 10 tokens ≈ 40 chars)
    result = render_conversation_compact(transcript, max_tokens=10)

    # Should be truncated (40 chars + "\n[...truncated]" = ~56 chars)
    assert len(result) <= 60
    assert "[...truncated]" in result


def test_render_conversation_compact_fallback_to_segments():
    """Test compact rendering falls back to segments if no turns."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="First.", speaker="spk_0"),
        Segment(id=1, start=2.5, end=4.0, text="Second.", speaker="spk_1"),
    ]
    transcript = Transcript(
        file_name="test.wav", language="en", segments=segments, turns=None, speakers=None
    )

    result = render_conversation_compact(transcript)

    assert result == "spk_0: First.\nspk_1: Second."


def test_render_segment_strips_whitespace():
    """Test that segment rendering strips leading/trailing whitespace from text."""
    segment = Segment(id=0, start=0.0, end=2.0, text="  Hello world.  ", speaker="spk_0")

    result = render_segment(segment, include_audio_cues=False, include_timestamps=False)
    assert result == "[spk_0] Hello world."


def test_render_conversation_metadata_duration_formatting():
    """Test that conversation metadata formats duration correctly."""
    segments = [Segment(id=0, start=0.0, end=3665.0, text="Long conversation.", speaker="spk_0")]
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        turns=None,
        speakers=["spk_0"],
    )

    result = render_conversation_for_llm(transcript, mode="segments", include_metadata=True)

    # 3665 seconds = 1:01:05
    assert "Duration: 01:01:05" in result


def test_render_turn_with_audio_aggregation():
    """Test that turn rendering aggregates audio cues from multiple segments."""
    segments = [
        Segment(
            id=0,
            start=0.0,
            end=2.0,
            text="I'm frustrated.",
            speaker="spk_1",
            audio_state={"rendering": "[audio: high pitch, fast rate]"},
        ),
        Segment(
            id=1,
            start=2.0,
            end=4.0,
            text="This is not working!",
            speaker="spk_1",
            audio_state={"rendering": "[audio: loud volume, fast rate]"},
        ),
    ]
    turns = [
        {
            "speaker_id": "spk_1",
            "start": 0.0,
            "end": 4.0,
            "segment_ids": [0, 1],
            "text": "I'm frustrated. This is not working!",
        }
    ]
    transcript = Transcript(
        file_name="test.wav", language="en", segments=segments, turns=turns, speakers=["spk_1"]
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


def test_render_turn_with_timestamps():
    """Test turn rendering with timestamp prefix."""
    segments = [Segment(id=0, start=125.0, end=127.0, text="At two minutes.", speaker="spk_0")]
    turns = [
        {
            "speaker_id": "spk_0",
            "start": 125.0,
            "end": 127.0,
            "segment_ids": [0],
            "text": "At two minutes.",
        }
    ]
    transcript = Transcript(
        file_name="test.wav", language="en", segments=segments, turns=turns, speakers=["spk_0"]
    )

    result = render_conversation_for_llm(
        transcript, mode="turns", include_timestamps=True, include_metadata=False
    )

    assert "[00:02:05]" in result
    assert "spk_0" in result
    assert "At two minutes." in result


def test_render_segment_with_speaker_labels():
    """Test that speaker_labels parameter maps speaker IDs to readable names."""
    segment = Segment(id=0, start=0.0, end=2.0, text="Hello there.", speaker="spk_0")

    result = render_segment(
        segment,
        include_audio_cues=False,
        include_timestamps=False,
        speaker_labels={"spk_0": "Agent", "spk_1": "Customer"},
    )
    assert result == "[Agent] Hello there."


def test_render_conversation_with_speaker_labels():
    """Test that speaker_labels work in full conversation rendering."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Hello.", speaker="spk_0"),
        Segment(id=1, start=2.5, end=4.0, text="Hi there!", speaker="spk_1"),
    ]
    turns = [
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
        speakers=["spk_0", "spk_1"],
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


def test_render_compact_with_speaker_labels():
    """Test that speaker_labels work in compact rendering."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Hello.", speaker="spk_0"),
        Segment(id=1, start=2.5, end=4.0, text="Hi!", speaker="spk_1"),
    ]
    transcript = Transcript(
        file_name="test.wav", language="en", segments=segments, speakers=["spk_0", "spk_1"]
    )

    result = render_conversation_compact(
        transcript, speaker_labels={"spk_0": "Agent", "spk_1": "Customer"}
    )

    assert "Agent: Hello." in result
    assert "Customer: Hi!" in result
    assert "spk_0" not in result
    assert "spk_1" not in result


def test_render_empty_transcript():
    """Test rendering a transcript with no segments."""
    transcript = Transcript(file_name="empty.wav", language="en", segments=[], speakers=[])

    result = render_conversation_for_llm(transcript, mode="segments", include_metadata=True)

    # Should still have metadata
    assert "empty.wav" in result
    assert "Duration: 00:00:00" in result


def test_render_no_turns_no_speakers():
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


def test_render_segment_none_speaker_with_audio():
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


def test_render_turns_mode_fallback_to_segments():
    """Test that turns mode gracefully falls back to segments when no turns exist."""
    segments = [
        Segment(id=0, start=0.0, end=2.0, text="Line one.", speaker="spk_0"),
        Segment(id=1, start=2.0, end=4.0, text="Line two.", speaker="spk_0"),
    ]
    transcript = Transcript(
        file_name="test.wav", language="en", segments=segments, speakers=["spk_0"]
    )

    # Request turns mode but transcript has no turns
    result = render_conversation_for_llm(transcript, mode="turns", include_metadata=False)

    # Should fall back to segment rendering
    assert "Line one." in result
    assert "Line two." in result
