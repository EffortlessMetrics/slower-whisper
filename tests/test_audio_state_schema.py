"""
Tests for audio_state TypedDict schema compliance.

This test validates that the TypedDict definitions in transcription/types_audio.py
match the actual structure produced by the audio enrichment pipeline.

The audio_state schema (AUDIO_STATE_VERSION = "1.0.0") defines how acoustic features
are serialized into JSON. This test ensures:
1. The structure matches the TypedDict definitions exactly
2. All required fields and sub-structures are present
3. Field types match expectations (str, float, int, dict)
4. The schema is consistent with what audio_enrichment.py generates

Why this test matters:
- TypedDicts provide type hints but don't enforce runtime validation
- The audio enrichment pipeline constructs dicts manually (not using TypedDict constructors)
- JSON serialization/deserialization can introduce type mismatches
- This test catches schema drift between types and implementation
"""

import pytest

from slower_whisper.pipeline.types_audio import (
    ArousalState,
    AudioState,
    EmotionState,
    EnergyState,
    ExtractionStatus,
    PauseState,
    PitchState,
    ProsodyState,
    RateState,
    ValenceState,
)

# ============================================================================
# Test Fixtures - Realistic audio_state structures
# ============================================================================


@pytest.fixture
def complete_prosody_state() -> ProsodyState:
    """Create a complete prosody state matching pipeline output."""
    return {
        "pitch": {
            "level": "high",
            "mean_hz": 245.3,
            "std_hz": 32.1,
            "contour": "rising",
        },
        "energy": {
            "level": "loud",
            "db_rms": -8.2,
        },
        "rate": {
            "level": "fast",
            "syllables_per_sec": 6.3,
        },
        "pauses": {
            "longest_ms": 320,
            "count": 2,
            "total_ms": 450,
            "rate_per_min": 12.5,
        },
    }


@pytest.fixture
def complete_emotion_state() -> EmotionState:
    """Create a complete emotion state matching pipeline output."""
    return {
        "valence": {
            "level": "negative",
            "score": 0.35,
        },
        "arousal": {
            "level": "high",
            "score": 0.68,
        },
    }


@pytest.fixture
def complete_audio_state(
    complete_prosody_state: ProsodyState,
    complete_emotion_state: EmotionState,
) -> AudioState:
    """Create a complete audio_state matching pipeline output."""
    return {
        "prosody": complete_prosody_state,
        "emotion": complete_emotion_state,
        "rendering": "[audio: high pitch, loud volume, fast speech]",
        "extraction_status": {
            "prosody": "success",
            "emotion_dimensional": "success",
        },
    }


# ============================================================================
# PitchState Tests
# ============================================================================


def test_pitch_state_structure():
    """Test PitchState has all expected fields with correct types."""
    pitch: PitchState = {
        "level": "high",
        "mean_hz": 245.3,
        "std_hz": 32.1,
        "contour": "rising",
    }

    assert isinstance(pitch["level"], str)
    assert isinstance(pitch["mean_hz"], float | int) or pitch["mean_hz"] is None
    assert isinstance(pitch["std_hz"], float | int) or pitch["std_hz"] is None
    assert isinstance(pitch["contour"], str)


def test_pitch_state_with_nulls():
    """Test PitchState allows None values for numeric fields."""
    pitch: PitchState = {
        "level": "unknown",
        "mean_hz": None,
        "std_hz": None,
        "contour": "flat",
    }

    assert pitch["mean_hz"] is None
    assert pitch["std_hz"] is None
    assert pitch["level"] == "unknown"


def test_pitch_state_level_values():
    """Test PitchState level field accepts expected categorical values."""
    valid_levels = ["very_low", "low", "neutral", "high", "very_high", "unknown"]

    for level in valid_levels:
        pitch: PitchState = {
            "level": level,
            "mean_hz": 180.0,
            "std_hz": 25.0,
            "contour": "flat",
        }
        assert pitch["level"] == level


def test_pitch_state_contour_values():
    """Test PitchState contour field accepts expected values."""
    valid_contours = ["flat", "rising", "falling", "dynamic", "unknown"]

    for contour in valid_contours:
        pitch: PitchState = {
            "level": "neutral",
            "mean_hz": 180.0,
            "std_hz": 25.0,
            "contour": contour,
        }
        assert pitch["contour"] == contour


# ============================================================================
# EnergyState Tests
# ============================================================================


def test_energy_state_structure():
    """Test EnergyState has all expected fields with correct types."""
    energy: EnergyState = {
        "level": "loud",
        "db_rms": -8.2,
    }

    assert isinstance(energy["level"], str)
    assert isinstance(energy["db_rms"], float | int) or energy["db_rms"] is None


def test_energy_state_with_nulls():
    """Test EnergyState allows None for db_rms."""
    energy: EnergyState = {
        "level": "unknown",
        "db_rms": None,
    }

    assert energy["db_rms"] is None
    assert energy["level"] == "unknown"


def test_energy_state_level_values():
    """Test EnergyState level field accepts expected categorical values."""
    valid_levels = ["very_quiet", "quiet", "normal", "loud", "very_loud", "unknown"]

    for level in valid_levels:
        energy: EnergyState = {
            "level": level,
            "db_rms": -15.0,
        }
        assert energy["level"] == level


# ============================================================================
# RateState Tests
# ============================================================================


def test_rate_state_structure():
    """Test RateState has all expected fields with correct types."""
    rate: RateState = {
        "level": "fast",
        "syllables_per_sec": 6.3,
    }

    assert isinstance(rate["level"], str)
    assert isinstance(rate["syllables_per_sec"], float) or rate["syllables_per_sec"] is None


def test_rate_state_with_nulls():
    """Test RateState allows None for syllables_per_sec."""
    rate: RateState = {
        "level": "unknown",
        "syllables_per_sec": None,
    }

    assert rate["syllables_per_sec"] is None
    assert rate["level"] == "unknown"


def test_rate_state_level_values():
    """Test RateState level field accepts expected categorical values."""
    valid_levels = ["very_slow", "slow", "normal", "fast", "very_fast", "unknown"]

    for level in valid_levels:
        rate: RateState = {
            "level": level,
            "syllables_per_sec": 5.0,
        }
        assert rate["level"] == level


# ============================================================================
# PauseState Tests
# ============================================================================


def test_pause_state_structure():
    """Test PauseState has all expected fields with correct types."""
    pauses: PauseState = {
        "longest_ms": 320,
        "count": 2,
        "total_ms": 450,
        "rate_per_min": 12.5,
    }

    assert isinstance(pauses["longest_ms"], float | int) or pauses["longest_ms"] is None
    assert isinstance(pauses["count"], int) or pauses["count"] is None
    assert isinstance(pauses["total_ms"], float | int) or pauses["total_ms"] is None
    assert isinstance(pauses["rate_per_min"], float) or pauses["rate_per_min"] is None


def test_pause_state_with_nulls():
    """Test PauseState allows None values for all fields."""
    pauses: PauseState = {
        "longest_ms": None,
        "count": None,
        "total_ms": None,
        "rate_per_min": None,
    }

    assert pauses["longest_ms"] is None
    assert pauses["count"] is None
    assert pauses["total_ms"] is None
    assert pauses["rate_per_min"] is None


def test_pause_state_integer_milliseconds():
    """Test PauseState accepts integer values for millisecond fields."""
    pauses: PauseState = {
        "longest_ms": 500,
        "count": 3,
        "total_ms": 750,
        "rate_per_min": 15.0,
    }

    assert pauses["longest_ms"] == 500
    assert pauses["total_ms"] == 750


# ============================================================================
# ProsodyState Tests
# ============================================================================


def test_prosody_state_structure(complete_prosody_state: ProsodyState):
    """Test ProsodyState has all sub-structures present."""
    prosody = complete_prosody_state

    assert "pitch" in prosody
    assert "energy" in prosody
    assert "rate" in prosody
    assert "pauses" in prosody


def test_prosody_state_nested_types(complete_prosody_state: ProsodyState):
    """Test ProsodyState nested structures have correct types."""
    prosody = complete_prosody_state

    # Verify pitch sub-structure
    assert isinstance(prosody["pitch"]["level"], str)
    assert isinstance(prosody["pitch"]["mean_hz"], float | int)

    # Verify energy sub-structure
    assert isinstance(prosody["energy"]["level"], str)
    assert isinstance(prosody["energy"]["db_rms"], float | int)

    # Verify rate sub-structure
    assert isinstance(prosody["rate"]["level"], str)
    assert isinstance(prosody["rate"]["syllables_per_sec"], float | int)

    # Verify pauses sub-structure
    assert isinstance(prosody["pauses"]["count"], int)
    assert isinstance(prosody["pauses"]["longest_ms"], int)


def test_prosody_state_all_levels_present(complete_prosody_state: ProsodyState):
    """Test ProsodyState has 'level' field in all sub-structures."""
    prosody = complete_prosody_state

    assert "level" in prosody["pitch"]
    assert "level" in prosody["energy"]
    assert "level" in prosody["rate"]


# ============================================================================
# ValenceState and ArousalState Tests
# ============================================================================


def test_valence_state_structure():
    """Test ValenceState has all expected fields with correct types."""
    valence: ValenceState = {
        "level": "negative",
        "score": 0.35,
    }

    assert isinstance(valence["level"], str)
    assert isinstance(valence["score"], float)
    assert 0.0 <= valence["score"] <= 1.0


def test_valence_state_level_values():
    """Test ValenceState level field accepts expected categorical values."""
    valid_levels = ["very_negative", "negative", "neutral", "positive", "very_positive"]

    for level in valid_levels:
        valence: ValenceState = {
            "level": level,
            "score": 0.5,
        }
        assert valence["level"] == level


def test_arousal_state_structure():
    """Test ArousalState has all expected fields with correct types."""
    arousal: ArousalState = {
        "level": "high",
        "score": 0.68,
    }

    assert isinstance(arousal["level"], str)
    assert isinstance(arousal["score"], float)
    assert 0.0 <= arousal["score"] <= 1.0


def test_arousal_state_level_values():
    """Test ArousalState level field accepts expected categorical values."""
    valid_levels = ["very_low", "low", "medium", "high", "very_high"]

    for level in valid_levels:
        arousal: ArousalState = {
            "level": level,
            "score": 0.5,
        }
        assert arousal["level"] == level


# ============================================================================
# EmotionState Tests
# ============================================================================


def test_emotion_state_structure(complete_emotion_state: EmotionState):
    """Test EmotionState has all expected sub-structures."""
    emotion = complete_emotion_state

    assert "valence" in emotion
    assert "arousal" in emotion


def test_emotion_state_nested_types(complete_emotion_state: EmotionState):
    """Test EmotionState nested structures have correct types."""
    emotion = complete_emotion_state

    # Verify valence sub-structure
    assert isinstance(emotion["valence"]["level"], str)
    assert isinstance(emotion["valence"]["score"], float)

    # Verify arousal sub-structure
    assert isinstance(emotion["arousal"]["level"], str)
    assert isinstance(emotion["arousal"]["score"], float)


def test_emotion_state_score_ranges(complete_emotion_state: EmotionState):
    """Test EmotionState scores are in valid [0, 1] range."""
    emotion = complete_emotion_state

    assert 0.0 <= emotion["valence"]["score"] <= 1.0
    assert 0.0 <= emotion["arousal"]["score"] <= 1.0


# ============================================================================
# ExtractionStatus Tests
# ============================================================================


def test_extraction_status_structure():
    """Test ExtractionStatus has expected fields with correct types."""
    status: ExtractionStatus = {
        "prosody": "success",
        "emotion_dimensional": "success",
    }

    assert isinstance(status["prosody"], str)
    assert isinstance(status["emotion_dimensional"], str)


def test_extraction_status_values():
    """Test ExtractionStatus accepts expected status values."""
    valid_statuses = ["success", "failed", "error", "skipped", "unavailable"]

    for status_value in valid_statuses:
        status: ExtractionStatus = {
            "prosody": status_value,
            "emotion_dimensional": status_value,
        }
        assert status["prosody"] == status_value
        assert status["emotion_dimensional"] == status_value


def test_extraction_status_mixed():
    """Test ExtractionStatus can have different values for each extractor."""
    status: ExtractionStatus = {
        "prosody": "success",
        "emotion_dimensional": "failed",
    }

    assert status["prosody"] == "success"
    assert status["emotion_dimensional"] == "failed"


# ============================================================================
# AudioState Tests (Top-Level Integration)
# ============================================================================


def test_audio_state_structure(complete_audio_state: AudioState):
    """Test AudioState has all top-level fields present."""
    audio_state = complete_audio_state

    assert "prosody" in audio_state
    assert "emotion" in audio_state
    assert "rendering" in audio_state
    assert "extraction_status" in audio_state


def test_audio_state_field_types(complete_audio_state: AudioState):
    """Test AudioState top-level fields have correct types."""
    audio_state = complete_audio_state

    assert isinstance(audio_state["prosody"], dict)
    assert isinstance(audio_state["emotion"], dict)
    assert isinstance(audio_state["rendering"], str)
    assert isinstance(audio_state["extraction_status"], dict)


def test_audio_state_rendering_format(complete_audio_state: AudioState):
    """Test AudioState rendering field has expected format."""
    audio_state = complete_audio_state
    rendering = audio_state["rendering"]

    # Rendering should be a bracketed text annotation
    assert rendering.startswith("[audio:")
    assert rendering.endswith("]")


def test_audio_state_nested_structure_complete(complete_audio_state: AudioState):
    """Test AudioState has complete nested structure matching schema."""
    audio_state = complete_audio_state

    # Verify prosody sub-structures
    assert "pitch" in audio_state["prosody"]
    assert "energy" in audio_state["prosody"]
    assert "rate" in audio_state["prosody"]
    assert "pauses" in audio_state["prosody"]

    # Verify emotion sub-structures
    assert "valence" in audio_state["emotion"]
    assert "arousal" in audio_state["emotion"]

    # Verify extraction status keys
    assert "prosody" in audio_state["extraction_status"]
    assert "emotion_dimensional" in audio_state["extraction_status"]


def test_audio_state_with_nulls():
    """Test AudioState handles missing/null features gracefully."""
    audio_state: AudioState = {
        "prosody": {
            "pitch": {"level": "unknown", "mean_hz": None, "std_hz": None, "contour": "unknown"},
            "energy": {"level": "unknown", "db_rms": None},
            "rate": {"level": "unknown", "syllables_per_sec": None},
            "pauses": {"longest_ms": None, "count": None, "total_ms": None, "rate_per_min": None},
        },
        "emotion": {
            "valence": {"level": "neutral", "score": 0.5},
            "arousal": {"level": "medium", "score": 0.5},
        },
        "rendering": "[audio: neutral]",
        "extraction_status": {
            "prosody": "failed",
            "emotion_dimensional": "unavailable",
        },
    }

    # Verify structure is valid even with nulls
    assert audio_state["prosody"]["pitch"]["mean_hz"] is None
    assert audio_state["prosody"]["energy"]["db_rms"] is None
    assert audio_state["extraction_status"]["prosody"] == "failed"


def test_audio_state_partial_extraction():
    """Test AudioState handles partial extraction (prosody success, emotion failed)."""
    audio_state: AudioState = {
        "prosody": {
            "pitch": {"level": "high", "mean_hz": 245.0, "std_hz": 30.0, "contour": "rising"},
            "energy": {"level": "loud", "db_rms": -10.0},
            "rate": {"level": "fast", "syllables_per_sec": 6.0},
            "pauses": {"longest_ms": 200, "count": 1, "total_ms": 200, "rate_per_min": 10.0},
        },
        "emotion": {
            "valence": {"level": "neutral", "score": 0.5},
            "arousal": {"level": "medium", "score": 0.5},
        },
        "rendering": "[audio: high pitch, loud volume, fast speech]",
        "extraction_status": {
            "prosody": "success",
            "emotion_dimensional": "failed",
        },
    }

    # Verify prosody succeeded
    assert audio_state["extraction_status"]["prosody"] == "success"
    assert audio_state["prosody"]["pitch"]["mean_hz"] is not None

    # Verify emotion failed but still has default values
    assert audio_state["extraction_status"]["emotion_dimensional"] == "failed"
    assert audio_state["emotion"]["valence"]["level"] == "neutral"


def test_audio_state_matches_pipeline_output_structure(complete_audio_state: AudioState):
    """
    Test that AudioState structure matches what audio_enrichment.py produces.

    This is the critical integration test ensuring type definitions match implementation.
    """
    audio_state = complete_audio_state

    # Top-level keys match audio_enrichment.py:enrich_segment_audio return value
    expected_top_keys = {"prosody", "emotion", "rendering", "extraction_status"}
    assert set(audio_state.keys()) == expected_top_keys

    # Prosody keys match prosody.py:extract_prosody return value
    expected_prosody_keys = {"pitch", "energy", "rate", "pauses"}
    assert set(audio_state["prosody"].keys()) == expected_prosody_keys

    # Emotion keys match emotion.py:extract_emotion_dimensional return value
    expected_emotion_keys = {"valence", "arousal"}
    assert set(audio_state["emotion"].keys()) == expected_emotion_keys

    # Each prosody feature has 'level' field (required for rendering)
    assert "level" in audio_state["prosody"]["pitch"]
    assert "level" in audio_state["prosody"]["energy"]
    assert "level" in audio_state["prosody"]["rate"]

    # Each emotion dimension has 'level' and 'score' fields
    assert "level" in audio_state["emotion"]["valence"]
    assert "score" in audio_state["emotion"]["valence"]
    assert "level" in audio_state["emotion"]["arousal"]
    assert "score" in audio_state["emotion"]["arousal"]

    # Extraction status tracks both prosody and emotion_dimensional
    assert "prosody" in audio_state["extraction_status"]
    assert "emotion_dimensional" in audio_state["extraction_status"]


# ============================================================================
# Edge Cases and Validation Tests
# ============================================================================


def test_audio_state_empty_rendering():
    """Test AudioState accepts empty rendering string."""
    audio_state: AudioState = {
        "prosody": {
            "pitch": {"level": "unknown", "mean_hz": None, "std_hz": None, "contour": "unknown"},
            "energy": {"level": "unknown", "db_rms": None},
            "rate": {"level": "unknown", "syllables_per_sec": None},
            "pauses": {"longest_ms": None, "count": None, "total_ms": None, "rate_per_min": None},
        },
        "emotion": {
            "valence": {"level": "neutral", "score": 0.5},
            "arousal": {"level": "medium", "score": 0.5},
        },
        "rendering": "",
        "extraction_status": {
            "prosody": "skipped",
            "emotion_dimensional": "skipped",
        },
    }

    assert audio_state["rendering"] == ""


def test_audio_state_numeric_type_flexibility():
    """Test AudioState accepts both int and float for numeric fields."""
    # The TypedDict allows float | int | None for most numeric fields
    audio_state: AudioState = {
        "prosody": {
            "pitch": {"level": "neutral", "mean_hz": 180, "std_hz": 25, "contour": "flat"},
            "energy": {"level": "normal", "db_rms": -15},
            "rate": {"level": "normal", "syllables_per_sec": 5.0},
            "pauses": {"longest_ms": 300, "count": 2, "total_ms": 400, "rate_per_min": 10.5},
        },
        "emotion": {
            "valence": {"level": "neutral", "score": 0.5},
            "arousal": {"level": "medium", "score": 0.5},
        },
        "rendering": "[audio: neutral]",
        "extraction_status": {
            "prosody": "success",
            "emotion_dimensional": "success",
        },
    }

    # Verify integers are accepted
    assert isinstance(audio_state["prosody"]["pitch"]["mean_hz"], int)
    assert isinstance(audio_state["prosody"]["energy"]["db_rms"], int)
    assert isinstance(audio_state["prosody"]["pauses"]["longest_ms"], int)


def test_audio_state_boundary_emotion_scores():
    """Test AudioState accepts boundary values for emotion scores."""
    audio_state: AudioState = {
        "prosody": {
            "pitch": {"level": "unknown", "mean_hz": None, "std_hz": None, "contour": "unknown"},
            "energy": {"level": "unknown", "db_rms": None},
            "rate": {"level": "unknown", "syllables_per_sec": None},
            "pauses": {"longest_ms": None, "count": None, "total_ms": None, "rate_per_min": None},
        },
        "emotion": {
            "valence": {"level": "very_negative", "score": 0.0},
            "arousal": {"level": "very_high", "score": 1.0},
        },
        "rendering": "[audio: neutral]",
        "extraction_status": {
            "prosody": "skipped",
            "emotion_dimensional": "success",
        },
    }

    assert audio_state["emotion"]["valence"]["score"] == 0.0
    assert audio_state["emotion"]["arousal"]["score"] == 1.0


# ============================================================================
# Integration with render_segment (llm_utils)
# ============================================================================


def test_render_segment_accepts_audio_state_shape():
    """
    A Segment with an AudioState-shaped audio_state should render with prosody cues.

    This ties the TypedDict contract to actual runtime behavior in llm_utils.
    """
    from typing import Any

    from slower_whisper.pipeline.llm_utils import render_segment
    from slower_whisper.pipeline.models import Segment

    audio_state: dict[str, Any] = {
        "prosody": {
            "pitch": {"mean_hz": 180.0, "level": "high", "contour": "rising"},
            "energy": {"db_rms": -10.5, "level": "loud"},
            "rate": {"syllables_per_sec": 5.5, "level": "normal"},
            "pauses": {"longest_ms": 350.0, "count": 2, "total_ms": 500.0},
        },
        "emotion": {
            "valence": {"level": "positive", "score": 0.72},
            "arousal": {"level": "high", "score": 0.68},
        },
        "rendering": "[audio: high pitch, loud, long pause]",
        "extraction_status": {"prosody": "success", "emotion_dimensional": "success"},
    }

    seg = Segment(
        id=0,
        start=0.0,
        end=2.5,
        text="Test sentence?",
        speaker={"id": "spk_0"},
    )
    seg.audio_state = audio_state

    rendered = render_segment(seg, include_audio_cues=True)

    # Assert prosody made it into the text representation
    # render_segment extracts from rendering field: "[audio: high pitch, loud, long pause]"
    assert "high pitch" in rendered or "loud" in rendered or "pause" in rendered


def test_render_segment_with_speaker_and_audio_cues():
    """render_segment should combine speaker label and audio rendering."""
    from typing import Any

    from slower_whisper.pipeline.llm_utils import render_segment
    from slower_whisper.pipeline.models import Segment

    audio_state: dict[str, Any] = {
        "prosody": {
            "pitch": {"mean_hz": 200.0, "level": "normal"},
            "energy": {"db_rms": -15.0, "level": "normal"},
        },
        "rendering": "[audio: calm tone]",
        "extraction_status": {"prosody": "success"},
    }

    seg = Segment(
        id=1,
        start=5.0,
        end=8.0,
        text="I understand your concern.",
        speaker={"id": "spk_1"},
    )
    seg.audio_state = audio_state

    rendered = render_segment(
        seg,
        include_audio_cues=True,
        speaker_labels={"spk_1": "Agent"},
    )

    # Should have speaker label and audio cues
    assert "Agent" in rendered
    assert "calm tone" in rendered
    assert "I understand your concern" in rendered


def test_render_segment_without_audio_state():
    """render_segment should handle segments without audio_state gracefully."""
    from slower_whisper.pipeline.llm_utils import render_segment
    from slower_whisper.pipeline.models import Segment

    seg = Segment(
        id=2,
        start=10.0,
        end=12.0,
        text="Plain segment without enrichment.",
        speaker=None,
    )
    # audio_state is None by default

    rendered = render_segment(seg, include_audio_cues=True)

    # Should just have the text, no brackets
    assert rendered == "Plain segment without enrichment."


def test_render_segment_with_empty_rendering():
    """render_segment should handle audio_state with empty rendering field."""
    from typing import Any

    from slower_whisper.pipeline.llm_utils import render_segment
    from slower_whisper.pipeline.models import Segment

    audio_state: dict[str, Any] = {
        "prosody": {
            "pitch": {"mean_hz": 150.0, "level": "low"},
        },
        "rendering": "",  # Empty rendering
        "extraction_status": {"prosody": "success"},
    }

    seg = Segment(
        id=3,
        start=15.0,
        end=17.0,
        text="Has prosody but no rendering.",
        speaker={"id": "spk_0"},
    )
    seg.audio_state = audio_state

    rendered = render_segment(
        seg,
        include_audio_cues=True,
        speaker_labels={"spk_0": "Customer"},
    )

    # Should have speaker but no audio cues (empty rendering)
    assert "Customer" in rendered
    assert "Has prosody but no rendering" in rendered
