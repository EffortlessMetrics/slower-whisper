"""
Tests and examples for audio_rendering module.

Demonstrates the render_audio_state function with various audio states
and feature combinations.
"""

from transcription.audio_rendering import render_audio_features_detailed, render_audio_state


def test_render_neutral_state():
    """Test rendering of neutral/empty audio state."""
    result = render_audio_state({})
    assert result == "[audio: neutral]"
    print(f"Empty state: {result}")

    # Explicitly neutral features
    result = render_audio_state(
        {"prosody": {"pitch": "neutral", "volume": "medium"}, "emotion": {"tone": "normal"}}
    )
    assert result == "[audio: neutral]"
    print(f"Explicitly neutral: {result}")


def test_render_excited_state():
    """Test rendering of excited/energetic audio state."""
    excited = {
        "prosody": {"pitch": "high", "volume": "loud", "speech_rate": "fast"},
        "emotion": {"tone": "excited", "confidence": 0.95},
    }
    result = render_audio_state(excited)
    assert result == "[audio: high pitch, loud volume, fast speech, excited tone]"
    print(f"Excited state: {result}")
    assert len(result) < 80  # Should be concise


def test_render_hesitant_state():
    """Test rendering of hesitant/uncertain audio state."""
    hesitant = {
        "prosody": {"pauses": "moderate"},
        "emotion": {"tone": "hesitant", "confidence": 0.45},
    }
    result = render_audio_state(hesitant)
    assert "[audio:" in result
    assert "moderate pauses" in result
    assert "hesitant tone" in result
    assert "possibly uncertain" in result
    print(f"Hesitant state: {result}")


def test_render_calm_state():
    """Test rendering of calm/relaxed audio state."""
    calm = {
        "prosody": {"pitch": "low", "volume": "quiet", "speech_rate": "slow"},
        "emotion": {"tone": "calm", "confidence": 0.85},
    }
    result = render_audio_state(calm)
    assert result == "[audio: low pitch, quiet volume, slow speech, calm tone]"
    print(f"Calm state: {result}")
    assert len(result) < 80


def test_render_stressed_state():
    """Test rendering of stressed/tense audio state."""
    stressed = {
        "prosody": {
            "pitch": "high",
            "volume": "loud",
            "pauses": "frequent",
            "speech_rate": "rapid",
        },
        "voice_quality": {"stress_level": "high", "energy": "high"},
        "emotion": {"tone": "stressed", "confidence": 0.78},
    }
    result = render_audio_state(stressed)
    print(f"Stressed state: {result}")
    assert "high pitch" in result
    assert "loud volume" in result
    assert "frequent pauses" in result
    assert "high stress" in result
    assert "stressed tone" in result


def test_render_sad_state():
    """Test rendering of sad/melancholic audio state."""
    sad = {
        "prosody": {"pitch": "low", "volume": "quiet", "speech_rate": "slow", "pauses": "long"},
        "emotion": {"tone": "sad", "confidence": 0.82},
    }
    result = render_audio_state(sad)
    print(f"Sad state: {result}")
    assert "low pitch" in result
    assert "quiet volume" in result
    assert "slow speech" in result
    assert "long pauses" in result
    assert "sad tone" in result


def test_render_uncertain_state():
    """Test rendering with low confidence uncertainty."""
    uncertain = {"emotion": {"tone": "confused", "confidence": 0.35}}
    result = render_audio_state(uncertain)
    print(f"Uncertain state: {result}")
    assert "confused tone" in result
    assert "possibly uncertain" in result


def test_render_partial_features():
    """Test rendering with only some features present."""
    partial1 = {"prosody": {"pitch": "high"}}
    result = render_audio_state(partial1)
    assert result == "[audio: high pitch]"
    print(f"Partial (pitch only): {result}")

    partial2 = {"emotion": {"tone": "enthusiastic"}}
    result = render_audio_state(partial2)
    assert result == "[audio: enthusiastic tone]"
    print(f"Partial (tone only): {result}")

    partial3 = {"voice_quality": {"clarity": "muffled"}}
    result = render_audio_state(partial3)
    assert result == "[audio: muffled clarity]"
    print(f"Partial (clarity only): {result}")


def test_render_mixed_confidence():
    """Test rendering with mixed confidence levels."""
    low_conf = {"emotion": {"tone": "angry", "confidence": 0.65}}
    result = render_audio_state(low_conf)
    print(f"Medium confidence: {result}")
    assert "somewhat uncertain" in result

    high_conf = {"emotion": {"tone": "happy", "confidence": 0.92}}
    result = render_audio_state(high_conf)
    print(f"High confidence: {result}")
    assert "uncertain" not in result


def test_render_none_values():
    """Test that None values are handled gracefully."""
    with_nones = {"prosody": {"pitch": None, "volume": "loud"}, "emotion": {"tone": None}}
    result = render_audio_state(with_nones)
    print(f"With None values: {result}")
    assert result == "[audio: loud volume]"


def test_detailed_rendering():
    """Test the detailed rendering function."""
    state = {
        "prosody": {"pitch": "high", "volume": "loud", "speech_rate": "fast"},
        "emotion": {"tone": "excited"},
        "voice_quality": {"energy": "high"},
    }
    detailed = render_audio_features_detailed(state)

    print("\nDetailed rendering:")
    print(f"  Prosody: {detailed['prosody']}")
    print(f"  Emotion: {detailed['emotion']}")
    print(f"  Voice quality: {detailed['voice_quality']}")

    assert detailed["prosody"] == ["high pitch", "loud volume", "fast speech"]
    assert detailed["emotion"] == ["excited tone"]
    assert detailed["voice_quality"] == ["high energy"]


def test_length_constraint():
    """Test that rendered annotations are reasonably concise."""
    # Most annotations should be under 80 characters
    test_cases = [
        {"prosody": {"pitch": "high", "volume": "loud"}},
        {"emotion": {"tone": "excited"}},
        {"prosody": {"speech_rate": "fast", "pauses": "minimal"}},
        {"emotion": {"tone": "calm"}, "voice_quality": {"stress_level": "low"}},
    ]

    for state in test_cases:
        result = render_audio_state(state)
        print(f"Length {len(result):2d}: {result}")
        # Most should be under 80 chars (excluding the most complex cases)
        assert len(result) < 100


if __name__ == "__main__":
    print("Audio Rendering Examples and Tests")
    print("=" * 60)

    print("\n1. Neutral state:")
    test_render_neutral_state()

    print("\n2. Excited state:")
    test_render_excited_state()

    print("\n3. Hesitant state:")
    test_render_hesitant_state()

    print("\n4. Calm state:")
    test_render_calm_state()

    print("\n5. Stressed state:")
    test_render_stressed_state()

    print("\n6. Sad state:")
    test_render_sad_state()

    print("\n7. Uncertain state:")
    test_render_uncertain_state()

    print("\n8. Partial features:")
    test_render_partial_features()

    print("\n9. Mixed confidence:")
    test_render_mixed_confidence()

    print("\n10. None values:")
    test_render_none_values()

    print("\n11. Detailed rendering:")
    test_detailed_rendering()

    print("\n12. Length constraint:")
    test_length_constraint()

    print("\n" + "=" * 60)
    print("All tests passed!")
