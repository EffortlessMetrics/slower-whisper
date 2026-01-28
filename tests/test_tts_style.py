"""Tests for TTS style metadata computation (v2.1.0).

This test suite validates the TTS style computation:
1. compute_tts_style with None inputs (returns neutral defaults)
2. compute_tts_style with prosody only
3. compute_tts_style with emotion only
4. compute_tts_style with both prosody and emotion
5. Energy thresholds (quiet/normal/loud)
6. Speech rate thresholds (slow/normal/fast)
7. Valence thresholds (negative/neutral/positive)
8. Arousal thresholds (low/medium/high)
9. Recommended response mode logic (de-escalate, calm, match, neutral)
10. to_dict() serialization
"""

from __future__ import annotations

from transcription.tts_style import (
    TTSStyleMetadata,
    compute_tts_style,
)

# =============================================================================
# 1. Test compute_tts_style with None Inputs
# =============================================================================


class TestComputeTtsStyleNoneInputs:
    """Tests for compute_tts_style with None inputs."""

    def test_both_none_returns_neutral_defaults(self) -> None:
        """Both prosody and emotion None returns neutral defaults."""
        result = compute_tts_style(None, None)

        assert result.energy_level == "normal"
        assert result.speech_rate == "normal"
        assert result.valence == "neutral"
        assert result.arousal == "medium"
        assert result.recommended_response_mode == "neutral"

    def test_empty_dicts_same_as_none(self) -> None:
        """Empty dicts behave same as None (missing keys)."""
        result = compute_tts_style({}, {})

        assert result.energy_level == "normal"
        assert result.speech_rate == "normal"
        assert result.valence == "neutral"
        assert result.arousal == "medium"

    def test_partial_prosody_missing_keys(self) -> None:
        """Prosody dict with missing keys uses defaults for missing."""
        result = compute_tts_style({"energy": 0.8}, None)  # No speech_rate

        assert result.energy_level == "loud"  # From provided energy
        assert result.speech_rate == "normal"  # Default (missing)

    def test_partial_emotion_missing_keys(self) -> None:
        """Emotion dict with missing keys uses defaults for missing."""
        result = compute_tts_style(None, {"valence": -0.5})  # No arousal

        assert result.valence == "negative"  # From provided valence
        assert result.arousal == "medium"  # Default (missing)


# =============================================================================
# 2. Test compute_tts_style with Prosody Only
# =============================================================================


class TestComputeTtsStyleProsodyOnly:
    """Tests for compute_tts_style with prosody only."""

    def test_prosody_only_energy(self) -> None:
        """Prosody with only energy is processed correctly."""
        result = compute_tts_style({"energy": 0.5}, None)

        assert result.energy_level == "normal"
        # Emotion defaults
        assert result.valence == "neutral"
        assert result.arousal == "medium"

    def test_prosody_only_speech_rate(self) -> None:
        """Prosody with only speech_rate is processed correctly."""
        result = compute_tts_style({"speech_rate": 1.5}, None)

        assert result.speech_rate == "fast"
        # Energy defaults
        assert result.energy_level == "normal"

    def test_prosody_with_both_values(self) -> None:
        """Prosody with both energy and speech_rate is processed."""
        result = compute_tts_style({"energy": 0.2, "speech_rate": 0.6}, None)

        assert result.energy_level == "quiet"
        assert result.speech_rate == "slow"


# =============================================================================
# 3. Test compute_tts_style with Emotion Only
# =============================================================================


class TestComputeTtsStyleEmotionOnly:
    """Tests for compute_tts_style with emotion only."""

    def test_emotion_only_valence(self) -> None:
        """Emotion with only valence is processed correctly."""
        result = compute_tts_style(None, {"valence": 0.5})

        assert result.valence == "positive"
        # Prosody defaults
        assert result.energy_level == "normal"
        assert result.speech_rate == "normal"

    def test_emotion_only_arousal(self) -> None:
        """Emotion with only arousal is processed correctly."""
        result = compute_tts_style(None, {"arousal": 0.9})

        assert result.arousal == "high"
        # Valence defaults
        assert result.valence == "neutral"

    def test_emotion_with_both_values(self) -> None:
        """Emotion with both valence and arousal is processed."""
        result = compute_tts_style(None, {"valence": -0.6, "arousal": 0.8})

        assert result.valence == "negative"
        assert result.arousal == "high"


# =============================================================================
# 4. Test compute_tts_style with Both Prosody and Emotion
# =============================================================================


class TestComputeTtsStyleBothInputs:
    """Tests for compute_tts_style with both prosody and emotion."""

    def test_both_inputs_all_values(self) -> None:
        """Both prosody and emotion with all values."""
        prosody = {"energy": 0.8, "speech_rate": 1.3}
        emotion = {"valence": 0.5, "arousal": 0.6}

        result = compute_tts_style(prosody, emotion)

        assert result.energy_level == "loud"
        assert result.speech_rate == "fast"
        assert result.valence == "positive"
        assert result.arousal == "medium"

    def test_mixed_extreme_values(self) -> None:
        """Mix of extreme values from both inputs."""
        prosody = {"energy": 0.1, "speech_rate": 1.5}
        emotion = {"valence": -0.8, "arousal": 0.1}

        result = compute_tts_style(prosody, emotion)

        assert result.energy_level == "quiet"
        assert result.speech_rate == "fast"
        assert result.valence == "negative"
        assert result.arousal == "low"


# =============================================================================
# 5. Test Energy Thresholds (quiet/normal/loud)
# =============================================================================


class TestEnergyThresholds:
    """Tests for energy level threshold mapping."""

    def test_energy_quiet_below_0_3(self) -> None:
        """Energy < 0.3 maps to 'quiet'."""
        assert compute_tts_style({"energy": 0.0}, None).energy_level == "quiet"
        assert compute_tts_style({"energy": 0.1}, None).energy_level == "quiet"
        assert compute_tts_style({"energy": 0.29}, None).energy_level == "quiet"

    def test_energy_normal_between_0_3_and_0_7(self) -> None:
        """Energy between 0.3 and 0.7 maps to 'normal'."""
        assert compute_tts_style({"energy": 0.3}, None).energy_level == "normal"
        assert compute_tts_style({"energy": 0.5}, None).energy_level == "normal"
        assert compute_tts_style({"energy": 0.7}, None).energy_level == "normal"

    def test_energy_loud_above_0_7(self) -> None:
        """Energy > 0.7 maps to 'loud'."""
        assert compute_tts_style({"energy": 0.71}, None).energy_level == "loud"
        assert compute_tts_style({"energy": 0.9}, None).energy_level == "loud"
        assert compute_tts_style({"energy": 1.0}, None).energy_level == "loud"

    def test_energy_boundary_0_3(self) -> None:
        """Energy exactly 0.3 is 'normal' (not quiet)."""
        result = compute_tts_style({"energy": 0.3}, None)
        assert result.energy_level == "normal"

    def test_energy_boundary_0_7(self) -> None:
        """Energy exactly 0.7 is 'normal' (not loud)."""
        result = compute_tts_style({"energy": 0.7}, None)
        assert result.energy_level == "normal"


# =============================================================================
# 6. Test Speech Rate Thresholds (slow/normal/fast)
# =============================================================================


class TestSpeechRateThresholds:
    """Tests for speech rate threshold mapping."""

    def test_speech_rate_slow_below_0_8(self) -> None:
        """Speech rate < 0.8 maps to 'slow'."""
        assert compute_tts_style({"speech_rate": 0.0}, None).speech_rate == "slow"
        assert compute_tts_style({"speech_rate": 0.5}, None).speech_rate == "slow"
        assert compute_tts_style({"speech_rate": 0.79}, None).speech_rate == "slow"

    def test_speech_rate_normal_between_0_8_and_1_2(self) -> None:
        """Speech rate between 0.8 and 1.2 maps to 'normal'."""
        assert compute_tts_style({"speech_rate": 0.8}, None).speech_rate == "normal"
        assert compute_tts_style({"speech_rate": 1.0}, None).speech_rate == "normal"
        assert compute_tts_style({"speech_rate": 1.2}, None).speech_rate == "normal"

    def test_speech_rate_fast_above_1_2(self) -> None:
        """Speech rate > 1.2 maps to 'fast'."""
        assert compute_tts_style({"speech_rate": 1.21}, None).speech_rate == "fast"
        assert compute_tts_style({"speech_rate": 1.5}, None).speech_rate == "fast"
        assert compute_tts_style({"speech_rate": 2.0}, None).speech_rate == "fast"

    def test_speech_rate_boundary_0_8(self) -> None:
        """Speech rate exactly 0.8 is 'normal' (not slow)."""
        result = compute_tts_style({"speech_rate": 0.8}, None)
        assert result.speech_rate == "normal"

    def test_speech_rate_boundary_1_2(self) -> None:
        """Speech rate exactly 1.2 is 'normal' (not fast)."""
        result = compute_tts_style({"speech_rate": 1.2}, None)
        assert result.speech_rate == "normal"


# =============================================================================
# 7. Test Valence Thresholds (negative/neutral/positive)
# =============================================================================


class TestValenceThresholds:
    """Tests for valence threshold mapping."""

    def test_valence_negative_below_minus_0_3(self) -> None:
        """Valence < -0.3 maps to 'negative'."""
        assert compute_tts_style(None, {"valence": -1.0}).valence == "negative"
        assert compute_tts_style(None, {"valence": -0.5}).valence == "negative"
        assert compute_tts_style(None, {"valence": -0.31}).valence == "negative"

    def test_valence_neutral_between_minus_0_3_and_0_3(self) -> None:
        """Valence between -0.3 and 0.3 maps to 'neutral'."""
        assert compute_tts_style(None, {"valence": -0.3}).valence == "neutral"
        assert compute_tts_style(None, {"valence": 0.0}).valence == "neutral"
        assert compute_tts_style(None, {"valence": 0.3}).valence == "neutral"

    def test_valence_positive_above_0_3(self) -> None:
        """Valence > 0.3 maps to 'positive'."""
        assert compute_tts_style(None, {"valence": 0.31}).valence == "positive"
        assert compute_tts_style(None, {"valence": 0.7}).valence == "positive"
        assert compute_tts_style(None, {"valence": 1.0}).valence == "positive"

    def test_valence_boundary_minus_0_3(self) -> None:
        """Valence exactly -0.3 is 'neutral' (not negative)."""
        result = compute_tts_style(None, {"valence": -0.3})
        assert result.valence == "neutral"

    def test_valence_boundary_0_3(self) -> None:
        """Valence exactly 0.3 is 'neutral' (not positive)."""
        result = compute_tts_style(None, {"valence": 0.3})
        assert result.valence == "neutral"


# =============================================================================
# 8. Test Arousal Thresholds (low/medium/high)
# =============================================================================


class TestArousalThresholds:
    """Tests for arousal threshold mapping."""

    def test_arousal_low_below_0_3(self) -> None:
        """Arousal < 0.3 maps to 'low'."""
        assert compute_tts_style(None, {"arousal": 0.0}).arousal == "low"
        assert compute_tts_style(None, {"arousal": 0.1}).arousal == "low"
        assert compute_tts_style(None, {"arousal": 0.29}).arousal == "low"

    def test_arousal_medium_between_0_3_and_0_7(self) -> None:
        """Arousal between 0.3 and 0.7 maps to 'medium'."""
        assert compute_tts_style(None, {"arousal": 0.3}).arousal == "medium"
        assert compute_tts_style(None, {"arousal": 0.5}).arousal == "medium"
        assert compute_tts_style(None, {"arousal": 0.7}).arousal == "medium"

    def test_arousal_high_above_0_7(self) -> None:
        """Arousal > 0.7 maps to 'high'."""
        assert compute_tts_style(None, {"arousal": 0.71}).arousal == "high"
        assert compute_tts_style(None, {"arousal": 0.9}).arousal == "high"
        assert compute_tts_style(None, {"arousal": 1.0}).arousal == "high"

    def test_arousal_boundary_0_3(self) -> None:
        """Arousal exactly 0.3 is 'medium' (not low)."""
        result = compute_tts_style(None, {"arousal": 0.3})
        assert result.arousal == "medium"

    def test_arousal_boundary_0_7(self) -> None:
        """Arousal exactly 0.7 is 'medium' (not high)."""
        result = compute_tts_style(None, {"arousal": 0.7})
        assert result.arousal == "medium"


# =============================================================================
# 9. Test Recommended Response Mode Logic
# =============================================================================


class TestRecommendedResponseMode:
    """Tests for recommended_response_mode computation."""

    def test_de_escalate_negative_high_arousal(self) -> None:
        """Negative valence + high arousal -> de-escalate."""
        result = compute_tts_style(None, {"valence": -0.5, "arousal": 0.9})

        assert result.valence == "negative"
        assert result.arousal == "high"
        assert result.recommended_response_mode == "de-escalate"

    def test_calm_neutral_high_arousal(self) -> None:
        """Neutral valence + high arousal -> calm."""
        result = compute_tts_style(None, {"valence": 0.0, "arousal": 0.9})

        assert result.valence == "neutral"
        assert result.arousal == "high"
        assert result.recommended_response_mode == "calm"

    def test_calm_positive_high_arousal(self) -> None:
        """Positive valence + high arousal -> calm (not de-escalate)."""
        result = compute_tts_style(None, {"valence": 0.5, "arousal": 0.9})

        assert result.valence == "positive"
        assert result.arousal == "high"
        assert result.recommended_response_mode == "calm"

    def test_match_positive_non_high_arousal(self) -> None:
        """Positive valence + non-high arousal -> match."""
        result = compute_tts_style(None, {"valence": 0.5, "arousal": 0.5})

        assert result.valence == "positive"
        assert result.arousal == "medium"
        assert result.recommended_response_mode == "match"

    def test_match_positive_low_arousal(self) -> None:
        """Positive valence + low arousal -> match."""
        result = compute_tts_style(None, {"valence": 0.5, "arousal": 0.1})

        assert result.valence == "positive"
        assert result.arousal == "low"
        assert result.recommended_response_mode == "match"

    def test_neutral_neutral_valence_medium_arousal(self) -> None:
        """Neutral valence + medium arousal -> neutral."""
        result = compute_tts_style(None, {"valence": 0.0, "arousal": 0.5})

        assert result.valence == "neutral"
        assert result.arousal == "medium"
        assert result.recommended_response_mode == "neutral"

    def test_neutral_negative_non_high_arousal(self) -> None:
        """Negative valence + non-high arousal -> neutral."""
        result = compute_tts_style(None, {"valence": -0.5, "arousal": 0.5})

        assert result.valence == "negative"
        assert result.arousal == "medium"
        assert result.recommended_response_mode == "neutral"

    def test_neutral_default_case(self) -> None:
        """Default case (no emotion) -> neutral."""
        result = compute_tts_style(None, None)

        assert result.recommended_response_mode == "neutral"

    def test_response_mode_priority_de_escalate(self) -> None:
        """De-escalate has highest priority over other modes."""
        # Negative + high arousal should always be de-escalate
        result = compute_tts_style(
            {"energy": 0.1, "speech_rate": 0.5},  # Quiet and slow
            {"valence": -0.5, "arousal": 0.9},  # Negative and high arousal
        )

        assert result.recommended_response_mode == "de-escalate"


# =============================================================================
# 10. Test to_dict() Serialization
# =============================================================================


class TestToDictSerialization:
    """Tests for TTSStyleMetadata.to_dict() method."""

    def test_to_dict_returns_dict(self) -> None:
        """to_dict returns a dictionary."""
        result = compute_tts_style(None, None)

        d = result.to_dict()

        assert isinstance(d, dict)

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict contains all expected fields."""
        result = compute_tts_style(
            {"energy": 0.5, "speech_rate": 1.0},
            {"valence": 0.2, "arousal": 0.5},
        )

        d = result.to_dict()

        assert "energy_level" in d
        assert "speech_rate" in d
        assert "valence" in d
        assert "arousal" in d
        assert "recommended_response_mode" in d
        assert len(d) == 5

    def test_to_dict_values_match_attributes(self) -> None:
        """to_dict values match the object attributes."""
        result = compute_tts_style(
            {"energy": 0.8, "speech_rate": 1.3},
            {"valence": -0.5, "arousal": 0.9},
        )

        d = result.to_dict()

        assert d["energy_level"] == result.energy_level
        assert d["speech_rate"] == result.speech_rate
        assert d["valence"] == result.valence
        assert d["arousal"] == result.arousal
        assert d["recommended_response_mode"] == result.recommended_response_mode

    def test_to_dict_values_are_strings(self) -> None:
        """to_dict values are all strings."""
        result = compute_tts_style({"energy": 0.5}, {"valence": 0.0})

        d = result.to_dict()

        for key, value in d.items():
            assert isinstance(value, str), f"{key} should be a string"

    def test_to_dict_serializable_to_json(self) -> None:
        """to_dict result is JSON-serializable."""
        import json

        result = compute_tts_style({"energy": 0.8}, {"valence": -0.6, "arousal": 0.9})

        d = result.to_dict()
        json_str = json.dumps(d)

        assert isinstance(json_str, str)
        # Verify round-trip
        parsed = json.loads(json_str)
        assert parsed == d


# =============================================================================
# Additional Tests
# =============================================================================


class TestTTSStyleMetadataDataclass:
    """Tests for TTSStyleMetadata dataclass behavior."""

    def test_dataclass_fields(self) -> None:
        """TTSStyleMetadata has expected fields."""
        style = TTSStyleMetadata(
            energy_level="normal",
            speech_rate="normal",
            valence="neutral",
            arousal="medium",
            recommended_response_mode="neutral",
        )

        assert style.energy_level == "normal"
        assert style.speech_rate == "normal"
        assert style.valence == "neutral"
        assert style.arousal == "medium"
        assert style.recommended_response_mode == "neutral"

    def test_dataclass_equality(self) -> None:
        """Two TTSStyleMetadata with same values are equal."""
        style1 = compute_tts_style({"energy": 0.5}, {"valence": 0.0})
        style2 = compute_tts_style({"energy": 0.5}, {"valence": 0.0})

        assert style1 == style2

    def test_dataclass_inequality(self) -> None:
        """Two TTSStyleMetadata with different values are not equal."""
        style1 = compute_tts_style({"energy": 0.2}, None)  # quiet
        style2 = compute_tts_style({"energy": 0.8}, None)  # loud

        assert style1 != style2


class TestEdgeCases:
    """Edge case tests."""

    def test_extreme_values_prosody(self) -> None:
        """Extreme prosody values are handled."""
        # Very high values
        result_high = compute_tts_style({"energy": 100.0, "speech_rate": 50.0}, None)
        assert result_high.energy_level == "loud"
        assert result_high.speech_rate == "fast"

        # Very low/negative values
        result_low = compute_tts_style({"energy": -1.0, "speech_rate": -1.0}, None)
        assert result_low.energy_level == "quiet"
        assert result_low.speech_rate == "slow"

    def test_extreme_values_emotion(self) -> None:
        """Extreme emotion values are handled."""
        # Beyond normal range
        result = compute_tts_style(None, {"valence": -5.0, "arousal": 10.0})
        assert result.valence == "negative"
        assert result.arousal == "high"

    def test_none_values_in_dict(self) -> None:
        """None values within dicts are handled like missing keys."""
        result = compute_tts_style(
            {"energy": None, "speech_rate": 1.5},
            {"valence": None, "arousal": 0.9},
        )

        # None values should give defaults
        assert result.energy_level == "normal"  # Default
        assert result.speech_rate == "fast"  # From provided value
        assert result.valence == "neutral"  # Default
        assert result.arousal == "high"  # From provided value
