"""
Comprehensive test suite for the emotion.py module.

Tests cover:
- EmotionRecognizer class (initialization, lazy loading, extraction methods)
- DummyEmotionRecognizer class (fallback behavior)
- Singleton management (get_emotion_recognizer, cleanup_emotion_models)
- Convenience functions (extract_emotion_dimensional, extract_emotion_categorical)
- Classification thresholds (_classify_valence, _classify_arousal, _classify_dominance)
- Resampling logic (_simple_resample)
- Thread-safety (concurrent access patterns)
- Edge cases (empty audio, short audio, normalization)
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import directly from the emotion module to avoid circular import issues
from transcription import emotion

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate a 1-second sample audio signal at 16kHz."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Simple sine wave at 440 Hz
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def short_audio() -> np.ndarray:
    """Generate a very short audio segment (0.1 seconds)."""
    sr = 16000
    duration = 0.1
    t = np.linspace(0, duration, int(sr * duration))
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def empty_audio() -> np.ndarray:
    """Generate empty audio array."""
    return np.array([], dtype=np.float32)


@pytest.fixture
def unnormalized_audio() -> np.ndarray:
    """Generate audio with values outside [-1, 1] range."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Multiply to exceed normalized range
    return (np.sin(2 * np.pi * 440 * t) * 5.0).astype(np.float32)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the emotion recognizer singleton before each test."""
    emotion._recognizer_instance = None
    yield
    emotion._recognizer_instance = None


# ============================================================================
# _clamp01 tests
# ============================================================================


class TestClamp01:
    """Tests for the _clamp01 helper function."""

    def test_clamp01_within_range(self):
        """Values within [0, 1] should be unchanged."""
        assert emotion._clamp01(0.5) == 0.5
        assert emotion._clamp01(0.0) == 0.0
        assert emotion._clamp01(1.0) == 1.0

    def test_clamp01_below_zero(self):
        """Values below 0 should be clamped to 0."""
        assert emotion._clamp01(-0.1) == 0.0
        assert emotion._clamp01(-100.0) == 0.0

    def test_clamp01_above_one(self):
        """Values above 1 should be clamped to 1."""
        assert emotion._clamp01(1.1) == 1.0
        assert emotion._clamp01(100.0) == 1.0


# ============================================================================
# Classification threshold tests
# ============================================================================


class TestClassifyValence:
    """Tests for _classify_valence threshold logic."""

    def test_very_negative_valence(self):
        """Scores < 0.3 should be very_negative."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_valence(0.0) == "very_negative"
        assert recognizer._classify_valence(0.29) == "very_negative"

    def test_negative_valence(self):
        """Scores in [0.3, 0.4) should be negative."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_valence(0.3) == "negative"
        assert recognizer._classify_valence(0.39) == "negative"

    def test_neutral_valence(self):
        """Scores in [0.4, 0.6) should be neutral."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_valence(0.4) == "neutral"
        assert recognizer._classify_valence(0.5) == "neutral"
        assert recognizer._classify_valence(0.59) == "neutral"

    def test_positive_valence(self):
        """Scores in [0.6, 0.7) should be positive."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_valence(0.6) == "positive"
        assert recognizer._classify_valence(0.69) == "positive"

    def test_very_positive_valence(self):
        """Scores >= 0.7 should be very_positive."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_valence(0.7) == "very_positive"
        assert recognizer._classify_valence(1.0) == "very_positive"


class TestClassifyArousal:
    """Tests for _classify_arousal threshold logic."""

    def test_very_low_arousal(self):
        """Scores < 0.3 should be very_low."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_arousal(0.0) == "very_low"
        assert recognizer._classify_arousal(0.29) == "very_low"

    def test_low_arousal(self):
        """Scores in [0.3, 0.4) should be low."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_arousal(0.3) == "low"
        assert recognizer._classify_arousal(0.39) == "low"

    def test_medium_arousal(self):
        """Scores in [0.4, 0.6) should be medium."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_arousal(0.4) == "medium"
        assert recognizer._classify_arousal(0.5) == "medium"
        assert recognizer._classify_arousal(0.59) == "medium"

    def test_high_arousal(self):
        """Scores in [0.6, 0.7) should be high."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_arousal(0.6) == "high"
        assert recognizer._classify_arousal(0.69) == "high"

    def test_very_high_arousal(self):
        """Scores >= 0.7 should be very_high."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_arousal(0.7) == "very_high"
        assert recognizer._classify_arousal(1.0) == "very_high"


class TestClassifyDominance:
    """Tests for _classify_dominance threshold logic."""

    def test_very_submissive_dominance(self):
        """Scores < 0.3 should be very_submissive."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_dominance(0.0) == "very_submissive"
        assert recognizer._classify_dominance(0.29) == "very_submissive"

    def test_submissive_dominance(self):
        """Scores in [0.3, 0.4) should be submissive."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_dominance(0.3) == "submissive"
        assert recognizer._classify_dominance(0.39) == "submissive"

    def test_neutral_dominance(self):
        """Scores in [0.4, 0.6) should be neutral."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_dominance(0.4) == "neutral"
        assert recognizer._classify_dominance(0.5) == "neutral"
        assert recognizer._classify_dominance(0.59) == "neutral"

    def test_dominant_dominance(self):
        """Scores in [0.6, 0.7) should be dominant."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_dominance(0.6) == "dominant"
        assert recognizer._classify_dominance(0.69) == "dominant"

    def test_very_dominant_dominance(self):
        """Scores >= 0.7 should be very_dominant."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        assert recognizer._classify_dominance(0.7) == "very_dominant"
        assert recognizer._classify_dominance(1.0) == "very_dominant"


# ============================================================================
# DummyEmotionRecognizer tests
# ============================================================================


class TestDummyEmotionRecognizer:
    """Tests for the DummyEmotionRecognizer fallback class."""

    def test_extract_emotion_dimensional_returns_neutral(self, sample_audio):
        """Dummy recognizer should return neutral dimensional values."""
        recognizer = emotion.DummyEmotionRecognizer()
        result = recognizer.extract_emotion_dimensional(sample_audio, 16000)

        assert result["valence"]["level"] == "neutral"
        assert result["valence"]["score"] == 0.5
        assert result["arousal"]["level"] == "medium"
        assert result["arousal"]["score"] == 0.5
        assert result["dominance"]["level"] == "neutral"
        assert result["dominance"]["score"] == 0.5

    def test_extract_emotion_categorical_returns_neutral(self, sample_audio):
        """Dummy recognizer should return neutral categorical values."""
        recognizer = emotion.DummyEmotionRecognizer()
        result = recognizer.extract_emotion_categorical(sample_audio, 16000)

        assert result["categorical"]["primary"] == "neutral"
        assert result["categorical"]["confidence"] == 1.0
        assert result["categorical"]["secondary"] is None
        assert result["categorical"]["secondary_confidence"] == 0.0
        assert result["categorical"]["all_scores"] == {"neutral": 1.0}

    def test_dummy_recognizer_handles_empty_audio(self, empty_audio):
        """Dummy recognizer should handle empty audio without error."""
        recognizer = emotion.DummyEmotionRecognizer()
        result_dim = recognizer.extract_emotion_dimensional(empty_audio, 16000)
        result_cat = recognizer.extract_emotion_categorical(empty_audio, 16000)

        assert result_dim["valence"]["score"] == 0.5
        assert result_cat["categorical"]["primary"] == "neutral"

    def test_dummy_recognizer_ignores_sample_rate(self, sample_audio):
        """Dummy recognizer should work with any sample rate."""
        recognizer = emotion.DummyEmotionRecognizer()

        result_16k = recognizer.extract_emotion_dimensional(sample_audio, 16000)
        result_8k = recognizer.extract_emotion_dimensional(sample_audio, 8000)
        result_48k = recognizer.extract_emotion_dimensional(sample_audio, 48000)

        # All should return identical neutral values
        assert result_16k == result_8k == result_48k


# ============================================================================
# Singleton management tests
# ============================================================================


class TestSingletonManagement:
    """Tests for get_emotion_recognizer and cleanup_emotion_models."""

    def test_get_emotion_recognizer_returns_consistent_instance(self):
        """Multiple calls should return the same instance."""
        recognizer1 = emotion.get_emotion_recognizer()
        recognizer2 = emotion.get_emotion_recognizer()
        assert recognizer1 is recognizer2

    def test_get_emotion_recognizer_returns_dummy_when_unavailable(self):
        """Should return DummyEmotionRecognizer when emotion deps are unavailable."""
        with patch.object(emotion, "EMOTION_AVAILABLE", False):
            emotion._recognizer_instance = None
            recognizer = emotion.get_emotion_recognizer()
            assert isinstance(recognizer, emotion.DummyEmotionRecognizer)

    def test_cleanup_emotion_models_resets_singleton(self):
        """cleanup_emotion_models should reset the singleton to None."""
        # Get an instance first
        _ = emotion.get_emotion_recognizer()
        assert emotion._recognizer_instance is not None

        # Cleanup
        emotion.cleanup_emotion_models()
        assert emotion._recognizer_instance is None

    def test_cleanup_when_no_instance_exists(self):
        """cleanup_emotion_models should handle case when no instance exists."""
        emotion._recognizer_instance = None
        # Should not raise
        emotion.cleanup_emotion_models()
        assert emotion._recognizer_instance is None

    def test_get_emotion_recognizer_after_cleanup(self):
        """Should create new instance after cleanup."""
        recognizer1 = emotion.get_emotion_recognizer()
        emotion.cleanup_emotion_models()
        recognizer2 = emotion.get_emotion_recognizer()

        # New instance should be created
        assert recognizer2 is not recognizer1


# ============================================================================
# Convenience function tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions that use the singleton."""

    def test_extract_emotion_dimensional_uses_singleton(self, sample_audio):
        """Convenience function should use the singleton recognizer."""
        result = emotion.extract_emotion_dimensional(sample_audio, 16000)

        # Should have expected structure
        assert "valence" in result
        assert "arousal" in result
        assert "dominance" in result
        assert "level" in result["valence"]
        assert "score" in result["valence"]

    def test_extract_emotion_categorical_uses_singleton(self, sample_audio):
        """Convenience function should use the singleton recognizer."""
        result = emotion.extract_emotion_categorical(sample_audio, 16000)

        # Should have expected structure
        assert "categorical" in result
        assert "primary" in result["categorical"]
        assert "confidence" in result["categorical"]

    def test_convenience_functions_share_singleton(self, sample_audio):
        """Both convenience functions should use the same singleton."""
        _ = emotion.extract_emotion_dimensional(sample_audio, 16000)
        recognizer1 = emotion._recognizer_instance

        _ = emotion.extract_emotion_categorical(sample_audio, 16000)
        recognizer2 = emotion._recognizer_instance

        assert recognizer1 is recognizer2


# ============================================================================
# Thread-safety tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe singleton initialization."""

    def test_concurrent_get_emotion_recognizer(self):
        """Concurrent calls should all get the same singleton instance."""
        results = []
        errors = []

        def get_recognizer():
            try:
                recognizer = emotion.get_emotion_recognizer()
                results.append(recognizer)
            except Exception as e:
                errors.append(e)

        # Run multiple threads concurrently
        threads = [threading.Thread(target=get_recognizer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 10
        # All should get the same instance
        assert all(r is results[0] for r in results)

    def test_concurrent_cleanup_is_safe(self):
        """Concurrent cleanup calls should not cause errors."""
        # Get an instance first
        _ = emotion.get_emotion_recognizer()
        errors = []

        def cleanup():
            try:
                emotion.cleanup_emotion_models()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=cleanup) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert emotion._recognizer_instance is None

    def test_concurrent_extraction_with_dummy(self, sample_audio):
        """Concurrent extraction calls should work safely with dummy recognizer."""
        errors = []
        results = []

        def extract():
            try:
                result = emotion.extract_emotion_dimensional(sample_audio, 16000)
                results.append(result)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(extract) for _ in range(10)]
            for f in futures:
                f.result()

        assert len(errors) == 0
        assert len(results) == 10
        # All results should be identical from dummy recognizer
        assert all(r == results[0] for r in results)


# ============================================================================
# Resampling logic tests
# ============================================================================


class TestResamplingLogic:
    """Tests for _simple_resample and _validate_audio resampling."""

    @pytest.fixture
    def recognizer_instance(self):
        """Create a recognizer instance for testing internal methods."""
        # Skip init to avoid model loading
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        recognizer._dimensional_model = None
        recognizer._dimensional_feature_extractor = None
        recognizer._categorical_model = None
        recognizer._categorical_feature_extractor = None
        recognizer._dimensional_lock = threading.Lock()
        recognizer._categorical_lock = threading.Lock()
        return recognizer

    def test_resample_same_rate_returns_unchanged(self, recognizer_instance, sample_audio):
        """Resampling from same rate to same rate should return unchanged audio."""
        result = recognizer_instance._simple_resample(sample_audio, 16000, 16000)
        np.testing.assert_array_almost_equal(result, sample_audio.astype(np.float32))

    def test_resample_changes_length(self, recognizer_instance, sample_audio):
        """Resampling should change audio length proportionally."""
        # Downsample from 16kHz to 8kHz should halve the length
        result = recognizer_instance._simple_resample(sample_audio, 16000, 8000)
        expected_length = len(sample_audio) // 2
        # Allow some tolerance due to resampling algorithm
        assert abs(len(result) - expected_length) <= 1

    def test_resample_upsample_increases_length(self, recognizer_instance, sample_audio):
        """Upsampling should increase audio length."""
        result = recognizer_instance._simple_resample(sample_audio, 16000, 32000)
        expected_length = len(sample_audio) * 2
        assert abs(len(result) - expected_length) <= 1

    def test_resample_returns_float32(self, recognizer_instance):
        """Resampled audio should be float32."""
        audio = np.random.randn(16000).astype(np.float64)
        result = recognizer_instance._simple_resample(audio, 16000, 8000)
        assert result.dtype == np.float32

    def test_resample_fallback_without_librosa(self, recognizer_instance, sample_audio):
        """Should fall back to linear interpolation if librosa unavailable."""
        with patch.dict("sys.modules", {"librosa": None}):
            with patch("transcription.emotion.logger"):
                # Force reimport to clear cache
                import sys

                # Remove librosa from modules temporarily to test fallback
                librosa_backup = sys.modules.get("librosa")
                sys.modules["librosa"] = None

                try:
                    result = recognizer_instance._simple_resample(sample_audio, 16000, 8000)
                    # Should still produce valid output
                    assert len(result) > 0
                    assert result.dtype == np.float32
                finally:
                    # Restore librosa
                    if librosa_backup:
                        sys.modules["librosa"] = librosa_backup


# ============================================================================
# Audio validation tests
# ============================================================================


class TestAudioValidation:
    """Tests for _validate_audio method."""

    @pytest.fixture
    def recognizer_instance(self):
        """Create a recognizer instance for testing internal methods."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)
        recognizer._dimensional_model = None
        recognizer._dimensional_feature_extractor = None
        recognizer._categorical_model = None
        recognizer._categorical_feature_extractor = None
        recognizer._dimensional_lock = threading.Lock()
        recognizer._categorical_lock = threading.Lock()
        return recognizer

    def test_validate_audio_short_segment_logs_warning(
        self, recognizer_instance, short_audio, caplog
    ):
        """Short audio segments should log a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            recognizer_instance._validate_audio(short_audio, 16000)

        # Should have logged a warning about short segment
        assert any("shorter than recommended" in rec.message for rec in caplog.records)

    def test_validate_audio_resamples_when_needed(self, recognizer_instance, sample_audio):
        """Audio should be resampled when sample rate differs from target."""
        # Audio at 8kHz should be resampled to 16kHz
        audio, is_valid = recognizer_instance._validate_audio(sample_audio, 8000)
        assert is_valid
        # Length should approximately double
        expected_length = len(sample_audio) * 2
        assert abs(len(audio) - expected_length) <= 1

    def test_validate_audio_normalizes_range(self, recognizer_instance, unnormalized_audio):
        """Audio with values outside [-1, 1] should be normalized."""
        audio, is_valid = recognizer_instance._validate_audio(unnormalized_audio, 16000)
        assert is_valid
        assert np.abs(audio).max() <= 1.0

    def test_validate_audio_converts_to_float32(self, recognizer_instance):
        """Audio should be converted to float32."""
        audio_int16 = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        result, is_valid = recognizer_instance._validate_audio(audio_int16, 16000)
        assert is_valid
        assert result.dtype == np.float32

    def test_validate_audio_preserves_normalized_audio(self, recognizer_instance, sample_audio):
        """Already normalized audio should be mostly unchanged."""
        audio, is_valid = recognizer_instance._validate_audio(sample_audio, 16000)
        assert is_valid
        # Should be same length (same sample rate)
        assert len(audio) == len(sample_audio)


# ============================================================================
# EmotionRecognizer initialization tests
# ============================================================================


class TestEmotionRecognizerInitialization:
    """Tests for EmotionRecognizer initialization and model loading."""

    def test_init_raises_without_dependencies(self):
        """Should raise RuntimeError when emotion deps unavailable."""
        with patch.object(emotion, "EMOTION_AVAILABLE", False):
            with pytest.raises(RuntimeError) as exc_info:
                emotion.EmotionRecognizer()
            assert "torch and transformers" in str(exc_info.value)

    @pytest.mark.heavy
    def test_init_with_mocked_torch(self):
        """Should initialize with mocked torch dependencies."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.object(emotion, "EMOTION_AVAILABLE", True):
            with patch.object(emotion, "torch", mock_torch):
                recognizer = emotion.EmotionRecognizer()
                assert recognizer._device == "cpu"
                assert recognizer._dimensional_model is None
                assert recognizer._categorical_model is None

    def test_lazy_loading_deferred(self):
        """Models should not be loaded during initialization."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.object(emotion, "EMOTION_AVAILABLE", True):
            with patch.object(emotion, "torch", mock_torch):
                recognizer = emotion.EmotionRecognizer()

                # Models should not be loaded yet
                assert recognizer._dimensional_model is None
                assert recognizer._categorical_model is None
                assert recognizer._dimensional_feature_extractor is None
                assert recognizer._categorical_feature_extractor is None


# ============================================================================
# EmotionRecognizerLike Protocol tests
# ============================================================================


class TestEmotionRecognizerLikeProtocol:
    """Tests verifying protocol compliance."""

    def test_emotion_recognizer_implements_protocol(self):
        """EmotionRecognizer should implement EmotionRecognizerLike."""
        # Create without actually initializing (avoid deps)
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)

        # Check method signatures exist
        assert hasattr(recognizer, "extract_emotion_dimensional")
        assert hasattr(recognizer, "extract_emotion_categorical")
        assert callable(recognizer.extract_emotion_dimensional)
        assert callable(recognizer.extract_emotion_categorical)

    def test_dummy_recognizer_implements_protocol(self):
        """DummyEmotionRecognizer should implement EmotionRecognizerLike."""
        recognizer = emotion.DummyEmotionRecognizer()

        assert hasattr(recognizer, "extract_emotion_dimensional")
        assert hasattr(recognizer, "extract_emotion_categorical")
        assert callable(recognizer.extract_emotion_dimensional)
        assert callable(recognizer.extract_emotion_categorical)


# ============================================================================
# Module constants tests
# ============================================================================


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_emotion_available_is_boolean(self):
        """EMOTION_AVAILABLE should be a boolean."""
        assert isinstance(emotion.EMOTION_AVAILABLE, bool)

    def test_model_constants_defined(self):
        """Model constants should be defined."""
        assert emotion.EmotionRecognizer.DIMENSIONAL_MODEL is not None
        assert emotion.EmotionRecognizer.CATEGORICAL_MODEL is not None
        assert emotion.EmotionRecognizer.MIN_SEGMENT_LENGTH == 0.5
        assert emotion.EmotionRecognizer.TARGET_SAMPLE_RATE == 16000

    def test_public_api_exports(self):
        """__all__ should export expected symbols."""
        expected_exports = {
            "DummyEmotionRecognizer",
            "EmotionRecognizer",
            "EmotionRecognizerLike",
            "cleanup_emotion_models",
            "extract_emotion_categorical",
            "extract_emotion_dimensional",
            "get_emotion_recognizer",
        }
        assert set(emotion.__all__) == expected_exports


# ============================================================================
# Return value structure tests
# ============================================================================


class TestReturnValueStructure:
    """Tests verifying return value structure compliance."""

    def test_dimensional_result_structure(self, sample_audio):
        """Dimensional result should have correct structure."""
        result = emotion.extract_emotion_dimensional(sample_audio, 16000)

        # Top-level keys
        assert set(result.keys()) == {"valence", "arousal", "dominance"}

        # Each dimension should have level and score
        for dim in ["valence", "arousal", "dominance"]:
            assert "level" in result[dim]
            assert "score" in result[dim]
            assert isinstance(result[dim]["level"], str)
            assert isinstance(result[dim]["score"], (int, float))

    def test_categorical_result_structure(self, sample_audio):
        """Categorical result should have correct structure."""
        result = emotion.extract_emotion_categorical(sample_audio, 16000)

        assert "categorical" in result
        cat = result["categorical"]

        assert "primary" in cat
        assert "confidence" in cat
        assert "secondary" in cat
        assert "secondary_confidence" in cat
        assert "all_scores" in cat

        # Type checks
        assert isinstance(cat["primary"], str)
        assert isinstance(cat["confidence"], (int, float))
        assert cat["secondary"] is None or isinstance(cat["secondary"], str)
        assert isinstance(cat["secondary_confidence"], (int, float))
        assert isinstance(cat["all_scores"], dict)


# ============================================================================
# Edge case tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_boundary_classification_values(self):
        """Test classification at exact boundary values."""
        recognizer = emotion.EmotionRecognizer.__new__(emotion.EmotionRecognizer)

        # Test exact boundaries
        assert recognizer._classify_valence(0.3) == "negative"  # Not very_negative
        assert recognizer._classify_valence(0.4) == "neutral"  # Not negative
        assert recognizer._classify_valence(0.6) == "positive"  # Not neutral
        assert recognizer._classify_valence(0.7) == "very_positive"  # Not positive

        assert recognizer._classify_arousal(0.3) == "low"
        assert recognizer._classify_arousal(0.4) == "medium"
        assert recognizer._classify_arousal(0.6) == "high"
        assert recognizer._classify_arousal(0.7) == "very_high"

        assert recognizer._classify_dominance(0.3) == "submissive"
        assert recognizer._classify_dominance(0.4) == "neutral"
        assert recognizer._classify_dominance(0.6) == "dominant"
        assert recognizer._classify_dominance(0.7) == "very_dominant"

    def test_zero_audio_values(self, sample_audio):
        """Test with all-zero audio (silence)."""
        silence = np.zeros(16000, dtype=np.float32)
        result = emotion.extract_emotion_dimensional(silence, 16000)

        # Should still return valid structure
        assert "valence" in result
        assert "arousal" in result
        assert "dominance" in result

    def test_very_long_audio(self):
        """Test with very long audio segment."""
        # 60 seconds of audio
        long_audio = np.random.randn(16000 * 60).astype(np.float32)
        result = emotion.extract_emotion_dimensional(long_audio, 16000)

        # Should handle without error
        assert "valence" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
