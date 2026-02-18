"""Tests for environment classifier module."""

from __future__ import annotations

from slower_whisper.pipeline.environment_classifier import (
    AudioHealthMetrics,
    EnvironmentClassifier,
    EnvironmentClassifierConfig,
    EnvironmentState,
    classify_environment,
)


class TestEnvironmentState:
    """Tests for EnvironmentState dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        state = EnvironmentState()
        assert state.tag == "unknown"
        assert state.confidence == 0.0
        assert state.contributing_factors == []

    def test_to_dict(self):
        """Converts to dictionary correctly."""
        state = EnvironmentState(
            tag="clean",
            confidence=0.9,
            contributing_factors=["high_quality", "good_snr"],
            secondary_tags=["quiet"],
            quality_score=0.85,
        )
        data = state.to_dict()

        assert data["tag"] == "clean"
        assert data["confidence"] == 0.9
        assert "high_quality" in data["contributing_factors"]

    def test_from_dict(self):
        """Creates from dictionary correctly."""
        data = {
            "tag": "noisy",
            "confidence": 0.7,
            "contributing_factors": ["low_snr"],
            "secondary_tags": [],
            "quality_score": 0.4,
        }
        state = EnvironmentState.from_dict(data)

        assert state.tag == "noisy"
        assert state.confidence == 0.7


class TestAudioHealthMetrics:
    """Tests for AudioHealthMetrics dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        metrics = AudioHealthMetrics()
        assert metrics.quality_score == 0.5
        assert metrics.snr_db is None
        assert metrics.clipping_ratio == 0.0

    def test_from_audio_health_snapshot(self):
        """Creates from snapshot dictionary."""
        snapshot = {
            "quality_score": 0.8,
            "snr_db": 25.0,
            "spectral_centroid_hz": 2000.0,
            "clipping_ratio": 0.01,
        }
        metrics = AudioHealthMetrics.from_audio_health_snapshot(snapshot)

        assert metrics.quality_score == 0.8
        assert metrics.snr_db == 25.0
        assert metrics.spectral_centroid_hz == 2000.0


class TestEnvironmentClassifierConfig:
    """Tests for EnvironmentClassifierConfig."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = EnvironmentClassifierConfig()
        assert config.enabled is True
        assert config.clean_quality_threshold == 0.7
        assert config.noisy_snr_threshold == 10.0

    def test_to_dict(self):
        """Converts to dictionary correctly."""
        config = EnvironmentClassifierConfig(clean_quality_threshold=0.8)
        data = config.to_dict()

        assert data["clean_quality_threshold"] == 0.8
        assert data["enabled"] is True


class TestEnvironmentClassifier:
    """Tests for EnvironmentClassifier class."""

    def test_disabled_returns_unknown(self):
        """Disabled classifier returns unknown."""
        config = EnvironmentClassifierConfig(enabled=False)
        classifier = EnvironmentClassifier(config)
        metrics = AudioHealthMetrics(quality_score=0.9)

        result = classifier.classify(metrics)
        assert result.tag == "unknown"
        assert result.confidence == 0.0

    def test_clean_detection(self):
        """Detects clean audio."""
        classifier = EnvironmentClassifier()
        metrics = AudioHealthMetrics(
            quality_score=0.85,
            snr_db=25.0,
            spectral_centroid_hz=2000.0,
            clipping_ratio=0.0,
        )

        result = classifier.classify(metrics)
        assert result.tag == "clean"
        assert result.confidence > 0.7

    def test_noisy_detection(self):
        """Detects noisy audio."""
        classifier = EnvironmentClassifier()
        metrics = AudioHealthMetrics(
            quality_score=0.4,
            snr_db=5.0,  # Low SNR
            spectral_centroid_hz=2000.0,
            clipping_ratio=0.0,
        )

        result = classifier.classify(metrics)
        assert result.tag == "noisy"
        assert "snr_db" in str(result.contributing_factors)

    def test_muffled_detection(self):
        """Detects muffled audio."""
        classifier = EnvironmentClassifier()
        metrics = AudioHealthMetrics(
            quality_score=0.5,
            snr_db=15.0,
            spectral_centroid_hz=300.0,  # Low centroid
            clipping_ratio=0.0,
        )

        result = classifier.classify(metrics)
        assert result.tag == "muffled"
        assert "spectral_centroid" in str(result.contributing_factors)

    def test_clipping_detection(self):
        """Detects clipping audio."""
        classifier = EnvironmentClassifier()
        metrics = AudioHealthMetrics(
            quality_score=0.6,
            snr_db=20.0,
            spectral_centroid_hz=2000.0,
            clipping_ratio=0.1,  # High clipping
        )

        result = classifier.classify(metrics)
        assert result.tag == "clipping"
        assert "clipping_ratio" in str(result.contributing_factors)

    def test_hissy_detection(self):
        """Detects hissy audio."""
        classifier = EnvironmentClassifier()
        metrics = AudioHealthMetrics(
            quality_score=0.4,  # Low quality
            snr_db=15.0,
            spectral_centroid_hz=5000.0,  # High centroid
            clipping_ratio=0.0,
        )

        result = classifier.classify(metrics)
        assert result.tag == "hissy"

    def test_insufficient_metrics(self):
        """Handles insufficient metrics."""
        classifier = EnvironmentClassifier()
        metrics = AudioHealthMetrics(
            quality_score=0.5,
            # No SNR or centroid
        )

        result = classifier.classify(metrics)
        # Should still produce a result
        assert isinstance(result, EnvironmentState)

    def test_multiple_issues_secondary_tags(self):
        """Multiple issues produce secondary tags."""
        classifier = EnvironmentClassifier()
        metrics = AudioHealthMetrics(
            quality_score=0.3,
            snr_db=5.0,  # Noisy
            spectral_centroid_hz=300.0,  # Muffled
            clipping_ratio=0.0,
        )

        result = classifier.classify(metrics)
        # Should have primary and potentially secondary tags
        assert result.tag in ["noisy", "muffled"]

    def test_classify_from_snapshot(self):
        """Classifies from snapshot dictionary."""
        classifier = EnvironmentClassifier()
        snapshot = {
            "quality_score": 0.85,
            "snr_db": 25.0,
            "spectral_centroid_hz": 2000.0,
            "clipping_ratio": 0.0,
        }

        result = classifier.classify_from_snapshot(snapshot)
        assert result.tag == "clean"


class TestClassifyEnvironmentFunction:
    """Tests for convenience function."""

    def test_basic_usage(self):
        """Basic usage works correctly."""
        result = classify_environment(
            quality_score=0.9,
            snr_db=25.0,
            spectral_centroid_hz=2000.0,
            clipping_ratio=0.0,
        )

        assert result.tag == "clean"

    def test_noisy_detection(self):
        """Detects noisy environment."""
        result = classify_environment(
            quality_score=0.4,
            snr_db=5.0,
        )

        assert result.tag == "noisy"

    def test_with_extra_kwargs(self):
        """Ignores extra kwargs."""
        result = classify_environment(
            quality_score=0.9,
            snr_db=25.0,
            extra_param="ignored",
        )

        # Should not crash
        assert isinstance(result, EnvironmentState)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_quality_score(self):
        """Handles zero quality score."""
        classifier = EnvironmentClassifier()
        metrics = AudioHealthMetrics(quality_score=0.0)

        result = classifier.classify(metrics)
        assert isinstance(result, EnvironmentState)

    def test_max_quality_score(self):
        """Handles max quality score."""
        classifier = EnvironmentClassifier()
        metrics = AudioHealthMetrics(
            quality_score=1.0,
            snr_db=50.0,
        )

        result = classifier.classify(metrics)
        assert result.tag == "clean"
        assert result.confidence >= 0.9

    def test_negative_snr(self):
        """Handles negative SNR."""
        classifier = EnvironmentClassifier()
        metrics = AudioHealthMetrics(
            quality_score=0.3,
            snr_db=-5.0,  # Very bad SNR
        )

        result = classifier.classify(metrics)
        assert result.tag == "noisy"

    def test_custom_thresholds(self):
        """Custom thresholds work correctly."""
        config = EnvironmentClassifierConfig(
            clean_quality_threshold=0.9,  # More strict
            noisy_snr_threshold=15.0,  # More strict
        )
        classifier = EnvironmentClassifier(config)

        # Would be clean with defaults, but not with strict config
        metrics = AudioHealthMetrics(
            quality_score=0.75,
            snr_db=12.0,
        )

        result = classifier.classify(metrics)
        assert result.tag != "clean"
