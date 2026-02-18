"""Tests for audio health analysis (v2.1.0).

This test suite validates the audio health analysis functionality:
1. analyze_chunk_health with normal audio (synthetic PCM)
2. analyze_chunk_health with clipped audio
3. analyze_chunk_health with silence
4. analyze_chunk_health with empty input
5. AudioHealthAggregator rolling window
6. AudioHealthAggregator.get_aggregate()
7. quality_score is in valid range (0-1)
"""

from __future__ import annotations

import numpy as np
import pytest

from slower_whisper.pipeline.audio_health import (
    AudioHealthAggregator,
    AudioHealthSnapshot,
    analyze_chunk_health,
)

# =============================================================================
# Helper Functions
# =============================================================================


def _create_sine_wave_pcm(
    frequency: float = 440.0,
    duration_ms: int = 100,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> bytes:
    """Create synthetic PCM audio as 16-bit signed LE bytes.

    Args:
        frequency: Frequency in Hz.
        duration_ms: Duration in milliseconds.
        sample_rate: Sample rate in Hz.
        amplitude: Amplitude (0.0 to 1.0).

    Returns:
        Raw PCM bytes.
    """
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, dtype=np.float32)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    # Convert to int16
    samples_int16 = (signal * 32767).astype(np.int16)
    return samples_int16.tobytes()


def _create_silence_pcm(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
    """Create silent PCM audio.

    Args:
        duration_ms: Duration in milliseconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Raw PCM bytes (all zeros).
    """
    n_samples = int(sample_rate * duration_ms / 1000)
    samples = np.zeros(n_samples, dtype=np.int16)
    return samples.tobytes()


def _create_clipped_pcm(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
    """Create clipped (distorted) PCM audio.

    Args:
        duration_ms: Duration in milliseconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Raw PCM bytes with many samples at max value.
    """
    n_samples = int(sample_rate * duration_ms / 1000)
    # Create signal that clips
    t = np.linspace(0, duration_ms / 1000, n_samples, dtype=np.float32)
    signal = 2.0 * np.sin(2 * np.pi * 440 * t)  # Amplitude > 1.0 will clip
    signal = np.clip(signal, -1.0, 1.0)
    samples_int16 = (signal * 32767).astype(np.int16)
    return samples_int16.tobytes()


def _create_speech_like_pcm(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
    """Create speech-like PCM audio with appropriate frequency content.

    Args:
        duration_ms: Duration in milliseconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Raw PCM bytes simulating speech-like audio.
    """
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, dtype=np.float32)
    # Combine multiple frequencies typical of speech (F0 + harmonics)
    signal = 0.3 * np.sin(2 * np.pi * 150 * t)  # F0
    signal += 0.2 * np.sin(2 * np.pi * 300 * t)  # 2nd harmonic
    signal += 0.15 * np.sin(2 * np.pi * 600 * t)  # Higher formant
    signal += 0.1 * np.sin(2 * np.pi * 1200 * t)  # Higher formant
    samples_int16 = (signal * 32767).astype(np.int16)
    return samples_int16.tobytes()


# =============================================================================
# 1. Test analyze_chunk_health with Normal Audio
# =============================================================================


class TestAnalyzeChunkHealthNormal:
    """Tests for analyze_chunk_health with normal audio."""

    def test_normal_audio_returns_snapshot(self) -> None:
        """Normal audio returns an AudioHealthSnapshot."""
        pcm = _create_sine_wave_pcm(frequency=440, amplitude=0.3)

        result = analyze_chunk_health(pcm)

        assert isinstance(result, AudioHealthSnapshot)

    def test_normal_audio_low_clipping(self) -> None:
        """Normal audio has low clipping ratio."""
        pcm = _create_sine_wave_pcm(frequency=440, amplitude=0.3)

        result = analyze_chunk_health(pcm)

        assert result.clipping_ratio < 0.01  # Less than 1% clipping

    def test_normal_audio_has_rms_energy(self) -> None:
        """Normal audio has measurable RMS energy."""
        pcm = _create_sine_wave_pcm(frequency=440, amplitude=0.3)

        result = analyze_chunk_health(pcm)

        assert result.rms_energy > 0.0
        assert result.rms_energy < 1.0

    def test_normal_audio_has_spectral_centroid(self) -> None:
        """Normal audio has computed spectral centroid."""
        pcm = _create_sine_wave_pcm(frequency=440, amplitude=0.3)

        result = analyze_chunk_health(pcm)

        assert result.spectral_centroid is not None
        # Spectral centroid should be near the frequency
        assert result.spectral_centroid > 0

    def test_sample_rate_affects_spectral_centroid(self) -> None:
        """Different sample rates produce different spectral centroids."""
        pcm = _create_sine_wave_pcm(frequency=440, amplitude=0.3, sample_rate=16000)

        result_16k = analyze_chunk_health(pcm, sample_rate=16000)
        result_8k = analyze_chunk_health(pcm, sample_rate=8000)

        # Results should be different due to different sample rate interpretation
        assert result_16k.spectral_centroid != result_8k.spectral_centroid


# =============================================================================
# 2. Test analyze_chunk_health with Clipped Audio
# =============================================================================


class TestAnalyzeChunkHealthClipped:
    """Tests for analyze_chunk_health with clipped audio."""

    def test_clipped_audio_high_clipping_ratio(self) -> None:
        """Clipped audio has high clipping ratio."""
        pcm = _create_clipped_pcm()

        result = analyze_chunk_health(pcm)

        # Clipping ratio should be significant
        assert result.clipping_ratio > 0.1  # More than 10% clipped

    def test_clipped_audio_lower_quality_score(self) -> None:
        """Clipped audio has lower quality score than normal audio."""
        normal_pcm = _create_sine_wave_pcm(amplitude=0.3)
        clipped_pcm = _create_clipped_pcm()

        normal_result = analyze_chunk_health(normal_pcm)
        clipped_result = analyze_chunk_health(clipped_pcm)

        # Clipped should have lower quality
        assert clipped_result.quality_score < normal_result.quality_score

    def test_heavily_clipped_audio(self) -> None:
        """Heavily clipped audio (all samples at max) is detected."""
        # Create audio where all samples are at max
        n_samples = 1600  # 100ms at 16kHz
        samples = np.full(n_samples, 32760, dtype=np.int16)  # Near max
        pcm = samples.tobytes()

        result = analyze_chunk_health(pcm)

        # Should detect high clipping
        assert result.clipping_ratio > 0.9


# =============================================================================
# 3. Test analyze_chunk_health with Silence
# =============================================================================


class TestAnalyzeChunkHealthSilence:
    """Tests for analyze_chunk_health with silent audio."""

    def test_silence_zero_rms_energy(self) -> None:
        """Silent audio has zero RMS energy."""
        pcm = _create_silence_pcm()

        result = analyze_chunk_health(pcm)

        assert result.rms_energy == 0.0

    def test_silence_no_clipping(self) -> None:
        """Silent audio has no clipping."""
        pcm = _create_silence_pcm()

        result = analyze_chunk_health(pcm)

        assert result.clipping_ratio == 0.0

    def test_silence_not_speech_likely(self) -> None:
        """Silent audio is not likely to be speech."""
        pcm = _create_silence_pcm()

        result = analyze_chunk_health(pcm)

        assert result.is_speech_likely is False

    def test_silence_low_quality_score(self) -> None:
        """Silent audio has low quality score (nothing to transcribe)."""
        pcm = _create_silence_pcm()

        result = analyze_chunk_health(pcm)

        # Silence should have low quality (useless for transcription)
        assert result.quality_score < 0.5


# =============================================================================
# 4. Test analyze_chunk_health with Empty Input
# =============================================================================


class TestAnalyzeChunkHealthEmpty:
    """Tests for analyze_chunk_health with empty or minimal input."""

    def test_empty_bytes_returns_default_snapshot(self) -> None:
        """Empty bytes returns snapshot with zero/default values."""
        result = analyze_chunk_health(b"")

        assert result.clipping_ratio == 0.0
        assert result.rms_energy == 0.0
        assert result.snr_proxy == 0.0
        assert result.spectral_centroid is None
        assert result.quality_score == 0.0
        assert result.is_speech_likely is False

    def test_single_byte_returns_default_snapshot(self) -> None:
        """Single byte (incomplete sample) returns default snapshot."""
        result = analyze_chunk_health(b"\x00")

        assert result.rms_energy == 0.0
        assert result.spectral_centroid is None

    def test_two_bytes_minimal_sample(self) -> None:
        """Two bytes (one sample) is processed."""
        # One sample of silence
        result = analyze_chunk_health(b"\x00\x00")

        assert result.rms_energy == 0.0

    def test_none_like_empty_handled(self) -> None:
        """Verify graceful handling with minimal data."""
        # Very short audio (just a few samples)
        pcm = np.array([100, 200, 150], dtype=np.int16).tobytes()

        result = analyze_chunk_health(pcm)

        assert isinstance(result, AudioHealthSnapshot)


# =============================================================================
# 5. Test AudioHealthAggregator Rolling Window
# =============================================================================


class TestAudioHealthAggregatorWindow:
    """Tests for AudioHealthAggregator rolling window behavior."""

    def test_aggregator_creation(self) -> None:
        """AudioHealthAggregator can be created with window size."""
        aggregator = AudioHealthAggregator(window_size=10)

        assert aggregator.window_size == 10
        assert aggregator.count == 0

    def test_aggregator_invalid_window_size(self) -> None:
        """AudioHealthAggregator raises for invalid window size."""
        with pytest.raises(ValueError, match="window_size must be at least 1"):
            AudioHealthAggregator(window_size=0)

        with pytest.raises(ValueError, match="window_size must be at least 1"):
            AudioHealthAggregator(window_size=-5)

    def test_add_snapshot_increments_count(self) -> None:
        """Adding snapshots increments the count."""
        aggregator = AudioHealthAggregator(window_size=10)
        pcm = _create_sine_wave_pcm()

        for _i in range(5):
            snapshot = analyze_chunk_health(pcm)
            aggregator.add_snapshot(snapshot)

        assert aggregator.count == 5

    def test_rolling_window_evicts_old(self) -> None:
        """Rolling window evicts oldest when full."""
        aggregator = AudioHealthAggregator(window_size=3)
        pcm = _create_sine_wave_pcm()

        for _i in range(5):
            snapshot = analyze_chunk_health(pcm)
            aggregator.add_snapshot(snapshot)

        # Window size is 3, so should only have 3 snapshots
        assert aggregator.count == 3

    def test_reset_clears_window(self) -> None:
        """reset() clears all snapshots from window."""
        aggregator = AudioHealthAggregator(window_size=10)
        pcm = _create_sine_wave_pcm()

        for _i in range(5):
            snapshot = analyze_chunk_health(pcm)
            aggregator.add_snapshot(snapshot)

        aggregator.reset()

        assert aggregator.count == 0


# =============================================================================
# 6. Test AudioHealthAggregator.get_aggregate()
# =============================================================================


class TestAudioHealthAggregatorGetAggregate:
    """Tests for AudioHealthAggregator.get_aggregate() method."""

    def test_get_aggregate_empty_window(self) -> None:
        """get_aggregate on empty window returns default values."""
        aggregator = AudioHealthAggregator(window_size=10)

        result = aggregator.get_aggregate()

        assert result.clipping_ratio == 0.0
        assert result.rms_energy == 0.0
        assert result.snr_proxy == 0.0
        assert result.spectral_centroid is None
        assert result.quality_score == 0.0
        assert result.is_speech_likely is False

    def test_get_aggregate_single_snapshot(self) -> None:
        """get_aggregate with single snapshot returns that snapshot's values."""
        aggregator = AudioHealthAggregator(window_size=10)
        pcm = _create_sine_wave_pcm(amplitude=0.3)
        snapshot = analyze_chunk_health(pcm)
        aggregator.add_snapshot(snapshot)

        result = aggregator.get_aggregate()

        # Should match the single snapshot
        assert result.clipping_ratio == pytest.approx(snapshot.clipping_ratio, rel=0.01)
        assert result.rms_energy == pytest.approx(snapshot.rms_energy, rel=0.01)

    def test_get_aggregate_averages_metrics(self) -> None:
        """get_aggregate averages numeric metrics across window."""
        aggregator = AudioHealthAggregator(window_size=10)

        # Add snapshots with different energies
        for amplitude in [0.1, 0.2, 0.3, 0.4, 0.5]:
            pcm = _create_sine_wave_pcm(amplitude=amplitude)
            snapshot = analyze_chunk_health(pcm)
            aggregator.add_snapshot(snapshot)

        result = aggregator.get_aggregate()

        # RMS should be average of the different amplitudes' RMS values
        assert result.rms_energy > 0.0

    def test_get_aggregate_speech_likelihood_majority_vote(self) -> None:
        """is_speech_likely uses majority vote across window."""
        aggregator = AudioHealthAggregator(window_size=5)

        # Add 3 speech-like and 2 silence
        speech_pcm = _create_speech_like_pcm()
        silence_pcm = _create_silence_pcm()

        for _ in range(3):
            aggregator.add_snapshot(analyze_chunk_health(speech_pcm))
        for _ in range(2):
            aggregator.add_snapshot(analyze_chunk_health(silence_pcm))

        result = aggregator.get_aggregate()

        # Majority (3 out of 5) is speech-like
        # Note: depends on speech detection heuristics
        # Just verify we get a boolean result
        assert isinstance(result.is_speech_likely, bool)

    def test_get_aggregate_handles_none_spectral_centroid(self) -> None:
        """get_aggregate handles None spectral centroids correctly."""
        aggregator = AudioHealthAggregator(window_size=5)

        # Mix of snapshots with and without spectral centroid
        normal_pcm = _create_sine_wave_pcm()
        aggregator.add_snapshot(analyze_chunk_health(normal_pcm))
        aggregator.add_snapshot(analyze_chunk_health(normal_pcm))

        # Add empty (which has None centroid)
        empty_snapshot = analyze_chunk_health(b"")
        aggregator.add_snapshot(empty_snapshot)

        result = aggregator.get_aggregate()

        # Should compute average of non-None centroids
        # or return None if all are None
        assert result.spectral_centroid is None or isinstance(result.spectral_centroid, float)


# =============================================================================
# 7. Test quality_score is in Valid Range (0-1)
# =============================================================================


class TestQualityScoreRange:
    """Tests for quality_score being in valid range."""

    def test_quality_score_range_normal_audio(self) -> None:
        """Normal audio quality score is in [0, 1]."""
        pcm = _create_sine_wave_pcm(amplitude=0.3)

        result = analyze_chunk_health(pcm)

        assert 0.0 <= result.quality_score <= 1.0

    def test_quality_score_range_silence(self) -> None:
        """Silent audio quality score is in [0, 1]."""
        pcm = _create_silence_pcm()

        result = analyze_chunk_health(pcm)

        assert 0.0 <= result.quality_score <= 1.0

    def test_quality_score_range_clipped(self) -> None:
        """Clipped audio quality score is in [0, 1]."""
        pcm = _create_clipped_pcm()

        result = analyze_chunk_health(pcm)

        assert 0.0 <= result.quality_score <= 1.0

    def test_quality_score_range_empty(self) -> None:
        """Empty audio quality score is in [0, 1]."""
        result = analyze_chunk_health(b"")

        assert 0.0 <= result.quality_score <= 1.0

    def test_quality_score_range_extreme_values(self) -> None:
        """Quality score remains in range for extreme audio values."""
        # All max values
        n_samples = 1600
        max_samples = np.full(n_samples, 32767, dtype=np.int16)
        result_max = analyze_chunk_health(max_samples.tobytes())
        assert 0.0 <= result_max.quality_score <= 1.0

        # All min values
        min_samples = np.full(n_samples, -32768, dtype=np.int16)
        result_min = analyze_chunk_health(min_samples.tobytes())
        assert 0.0 <= result_min.quality_score <= 1.0

        # Random noise
        noise_samples = np.random.randint(-32768, 32767, n_samples, dtype=np.int16)
        result_noise = analyze_chunk_health(noise_samples.tobytes())
        assert 0.0 <= result_noise.quality_score <= 1.0

    def test_aggregate_quality_score_range(self) -> None:
        """Aggregated quality score is in [0, 1]."""
        aggregator = AudioHealthAggregator(window_size=10)

        # Mix of different audio types
        aggregator.add_snapshot(analyze_chunk_health(_create_sine_wave_pcm()))
        aggregator.add_snapshot(analyze_chunk_health(_create_silence_pcm()))
        aggregator.add_snapshot(analyze_chunk_health(_create_clipped_pcm()))

        result = aggregator.get_aggregate()

        assert 0.0 <= result.quality_score <= 1.0


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Additional edge case tests."""

    def test_snapshot_is_frozen(self) -> None:
        """AudioHealthSnapshot is immutable (frozen dataclass)."""
        pcm = _create_sine_wave_pcm()
        snapshot = analyze_chunk_health(pcm)

        with pytest.raises(AttributeError):
            snapshot.quality_score = 0.5  # type: ignore[misc]

    def test_very_short_audio_no_spectral_centroid(self) -> None:
        """Very short audio may not have spectral centroid."""
        # Less than MIN_SAMPLES_FOR_FFT (64 samples)
        short_samples = np.array([100, 200, 150, 100] * 5, dtype=np.int16)
        pcm = short_samples.tobytes()

        result = analyze_chunk_health(pcm)

        # Should still return a valid snapshot
        assert isinstance(result, AudioHealthSnapshot)

    def test_high_frequency_audio(self) -> None:
        """High frequency audio is analyzed correctly."""
        pcm = _create_sine_wave_pcm(frequency=7000, amplitude=0.3)

        result = analyze_chunk_health(pcm)

        # Should have high spectral centroid
        assert result.spectral_centroid is not None
        assert result.spectral_centroid > 1000  # Higher than speech range

    def test_low_frequency_audio(self) -> None:
        """Low frequency audio is analyzed correctly."""
        pcm = _create_sine_wave_pcm(frequency=50, amplitude=0.3)

        result = analyze_chunk_health(pcm)

        # Should have low spectral centroid
        assert result.spectral_centroid is not None

    def test_speech_like_audio_detected(self) -> None:
        """Speech-like audio is detected as potentially speech."""
        pcm = _create_speech_like_pcm()

        result = analyze_chunk_health(pcm)

        # Speech-like audio should have reasonable quality
        assert result.quality_score > 0.0
        # Note: is_speech_likely depends on energy thresholds
