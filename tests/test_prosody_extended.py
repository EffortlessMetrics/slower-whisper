"""Tests for extended prosody module."""

from __future__ import annotations

import numpy as np

from transcription.prosody_extended import (
    BoundaryToneState,
    MonotonyState,
    PitchSlopeState,
    ProsodyExtendedState,
    analyze_boundary_tone,
    analyze_monotony,
    analyze_pitch_slope,
    extract_prosody_extended,
)


class TestBoundaryToneState:
    """Tests for BoundaryToneState dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        state = BoundaryToneState()
        assert state.tone == "unknown"
        assert state.confidence == 0.0
        assert state.window_duration_ms == 200.0

    def test_to_dict(self):
        """Converts to dictionary correctly."""
        state = BoundaryToneState(
            tone="rising",
            final_slope_hz_per_sec=50.0,
            confidence=0.8,
        )
        data = state.to_dict()

        assert data["tone"] == "rising"
        assert data["final_slope_hz_per_sec"] == 50.0
        assert data["confidence"] == 0.8


class TestMonotonyState:
    """Tests for MonotonyState dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        state = MonotonyState()
        assert state.level == "normal"
        assert state.range_utilization == 50.0

    def test_to_dict(self):
        """Converts to dictionary correctly."""
        state = MonotonyState(
            level="expressive",
            range_utilization=85.0,
            pitch_range_hz=120.0,
        )
        data = state.to_dict()

        assert data["level"] == "expressive"
        assert data["range_utilization"] == 85.0


class TestPitchSlopeState:
    """Tests for PitchSlopeState dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        state = PitchSlopeState()
        assert state.slope_hz_per_sec == 0.0
        assert state.direction == "flat"

    def test_to_dict(self):
        """Converts to dictionary correctly."""
        state = PitchSlopeState(
            slope_hz_per_sec=-25.0,
            direction="falling",
            r_squared=0.85,
        )
        data = state.to_dict()

        assert data["slope_hz_per_sec"] == -25.0
        assert data["direction"] == "falling"


class TestProsodyExtendedState:
    """Tests for ProsodyExtendedState dataclass."""

    def test_default_values(self):
        """Default values create valid nested objects."""
        state = ProsodyExtendedState()
        assert state.boundary_tone.tone == "unknown"
        assert state.monotony.level == "normal"
        assert state.pitch_slope.direction == "flat"
        assert state.extraction_status == "success"

    def test_to_dict(self):
        """Converts to nested dictionary correctly."""
        state = ProsodyExtendedState(
            boundary_tone=BoundaryToneState(tone="falling"),
            monotony=MonotonyState(level="expressive"),
            pitch_slope=PitchSlopeState(direction="rising"),
        )
        data = state.to_dict()

        assert "boundary_tone" in data
        assert "monotony" in data
        assert "pitch_slope" in data
        assert data["boundary_tone"]["tone"] == "falling"


class TestAnalyzeBoundaryTone:
    """Tests for analyze_boundary_tone function."""

    def test_none_input(self):
        """Handles None input gracefully."""
        result = analyze_boundary_tone(None, None)
        assert result.tone == "unknown"
        assert result.confidence == 0.0

    def test_insufficient_points(self):
        """Handles insufficient data points."""
        times = np.array([0.0, 0.1])
        pitches = np.array([100.0, 110.0])
        result = analyze_boundary_tone(times, pitches)
        # May or may not classify depending on window
        assert isinstance(result, BoundaryToneState)

    def test_rising_tone_detection(self):
        """Detects rising tone from increasing pitch."""
        times = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
        # Rising pitch: 100 -> 200 Hz over 200ms = 500 Hz/sec
        pitches = np.array([100.0, 125.0, 150.0, 175.0, 200.0])
        result = analyze_boundary_tone(times, pitches, window_ms=200.0)

        assert result.tone == "rising"
        assert result.final_slope_hz_per_sec is not None
        assert result.final_slope_hz_per_sec > 0

    def test_falling_tone_detection(self):
        """Detects falling tone from decreasing pitch."""
        times = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
        # Falling pitch: 200 -> 100 Hz over 200ms = -500 Hz/sec
        pitches = np.array([200.0, 175.0, 150.0, 125.0, 100.0])
        result = analyze_boundary_tone(times, pitches, window_ms=200.0)

        assert result.tone == "falling"
        assert result.final_slope_hz_per_sec is not None
        assert result.final_slope_hz_per_sec < 0

    def test_flat_tone_detection(self):
        """Detects flat tone from constant pitch."""
        times = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
        # Flat pitch with small variation
        pitches = np.array([150.0, 152.0, 148.0, 151.0, 149.0])
        result = analyze_boundary_tone(times, pitches, window_ms=200.0)

        assert result.tone == "flat"


class TestAnalyzeMonotony:
    """Tests for analyze_monotony function."""

    def test_none_input(self):
        """Handles None input gracefully."""
        result = analyze_monotony(None)
        assert result.level == "normal"
        assert result.range_utilization == 50.0

    def test_monotone_detection(self):
        """Detects monotone speech with low range."""
        # Very narrow pitch range
        pitches = np.array([150.0, 152.0, 148.0, 151.0, 149.0] * 10)
        result = analyze_monotony(pitches)

        assert result.level in ["very_monotone", "monotone"]
        assert result.range_utilization < 50.0

    def test_expressive_detection(self):
        """Detects expressive speech with wide range."""
        # Wide pitch range
        pitches = np.array([100.0, 200.0, 150.0, 250.0, 120.0] * 10)
        result = analyze_monotony(pitches)

        assert result.level in ["expressive", "very_expressive"]
        assert result.range_utilization > 50.0

    def test_speaker_type_affects_expectation(self):
        """Speaker type affects expected range."""
        pitches = np.array([100.0, 150.0, 125.0, 140.0, 110.0] * 10)

        result_male = analyze_monotony(pitches, speaker_type="male")
        result_female = analyze_monotony(pitches, speaker_type="female")

        # Same absolute range, but different utilization due to expectations
        assert result_male.expected_range_hz == 80.0
        assert result_female.expected_range_hz == 100.0


class TestAnalyzePitchSlope:
    """Tests for analyze_pitch_slope function."""

    def test_none_input(self):
        """Handles None input gracefully."""
        result = analyze_pitch_slope(None, None)
        assert result.direction == "flat"
        assert result.slope_hz_per_sec == 0.0

    def test_rising_slope(self):
        """Detects rising slope."""
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        # 100 -> 200 Hz over 2 seconds = 50 Hz/sec
        pitches = np.array([100.0, 125.0, 150.0, 175.0, 200.0])
        result = analyze_pitch_slope(times, pitches)

        assert result.direction == "rising"
        assert result.slope_hz_per_sec > 10.0

    def test_falling_slope(self):
        """Detects falling slope."""
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        pitches = np.array([200.0, 175.0, 150.0, 125.0, 100.0])
        result = analyze_pitch_slope(times, pitches)

        assert result.direction == "falling"
        assert result.slope_hz_per_sec < -10.0

    def test_flat_slope(self):
        """Detects flat slope."""
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        pitches = np.array([150.0, 152.0, 148.0, 151.0, 149.0])
        result = analyze_pitch_slope(times, pitches)

        assert result.direction == "flat"
        assert abs(result.slope_hz_per_sec) < 10.0


class TestExtractProsodyExtended:
    """Tests for extract_prosody_extended function."""

    def test_synthetic_audio(self):
        """Tests with synthetic audio (may fail without parselmouth)."""
        # Generate simple sine wave audio
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        # 200 Hz sine wave
        audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)

        result = extract_prosody_extended(audio, sr)

        assert isinstance(result, ProsodyExtendedState)
        # May have success or no_pitch_detected depending on parselmouth availability
        assert result.extraction_status in ["success", "no_pitch_detected"]

    def test_empty_audio(self):
        """Handles empty audio."""
        audio = np.array([], dtype=np.float32)
        result = extract_prosody_extended(audio, 16000)

        assert isinstance(result, ProsodyExtendedState)
        assert result.extraction_status == "no_pitch_detected"

    def test_silent_audio(self):
        """Handles silent audio."""
        audio = np.zeros(16000, dtype=np.float32)
        result = extract_prosody_extended(audio, 16000)

        assert isinstance(result, ProsodyExtendedState)
        # Silent audio has no pitch
        assert result.extraction_status == "no_pitch_detected"
