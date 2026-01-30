"""Extended prosody features (Prosody 2.0).

This module provides advanced prosodic analysis features:
- Boundary tone: Final 200ms pitch slope → rising/falling/flat
- Monotony: Pitch range utilization % (engagement indicator)
- Pitch slope: Hz/sec trend over utterance

These features extend the base prosody module with more nuanced
analysis for conversation intelligence applications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

# Try to import optional audio processing libraries
try:
    import parselmouth
    from parselmouth.praat import call

    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    logger.debug("parselmouth not available - extended prosody features limited")


BoundaryToneType = Literal["rising", "falling", "flat", "unknown"]
MonotonyLevel = Literal["very_monotone", "monotone", "normal", "expressive", "very_expressive"]


@dataclass(slots=True)
class BoundaryToneState:
    """Boundary tone analysis result.

    Attributes:
        tone: Detected boundary tone type.
        final_slope_hz_per_sec: Pitch slope in final window (Hz/sec).
        confidence: Detection confidence (0.0-1.0).
        window_duration_ms: Duration of analysis window.
    """

    tone: BoundaryToneType = "unknown"
    final_slope_hz_per_sec: float | None = None
    confidence: float = 0.0
    window_duration_ms: float = 200.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tone": self.tone,
            "final_slope_hz_per_sec": self.final_slope_hz_per_sec,
            "confidence": self.confidence,
            "window_duration_ms": self.window_duration_ms,
        }


@dataclass(slots=True)
class MonotonyState:
    """Monotony analysis result.

    Attributes:
        level: Categorical monotony level.
        range_utilization: Pitch range utilization percentage (0-100).
        pitch_range_hz: Actual pitch range in Hz.
        expected_range_hz: Expected pitch range for speaker type.
    """

    level: MonotonyLevel = "normal"
    range_utilization: float = 50.0
    pitch_range_hz: float | None = None
    expected_range_hz: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "range_utilization": self.range_utilization,
            "pitch_range_hz": self.pitch_range_hz,
            "expected_range_hz": self.expected_range_hz,
        }


@dataclass(slots=True)
class PitchSlopeState:
    """Pitch slope analysis result.

    Attributes:
        slope_hz_per_sec: Overall pitch slope in Hz/sec.
        direction: Slope direction (rising/falling/flat).
        r_squared: Goodness of fit for linear regression.
    """

    slope_hz_per_sec: float = 0.0
    direction: Literal["rising", "falling", "flat"] = "flat"
    r_squared: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slope_hz_per_sec": self.slope_hz_per_sec,
            "direction": self.direction,
            "r_squared": self.r_squared,
        }


@dataclass(slots=True)
class ProsodyExtendedState:
    """Extended prosody analysis result.

    Attributes:
        boundary_tone: Boundary tone analysis.
        monotony: Monotony analysis.
        pitch_slope: Pitch slope analysis.
        extraction_status: Success/failure status.
    """

    boundary_tone: BoundaryToneState = field(default_factory=BoundaryToneState)
    monotony: MonotonyState = field(default_factory=MonotonyState)
    pitch_slope: PitchSlopeState = field(default_factory=PitchSlopeState)
    extraction_status: str = "success"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "boundary_tone": self.boundary_tone.to_dict(),
            "monotony": self.monotony.to_dict(),
            "pitch_slope": self.pitch_slope.to_dict(),
            "extraction_status": self.extraction_status,
        }


# Thresholds for boundary tone classification
BOUNDARY_SLOPE_THRESHOLDS = {
    "rising": 30.0,  # Hz/sec threshold for rising
    "falling": -30.0,  # Hz/sec threshold for falling
}

# Thresholds for monotony classification (range utilization %)
MONOTONY_THRESHOLDS = {
    "very_monotone": 20.0,
    "monotone": 40.0,
    "normal": 60.0,
    "expressive": 80.0,
    # Above 80% is very_expressive
}

# Expected pitch range for different speaker types
EXPECTED_PITCH_RANGES = {
    "male": 80.0,  # Expected range in Hz
    "female": 100.0,
    "default": 90.0,
}


def extract_pitch_contour(
    audio: np.ndarray,
    sr: int,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 500.0,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Extract pitch contour from audio using Parselmouth.

    Args:
        audio: Audio samples (mono).
        sr: Sample rate.
        time_step: Time step for pitch extraction (seconds).
        pitch_floor: Minimum pitch to detect (Hz).
        pitch_ceiling: Maximum pitch to detect (Hz).

    Returns:
        Tuple of (times, pitch_values) arrays, or (None, None) if extraction fails.
    """
    if not PARSELMOUTH_AVAILABLE:
        return None, None

    try:
        # Convert to float64 if needed
        if audio.dtype != np.float64:
            audio = audio.astype(np.float64)

        # Create Parselmouth Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)

        # Extract pitch
        pitch = call(
            sound,
            "To Pitch",
            time_step,
            pitch_floor,
            pitch_ceiling,
        )

        # Get pitch values
        times = pitch.xs()
        pitch_values = np.array([pitch.get_value_at_time(t) for t in times])

        # Filter out undefined values (0 or NaN)
        valid_mask = ~np.isnan(pitch_values) & (pitch_values > 0)

        if np.sum(valid_mask) < 3:
            return None, None

        return times[valid_mask], pitch_values[valid_mask]

    except Exception as e:
        logger.debug(f"Pitch extraction failed: {e}")
        return None, None


def analyze_boundary_tone(
    pitch_times: np.ndarray,
    pitch_values: np.ndarray,
    window_ms: float = 200.0,
) -> BoundaryToneState:
    """Analyze boundary tone from pitch contour.

    Examines the final portion of the pitch contour to determine
    if the utterance ends with rising, falling, or flat intonation.

    Args:
        pitch_times: Time points for pitch values.
        pitch_values: Pitch values in Hz.
        window_ms: Duration of final window to analyze (ms).

    Returns:
        BoundaryToneState with analysis results.
    """
    if pitch_times is None or len(pitch_times) < 3:
        return BoundaryToneState(tone="unknown", confidence=0.0)

    # Get final window
    end_time = pitch_times[-1]
    window_start = end_time - (window_ms / 1000.0)

    # Find points in final window
    window_mask = pitch_times >= window_start
    window_times = pitch_times[window_mask]
    window_pitches = pitch_values[window_mask]

    if len(window_times) < 2:
        return BoundaryToneState(tone="unknown", confidence=0.0)

    # Compute slope via linear regression
    slope, intercept, r_squared = _linear_regression(window_times, window_pitches)

    # Classify based on slope
    if slope >= BOUNDARY_SLOPE_THRESHOLDS["rising"]:
        tone: BoundaryToneType = "rising"
    elif slope <= BOUNDARY_SLOPE_THRESHOLDS["falling"]:
        tone = "falling"
    else:
        tone = "flat"

    # Confidence based on R-squared and number of points
    confidence = min(r_squared, 1.0) * min(len(window_times) / 5, 1.0)

    return BoundaryToneState(
        tone=tone,
        final_slope_hz_per_sec=slope,
        confidence=confidence,
        window_duration_ms=window_ms,
    )


def analyze_monotony(
    pitch_values: np.ndarray,
    speaker_type: str = "default",
) -> MonotonyState:
    """Analyze pitch range utilization (monotony).

    Compares actual pitch range to expected range for speaker type
    to determine how expressive/monotone the speech is.

    Args:
        pitch_values: Pitch values in Hz.
        speaker_type: Speaker type ("male", "female", "default").

    Returns:
        MonotonyState with analysis results.
    """
    if pitch_values is None or len(pitch_values) < 3:
        return MonotonyState(level="normal", range_utilization=50.0)

    # Calculate actual range (use percentiles to be robust to outliers)
    p10 = np.percentile(pitch_values, 10)
    p90 = np.percentile(pitch_values, 90)
    actual_range = p90 - p10

    # Get expected range
    expected_range = EXPECTED_PITCH_RANGES.get(speaker_type, EXPECTED_PITCH_RANGES["default"])

    # Calculate utilization percentage
    range_utilization = (actual_range / expected_range) * 100.0
    range_utilization = min(range_utilization, 150.0)  # Cap at 150%

    # Classify monotony level
    if range_utilization < MONOTONY_THRESHOLDS["very_monotone"]:
        level: MonotonyLevel = "very_monotone"
    elif range_utilization < MONOTONY_THRESHOLDS["monotone"]:
        level = "monotone"
    elif range_utilization < MONOTONY_THRESHOLDS["normal"]:
        level = "normal"
    elif range_utilization < MONOTONY_THRESHOLDS["expressive"]:
        level = "expressive"
    else:
        level = "very_expressive"

    return MonotonyState(
        level=level,
        range_utilization=range_utilization,
        pitch_range_hz=actual_range,
        expected_range_hz=expected_range,
    )


def analyze_pitch_slope(
    pitch_times: np.ndarray,
    pitch_values: np.ndarray,
) -> PitchSlopeState:
    """Analyze overall pitch slope across utterance.

    Fits a linear regression to the pitch contour to determine
    the overall trend direction and strength.

    Args:
        pitch_times: Time points for pitch values.
        pitch_values: Pitch values in Hz.

    Returns:
        PitchSlopeState with analysis results.
    """
    if pitch_times is None or len(pitch_times) < 3:
        return PitchSlopeState(slope_hz_per_sec=0.0, direction="flat", r_squared=0.0)

    # Compute linear regression
    slope, intercept, r_squared = _linear_regression(pitch_times, pitch_values)

    # Classify direction
    if slope > 10.0:
        direction: Literal["rising", "falling", "flat"] = "rising"
    elif slope < -10.0:
        direction = "falling"
    else:
        direction = "flat"

    return PitchSlopeState(
        slope_hz_per_sec=slope,
        direction=direction,
        r_squared=r_squared,
    )


def _linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Compute simple linear regression.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        Tuple of (slope, intercept, r_squared).
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Compute slope
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0, y_mean, 0.0

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Compute R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    if ss_tot == 0:
        r_squared = 0.0
    else:
        r_squared = 1 - (ss_res / ss_tot)

    return slope, intercept, max(0.0, r_squared)


def extract_prosody_extended(
    audio: np.ndarray,
    sr: int,
    speaker_type: str = "default",
    boundary_window_ms: float = 200.0,
) -> ProsodyExtendedState:
    """Extract extended prosody features from audio.

    Main entry point for extended prosody analysis.

    Args:
        audio: Audio samples (mono, float).
        sr: Sample rate.
        speaker_type: Speaker type for monotony analysis.
        boundary_window_ms: Window size for boundary tone analysis.

    Returns:
        ProsodyExtendedState with all analysis results.
    """
    result = ProsodyExtendedState()

    # Extract pitch contour
    pitch_times, pitch_values = extract_pitch_contour(audio, sr)

    if pitch_times is None or pitch_values is None:
        result.extraction_status = "no_pitch_detected"
        return result

    # Analyze boundary tone
    result.boundary_tone = analyze_boundary_tone(
        pitch_times, pitch_values, boundary_window_ms
    )

    # Analyze monotony
    result.monotony = analyze_monotony(pitch_values, speaker_type)

    # Analyze pitch slope
    result.pitch_slope = analyze_pitch_slope(pitch_times, pitch_values)

    result.extraction_status = "success"
    return result


__all__ = [
    "ProsodyExtendedState",
    "BoundaryToneState",
    "MonotonyState",
    "PitchSlopeState",
    "BoundaryToneType",
    "MonotonyLevel",
    "extract_prosody_extended",
    "analyze_boundary_tone",
    "analyze_monotony",
    "analyze_pitch_slope",
    "extract_pitch_contour",
    "PARSELMOUTH_AVAILABLE",
]
