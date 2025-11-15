"""
Prosody feature extraction module.

This module extracts prosodic features from audio segments including:
- Pitch (fundamental frequency): mean, std, contour
- Energy (intensity): RMS mean, std, dB levels
- Speech rate: syllables per second, words per second
- Pauses: count, duration, density

Features can be normalized relative to speaker baselines and mapped to
categorical levels for easier interpretation.
"""

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import parselmouth
    from parselmouth.praat import call

    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    logger.warning("parselmouth not available. Pitch extraction will be limited.")

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available. Energy extraction will be limited.")


# Threshold constants for categorical mapping (based on typical speech)
PITCH_THRESHOLDS = {
    "very_low": 80,  # Hz
    "low": 120,
    "neutral": 180,
    "high": 250,
    "very_high": float("inf"),
}

ENERGY_THRESHOLDS = {
    "very_quiet": -35,  # dB RMS
    "quiet": -25,
    "normal": -15,
    "loud": -8,
    "very_loud": float("inf"),
}

RATE_THRESHOLDS = {
    "very_slow": 3.0,  # syllables per second
    "slow": 4.5,
    "normal": 5.5,
    "fast": 7.0,
    "very_fast": float("inf"),
}

PAUSE_DENSITY_THRESHOLDS = {
    "very_sparse": 0.5,  # pauses per second
    "sparse": 1.0,
    "moderate": 1.5,
    "frequent": 2.5,
    "very_frequent": float("inf"),
}


def categorize_value(value: float, thresholds: dict[str, float]) -> str:
    """
    Map a numeric value to a categorical level based on thresholds.

    Args:
        value: Numeric value to categorize
        thresholds: Dict mapping category names to upper bounds (sorted)

    Returns:
        Category name (e.g., 'low', 'normal', 'high')
    """
    for category, threshold in thresholds.items():
        if value < threshold:
            return category
    return list(thresholds.keys())[-1]


def normalize_to_baseline(value: float, baseline_median: float, baseline_std: float = None) -> str:
    """
    Normalize a value relative to speaker baseline and categorize.

    Args:
        value: Current value
        baseline_median: Speaker's median value
        baseline_std: Speaker's standard deviation (optional)

    Returns:
        Relative category: 'very_low', 'low', 'neutral', 'high', 'very_high'
    """
    if baseline_std and baseline_std > 0:
        # Use z-score if we have std
        z_score = (value - baseline_median) / baseline_std
        if z_score < -1.5:
            return "very_low"
        if z_score < -0.5:
            return "low"
        if z_score < 0.5:
            return "neutral"
        if z_score < 1.5:
            return "high"
        return "very_high"
    # Use percentage difference if no std
    ratio = value / baseline_median if baseline_median > 0 else 1.0
    if ratio < 0.7:
        return "very_low"
    if ratio < 0.9:
        return "low"
    if ratio < 1.1:
        return "neutral"
    if ratio < 1.3:
        return "high"
    return "very_high"


def extract_pitch_features(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """
    Extract pitch-related features using Parselmouth.

    Args:
        audio: Audio signal as numpy array
        sr: Sample rate in Hz

    Returns:
        Dict with pitch features: mean_hz, std_hz, contour
    """
    if not PARSELMOUTH_AVAILABLE:
        logger.warning("Parselmouth not available, returning default pitch features")
        return {
            "mean_hz": None,
            "std_hz": None,
            "contour": "unknown",
            "min_hz": None,
            "max_hz": None,
        }

    try:
        # Convert to Parselmouth Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)

        # Extract pitch using autocorrelation (standard Praat method)
        pitch = call(sound, "To Pitch", 0.0, 75, 600)  # 75-600 Hz range

        # Get pitch values at regular intervals
        pitch_values = []
        for t in np.arange(sound.xmin, sound.xmax, 0.01):  # 10ms steps
            f0 = call(pitch, "Get value at time", t, "Hertz", "Linear")
            if f0 and not np.isnan(f0) and f0 > 0:
                pitch_values.append(f0)

        if not pitch_values:
            return {
                "mean_hz": None,
                "std_hz": None,
                "contour": "flat",
                "min_hz": None,
                "max_hz": None,
            }

        pitch_array = np.array(pitch_values)
        mean_pitch = float(np.mean(pitch_array))
        std_pitch = float(np.std(pitch_array))
        min_pitch = float(np.min(pitch_array))
        max_pitch = float(np.max(pitch_array))

        # Determine pitch contour (rising/falling/flat)
        # Use linear regression on pitch values over time
        if len(pitch_values) > 2:
            x = np.arange(len(pitch_values))
            slope = np.polyfit(x, pitch_values, 1)[0]

            # Threshold: >0.5 Hz per 10ms interval = rising
            if slope > 0.5:
                contour = "rising"
            elif slope < -0.5:
                contour = "falling"
            else:
                contour = "flat"
        else:
            contour = "flat"

        return {
            "mean_hz": mean_pitch,
            "std_hz": std_pitch,
            "contour": contour,
            "min_hz": min_pitch,
            "max_hz": max_pitch,
        }

    except Exception as e:
        logger.error(f"Error extracting pitch features: {e}")
        return {
            "mean_hz": None,
            "std_hz": None,
            "contour": "unknown",
            "min_hz": None,
            "max_hz": None,
        }


def extract_energy_features(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """
    Extract energy/intensity features using librosa.

    Args:
        audio: Audio signal as numpy array
        sr: Sample rate in Hz

    Returns:
        Dict with energy features: rms_mean, rms_std, db_rms
    """
    if not LIBROSA_AVAILABLE:
        logger.warning("Librosa not available, using numpy for energy extraction")
        # Fallback to simple RMS calculation
        try:
            rms = np.sqrt(np.mean(audio**2))
            db_rms = 20 * np.log10(rms + 1e-10) if rms > 0 else -100
            return {"rms_mean": float(rms), "rms_std": 0.0, "db_rms": float(db_rms)}
        except Exception as e:
            logger.error(f"Error in fallback energy extraction: {e}")
            return {"rms_mean": None, "rms_std": None, "db_rms": None}

    try:
        # Extract RMS energy with frame length of 2048 samples
        frame_length = min(2048, len(audio))
        hop_length = frame_length // 4

        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        # Convert to dB
        db_rms = 20 * np.log10(rms_mean + 1e-10)

        return {"rms_mean": rms_mean, "rms_std": rms_std, "db_rms": float(db_rms)}

    except Exception as e:
        logger.error(f"Error extracting energy features: {e}")
        return {"rms_mean": None, "rms_std": None, "db_rms": None}


def count_syllables(text: str) -> int:
    """
    Estimate syllable count using simple heuristic.

    This uses a vowel-counting approximation. More sophisticated
    methods could use NLP libraries like pyphen or nltk.

    Args:
        text: Text string

    Returns:
        Estimated syllable count
    """
    if not text:
        return 0

    text = text.lower().strip()

    # Remove punctuation
    text = re.sub(r"[^a-z\s]", "", text)

    # Count vowel groups
    vowels = "aeiouy"
    syllable_count = 0
    previous_was_vowel = False

    for char in text:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel

    # Adjust for common patterns
    # Remove silent 'e' at end of words
    words = text.split()
    for word in words:
        if len(word) > 2 and word.endswith("e"):
            syllable_count -= 1

    # At least one syllable per word
    syllable_count = max(syllable_count, len(words))

    return syllable_count


def detect_pauses(
    audio: np.ndarray, sr: int, silence_threshold: float = -40, min_pause_duration: float = 0.15
) -> list[tuple[float, float]]:
    """
    Detect pauses (silence regions) in audio.

    Args:
        audio: Audio signal as numpy array
        sr: Sample rate in Hz
        silence_threshold: Threshold in dB below which audio is considered silence
        min_pause_duration: Minimum pause duration in seconds to count

    Returns:
        List of (start_time, end_time) tuples for each pause
    """
    try:
        # Calculate frame energy
        frame_length = 2048
        hop_length = frame_length // 4

        if LIBROSA_AVAILABLE:
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        else:
            # Fallback: manual frame-based RMS
            num_frames = 1 + (len(audio) - frame_length) // hop_length
            rms = np.zeros(num_frames)
            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                frame = audio[start:end]
                rms[i] = np.sqrt(np.mean(frame**2))

        # Convert to dB
        db = 20 * np.log10(rms + 1e-10)

        # Detect silent frames
        is_silent = db < silence_threshold

        # Convert frame indices to time
        times = (
            librosa.frames_to_time(np.arange(len(is_silent)), sr=sr, hop_length=hop_length)
            if LIBROSA_AVAILABLE
            else np.arange(len(is_silent)) * hop_length / sr
        )

        # Find continuous silent regions
        pauses = []
        in_pause = False
        pause_start = 0

        for _i, (t, silent) in enumerate(zip(times, is_silent, strict=False)):
            if silent and not in_pause:
                # Start of pause
                in_pause = True
                pause_start = t
            elif not silent and in_pause:
                # End of pause
                pause_duration = t - pause_start
                if pause_duration >= min_pause_duration:
                    pauses.append((pause_start, t))
                in_pause = False

        # Handle pause at end
        if in_pause:
            pause_duration = times[-1] - pause_start
            if pause_duration >= min_pause_duration:
                pauses.append((pause_start, times[-1]))

        return pauses

    except Exception as e:
        logger.error(f"Error detecting pauses: {e}")
        return []


def extract_speech_rate(audio: np.ndarray, sr: int, text: str, duration: float) -> dict[str, Any]:
    """
    Extract speech rate features.

    Args:
        audio: Audio signal as numpy array
        sr: Sample rate in Hz
        text: Transcribed text
        duration: Duration in seconds

    Returns:
        Dict with rate features: syllables_per_sec, words_per_sec
    """
    if duration <= 0:
        return {"syllables_per_sec": None, "words_per_sec": None}

    try:
        # Count syllables and words
        syllable_count = count_syllables(text)
        word_count = len(text.split())

        # Calculate rates
        syllables_per_sec = syllable_count / duration
        words_per_sec = word_count / duration

        return {
            "syllables_per_sec": float(syllables_per_sec),
            "words_per_sec": float(words_per_sec),
        }

    except Exception as e:
        logger.error(f"Error extracting speech rate: {e}")
        return {"syllables_per_sec": None, "words_per_sec": None}


def extract_pause_features(audio: np.ndarray, sr: int, duration: float) -> dict[str, Any]:
    """
    Extract pause-related features.

    Args:
        audio: Audio signal as numpy array
        sr: Sample rate in Hz
        duration: Duration in seconds

    Returns:
        Dict with pause features: count, longest_ms, total_duration_ms, density
    """
    pauses = detect_pauses(audio, sr)

    if not pauses:
        return {"count": 0, "longest_ms": 0, "total_duration_ms": 0, "density_per_sec": 0.0}

    pause_durations = [(end - start) for start, end in pauses]
    longest_pause = max(pause_durations)
    total_pause_duration = sum(pause_durations)
    pause_count = len(pauses)

    density = pause_count / duration if duration > 0 else 0.0

    return {
        "count": pause_count,
        "longest_ms": int(longest_pause * 1000),
        "total_duration_ms": int(total_pause_duration * 1000),
        "density_per_sec": float(density),
    }


def extract_prosody(
    audio: np.ndarray,
    sr: int,
    text: str,
    speaker_baseline: dict[str, Any] | None = None,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> dict[str, Any]:
    """
    Extract comprehensive prosodic features from an audio segment.

    This is the main entry point for prosody extraction. It combines
    pitch, energy, rate, and pause features into a structured dictionary.

    Args:
        audio: Audio signal as numpy array (mono)
        sr: Sample rate in Hz
        text: Transcribed text for this segment
        speaker_baseline: Optional dict with speaker baseline statistics
                         e.g., {'pitch_median': 180, 'pitch_std': 30,
                                'energy_median': -15, 'rate_median': 5.2}
        start_time: Start time of segment (for logging)
        end_time: End time of segment (for logging)

    Returns:
        Dict matching the JSON schema:
        {
            "pitch": {
                "level": "high",
                "mean_hz": 245.3,
                "std_hz": 32.1,
                "variation": "moderate",
                "contour": "rising"
            },
            "energy": {
                "level": "loud",
                "db_rms": -8.2,
                "variation": "low"
            },
            "rate": {
                "level": "fast",
                "syllables_per_sec": 6.3,
                "words_per_sec": 3.1
            },
            "pauses": {
                "count": 2,
                "longest_ms": 320,
                "density": "sparse"
            }
        }
    """
    # Calculate duration
    duration = len(audio) / sr if sr > 0 else 0

    # Initialize result with defaults
    result = {
        "pitch": {
            "level": "unknown",
            "mean_hz": None,
            "std_hz": None,
            "variation": "unknown",
            "contour": "unknown",
        },
        "energy": {"level": "unknown", "db_rms": None, "variation": "unknown"},
        "rate": {"level": "unknown", "syllables_per_sec": None, "words_per_sec": None},
        "pauses": {"count": 0, "longest_ms": 0, "density": "unknown"},
    }

    try:
        # Extract pitch features
        pitch_features = extract_pitch_features(audio, sr)
        result["pitch"]["mean_hz"] = pitch_features["mean_hz"]
        result["pitch"]["std_hz"] = pitch_features["std_hz"]
        result["pitch"]["contour"] = pitch_features["contour"]

        # Categorize pitch level
        if pitch_features["mean_hz"] is not None:
            if speaker_baseline and "pitch_median" in speaker_baseline:
                result["pitch"]["level"] = normalize_to_baseline(
                    pitch_features["mean_hz"],
                    speaker_baseline["pitch_median"],
                    speaker_baseline.get("pitch_std"),
                )
            else:
                result["pitch"]["level"] = categorize_value(
                    pitch_features["mean_hz"], PITCH_THRESHOLDS
                )

        # Categorize pitch variation
        if pitch_features["std_hz"] is not None:
            if pitch_features["std_hz"] < 15:
                result["pitch"]["variation"] = "low"
            elif pitch_features["std_hz"] < 30:
                result["pitch"]["variation"] = "moderate"
            else:
                result["pitch"]["variation"] = "high"

        # Extract energy features
        energy_features = extract_energy_features(audio, sr)
        result["energy"]["db_rms"] = energy_features["db_rms"]

        # Categorize energy level
        if energy_features["db_rms"] is not None:
            if speaker_baseline and "energy_median" in speaker_baseline:
                result["energy"]["level"] = normalize_to_baseline(
                    energy_features["db_rms"],
                    speaker_baseline["energy_median"],
                    speaker_baseline.get("energy_std"),
                )
            else:
                result["energy"]["level"] = categorize_value(
                    energy_features["db_rms"], ENERGY_THRESHOLDS
                )

        # Categorize energy variation
        if energy_features["rms_std"] is not None and energy_features["rms_mean"] is not None:
            # Coefficient of variation
            cv = (
                energy_features["rms_std"] / energy_features["rms_mean"]
                if energy_features["rms_mean"] > 0
                else 0
            )
            if cv < 0.2:
                result["energy"]["variation"] = "low"
            elif cv < 0.4:
                result["energy"]["variation"] = "moderate"
            else:
                result["energy"]["variation"] = "high"

        # Extract speech rate
        rate_features = extract_speech_rate(audio, sr, text, duration)
        result["rate"]["syllables_per_sec"] = rate_features["syllables_per_sec"]
        result["rate"]["words_per_sec"] = rate_features["words_per_sec"]

        # Categorize speech rate
        if rate_features["syllables_per_sec"] is not None:
            if speaker_baseline and "rate_median" in speaker_baseline:
                result["rate"]["level"] = normalize_to_baseline(
                    rate_features["syllables_per_sec"],
                    speaker_baseline["rate_median"],
                    speaker_baseline.get("rate_std"),
                )
            else:
                result["rate"]["level"] = categorize_value(
                    rate_features["syllables_per_sec"], RATE_THRESHOLDS
                )

        # Extract pause features
        pause_features = extract_pause_features(audio, sr, duration)
        result["pauses"]["count"] = pause_features["count"]
        result["pauses"]["longest_ms"] = pause_features["longest_ms"]

        # Categorize pause density
        if pause_features["density_per_sec"] is not None:
            result["pauses"]["density"] = categorize_value(
                pause_features["density_per_sec"], PAUSE_DENSITY_THRESHOLDS
            )

    except Exception as e:
        logger.error(f"Error in prosody extraction: {e}")

    return result


def compute_speaker_baseline(segments_data: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute baseline statistics across multiple segments for a speaker.

    This can be used to normalize prosodic features relative to a speaker's
    typical patterns.

    Args:
        segments_data: List of dicts, each containing:
                      {'audio': np.ndarray, 'sr': int, 'text': str}

    Returns:
        Dict with baseline statistics:
        {
            'pitch_median': 180.0,
            'pitch_std': 28.5,
            'energy_median': -15.2,
            'energy_std': 3.1,
            'rate_median': 5.3,
            'rate_std': 0.8
        }
    """
    pitch_values = []
    energy_values = []
    rate_values = []

    for segment in segments_data:
        audio = segment["audio"]
        sr = segment["sr"]
        text = segment.get("text", "")
        duration = len(audio) / sr

        # Extract features
        pitch_features = extract_pitch_features(audio, sr)
        if pitch_features["mean_hz"] is not None:
            pitch_values.append(pitch_features["mean_hz"])

        energy_features = extract_energy_features(audio, sr)
        if energy_features["db_rms"] is not None:
            energy_values.append(energy_features["db_rms"])

        rate_features = extract_speech_rate(audio, sr, text, duration)
        if rate_features["syllables_per_sec"] is not None:
            rate_values.append(rate_features["syllables_per_sec"])

    baseline = {}

    if pitch_values:
        baseline["pitch_median"] = float(np.median(pitch_values))
        baseline["pitch_std"] = float(np.std(pitch_values))

    if energy_values:
        baseline["energy_median"] = float(np.median(energy_values))
        baseline["energy_std"] = float(np.std(energy_values))

    if rate_values:
        baseline["rate_median"] = float(np.median(rate_values))
        baseline["rate_std"] = float(np.std(rate_values))

    return baseline
