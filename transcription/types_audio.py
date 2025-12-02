"""
TypedDict definitions for audio_state structure.

These types match AUDIO_STATE_VERSION = "1.0.0" schema as defined in the
audio enrichment pipeline (transcription/audio_enrichment.py).

All TypedDicts use total=False since fields are optional in practice
(features may be missing due to extraction failures or partial enrichment).
"""

from typing import TypedDict


class PauseState(TypedDict, total=False):
    """Pause/silence metrics extracted from audio.

    Attributes:
        longest_ms: Duration of longest pause in milliseconds
        count: Number of pauses detected
        total_ms: Total duration of all pauses
        rate_per_min: Pause frequency (pauses per minute)
    """

    longest_ms: float | int | None
    count: int | None
    total_ms: float | int | None
    rate_per_min: float | None


class PitchState(TypedDict, total=False):
    """Pitch/F0 metrics extracted from audio.

    Attributes:
        mean_hz: Mean fundamental frequency in Hz
        level: Categorical level relative to speaker baseline (low/normal/high)
        contour: Pitch movement pattern (flat/rising/falling/dynamic)
        std_hz: Standard deviation of pitch in Hz
    """

    mean_hz: float | int | None
    level: str
    contour: str
    std_hz: float | int | None


class EnergyState(TypedDict, total=False):
    """Energy/loudness metrics extracted from audio.

    Attributes:
        db_rms: RMS energy in decibels
        level: Categorical level relative to speaker baseline (quiet/normal/loud)
    """

    db_rms: float | int | None
    level: str


class RateState(TypedDict, total=False):
    """Speech rate metrics extracted from audio.

    Attributes:
        syllables_per_sec: Estimated syllable rate
        level: Categorical level relative to speaker baseline (slow/normal/fast)
    """

    syllables_per_sec: float | None
    level: str


class ProsodyState(TypedDict, total=False):
    """Prosodic features container.

    Groups all prosody-related metrics (pitch, energy, rate, pauses).
    Extracted by transcription/prosody.py.
    """

    pitch: PitchState
    energy: EnergyState
    rate: RateState
    pauses: PauseState


class ValenceState(TypedDict, total=False):
    """Valence (positive/negative emotion) dimension.

    Attributes:
        level: Categorical level (negative/neutral/positive)
        score: Continuous score in [0, 1] range
    """

    level: str
    score: float


class ArousalState(TypedDict, total=False):
    """Arousal (calm/excited) emotion dimension.

    Attributes:
        level: Categorical level (low/medium/high)
        score: Continuous score in [0, 1] range
    """

    level: str
    score: float


class EmotionState(TypedDict, total=False):
    """Emotion recognition features container.

    Uses dimensional model (valence/arousal) from wav2vec2 models.
    Extracted by transcription/emotion.py.
    """

    valence: ValenceState
    arousal: ArousalState


class ExtractionStatus(TypedDict, total=False):
    """Status tracking for audio feature extraction.

    Records success/failure of individual feature extractors.
    Typical values: "success", "error", "skipped", "unavailable"

    Attributes:
        prosody: Status of prosody extraction
        emotion_dimensional: Status of dimensional emotion extraction
        emotion_categorical: Status of categorical emotion extraction
        errors: List of error messages from failed extractions
    """

    prosody: str
    emotion_dimensional: str
    emotion_categorical: str
    errors: list[str]


class AudioState(TypedDict, total=False):
    """Top-level audio enrichment container.

    Attached to Segment.audio_state when audio enrichment is enabled.
    Version controlled by AUDIO_STATE_VERSION in transcription/models.py.

    Attributes:
        prosody: Prosodic features (pitch, energy, rate, pauses)
        emotion: Emotional features (valence, arousal)
        rendering: Human-readable text rendering of audio features
        extraction_status: Per-feature extraction success/failure tracking
    """

    prosody: ProsodyState
    emotion: EmotionState
    rendering: str
    extraction_status: ExtractionStatus
