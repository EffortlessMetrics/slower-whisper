"""TTS style metadata computation from prosody and emotion data.

This module provides utilities for computing TTS (Text-to-Speech) style
metadata based on prosodic features and emotional characteristics of speech.
The computed metadata can be used to guide response generation in
conversational AI systems.
"""

from dataclasses import dataclass

__all__ = ["TTSStyleMetadata", "compute_tts_style"]


@dataclass
class TTSStyleMetadata:
    """Metadata describing the style characteristics for TTS response generation.

    This dataclass captures the acoustic and emotional profile of speech input
    to inform appropriate response styling.

    Attributes:
        energy_level: Volume/intensity level - "quiet", "normal", or "loud".
        speech_rate: Speaking pace - "slow", "normal", or "fast".
        valence: Emotional valence - "negative", "neutral", or "positive".
        arousal: Emotional activation level - "low", "medium", or "high".
        recommended_response_mode: Suggested response strategy -
            "de-escalate", "match", "calm", or "neutral".
    """

    energy_level: str
    speech_rate: str
    valence: str
    arousal: str
    recommended_response_mode: str

    def to_dict(self) -> dict[str, str]:
        """Convert the metadata to a dictionary.

        Returns:
            Dictionary with all style metadata fields.
        """
        return {
            "energy_level": self.energy_level,
            "speech_rate": self.speech_rate,
            "valence": self.valence,
            "arousal": self.arousal,
            "recommended_response_mode": self.recommended_response_mode,
        }


def compute_tts_style(
    prosody: dict | None,
    emotion: dict | None,
) -> TTSStyleMetadata:
    """Compute TTS style metadata from prosody and emotion data.

    Maps raw prosodic features and emotional characteristics to discrete
    style categories suitable for guiding TTS response generation.

    Args:
        prosody: Dictionary containing prosodic features. Expected keys:
            - "energy": float in [0, 1] representing volume/intensity
            - "speech_rate": float where 1.0 is normal pace
            May be None or empty if prosody data is unavailable.
        emotion: Dictionary containing emotional characteristics. Expected keys:
            - "valence": float in [-1, 1] from negative to positive
            - "arousal": float in [0, 1] representing activation level
            May be None or empty if emotion data is unavailable.

    Returns:
        TTSStyleMetadata with computed style characteristics.
        Returns neutral defaults when input data is missing.

    Examples:
        >>> # With full data
        >>> prosody = {"energy": 0.8, "speech_rate": 1.3}
        >>> emotion = {"valence": -0.5, "arousal": 0.9}
        >>> style = compute_tts_style(prosody, emotion)
        >>> style.energy_level
        'loud'
        >>> style.recommended_response_mode
        'de-escalate'

        >>> # With missing data
        >>> style = compute_tts_style(None, None)
        >>> style.energy_level
        'normal'
        >>> style.recommended_response_mode
        'neutral'
    """
    # Extract prosody values with defaults
    if prosody:
        energy = prosody.get("energy")
        rate = prosody.get("speech_rate")
    else:
        energy = None
        rate = None

    # Extract emotion values with defaults
    if emotion:
        valence_value = emotion.get("valence")
        arousal_value = emotion.get("arousal")
    else:
        valence_value = None
        arousal_value = None

    # Map energy to energy_level
    if energy is None:
        energy_level = "normal"
    elif energy < 0.3:
        energy_level = "quiet"
    elif energy > 0.7:
        energy_level = "loud"
    else:
        energy_level = "normal"

    # Map speech_rate to speech_rate category
    if rate is None:
        speech_rate = "normal"
    elif rate < 0.8:
        speech_rate = "slow"
    elif rate > 1.2:
        speech_rate = "fast"
    else:
        speech_rate = "normal"

    # Map valence to valence category
    if valence_value is None:
        valence = "neutral"
    elif valence_value < -0.3:
        valence = "negative"
    elif valence_value > 0.3:
        valence = "positive"
    else:
        valence = "neutral"

    # Map arousal to arousal category
    if arousal_value is None:
        arousal = "medium"
    elif arousal_value < 0.3:
        arousal = "low"
    elif arousal_value > 0.7:
        arousal = "high"
    else:
        arousal = "medium"

    # Compute recommended response mode based on valence and arousal
    if valence == "negative" and arousal == "high":
        recommended_response_mode = "de-escalate"
    elif arousal == "high":
        recommended_response_mode = "calm"
    elif valence == "positive":
        recommended_response_mode = "match"
    else:
        recommended_response_mode = "neutral"

    return TTSStyleMetadata(
        energy_level=energy_level,
        speech_rate=speech_rate,
        valence=valence,
        arousal=arousal,
        recommended_response_mode=recommended_response_mode,
    )
