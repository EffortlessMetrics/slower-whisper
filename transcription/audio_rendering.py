"""
Audio state rendering utilities.

This module provides functions to render audio_state dictionaries as
human/LLM-friendly text annotations. The annotations capture prosodic,
emotional, and voice quality features in a compact, natural language format.
"""

from typing import Any


def render_audio_state(audio_state: dict[str, Any]) -> str:
    """
    Render an audio_state dictionary as a human-friendly text annotation.

    Converts detailed audio feature measurements (prosody, emotion, voice_quality)
    into a compact, natural language string suitable for display or LLM context.

    Args:
        audio_state: Dictionary containing optional keys:
            - prosody: dict with pitch, volume, speech_rate, pauses
            - emotion: dict with primary tone and confidence
            - voice_quality: dict with clarity, energy, stress level

    Returns:
        A compact string annotation like "[audio: neutral]" or
        "[audio: high pitch, loud volume, fast speech, excited tone]"

    Examples:
        >>> render_audio_state({})
        '[audio: neutral]'

        >>> render_audio_state({'prosody': {'pitch': 'high', 'volume': 'loud', 'speech_rate': 'fast'}, 'emotion': {'tone': 'excited'}})
        '[audio: high pitch, loud volume, fast speech, excited tone]'

        >>> render_audio_state({'prosody': {'pauses': 'moderate'}, 'emotion': {'tone': 'hesitant', 'confidence': 0.6}})
        '[audio: moderate pauses, hesitant tone]'
    """
    features: list[str] = []

    # Extract prosody features
    prosody = audio_state.get("prosody") or {}
    if isinstance(prosody, dict):
        # Pitch descriptor
        if "pitch" in prosody and prosody["pitch"] not in (None, "neutral", "medium"):
            pitch = prosody["pitch"].lower()
            features.append(f"{pitch} pitch")

        # Volume descriptor
        if "volume" in prosody and prosody["volume"] not in (None, "neutral", "medium"):
            volume = prosody["volume"].lower()
            features.append(f"{volume} volume")

        # Speech rate descriptor
        if "speech_rate" in prosody and prosody["speech_rate"] not in (None, "neutral", "normal"):
            rate = prosody["speech_rate"].lower()
            # Convert technical term to natural language
            if rate == "fast":
                features.append("fast speech")
            elif rate == "slow":
                features.append("slow speech")
            elif rate == "rapid":
                features.append("rapid speech")
            else:
                features.append(f"{rate} speech")

        # Pauses descriptor
        if "pauses" in prosody and prosody["pauses"] not in (None, "neutral", "normal", "minimal"):
            pauses = prosody["pauses"].lower()
            if pauses == "frequent":
                features.append("frequent pauses")
            elif pauses == "moderate":
                features.append("moderate pauses")
            elif pauses == "long":
                features.append("long pauses")
            else:
                features.append(f"{pauses} pauses")

    # Extract emotion/tone features
    emotion = audio_state.get("emotion") or {}
    if isinstance(emotion, dict):
        # Tone descriptor
        if "tone" in emotion and emotion["tone"] not in (None, "neutral", "normal"):
            tone = emotion["tone"].lower()
            features.append(f"{tone} tone")

        # Confidence indicator (only if confidence is low, indicating uncertainty)
        confidence = emotion.get("confidence")
        if confidence is not None and isinstance(confidence, int | float):
            if confidence < 0.5:
                features.append("possibly uncertain")
            elif confidence < 0.7:
                features.append("somewhat uncertain")

    # Extract voice quality features
    voice_quality = audio_state.get("voice_quality") or {}
    if isinstance(voice_quality, dict):
        # Clarity descriptor
        if "clarity" in voice_quality and voice_quality["clarity"] not in (
            None,
            "neutral",
            "clear",
            "normal",
        ):
            clarity = voice_quality["clarity"].lower()
            features.append(f"{clarity} clarity")

        # Energy descriptor
        if "energy" in voice_quality and voice_quality["energy"] not in (
            None,
            "neutral",
            "normal",
            "moderate",
        ):
            energy = voice_quality["energy"].lower()
            features.append(f"{energy} energy")

        # Stress level descriptor
        if "stress_level" in voice_quality and voice_quality["stress_level"] not in (
            None,
            "neutral",
            "normal",
            "low",
        ):
            stress = voice_quality["stress_level"].lower()
            if stress == "high":
                features.append("high stress")
            elif stress == "elevated":
                features.append("elevated stress")
            else:
                features.append(f"{stress} stress")

    # Build final annotation
    if not features:
        return "[audio: neutral]"

    # Join features with commas and wrap in annotation brackets
    annotation = ", ".join(features)
    return f"[audio: {annotation}]"


def render_audio_features_detailed(audio_state: dict[str, Any]) -> dict[str, list[str]]:
    """
    Render audio features in a structured, detailed format for analysis.

    Returns a dictionary with separate lists for prosody, emotion, and voice
    quality features. Useful for programmatic processing or detailed logging.

    Args:
        audio_state: Dictionary with optional prosody, emotion, voice_quality keys.

    Returns:
        Dictionary with keys 'prosody', 'emotion', 'voice_quality' containing
        lists of descriptive strings for non-neutral features.

    Example:
        >>> result = render_audio_features_detailed({
        ...     'prosody': {'pitch': 'high', 'volume': 'loud'},
        ...     'emotion': {'tone': 'excited'}
        ... })
        >>> result['prosody']
        ['high pitch', 'loud volume']
        >>> result['emotion']
        ['excited tone']
    """
    result = {"prosody": [], "emotion": [], "voice_quality": []}

    # Process prosody features
    prosody = audio_state.get("prosody") or {}
    if isinstance(prosody, dict):
        if "pitch" in prosody and prosody["pitch"] not in (None, "neutral", "medium"):
            result["prosody"].append(f"{prosody['pitch'].lower()} pitch")
        if "volume" in prosody and prosody["volume"] not in (None, "neutral", "medium"):
            result["prosody"].append(f"{prosody['volume'].lower()} volume")
        if "speech_rate" in prosody and prosody["speech_rate"] not in (None, "neutral", "normal"):
            rate = prosody["speech_rate"].lower()
            if rate == "fast":
                result["prosody"].append("fast speech")
            elif rate == "slow":
                result["prosody"].append("slow speech")
            else:
                result["prosody"].append(f"{rate} speech")
        if "pauses" in prosody and prosody["pauses"] not in (None, "neutral", "normal", "minimal"):
            pauses = prosody["pauses"].lower()
            if pauses == "frequent":
                result["prosody"].append("frequent pauses")
            elif pauses == "moderate":
                result["prosody"].append("moderate pauses")
            else:
                result["prosody"].append(f"{pauses} pauses")

    # Process emotion features
    emotion = audio_state.get("emotion") or {}
    if isinstance(emotion, dict):
        if "tone" in emotion and emotion["tone"] not in (None, "neutral", "normal"):
            result["emotion"].append(f"{emotion['tone'].lower()} tone")
        confidence = emotion.get("confidence")
        if confidence is not None and isinstance(confidence, int | float):
            if confidence < 0.5:
                result["emotion"].append("possibly uncertain")
            elif confidence < 0.7:
                result["emotion"].append("somewhat uncertain")

    # Process voice quality features
    voice_quality = audio_state.get("voice_quality") or {}
    if isinstance(voice_quality, dict):
        if "clarity" in voice_quality and voice_quality["clarity"] not in (
            None,
            "neutral",
            "clear",
            "normal",
        ):
            result["voice_quality"].append(f"{voice_quality['clarity'].lower()} clarity")
        if "energy" in voice_quality and voice_quality["energy"] not in (
            None,
            "neutral",
            "normal",
            "moderate",
        ):
            result["voice_quality"].append(f"{voice_quality['energy'].lower()} energy")
        if "stress_level" in voice_quality and voice_quality["stress_level"] not in (
            None,
            "neutral",
            "normal",
            "low",
        ):
            stress = voice_quality["stress_level"].lower()
            if stress == "high":
                result["voice_quality"].append("high stress")
            else:
                result["voice_quality"].append(f"{stress} stress")

    return result
