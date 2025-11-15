"""
Audio enrichment orchestrator module.

This module combines all feature extraction modules (audio_utils, prosody,
emotion, audio_rendering) to enrich transcription segments with comprehensive
audio state information including:
- Prosodic features (pitch, energy, rate, pauses)
- Emotional features (valence, arousal, categorical emotions)
- Rendered text annotations for human/LLM readability

The orchestrator handles extraction failures gracefully, logs errors, and
returns partial results when some features cannot be extracted.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio_rendering import render_audio_state
from .audio_utils import AudioSegmentExtractor
from .emotion import extract_emotion_categorical, extract_emotion_dimensional
from .models import Segment, Transcript
from .prosody import compute_speaker_baseline, extract_prosody

logger = logging.getLogger(__name__)


def enrich_segment_audio(
    wav_path: Path,
    segment: Segment,
    enable_prosody: bool = True,
    enable_emotion: bool = True,
    enable_categorical_emotion: bool = False,
    speaker_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Enrich a single segment with audio features.

    This function extracts prosodic and emotional features from the audio
    corresponding to a transcript segment and creates a comprehensive audio_state
    dictionary matching the JSON schema.

    Args:
        wav_path: Path to the normalized 16kHz WAV file.
        segment: Segment object containing start/end times and text.
        enable_prosody: If True, extract prosodic features (pitch, energy, rate, pauses).
        enable_emotion: If True, extract dimensional emotion features (valence, arousal).
        enable_categorical_emotion: If True, also extract categorical emotion labels.
        speaker_baseline: Optional baseline statistics for speaker-relative normalization.
                         Should contain keys: pitch_median, energy_median, rate_median, etc.

    Returns:
        Dictionary matching the audio_state schema:
        {
            "prosody": {
                "pitch": {"level": "high", "mean_hz": 245.3, "std_hz": 32.1, ...},
                "energy": {"level": "loud", "db_rms": -8.2, ...},
                "rate": {"level": "fast", "syllables_per_sec": 6.3, ...},
                "pauses": {"count": 2, "longest_ms": 320, ...}
            },
            "emotion": {
                "valence": {"level": "negative", "score": 0.35},
                "arousal": {"level": "high", "score": 0.68},
                "dominance": {"level": "neutral", "score": 0.52},
                "categorical": {...}  # Optional
            },
            "rendering": "[audio: high pitch, loud volume, fast speech, excited tone]",
            "extraction_status": {
                "prosody": "success",
                "emotion_dimensional": "success",
                "emotion_categorical": "skipped",
                "errors": []
            }
        }

    Raises:
        FileNotFoundError: If wav_path does not exist.
        ValueError: If segment times are invalid.
    """
    # Initialize result structure
    audio_state: dict[str, Any] = {
        "prosody": None,
        "emotion": None,
        "rendering": "[audio: neutral]",
        "extraction_status": {
            "prosody": "skipped",
            "emotion_dimensional": "skipped",
            "emotion_categorical": "skipped",
            "errors": [],
        },
    }

    try:
        # Extract audio segment
        extractor = AudioSegmentExtractor(wav_path)
        audio_data, sample_rate = extractor.extract_segment(
            segment.start,
            segment.end,
            clamp=True,
            min_duration=0.0,  # Allow very short segments
        )

        logger.debug(
            f"Extracted segment [{segment.start:.2f}s - {segment.end:.2f}s]: "
            f"{len(audio_data)} samples at {sample_rate} Hz"
        )

    except Exception as e:
        error_msg = f"Failed to extract audio segment: {e}"
        logger.error(error_msg)
        audio_state["extraction_status"]["errors"].append(error_msg)
        return audio_state

    # Extract prosodic features
    if enable_prosody:
        try:
            prosody_result = extract_prosody(
                audio_data,
                sample_rate,
                segment.text,
                speaker_baseline=speaker_baseline,
                start_time=segment.start,
                end_time=segment.end,
            )
            audio_state["prosody"] = prosody_result
            audio_state["extraction_status"]["prosody"] = "success"
            logger.debug(f"Prosody extraction succeeded for segment {segment.id}")

        except Exception as e:
            error_msg = f"Prosody extraction failed: {e}"
            logger.error(error_msg)
            audio_state["extraction_status"]["prosody"] = "failed"
            audio_state["extraction_status"]["errors"].append(error_msg)

    # Extract emotional features
    emotion_data = {}

    if enable_emotion:
        # Extract dimensional emotion (valence, arousal, dominance)
        try:
            dimensional_result = extract_emotion_dimensional(audio_data, sample_rate)
            emotion_data.update(dimensional_result)
            audio_state["extraction_status"]["emotion_dimensional"] = "success"
            logger.debug(f"Dimensional emotion extraction succeeded for segment {segment.id}")

        except Exception as e:
            error_msg = f"Dimensional emotion extraction failed: {e}"
            logger.error(error_msg)
            audio_state["extraction_status"]["emotion_dimensional"] = "failed"
            audio_state["extraction_status"]["errors"].append(error_msg)

    if enable_categorical_emotion:
        # Extract categorical emotion (angry, happy, sad, etc.)
        try:
            categorical_result = extract_emotion_categorical(audio_data, sample_rate)
            emotion_data.update(categorical_result)
            audio_state["extraction_status"]["emotion_categorical"] = "success"
            logger.debug(f"Categorical emotion extraction succeeded for segment {segment.id}")

        except Exception as e:
            error_msg = f"Categorical emotion extraction failed: {e}"
            logger.error(error_msg)
            audio_state["extraction_status"]["emotion_categorical"] = "failed"
            audio_state["extraction_status"]["errors"].append(error_msg)

    # Store emotion data if any was extracted
    if emotion_data:
        audio_state["emotion"] = emotion_data

    # Render audio state as text annotation
    try:
        # Build simplified state for rendering
        render_input = _build_render_input(audio_state)
        audio_state["rendering"] = render_audio_state(render_input)

    except Exception as e:
        error_msg = f"Audio rendering failed: {e}"
        logger.warning(error_msg)
        audio_state["extraction_status"]["errors"].append(error_msg)
        audio_state["rendering"] = "[audio: neutral]"

    return audio_state


def enrich_transcript_audio(
    transcript: Transcript,
    wav_path: Path,
    enable_prosody: bool = True,
    enable_emotion: bool = True,
    enable_categorical_emotion: bool = False,
    compute_baseline: bool = True,
) -> Transcript:
    """
    Enrich an entire transcript with audio features for all segments.

    This function processes all segments in a transcript, extracting audio
    features and populating the audio_state field of each segment. It optionally
    computes speaker baseline statistics for speaker-relative normalization.

    Args:
        transcript: Transcript object to enrich (modified in-place).
        wav_path: Path to the normalized 16kHz WAV file.
        enable_prosody: If True, extract prosodic features.
        enable_emotion: If True, extract dimensional emotion features.
        enable_categorical_emotion: If True, also extract categorical emotions.
        compute_baseline: If True, compute speaker baseline from all segments
                         for relative normalization.

    Returns:
        The enriched Transcript object (same as input, modified in-place).
        Also updates transcript.meta with audio enrichment metadata.

    Example:
        >>> from pathlib import Path
        >>> from transcription.models import Transcript, Segment
        >>>
        >>> # Load existing transcript
        >>> transcript = Transcript(
        ...     file_name="audio.wav",
        ...     language="en",
        ...     segments=[
        ...         Segment(id=0, start=0.0, end=2.5, text="Hello world"),
        ...         Segment(id=1, start=2.5, end=5.0, text="How are you?")
        ...     ]
        ... )
        >>>
        >>> # Enrich with audio features
        >>> wav_path = Path("normalized/audio.wav")
        >>> enriched = enrich_transcript_audio(transcript, wav_path)
        >>>
        >>> # Check results
        >>> print(enriched.segments[0].audio_state["rendering"])
        [audio: neutral]
    """
    if not transcript.segments:
        logger.warning("Transcript has no segments to enrich")
        return transcript

    logger.info(
        f"Starting audio enrichment for {len(transcript.segments)} segments "
        f"(prosody={enable_prosody}, emotion={enable_emotion}, "
        f"categorical={enable_categorical_emotion})"
    )

    # Compute speaker baseline if requested
    speaker_baseline: dict[str, Any] | None = None
    if compute_baseline and enable_prosody:
        try:
            logger.info("Computing speaker baseline statistics...")
            extractor = AudioSegmentExtractor(wav_path)

            # Sample up to 20 segments for baseline (or all if fewer)
            sample_size = min(20, len(transcript.segments))
            sample_indices = range(
                0, len(transcript.segments), max(1, len(transcript.segments) // sample_size)
            )

            segments_data = []
            for idx in sample_indices:
                seg = transcript.segments[idx]
                try:
                    audio, sr = extractor.extract_segment(seg.start, seg.end, clamp=True)
                    segments_data.append({"audio": audio, "sr": sr, "text": seg.text})
                except Exception as e:
                    logger.warning(f"Failed to extract segment {idx} for baseline: {e}")
                    continue

            if segments_data:
                speaker_baseline = compute_speaker_baseline(segments_data)
                logger.info(f"Speaker baseline computed from {len(segments_data)} segments")
                logger.debug(f"Baseline stats: {speaker_baseline}")
            else:
                logger.warning("Could not compute speaker baseline (no valid segments)")

        except Exception as e:
            logger.error(f"Failed to compute speaker baseline: {e}")

    # Enrich each segment
    success_count = 0
    partial_count = 0
    failed_count = 0

    for idx, segment in enumerate(transcript.segments):
        logger.debug(f"Enriching segment {idx + 1}/{len(transcript.segments)}: {segment.id}")

        try:
            audio_state = enrich_segment_audio(
                wav_path,
                segment,
                enable_prosody=enable_prosody,
                enable_emotion=enable_emotion,
                enable_categorical_emotion=enable_categorical_emotion,
                speaker_baseline=speaker_baseline,
            )

            # Attach audio_state to segment
            segment.audio_state = audio_state

            # Count success/partial/failure
            status = audio_state.get("extraction_status", {})
            errors = status.get("errors", [])

            if not errors:
                success_count += 1
            elif any(status.get(k) == "success" for k in ["prosody", "emotion_dimensional"]):
                partial_count += 1
            else:
                failed_count += 1

        except Exception as e:
            logger.error(f"Failed to enrich segment {segment.id}: {e}")
            segment.audio_state = {
                "prosody": None,
                "emotion": None,
                "rendering": "[audio: neutral]",
                "extraction_status": {
                    "prosody": "failed",
                    "emotion_dimensional": "failed",
                    "emotion_categorical": "failed",
                    "errors": [str(e)],
                },
            }
            failed_count += 1

    # Update transcript metadata
    if transcript.meta is None:
        transcript.meta = {}

    transcript.meta["audio_enrichment"] = {
        "enriched_at": datetime.now(timezone.utc).isoformat(),
        "total_segments": len(transcript.segments),
        "success_count": success_count,
        "partial_count": partial_count,
        "failed_count": failed_count,
        "features_enabled": {
            "prosody": enable_prosody,
            "emotion_dimensional": enable_emotion,
            "emotion_categorical": enable_categorical_emotion,
        },
        "speaker_baseline_computed": speaker_baseline is not None,
        "speaker_baseline": speaker_baseline,
    }

    logger.info(
        f"Audio enrichment complete: {success_count} success, "
        f"{partial_count} partial, {failed_count} failed"
    )

    return transcript


def _build_render_input(audio_state: dict[str, Any]) -> dict[str, Any]:
    """
    Build a simplified audio state dict for rendering.

    The audio_rendering module expects a different structure than our
    full audio_state schema. This function maps our schema to the
    rendering module's expected format.

    Args:
        audio_state: Full audio_state dict from enrich_segment_audio.

    Returns:
        Simplified dict for render_audio_state:
        {
            "prosody": {
                "pitch": "high",
                "volume": "loud",
                "speech_rate": "fast",
                "pauses": "moderate"
            },
            "emotion": {
                "tone": "excited"
            }
        }
    """
    render_input: dict[str, Any] = {}

    # Map prosody features
    prosody = audio_state.get("prosody")
    if prosody and isinstance(prosody, dict):
        render_prosody = {}

        # Map pitch
        if "pitch" in prosody and prosody["pitch"].get("level"):
            render_prosody["pitch"] = prosody["pitch"]["level"]

        # Map energy to volume
        if "energy" in prosody and prosody["energy"].get("level"):
            energy_level = prosody["energy"]["level"]
            # Map energy levels to volume levels
            volume_map = {
                "very_quiet": "very_quiet",
                "quiet": "quiet",
                "normal": "normal",
                "loud": "loud",
                "very_loud": "very_loud",
            }
            render_prosody["volume"] = volume_map.get(energy_level, energy_level)

        # Map rate to speech_rate
        if "rate" in prosody and prosody["rate"].get("level"):
            rate_level = prosody["rate"]["level"]
            # Map rate levels to speech_rate levels
            rate_map = {
                "very_slow": "very_slow",
                "slow": "slow",
                "normal": "normal",
                "fast": "fast",
                "very_fast": "very_fast",
            }
            render_prosody["speech_rate"] = rate_map.get(rate_level, rate_level)

        # Map pauses
        if "pauses" in prosody and prosody["pauses"].get("density"):
            render_prosody["pauses"] = prosody["pauses"]["density"]

        if render_prosody:
            render_input["prosody"] = render_prosody

    # Map emotion features
    emotion = audio_state.get("emotion")
    if emotion and isinstance(emotion, dict):
        render_emotion = {}

        # Map categorical emotion to tone
        if "categorical" in emotion:
            categorical = emotion["categorical"]
            if isinstance(categorical, dict) and "primary" in categorical:
                render_emotion["tone"] = categorical["primary"]
                if "confidence" in categorical:
                    render_emotion["confidence"] = categorical["confidence"]

        # If no categorical, try to infer tone from valence/arousal
        elif "valence" in emotion and "arousal" in emotion:
            valence = emotion["valence"].get("level", "neutral")
            arousal = emotion["arousal"].get("level", "medium")

            # Simple mapping of valence/arousal to tone
            if valence in ["positive", "very_positive"]:
                if arousal in ["high", "very_high"]:
                    render_emotion["tone"] = "excited"
                else:
                    render_emotion["tone"] = "calm"
            elif valence in ["negative", "very_negative"]:
                if arousal in ["high", "very_high"]:
                    render_emotion["tone"] = "agitated"
                else:
                    render_emotion["tone"] = "sad"
            else:
                render_emotion["tone"] = "neutral"

        if render_emotion:
            render_input["emotion"] = render_emotion

    return render_input


# Export public API
__all__ = [
    "enrich_segment_audio",
    "enrich_transcript_audio",
]
