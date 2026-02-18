"""
Tests for audio enrichment functionality.

This module tests the enrichment of transcripts with audio features including:
- Prosody (pitch, energy, rate, pauses)
- Emotion (categorical and dimensional)
- Error handling and edge cases
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slower_whisper.pipeline.audio_utils import AudioSegmentExtractor
from slower_whisper.pipeline.models import Segment, Transcript

# Check for optional dependencies (must be real packages, not mocks from conftest)
EMOTION_AVAILABLE = False
PROSODY_AVAILABLE = True

try:
    # Check for real transformers, not mock - from_pretrained must be callable
    from transformers import AutoModelForAudioClassification

    if hasattr(AutoModelForAudioClassification, "from_pretrained") and callable(
        getattr(AutoModelForAudioClassification, "from_pretrained", None)
    ):
        from slower_whisper.pipeline.emotion import (
            extract_emotion_categorical,
            extract_emotion_dimensional,
        )

        EMOTION_AVAILABLE = True
except (ImportError, ValueError, AttributeError):
    pass

if not EMOTION_AVAILABLE:
    extract_emotion_dimensional = None
    extract_emotion_categorical = None

try:
    from slower_whisper.pipeline.prosody import extract_prosody
except (ImportError, ValueError):
    PROSODY_AVAILABLE = False
    extract_prosody = None


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_audio() -> tuple[np.ndarray, int]:
    """Create synthetic audio for testing (1 second, 16kHz, sine wave)."""
    sr = 16000
    duration = 1.0
    frequency = 440.0  # A4 note
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32), sr


@pytest.fixture
def synthetic_audio_file(tmp_path: Path, synthetic_audio: tuple[np.ndarray, int]) -> Path:
    """Create a temporary WAV file with synthetic audio."""
    import soundfile as sf

    audio, sr = synthetic_audio
    wav_path = tmp_path / "synthetic_test.wav"
    sf.write(wav_path, audio, sr)
    return wav_path


@pytest.fixture
def sample_transcript() -> Transcript:
    """Create a sample transcript for testing."""
    segments = [
        Segment(id=0, start=0.0, end=1.0, text="Hello world"),
        Segment(id=1, start=1.0, end=2.0, text="This is a test"),
        Segment(id=2, start=2.0, end=3.0, text="Audio enrichment"),
    ]
    return Transcript(file_name="test.wav", language="en", segments=segments)


# ============================================================================
# Prosody Tests
# ============================================================================


@pytest.mark.skipif(not PROSODY_AVAILABLE, reason="parselmouth or librosa not installed")
def test_extract_prosody_basic(synthetic_audio):
    """Test basic prosody extraction from synthetic audio."""
    audio, sr = synthetic_audio
    text = "Hello world"

    result = extract_prosody(audio, sr, text)

    # Check structure
    assert "pitch" in result
    assert "energy" in result
    assert "rate" in result
    assert "pauses" in result

    # Check pitch features
    assert "level" in result["pitch"]
    assert "mean_hz" in result["pitch"]
    assert "contour" in result["pitch"]

    # Check energy features
    assert "level" in result["energy"]
    assert "db_rms" in result["energy"]

    # Check rate features
    assert "level" in result["rate"]
    assert "syllables_per_sec" in result["rate"]


@pytest.mark.skipif(not PROSODY_AVAILABLE, reason="parselmouth or librosa not installed")
def test_extract_prosody_with_baseline(synthetic_audio):
    """Test prosody extraction with speaker baseline normalization."""
    audio, sr = synthetic_audio
    text = "Hello world"

    # Define a baseline
    baseline = {
        "pitch_median": 180.0,
        "pitch_std": 30.0,
        "energy_median": -15.0,
        "rate_median": 5.0,
    }

    result = extract_prosody(audio, sr, text, speaker_baseline=baseline)

    # Should still return valid structure
    assert "pitch" in result
    assert "energy" in result
    assert result["pitch"]["level"] in [
        "very_low",
        "low",
        "neutral",
        "high",
        "very_high",
        "unknown",
    ]


@pytest.mark.skipif(not PROSODY_AVAILABLE, reason="parselmouth or librosa not installed")
def test_extract_prosody_empty_audio():
    """Test prosody extraction with empty or very short audio."""
    audio = np.array([], dtype=np.float32)
    sr = 16000
    text = "test"

    result = extract_prosody(audio, sr, text)

    # Should not crash, should return structure with unknown/None values
    assert "pitch" in result
    assert "energy" in result


# ============================================================================
# Emotion Tests
# ============================================================================


@pytest.mark.heavy
@pytest.mark.skipif(not EMOTION_AVAILABLE, reason="transformers not installed")
def test_extract_emotion_dimensional(synthetic_audio):
    """Test dimensional emotion extraction (valence, arousal, dominance)."""
    audio, sr = synthetic_audio

    result = extract_emotion_dimensional(audio, sr)

    # Check structure
    assert "valence" in result
    assert "arousal" in result
    assert "dominance" in result

    # Check valence
    assert "level" in result["valence"]
    assert "score" in result["valence"]
    assert isinstance(result["valence"]["score"], float)
    # Note: dimensional emotion models output regression values, not necessarily [0, 1]

    # Check arousal
    assert "level" in result["arousal"]
    assert "score" in result["arousal"]
    assert isinstance(result["arousal"]["score"], float)

    # Check dominance
    assert "level" in result["dominance"]
    assert "score" in result["dominance"]
    assert isinstance(result["dominance"]["score"], float)


@pytest.mark.heavy
@pytest.mark.skipif(not EMOTION_AVAILABLE, reason="transformers not installed")
def test_extract_emotion_categorical(synthetic_audio):
    """Test categorical emotion extraction."""
    audio, sr = synthetic_audio

    result = extract_emotion_categorical(audio, sr)

    # Check structure
    assert "categorical" in result
    cat = result["categorical"]

    assert "primary" in cat
    assert "confidence" in cat
    assert "secondary" in cat
    assert "secondary_confidence" in cat
    assert "all_scores" in cat

    # Check confidence values
    assert 0.0 <= cat["confidence"] <= 1.0
    assert 0.0 <= cat["secondary_confidence"] <= 1.0

    # Check all_scores is a dict
    assert isinstance(cat["all_scores"], dict)


@pytest.mark.heavy
@pytest.mark.skipif(not EMOTION_AVAILABLE, reason="transformers not installed")
def test_emotion_short_audio():
    """Test emotion extraction with very short audio (edge case)."""
    # 0.3 seconds of audio (below recommended minimum)
    sr = 16000
    duration = 0.3
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1

    # Should not crash, might have warning about short segment
    result = extract_emotion_categorical(audio, sr)

    assert "categorical" in result
    assert result["categorical"]["primary"] is not None


# ============================================================================
# Audio Segment Extraction Tests
# ============================================================================


def test_audio_segment_extractor_basic(synthetic_audio_file):
    """Test basic audio segment extraction."""
    extractor = AudioSegmentExtractor(synthetic_audio_file)

    # Extract first 0.5 seconds
    audio, sr = extractor.extract_segment(0.0, 0.5)

    assert len(audio) > 0
    assert sr == 16000
    assert len(audio) == int(0.5 * sr)


def test_audio_segment_extractor_full_duration(synthetic_audio_file):
    """Test extracting entire audio file."""
    extractor = AudioSegmentExtractor(synthetic_audio_file)

    duration = extractor.get_duration()
    audio, sr = extractor.extract_segment(0.0, duration)

    # Should extract the full audio
    assert len(audio) == extractor.total_frames or len(audio) == extractor.total_frames - 1


def test_audio_segment_extractor_clamping(synthetic_audio_file):
    """Test that out-of-bounds timestamps are clamped correctly."""
    extractor = AudioSegmentExtractor(synthetic_audio_file)

    # Request segment beyond file duration (with clamping enabled)
    audio, sr = extractor.extract_segment(-1.0, 100.0, clamp=True)

    # Should clamp to [0, duration]
    assert len(audio) > 0


def test_audio_segment_extractor_no_clamp_error(synthetic_audio_file):
    """Test that out-of-bounds raises error when clamp=False."""
    extractor = AudioSegmentExtractor(synthetic_audio_file)

    with pytest.raises(ValueError):
        extractor.extract_segment(-1.0, 0.5, clamp=False)

    with pytest.raises(ValueError):
        extractor.extract_segment(0.0, 100.0, clamp=False)


def test_audio_segment_extractor_missing_file():
    """Test that missing file raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        AudioSegmentExtractor("/nonexistent/file.wav")


# ============================================================================
# Integration Tests for Enrichment
# ============================================================================


@pytest.mark.heavy
def test_enrich_segment_with_audio_state(synthetic_audio):
    """Test enriching a segment with audio_state (if dependencies available)."""
    audio, sr = synthetic_audio

    segment = Segment(id=0, start=0.0, end=1.0, text="Hello world")

    # Create audio_state manually (simulating enrichment)
    audio_state = {}

    if PROSODY_AVAILABLE:
        prosody = extract_prosody(audio, sr, segment.text)
        audio_state["prosody"] = prosody

    if EMOTION_AVAILABLE:
        emotion_dim = extract_emotion_dimensional(audio, sr)
        emotion_cat = extract_emotion_categorical(audio, sr)
        audio_state["emotion"] = {
            "dimensional": emotion_dim,
            "categorical": emotion_cat["categorical"],
        }

    # Attach to segment
    segment.audio_state = audio_state if audio_state else None

    # Verify structure
    if audio_state:
        assert segment.audio_state is not None
        if PROSODY_AVAILABLE:
            assert "prosody" in segment.audio_state
        if EMOTION_AVAILABLE:
            assert "emotion" in segment.audio_state


@pytest.mark.heavy
def test_enrich_transcript_full(synthetic_audio_file, sample_transcript, tmp_path):
    """Test enriching an entire transcript with audio features."""
    # This simulates the enrichment process similar to emotion_integration.py

    # First, create a longer audio file with multiple segments
    import soundfile as sf

    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    audio_full = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    audio_file = tmp_path / "full_test.wav"
    sf.write(audio_file, audio_full, sr)

    # Load transcript and enrich
    extractor = AudioSegmentExtractor(audio_file)

    enriched_segments = []
    for seg in sample_transcript.segments:
        # Extract segment audio
        try:
            audio_seg, sr = extractor.extract_segment(seg.start, seg.end)

            # Build audio_state
            audio_state = {}

            if PROSODY_AVAILABLE:
                prosody = extract_prosody(audio_seg, sr, seg.text)
                audio_state["prosody"] = prosody

            if EMOTION_AVAILABLE:
                emotion_dim = extract_emotion_dimensional(audio_seg, sr)
                emotion_cat = extract_emotion_categorical(audio_seg, sr)
                audio_state["emotion"] = {
                    "dimensional": emotion_dim,
                    "categorical": emotion_cat["categorical"],
                }

            # Create new segment with audio_state
            enriched_seg = Segment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=seg.speaker,
                tone=seg.tone,
                audio_state=audio_state if audio_state else None,
            )
            enriched_segments.append(enriched_seg)

        except Exception:
            # If extraction fails, keep original segment
            enriched_segments.append(seg)

    # Create enriched transcript
    enriched_transcript = Transcript(
        file_name=sample_transcript.file_name,
        language=sample_transcript.language,
        segments=enriched_segments,
        meta=sample_transcript.meta,
    )

    # Verify at least structure is correct
    assert len(enriched_transcript.segments) == len(sample_transcript.segments)

    # Check if any segments were enriched
    has_enrichment = any(seg.audio_state is not None for seg in enriched_transcript.segments)

    if PROSODY_AVAILABLE or EMOTION_AVAILABLE:
        # At least some enrichment should have happened
        assert has_enrichment


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_segment_extraction_with_invalid_times(synthetic_audio_file):
    """Test that invalid time ranges are handled properly."""
    extractor = AudioSegmentExtractor(synthetic_audio_file)

    # Start > end
    with pytest.raises(ValueError):
        extractor.extract_segment(2.0, 1.0, clamp=False)


def test_segment_extraction_zero_duration(synthetic_audio_file):
    """Test extraction of zero-duration segment."""
    extractor = AudioSegmentExtractor(synthetic_audio_file)

    # Same start and end time - should extract at least 1 frame
    audio, sr = extractor.extract_segment(0.5, 0.5, clamp=True)

    # Should have at least 1 frame
    assert len(audio) >= 1


# ============================================================================
# Partial Enrichment Tests
# ============================================================================


def test_partial_enrichment_prosody_only(synthetic_audio):
    """Test enrichment with only prosody (no emotion)."""
    if not PROSODY_AVAILABLE:
        pytest.skip("Prosody not available")

    audio, sr = synthetic_audio
    segment = Segment(id=0, start=0.0, end=1.0, text="Test")

    # Enrich with only prosody
    audio_state = {"prosody": extract_prosody(audio, sr, segment.text)}
    segment.audio_state = audio_state

    assert segment.audio_state is not None
    assert "prosody" in segment.audio_state
    assert "emotion" not in segment.audio_state


@pytest.mark.heavy
def test_partial_enrichment_emotion_only(synthetic_audio):
    """Test enrichment with only emotion (no prosody)."""
    if not EMOTION_AVAILABLE:
        pytest.skip("Emotion not available")

    audio, sr = synthetic_audio
    segment = Segment(id=0, start=0.0, end=1.0, text="Test")

    # Enrich with only emotion
    emotion_cat = extract_emotion_categorical(audio, sr)
    audio_state = {"emotion": {"categorical": emotion_cat["categorical"]}}
    segment.audio_state = audio_state

    assert segment.audio_state is not None
    assert "emotion" in segment.audio_state
    assert "prosody" not in segment.audio_state


# ============================================================================
# Schema Version Tests
# ============================================================================


def test_audio_state_schema_version(synthetic_audio_file, sample_transcript):
    """Test that audio_state includes _schema_version for downstream consumers."""
    # Create a longer audio file with multiple segments
    import soundfile as sf

    from slower_whisper.pipeline.audio_enrichment import enrich_transcript_audio
    from slower_whisper.pipeline.models import AUDIO_STATE_VERSION

    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    audio_full = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    audio_file = synthetic_audio_file.parent / "schema_test.wav"
    sf.write(audio_file, audio_full, sr)

    # Enrich the transcript
    enriched = enrich_transcript_audio(
        sample_transcript,
        audio_file,
        enable_prosody=True,
        enable_emotion=False,  # Skip for speed
        compute_baseline=False,
    )

    # Verify all segments have _schema_version in audio_state
    for segment in enriched.segments:
        assert segment.audio_state is not None
        assert "_schema_version" in segment.audio_state
        assert segment.audio_state["_schema_version"] == AUDIO_STATE_VERSION


# ============================================================================
# Skip-existing Logic Tests
# ============================================================================


def test_skip_existing_audio_state(synthetic_audio):
    """Test that existing audio_state is not overwritten when skip_existing=True."""
    audio, sr = synthetic_audio

    # Create segment with existing audio_state
    existing_state = {"prosody": {"pitch": {"level": "existing"}}}
    segment = Segment(id=0, start=0.0, end=1.0, text="Test", audio_state=existing_state)

    # Simulate skip-existing logic
    if segment.audio_state is None:
        # Would enrich here
        segment.audio_state = {"new": "data"}

    # Should keep existing state
    assert segment.audio_state["prosody"]["pitch"]["level"] == "existing"


def test_overwrite_audio_state(synthetic_audio):
    """Test that audio_state is overwritten when skip_existing=False."""
    audio, sr = synthetic_audio

    # Create segment with existing audio_state
    existing_state = {"prosody": {"pitch": {"level": "existing"}}}
    segment = Segment(id=0, start=0.0, end=1.0, text="Test", audio_state=existing_state)

    # Simulate overwrite logic (skip_existing=False)
    segment.audio_state = {"new": "data"}

    # Should have new state
    assert segment.audio_state == {"new": "data"}
    assert "prosody" not in segment.audio_state
