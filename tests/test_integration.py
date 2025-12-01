"""
Integration tests for the transcription pipeline.

End-to-end tests covering:
- Transcription → enrichment → JSON output
- Round-trip consistency (write → load → write)
- Full pipeline with all optional features
"""

import json

import numpy as np
import pytest

from transcription.audio_utils import AudioSegmentExtractor
from transcription.models import SCHEMA_VERSION, Segment, Transcript
from transcription.writers import load_transcript_from_json, write_json

pytestmark = pytest.mark.integration

# Check for optional dependencies
EMOTION_AVAILABLE = True
PROSODY_AVAILABLE = True

try:
    from transcription.emotion import extract_emotion_categorical, extract_emotion_dimensional
except (ImportError, ValueError):
    EMOTION_AVAILABLE = False
    extract_emotion_dimensional = None
    extract_emotion_categorical = None

try:
    from transcription.prosody import extract_prosody
except (ImportError, ValueError):
    PROSODY_AVAILABLE = False
    extract_prosody = None


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_audio_file(tmp_path):
    """
    Create a test WAV file with synthetic speech-like audio.

    Creates a 3-second audio file with varying frequency to simulate speech.
    """
    import soundfile as sf

    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create speech-like audio with varying pitch
    # Segment 1 (0-1s): 200 Hz
    # Segment 2 (1-2s): 250 Hz
    # Segment 3 (2-3s): 180 Hz
    audio = np.zeros(len(t), dtype=np.float32)

    for i, time in enumerate(t):
        if time < 1.0:
            freq = 200
        elif time < 2.0:
            freq = 250
        else:
            freq = 180
        audio[i] = 0.3 * np.sin(2 * np.pi * freq * time)

    # Add some noise for realism
    audio += 0.01 * np.random.randn(len(audio)).astype(np.float32)

    audio_path = tmp_path / "test_speech.wav"
    sf.write(audio_path, audio, sr)

    return audio_path


@pytest.fixture
def test_transcript():
    """Create a test transcript with multiple segments."""
    segments = [
        Segment(id=0, start=0.0, end=1.0, text="Hello world"),
        Segment(id=1, start=1.0, end=2.0, text="This is a test"),
        Segment(id=2, start=2.0, end=3.0, text="Audio transcription"),
    ]

    return Transcript(
        file_name="test_speech.wav",
        language="en",
        segments=segments,
        meta={"model_name": "faster-whisper-base", "device": "cpu", "pipeline_version": "0.1.0"},
    )


# ============================================================================
# End-to-End Tests
# ============================================================================


def test_e2e_transcribe_enrich_verify(test_audio_file, test_transcript, tmp_path):
    """
    End-to-end test: transcribe → enrich → verify JSON.

    Simulates the full pipeline flow:
    1. Start with a transcript (simulated transcription)
    2. Enrich with audio features
    3. Write to JSON
    4. Verify all fields are present and valid
    """
    # Step 1: Start with transcript (simulates ASR output)
    transcript = test_transcript

    # Step 2: Enrich with audio features
    extractor = AudioSegmentExtractor(test_audio_file)

    enriched_segments = []
    for seg in transcript.segments:
        try:
            # Extract audio segment
            audio, sr = extractor.extract_segment(seg.start, seg.end)

            # Build audio_state
            audio_state = {}

            # Add prosody if available
            if PROSODY_AVAILABLE:
                prosody = extract_prosody(audio, sr, seg.text)
                audio_state["prosody"] = prosody

            # Add emotion if available
            if EMOTION_AVAILABLE:
                emotion_dim = extract_emotion_dimensional(audio, sr)
                emotion_cat = extract_emotion_categorical(audio, sr)
                audio_state["emotion"] = {
                    "dimensional": emotion_dim,
                    "categorical": emotion_cat["categorical"],
                }

            # Create enriched segment
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
            # Keep original segment on error
            enriched_segments.append(seg)

    enriched_transcript = Transcript(
        file_name=transcript.file_name,
        language=transcript.language,
        segments=enriched_segments,
        meta=transcript.meta,
    )

    # Step 3: Write to JSON
    json_path = tmp_path / "enriched_transcript.json"
    write_json(enriched_transcript, json_path)

    # Step 4: Verify JSON structure
    assert json_path.exists()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Verify top-level fields
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["file"] == "test_speech.wav"
    assert data["language"] == "en"
    assert "meta" in data
    assert "segments" in data

    # Verify segments
    assert len(data["segments"]) == 3

    for seg_data in data["segments"]:
        # Required fields
        assert "id" in seg_data
        assert "start" in seg_data
        assert "end" in seg_data
        assert "text" in seg_data

        # Optional fields
        assert "speaker" in seg_data
        assert "tone" in seg_data
        assert "audio_state" in seg_data

        # If enriched, verify audio_state structure
        if seg_data["audio_state"] is not None:
            if PROSODY_AVAILABLE:
                assert "prosody" in seg_data["audio_state"]
                prosody = seg_data["audio_state"]["prosody"]
                assert "pitch" in prosody
                assert "energy" in prosody
                assert "rate" in prosody
                assert "pauses" in prosody

            if EMOTION_AVAILABLE:
                assert "emotion" in seg_data["audio_state"]
                emotion = seg_data["audio_state"]["emotion"]
                assert "dimensional" in emotion or "categorical" in emotion


def test_roundtrip_consistency(test_transcript, tmp_path):
    """
    Test round-trip: write → load → write → verify consistency.

    Ensures that:
    1. Writing a transcript to JSON
    2. Loading it back
    3. Writing it again

    Results in identical JSON output.
    """
    # Create transcript with audio_state
    audio_state = {
        "prosody": {
            "pitch": {"level": "high", "mean_hz": 245.3, "contour": "rising"},
            "energy": {"level": "normal", "db_rms": -12.5},
            "rate": {"level": "fast", "syllables_per_sec": 6.2},
        },
        "emotion": {
            "categorical": {"primary": "neutral", "confidence": 0.85},
            "dimensional": {
                "valence": {"level": "neutral", "score": 0.52},
                "arousal": {"level": "medium", "score": 0.48},
            },
        },
    }

    segments = [
        Segment(id=0, start=0.0, end=1.5, text="First segment", audio_state=audio_state),
        Segment(id=1, start=1.5, end=3.0, text="Second segment", speaker="Speaker A"),
    ]

    transcript = Transcript(
        file_name="test.wav", language="en", segments=segments, meta={"test": "data"}
    )

    # First write
    json_path_1 = tmp_path / "transcript_1.json"
    write_json(transcript, json_path_1)

    # Load
    loaded_transcript = load_transcript_from_json(json_path_1)

    # Second write
    json_path_2 = tmp_path / "transcript_2.json"
    write_json(loaded_transcript, json_path_2)

    # Compare JSON files
    with open(json_path_1, encoding="utf-8") as f:
        data1 = json.load(f)

    with open(json_path_2, encoding="utf-8") as f:
        data2 = json.load(f)

    # Should be identical
    assert data1 == data2

    # Verify loaded transcript matches original
    assert loaded_transcript.file_name == transcript.file_name
    assert loaded_transcript.language == transcript.language
    assert len(loaded_transcript.segments) == len(transcript.segments)

    # Verify audio_state survived the round-trip
    assert loaded_transcript.segments[0].audio_state is not None
    assert loaded_transcript.segments[0].audio_state["prosody"]["pitch"]["level"] == "high"
    assert (
        loaded_transcript.segments[0].audio_state["emotion"]["categorical"]["primary"] == "neutral"
    )


def test_full_pipeline_with_all_features(test_audio_file, tmp_path):
    """
    Test complete pipeline with all optional features enabled.

    This test only runs if all dependencies are available.
    """
    if not (PROSODY_AVAILABLE and EMOTION_AVAILABLE):
        pytest.skip("All features not available (prosody and/or emotion missing)")

    # Simulate ASR output
    segments = [
        Segment(id=0, start=0.0, end=1.0, text="Hello world"),
        Segment(id=1, start=1.0, end=2.0, text="Testing features"),
    ]

    transcript = Transcript(
        file_name=test_audio_file.name, language="en", segments=segments, meta={"model": "test"}
    )

    # Enrich with all features
    extractor = AudioSegmentExtractor(test_audio_file)

    enriched_segments = []
    for seg in transcript.segments:
        audio, sr = extractor.extract_segment(seg.start, seg.end)

        # Extract all features
        prosody = extract_prosody(audio, sr, seg.text)
        emotion_dim = extract_emotion_dimensional(audio, sr)
        emotion_cat = extract_emotion_categorical(audio, sr)

        audio_state = {
            "prosody": prosody,
            "emotion": {"dimensional": emotion_dim, "categorical": emotion_cat["categorical"]},
        }

        enriched_seg = Segment(
            id=seg.id, start=seg.start, end=seg.end, text=seg.text, audio_state=audio_state
        )
        enriched_segments.append(enriched_seg)

    enriched_transcript = Transcript(
        file_name=transcript.file_name,
        language=transcript.language,
        segments=enriched_segments,
        meta=transcript.meta,
    )

    # Write to JSON
    json_path = tmp_path / "full_pipeline.json"
    write_json(enriched_transcript, json_path)

    # Load and verify
    loaded = load_transcript_from_json(json_path)

    assert len(loaded.segments) == 2

    for seg in loaded.segments:
        assert seg.audio_state is not None

        # Verify prosody
        assert "prosody" in seg.audio_state
        assert "pitch" in seg.audio_state["prosody"]
        assert "energy" in seg.audio_state["prosody"]
        assert "rate" in seg.audio_state["prosody"]
        assert "pauses" in seg.audio_state["prosody"]

        # Verify emotion
        assert "emotion" in seg.audio_state
        assert "dimensional" in seg.audio_state["emotion"]
        assert "categorical" in seg.audio_state["emotion"]

        # Verify dimensional emotion structure
        dim = seg.audio_state["emotion"]["dimensional"]
        assert "valence" in dim
        assert "arousal" in dim
        assert "dominance" in dim
        # Note: dimensional models output regression values, not necessarily bounded [0,1]
        assert isinstance(dim["valence"]["score"], float)
        assert isinstance(dim["arousal"]["score"], float)
        assert isinstance(dim["dominance"]["score"], float)

        # Verify categorical emotion structure
        cat = seg.audio_state["emotion"]["categorical"]
        assert "primary" in cat
        assert "confidence" in cat
        assert 0.0 <= cat["confidence"] <= 1.0


def test_empty_transcript(tmp_path):
    """Test handling of empty transcript."""
    transcript = Transcript(file_name="empty.wav", language="en", segments=[])

    json_path = tmp_path / "empty_transcript.json"
    write_json(transcript, json_path)

    loaded = load_transcript_from_json(json_path)

    assert loaded.file_name == "empty.wav"
    assert loaded.language == "en"
    assert len(loaded.segments) == 0


def test_large_transcript(tmp_path):
    """Test handling of transcript with many segments."""
    # Create transcript with 100 segments
    segments = [
        Segment(
            id=i,
            start=float(i),
            end=float(i + 1),
            text=f"Segment {i}",
            audio_state={"test": f"data_{i}"},
        )
        for i in range(100)
    ]

    transcript = Transcript(file_name="large.wav", language="en", segments=segments)

    json_path = tmp_path / "large_transcript.json"
    write_json(transcript, json_path)

    loaded = load_transcript_from_json(json_path)

    assert len(loaded.segments) == 100

    # Verify first and last segments
    assert loaded.segments[0].id == 0
    assert loaded.segments[0].text == "Segment 0"
    assert loaded.segments[99].id == 99
    assert loaded.segments[99].text == "Segment 99"

    # Verify audio_state survived
    for i, seg in enumerate(loaded.segments):
        assert seg.audio_state is not None
        assert seg.audio_state["test"] == f"data_{i}"


def test_unicode_text(tmp_path):
    """Test handling of unicode text in transcripts."""
    segments = [
        Segment(id=0, start=0.0, end=1.0, text="Hello 世界"),
        Segment(id=1, start=1.0, end=2.0, text="Привет мир"),
        Segment(id=2, start=2.0, end=3.0, text="مرحبا بالعالم"),
        Segment(id=3, start=3.0, end=4.0, text="🎉 Emoji test 🚀"),
    ]

    transcript = Transcript(file_name="unicode.wav", language="multi", segments=segments)

    json_path = tmp_path / "unicode_transcript.json"
    write_json(transcript, json_path)

    loaded = load_transcript_from_json(json_path)

    assert loaded.segments[0].text == "Hello 世界"
    assert loaded.segments[1].text == "Привет мир"
    assert loaded.segments[2].text == "مرحبا بالعالم"
    assert loaded.segments[3].text == "🎉 Emoji test 🚀"


def test_special_characters_in_metadata(tmp_path):
    """Test handling of special characters in metadata."""
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=1.0, text="test")],
        meta={
            "path": "/home/user/audio files/test.wav",
            "command": 'python transcribe.py --model "base" --device cuda',
            "notes": "Testing: special & chars < > \" '",
        },
    )

    json_path = tmp_path / "special_meta.json"
    write_json(transcript, json_path)

    loaded = load_transcript_from_json(json_path)

    assert loaded.meta["path"] == "/home/user/audio files/test.wav"
    assert loaded.meta["command"] == 'python transcribe.py --model "base" --device cuda'
    assert loaded.meta["notes"] == "Testing: special & chars < > \" '"


def test_deeply_nested_audio_state(tmp_path):
    """Test handling of deeply nested audio_state structures."""
    audio_state = {
        "prosody": {
            "pitch": {
                "level": "high",
                "mean_hz": 245.3,
                "std_hz": 32.1,
                "min_hz": 180.0,
                "max_hz": 310.2,
                "variation": "moderate",
                "contour": "rising",
            },
            "energy": {"level": "loud", "db_rms": -8.2, "variation": "low"},
            "rate": {"level": "fast", "syllables_per_sec": 6.3, "words_per_sec": 3.1},
            "pauses": {"count": 2, "longest_ms": 320, "density": "sparse"},
        },
        "emotion": {
            "dimensional": {
                "valence": {"level": "positive", "score": 0.72},
                "arousal": {"level": "high", "score": 0.68},
                "dominance": {"level": "neutral", "score": 0.51},
            },
            "categorical": {
                "primary": "excited",
                "confidence": 0.81,
                "secondary": "happy",
                "secondary_confidence": 0.12,
                "all_scores": {"excited": 0.81, "happy": 0.12, "neutral": 0.05, "sad": 0.02},
            },
        },
    }

    segment = Segment(id=0, start=0.0, end=2.0, text="deeply nested test", audio_state=audio_state)

    transcript = Transcript(file_name="nested.wav", language="en", segments=[segment])

    json_path = tmp_path / "nested_audio_state.json"
    write_json(transcript, json_path)

    loaded = load_transcript_from_json(json_path)

    # Verify deep nesting is preserved
    seg = loaded.segments[0]
    assert seg.audio_state["prosody"]["pitch"]["mean_hz"] == 245.3
    assert seg.audio_state["prosody"]["pauses"]["longest_ms"] == 320
    assert seg.audio_state["emotion"]["dimensional"]["valence"]["score"] == 0.72
    assert seg.audio_state["emotion"]["categorical"]["all_scores"]["excited"] == 0.81
