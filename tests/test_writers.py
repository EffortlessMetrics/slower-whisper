from pathlib import Path
import json
from transcription.models import Segment, Transcript, SCHEMA_VERSION
from transcription import __version__
from transcription.config import AppConfig
from transcription.pipeline import _build_meta
from transcription import writers  # type: ignore[attr-defined]


def test_write_json_shape(tmp_path: Path) -> None:
    seg = Segment(id=0, start=0.0, end=1.0, text="hello")
    t = Transcript(file_name="test.wav", language="en", segments=[seg])
    cfg = AppConfig()
    t.meta = _build_meta(cfg, t, Path("test.wav"), duration_sec=1.0)

    out_path = tmp_path / "out.json"
    writers.write_json(t, out_path)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION  # Uses current schema version (2)
    assert data["file"] == "test.wav"
    assert data["language"] == "en"
    assert "meta" in data
    assert data["meta"]["model_name"] == cfg.asr.model_name
    assert data["meta"]["pipeline_version"] == __version__
    assert len(data["segments"]) == 1
    seg0 = data["segments"][0]
    assert seg0["id"] == 0
    assert seg0["text"] == "hello"


def test_audio_state_serialization(tmp_path: Path) -> None:
    """Test that audio_state field is properly serialized and deserialized."""
    # Create segment with audio_state
    audio_state = {
        "prosody": {
            "pitch": {"level": "high", "mean_hz": 245.3},
            "energy": {"level": "normal", "db_rms": -12.5},
        },
        "emotion": {
            "categorical": {"primary": "neutral", "confidence": 0.85}
        }
    }
    seg = Segment(
        id=0,
        start=0.0,
        end=1.5,
        text="hello world",
        audio_state=audio_state
    )
    t = Transcript(file_name="test.wav", language="en", segments=[seg])
    cfg = AppConfig()
    t.meta = _build_meta(cfg, t, Path("test.wav"), duration_sec=1.5)

    # Write JSON
    out_path = tmp_path / "out_audio_state.json"
    writers.write_json(t, out_path)

    # Verify serialization
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(data["segments"]) == 1
    seg0 = data["segments"][0]
    assert seg0["audio_state"] is not None
    assert "prosody" in seg0["audio_state"]
    assert "emotion" in seg0["audio_state"]
    assert seg0["audio_state"]["prosody"]["pitch"]["level"] == "high"
    assert seg0["audio_state"]["emotion"]["categorical"]["primary"] == "neutral"


def test_audio_state_none(tmp_path: Path) -> None:
    """Test that segments without audio_state have None value."""
    seg = Segment(id=0, start=0.0, end=1.0, text="hello")
    t = Transcript(file_name="test.wav", language="en", segments=[seg])

    out_path = tmp_path / "out_no_audio.json"
    writers.write_json(t, out_path)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["segments"][0]["audio_state"] is None


def test_load_transcript_from_json(tmp_path: Path) -> None:
    """Test loading transcript from JSON file."""
    # Create and write a transcript
    seg1 = Segment(id=0, start=0.0, end=1.5, text="first segment")
    seg2 = Segment(
        id=1,
        start=1.5,
        end=3.0,
        text="second segment",
        speaker="Speaker A",
        tone="positive"
    )
    original_transcript = Transcript(
        file_name="test_audio.wav",
        language="en",
        segments=[seg1, seg2],
        meta={"model": "test-model", "version": "1.0"}
    )

    json_path = tmp_path / "test_transcript.json"
    writers.write_json(original_transcript, json_path)

    # Load it back
    loaded_transcript = writers.load_transcript_from_json(json_path)

    # Verify basic fields
    assert loaded_transcript.file_name == "test_audio.wav"
    assert loaded_transcript.language == "en"
    assert len(loaded_transcript.segments) == 2
    assert loaded_transcript.meta == {"model": "test-model", "version": "1.0"}

    # Verify first segment
    assert loaded_transcript.segments[0].id == 0
    assert loaded_transcript.segments[0].start == 0.0
    assert loaded_transcript.segments[0].end == 1.5
    assert loaded_transcript.segments[0].text == "first segment"
    assert loaded_transcript.segments[0].speaker is None
    assert loaded_transcript.segments[0].tone is None
    assert loaded_transcript.segments[0].audio_state is None

    # Verify second segment
    assert loaded_transcript.segments[1].id == 1
    assert loaded_transcript.segments[1].speaker == "Speaker A"
    assert loaded_transcript.segments[1].tone == "positive"


def test_load_transcript_with_audio_state(tmp_path: Path) -> None:
    """Test loading transcript with audio_state from JSON."""
    audio_state = {
        "prosody": {"pitch": {"level": "high"}},
        "emotion": {"categorical": {"primary": "happy", "confidence": 0.9}}
    }
    seg = Segment(
        id=0,
        start=0.0,
        end=2.0,
        text="enriched segment",
        audio_state=audio_state
    )
    original_transcript = Transcript(
        file_name="enriched.wav",
        language="en",
        segments=[seg]
    )

    json_path = tmp_path / "enriched_transcript.json"
    writers.write_json(original_transcript, json_path)

    # Load it back
    loaded_transcript = writers.load_transcript_from_json(json_path)

    # Verify audio_state is properly loaded
    assert loaded_transcript.segments[0].audio_state is not None
    assert loaded_transcript.segments[0].audio_state["prosody"]["pitch"]["level"] == "high"
    assert loaded_transcript.segments[0].audio_state["emotion"]["categorical"]["primary"] == "happy"
    assert loaded_transcript.segments[0].audio_state["emotion"]["categorical"]["confidence"] == 0.9


def test_load_transcript_backward_compatibility(tmp_path: Path) -> None:
    """Test that loading old JSON files (without audio_state) works."""
    # Manually create an old-format JSON without audio_state fields
    old_format_data = {
        "schema_version": 1,
        "file": "old_format.wav",
        "language": "en",
        "meta": {"model": "old"},
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "old segment",
                "speaker": None,
                "tone": None
                # Note: no audio_state field
            }
        ]
    }

    json_path = tmp_path / "old_format.json"
    json_path.write_text(json.dumps(old_format_data, indent=2), encoding="utf-8")

    # Load it - should not fail
    loaded_transcript = writers.load_transcript_from_json(json_path)

    assert loaded_transcript.file_name == "old_format.wav"
    assert loaded_transcript.segments[0].text == "old segment"
    assert loaded_transcript.segments[0].audio_state is None  # Should default to None
