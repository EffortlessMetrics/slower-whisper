import json
from pathlib import Path

from transcription import (
    __version__,
    writers,  # type: ignore[attr-defined]
)
from transcription.config import AppConfig
from transcription.models import SCHEMA_VERSION, Segment, Transcript
from transcription.pipeline import _build_meta


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
        "emotion": {"categorical": {"primary": "neutral", "confidence": 0.85}},
    }
    seg = Segment(id=0, start=0.0, end=1.5, text="hello world", audio_state=audio_state)
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
        id=1, start=1.5, end=3.0, text="second segment", speaker="Speaker A", tone="positive"
    )
    original_transcript = Transcript(
        file_name="test_audio.wav",
        language="en",
        segments=[seg1, seg2],
        meta={"model": "test-model", "version": "1.0"},
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
        "emotion": {"categorical": {"primary": "happy", "confidence": 0.9}},
    }
    seg = Segment(id=0, start=0.0, end=2.0, text="enriched segment", audio_state=audio_state)
    original_transcript = Transcript(file_name="enriched.wav", language="en", segments=[seg])

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
                "tone": None,
                # Note: no audio_state field
            }
        ],
    }

    json_path = tmp_path / "old_format.json"
    json_path.write_text(json.dumps(old_format_data, indent=2), encoding="utf-8")

    # Load it - should not fail
    loaded_transcript = writers.load_transcript_from_json(json_path)

    assert loaded_transcript.file_name == "old_format.wav"
    assert loaded_transcript.segments[0].text == "old segment"
    assert loaded_transcript.segments[0].audio_state is None  # Should default to None


def test_segment_speaker_dict_serialization(tmp_path: Path) -> None:
    """Test serialization of v1.1 speaker dict structure in segments."""
    # Create segment with speaker dict (v1.1 format)
    seg = Segment(
        id=0,
        start=0.0,
        end=2.0,
        text="Hello world",
        speaker={"id": "spk_0", "confidence": 0.95},
    )
    transcript = Transcript(file_name="test.wav", language="en", segments=[seg])

    # Write and verify JSON
    json_path = tmp_path / "test_speaker.json"
    writers.write_json(transcript, json_path)

    # Verify JSON structure
    data = json.loads(json_path.read_text())
    assert data["segments"][0]["speaker"]["id"] == "spk_0"
    assert data["segments"][0]["speaker"]["confidence"] == 0.95

    # Test round-trip
    loaded = writers.load_transcript_from_json(json_path)
    assert loaded.segments[0].speaker == {"id": "spk_0", "confidence": 0.95}


def test_speakers_array_serialization(tmp_path: Path) -> None:
    """Test serialization of speakers[] array (v1.1)."""
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[],
        speakers=[
            {"id": "spk_0", "label": None, "total_speech_time": 5.5, "num_segments": 3},
            {"id": "spk_1", "label": "Alice", "total_speech_time": 3.2, "num_segments": 2},
        ],
    )

    json_path = tmp_path / "test_speakers.json"
    writers.write_json(transcript, json_path)

    # Verify JSON structure
    data = json.loads(json_path.read_text())
    assert len(data["speakers"]) == 2
    assert data["speakers"][0]["id"] == "spk_0"
    assert data["speakers"][0]["total_speech_time"] == 5.5
    assert data["speakers"][0]["num_segments"] == 3
    assert data["speakers"][1]["label"] == "Alice"

    # Test round-trip
    loaded = writers.load_transcript_from_json(json_path)
    assert loaded.speakers is not None
    assert len(loaded.speakers) == 2
    assert loaded.speakers[0]["id"] == "spk_0"
    assert loaded.speakers[1]["label"] == "Alice"


def test_turns_array_serialization(tmp_path: Path) -> None:
    """Test serialization of turns[] array (v1.1)."""
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[],
        turns=[
            {
                "id": "turn_0",
                "speaker_id": "spk_0",
                "start": 0.0,
                "end": 3.5,
                "segment_ids": [0, 1],
                "text": "Hello there",
            },
            {
                "id": "turn_1",
                "speaker_id": "spk_1",
                "start": 3.6,
                "end": 5.0,
                "segment_ids": [2],
                "text": "Hi",
            },
        ],
    )

    json_path = tmp_path / "test_turns.json"
    writers.write_json(transcript, json_path)

    # Verify JSON structure
    data = json.loads(json_path.read_text())
    assert len(data["turns"]) == 2
    assert data["turns"][0]["speaker_id"] == "spk_0"
    assert data["turns"][0]["segment_ids"] == [0, 1]
    assert data["turns"][0]["text"] == "Hello there"
    assert data["turns"][1]["id"] == "turn_1"

    # Test round-trip
    loaded = writers.load_transcript_from_json(json_path)
    assert loaded.turns is not None
    assert len(loaded.turns) == 2
    assert loaded.turns[0]["speaker_id"] == "spk_0"
    assert loaded.turns[1]["text"] == "Hi"


def test_null_speakers_and_turns_serialization(tmp_path: Path) -> None:
    """Test that null speakers and turns serialize correctly (omitted from JSON)."""
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=1.0, text="test")],
        speakers=None,
        turns=None,
    )

    json_path = tmp_path / "test_null.json"
    writers.write_json(transcript, json_path)

    # Verify JSON structure - speakers/turns keys should be omitted when None
    data = json.loads(json_path.read_text())
    assert "speakers" not in data  # None values are omitted
    assert "turns" not in data  # None values are omitted

    # Test round-trip - missing keys should load as None
    loaded = writers.load_transcript_from_json(json_path)
    assert loaded.speakers is None
    assert loaded.turns is None
