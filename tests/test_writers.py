import json
import logging
from pathlib import Path

import pytest

from transcription import (
    __version__,
    writers,
)
from transcription.config import AppConfig
from transcription.models import SCHEMA_VERSION, Segment, Transcript, Turn
from transcription.pipeline import _build_meta
from transcription.writers import _to_dict


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
        id=1,
        start=1.5,
        end=3.0,
        text="second segment",
        speaker={"id": "Speaker A"},
        tone="positive",
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
    assert loaded_transcript.segments[1].speaker == {"id": "Speaker A"}
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


def test_load_transcript_accepts_file_name_key(tmp_path: Path) -> None:
    """API responses use file_name; loader should handle it as well."""
    api_payload = {
        "schema_version": 2,
        "file_name": "api_response.wav",
        "language": "en",
        "meta": {"model_name": "tiny"},
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "hello", "speaker": None, "tone": None}
        ],
    }

    json_path = tmp_path / "api_response.json"
    json_path.write_text(json.dumps(api_payload, indent=2), encoding="utf-8")

    loaded = writers.load_transcript_from_json(json_path)
    assert loaded.file_name == "api_response.wav"
    assert loaded.language == "en"
    assert loaded.segments[0].text == "hello"


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
    turn_0 = loaded.turns[0]
    turn_1 = loaded.turns[1]
    assert isinstance(turn_0, dict)
    assert isinstance(turn_1, dict)
    assert turn_0["speaker_id"] == "spk_0"
    assert turn_1["text"] == "Hi"


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


# ============================================================================
# Tests for _to_dict helper function
# ============================================================================


def test_to_dict_with_plain_dict() -> None:
    """_to_dict should return plain dicts unchanged."""
    input_dict = {"key": "value", "number": 42}
    result = _to_dict(input_dict)
    assert result == input_dict
    assert result is input_dict  # Should be the same object


def test_to_dict_with_to_dict_object() -> None:
    """_to_dict should call to_dict() on objects that have it."""
    turn = Turn(
        id="turn_0",
        speaker_id="spk_0",
        segment_ids=[0, 1],
        start=0.0,
        end=2.5,
        text="Hello world",
        metadata={"question_count": 1},
    )
    result = _to_dict(turn)

    assert isinstance(result, dict)
    assert result["id"] == "turn_0"
    assert result["speaker_id"] == "spk_0"
    assert result["segment_ids"] == [0, 1]
    assert result["text"] == "Hello world"


def test_to_dict_with_dataclass_instance() -> None:
    """_to_dict should convert dataclass instances using asdict."""
    segment = Segment(
        id=0,
        start=0.0,
        end=1.5,
        text="Test segment",
        speaker={"id": "spk_0"},
    )
    result = _to_dict(segment)

    assert isinstance(result, dict)
    assert result["id"] == 0
    assert result["start"] == 0.0
    assert result["end"] == 1.5
    assert result["text"] == "Test segment"
    assert result["speaker"] == {"id": "spk_0"}


def test_to_dict_with_dataclass_class_returns_unchanged() -> None:
    """_to_dict should return dataclass classes unchanged (not instances)."""
    result = _to_dict(Turn)
    assert result is Turn  # Class itself, not converted


def test_to_dict_with_unsupported_type_returns_unchanged() -> None:
    """_to_dict should return unsupported types unchanged."""
    assert _to_dict("a string") == "a string"
    assert _to_dict(42) == 42
    assert _to_dict([1, 2, 3]) == [1, 2, 3]
    assert _to_dict(None) is None


class CustomObjectWithToDict:
    """Test object with a to_dict method."""

    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def test_to_dict_with_custom_to_dict() -> None:
    """_to_dict should work with any object that has to_dict()."""
    custom = CustomObjectWithToDict({"custom": "data", "count": 5})
    result = _to_dict(custom)

    assert result == {"custom": "data", "count": 5}


def test_write_json_logs_warning_for_unexpected_meta(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """write_json should log a warning when meta can't be converted to dict."""
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=1.0, text="test")],
        meta="invalid meta type",  # type: ignore[arg-type]
    )

    json_path = tmp_path / "test_meta_warning.json"

    with caplog.at_level(logging.WARNING, logger="transcription.writers"):
        writers.write_json(transcript, json_path)

    # Verify warning was logged
    assert any("Unexpected transcript.meta type" in record.message for record in caplog.records)

    # Verify the file was still written with empty meta
    data = json.loads(json_path.read_text())
    assert data["meta"] == {}


def test_turns_as_turn_dataclass_roundtrip(tmp_path: Path) -> None:
    """Test round-trip serialization when turns are Turn dataclass objects.

    This verifies that _to_dict properly converts Turn dataclasses during
    write_json, and that the round-trip preserves all data.
    """
    from typing import Any

    # Create turns as Turn dataclass objects, not dicts
    turns: list[Turn | dict[str, Any]] = [
        Turn(
            id="turn_0",
            speaker_id="spk_0",
            segment_ids=[0, 1],
            start=0.0,
            end=3.5,
            text="Hello there",
            metadata={"question_count": 1},
        ),
        Turn(
            id="turn_1",
            speaker_id="spk_1",
            segment_ids=[2],
            start=3.6,
            end=5.0,
            text="Hi",
            metadata={},
        ),
    ]

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=1.5, text="Hello", speaker={"id": "spk_0"}),
            Segment(id=1, start=1.5, end=3.5, text="there", speaker={"id": "spk_0"}),
            Segment(id=2, start=3.6, end=5.0, text="Hi", speaker={"id": "spk_1"}),
        ],
        turns=turns,
    )

    json_path = tmp_path / "test_turn_dataclass.json"
    writers.write_json(transcript, json_path)

    # Verify JSON structure - Turn.to_dict() was called
    data = json.loads(json_path.read_text())
    assert len(data["turns"]) == 2
    assert data["turns"][0]["speaker_id"] == "spk_0"
    assert data["turns"][0]["segment_ids"] == [0, 1]
    assert data["turns"][0]["metadata"]["question_count"] == 1
    assert data["turns"][1]["id"] == "turn_1"

    # Test round-trip
    loaded = writers.load_transcript_from_json(json_path)
    assert loaded.turns is not None
    assert len(loaded.turns) == 2
    turn_0 = loaded.turns[0]
    turn_1 = loaded.turns[1]
    assert isinstance(turn_0, dict)
    assert isinstance(turn_1, dict)
    assert turn_0["speaker_id"] == "spk_0"
    assert turn_0["metadata"]["question_count"] == 1
    assert turn_1["text"] == "Hi"


def test_meta_with_to_dict_object_roundtrip(tmp_path: Path) -> None:
    """Test that meta containing objects with to_dict() is properly serialized.

    This verifies the _to_dict helper works for nested meta objects like
    DiarizationMeta, not just top-level fields.
    """
    from transcription.models import DiarizationMeta

    diar_meta = DiarizationMeta(
        requested=True,
        status="ok",
        backend="pyannote",
        num_speakers=2,
        error_type=None,
        message=None,
    )

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[Segment(id=0, start=0.0, end=1.0, text="test")],
        meta={"diarization": diar_meta, "custom_key": "value"},
    )

    json_path = tmp_path / "test_meta_to_dict.json"
    writers.write_json(transcript, json_path)

    # Verify JSON structure - DiarizationMeta.to_dict() was called
    data = json.loads(json_path.read_text())
    assert data["meta"]["diarization"]["requested"] is True
    assert data["meta"]["diarization"]["status"] == "ok"
    assert data["meta"]["diarization"]["backend"] == "pyannote"
    assert data["meta"]["diarization"]["num_speakers"] == 2
    assert data["meta"]["custom_key"] == "value"

    # Round-trip loads as plain dict (not DiarizationMeta)
    loaded = writers.load_transcript_from_json(json_path)
    assert loaded.meta is not None
    assert loaded.meta["diarization"]["status"] == "ok"
    assert loaded.meta["custom_key"] == "value"
