"""Tests for the validation module.

This module tests JSON loading, schema validation, and error handling
for the validation.py module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# Import directly from submodules to avoid circular import issues
# in the main transcription package
from slower_whisper.pipeline.exceptions import ConfigurationError
from slower_whisper.pipeline.validation import (
    DEFAULT_SCHEMA_PATH,
    _require_jsonschema,
    load_json,
    validate_many,
    validate_transcript_json,
)

# ============================================================================
# Tests for load_json()
# ============================================================================


class TestLoadJson:
    """Test suite for the load_json() function."""

    def test_load_json_valid_file(self, tmp_path: Path) -> None:
        """load_json should successfully load a valid JSON file."""
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        json_path = tmp_path / "valid.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        result = load_json(json_path)

        assert result == data
        assert result["key"] == "value"
        assert result["number"] == 42
        assert result["nested"]["a"] == 1

    def test_load_json_empty_object(self, tmp_path: Path) -> None:
        """load_json should handle empty JSON objects."""
        json_path = tmp_path / "empty.json"
        json_path.write_text("{}", encoding="utf-8")

        result = load_json(json_path)

        assert result == {}

    def test_load_json_unicode_content(self, tmp_path: Path) -> None:
        """load_json should handle unicode content correctly."""
        data = {"message": "Hello, World!", "unicode": "Cafe \u2615"}
        json_path = tmp_path / "unicode.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        result = load_json(json_path)

        assert result["unicode"] == "Cafe \u2615"

    def test_load_json_file_not_found_raises_configuration_error(self) -> None:
        """load_json should raise ConfigurationError for missing files."""
        nonexistent_path = Path("/nonexistent/path/to/file.json")

        with pytest.raises(ConfigurationError) as exc_info:
            load_json(nonexistent_path)

        error_msg = str(exc_info.value)
        assert "File not found" in error_msg
        assert str(nonexistent_path) in error_msg

    def test_load_json_invalid_json_raises_configuration_error(self, tmp_path: Path) -> None:
        """load_json should raise ConfigurationError for invalid JSON."""
        json_path = tmp_path / "invalid.json"
        json_path.write_text("{invalid json content}", encoding="utf-8")

        with pytest.raises(ConfigurationError) as exc_info:
            load_json(json_path)

        error_msg = str(exc_info.value)
        assert "Invalid JSON" in error_msg
        assert str(json_path) in error_msg

    def test_load_json_truncated_json_raises_configuration_error(self, tmp_path: Path) -> None:
        """load_json should raise ConfigurationError for truncated JSON."""
        json_path = tmp_path / "truncated.json"
        json_path.write_text('{"key": "value",', encoding="utf-8")

        with pytest.raises(ConfigurationError) as exc_info:
            load_json(json_path)

        assert "Invalid JSON" in str(exc_info.value)

    def test_load_json_array_at_root(self, tmp_path: Path) -> None:
        """load_json should handle JSON arrays at root (returns dict type annotation)."""
        # Note: The function annotates return as dict, but JSON can be any type.
        # This test documents the actual behavior when loading arrays.
        json_path = tmp_path / "array.json"
        json_path.write_text("[1, 2, 3]", encoding="utf-8")

        # Function returns Any in practice even though annotated as dict
        result = load_json(json_path)
        assert result == [1, 2, 3]

    def test_load_json_preserves_json_types(self, tmp_path: Path) -> None:
        """load_json should preserve all JSON types correctly."""
        data: dict[str, Any] = {
            "string": "text",
            "integer": 123,
            "float": 3.14,
            "boolean_true": True,
            "boolean_false": False,
            "null": None,
            "array": [1, "two", 3.0],
            "object": {"nested": "value"},
        }
        json_path = tmp_path / "types.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        result = load_json(json_path)

        assert result["string"] == "text"
        assert result["integer"] == 123
        assert result["float"] == 3.14
        assert result["boolean_true"] is True
        assert result["boolean_false"] is False
        assert result["null"] is None
        assert result["array"] == [1, "two", 3.0]
        assert result["object"] == {"nested": "value"}


# ============================================================================
# Tests for _require_jsonschema()
# ============================================================================


class TestRequireJsonschema:
    """Test suite for the _require_jsonschema() function."""

    def test_require_jsonschema_returns_module_when_available(self) -> None:
        """_require_jsonschema should return jsonschema module when installed."""
        # jsonschema is installed in the test environment
        result = _require_jsonschema()

        assert result is not None
        assert hasattr(result, "Draft7Validator")

    def test_require_jsonschema_raises_when_unavailable(self) -> None:
        """_require_jsonschema should raise ConfigurationError when not installed."""
        # Temporarily remove jsonschema from modules to simulate missing dependency
        original_modules = sys.modules.copy()

        try:
            # Remove jsonschema and any cached imports
            modules_to_remove = [k for k in sys.modules if k.startswith("jsonschema")]
            for mod in modules_to_remove:
                del sys.modules[mod]

            # Patch the import to raise ImportError
            with patch.dict(sys.modules, {"jsonschema": None}):
                with patch("builtins.__import__", side_effect=ImportError("No module")):
                    with pytest.raises(ConfigurationError) as exc_info:
                        _require_jsonschema()

                    error_msg = str(exc_info.value)
                    assert "jsonschema is required" in error_msg
                    assert "uv pip install jsonschema" in error_msg
        finally:
            # Restore original modules
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_require_jsonschema_handles_generic_exception(self) -> None:
        """_require_jsonschema should handle any exception during import."""
        original_modules = sys.modules.copy()

        try:
            modules_to_remove = [k for k in sys.modules if k.startswith("jsonschema")]
            for mod in modules_to_remove:
                del sys.modules[mod]

            with patch.dict(sys.modules, {"jsonschema": None}):
                with patch("builtins.__import__", side_effect=RuntimeError("Unexpected error")):
                    with pytest.raises(ConfigurationError) as exc_info:
                        _require_jsonschema()

                    assert "jsonschema is required" in str(exc_info.value)
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)


# ============================================================================
# Tests for validate_transcript_json()
# ============================================================================


class TestValidateTranscriptJson:
    """Test suite for the validate_transcript_json() function."""

    def test_validate_valid_transcript(self, tmp_path: Path) -> None:
        """validate_transcript_json should return (True, []) for valid transcript."""
        valid_transcript = {
            "schema_version": 2,
            "file": "test.wav",
            "language": "en",
            "segments": [{"id": 0, "start": 0.0, "end": 1.5, "text": "Hello world"}],
        }
        transcript_path = tmp_path / "valid_transcript.json"
        transcript_path.write_text(json.dumps(valid_transcript), encoding="utf-8")

        is_valid, errors = validate_transcript_json(transcript_path)

        assert is_valid is True
        assert errors == []

    def test_validate_transcript_with_all_fields(self, tmp_path: Path) -> None:
        """validate_transcript_json should accept transcript with all optional fields."""
        full_transcript = {
            "schema_version": 2,
            "file": "meeting.wav",
            "language": "en",
            "speakers": [{"id": "spk_0", "total_speech_time": 5.5, "num_segments": 3}],
            "turns": [
                {
                    "id": "turn_0",
                    "speaker_id": "spk_0",
                    "start": 0.0,
                    "end": 2.5,
                    "segment_ids": [0],
                }
            ],
            "meta": {
                "model_name": "large-v3",
                "device": "cuda",
                "duration_sec": 10.0,
            },
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello",
                    "speaker": {"id": "spk_0", "confidence": 0.95},
                }
            ],
        }
        transcript_path = tmp_path / "full_transcript.json"
        transcript_path.write_text(json.dumps(full_transcript), encoding="utf-8")

        is_valid, errors = validate_transcript_json(transcript_path)

        assert is_valid is True
        assert errors == []

    def test_validate_missing_required_field_schema_version(self, tmp_path: Path) -> None:
        """validate_transcript_json should detect missing schema_version."""
        invalid_transcript = {
            "file": "test.wav",
            "language": "en",
            "segments": [],
        }
        transcript_path = tmp_path / "missing_version.json"
        transcript_path.write_text(json.dumps(invalid_transcript), encoding="utf-8")

        is_valid, errors = validate_transcript_json(transcript_path)

        assert is_valid is False
        assert len(errors) >= 1
        assert any("schema_version" in err for err in errors)

    def test_validate_missing_required_field_file(self, tmp_path: Path) -> None:
        """validate_transcript_json should detect missing file field."""
        invalid_transcript = {
            "schema_version": 2,
            "language": "en",
            "segments": [],
        }
        transcript_path = tmp_path / "missing_file.json"
        transcript_path.write_text(json.dumps(invalid_transcript), encoding="utf-8")

        is_valid, errors = validate_transcript_json(transcript_path)

        assert is_valid is False
        assert any("file" in err for err in errors)

    def test_validate_missing_required_field_language(self, tmp_path: Path) -> None:
        """validate_transcript_json should detect missing language field."""
        invalid_transcript = {
            "schema_version": 2,
            "file": "test.wav",
            "segments": [],
        }
        transcript_path = tmp_path / "missing_language.json"
        transcript_path.write_text(json.dumps(invalid_transcript), encoding="utf-8")

        is_valid, errors = validate_transcript_json(transcript_path)

        assert is_valid is False
        assert any("language" in err for err in errors)

    def test_validate_missing_required_field_segments(self, tmp_path: Path) -> None:
        """validate_transcript_json should detect missing segments field."""
        invalid_transcript = {
            "schema_version": 2,
            "file": "test.wav",
            "language": "en",
        }
        transcript_path = tmp_path / "missing_segments.json"
        transcript_path.write_text(json.dumps(invalid_transcript), encoding="utf-8")

        is_valid, errors = validate_transcript_json(transcript_path)

        assert is_valid is False
        assert any("segments" in err for err in errors)

    def test_validate_wrong_schema_version(self, tmp_path: Path) -> None:
        """validate_transcript_json should detect wrong schema version."""
        invalid_transcript = {
            "schema_version": 1,  # Should be 2
            "file": "test.wav",
            "language": "en",
            "segments": [],
        }
        transcript_path = tmp_path / "wrong_version.json"
        transcript_path.write_text(json.dumps(invalid_transcript), encoding="utf-8")

        is_valid, errors = validate_transcript_json(transcript_path)

        assert is_valid is False
        assert len(errors) >= 1

    def test_validate_invalid_language_format(self, tmp_path: Path) -> None:
        """validate_transcript_json should detect invalid language format."""
        invalid_transcript = {
            "schema_version": 2,
            "file": "test.wav",
            "language": "english",  # Should be 'en' (2-letter code)
            "segments": [],
        }
        transcript_path = tmp_path / "invalid_language.json"
        transcript_path.write_text(json.dumps(invalid_transcript), encoding="utf-8")

        is_valid, errors = validate_transcript_json(transcript_path)

        assert is_valid is False
        assert any("language" in err for err in errors)

    def test_validate_invalid_segment_missing_id(self, tmp_path: Path) -> None:
        """validate_transcript_json should detect segment missing id field."""
        invalid_transcript = {
            "schema_version": 2,
            "file": "test.wav",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello"}  # Missing 'id'
            ],
        }
        transcript_path = tmp_path / "segment_no_id.json"
        transcript_path.write_text(json.dumps(invalid_transcript), encoding="utf-8")

        is_valid, errors = validate_transcript_json(transcript_path)

        assert is_valid is False
        assert any("id" in err for err in errors)

    def test_validate_invalid_segment_negative_time(self, tmp_path: Path) -> None:
        """validate_transcript_json should detect negative time values in segments."""
        invalid_transcript = {
            "schema_version": 2,
            "file": "test.wav",
            "language": "en",
            "segments": [
                {"id": 0, "start": -1.0, "end": 1.0, "text": "Hello"}  # Negative start
            ],
        }
        transcript_path = tmp_path / "negative_time.json"
        transcript_path.write_text(json.dumps(invalid_transcript), encoding="utf-8")

        is_valid, errors = validate_transcript_json(transcript_path)

        assert is_valid is False
        assert len(errors) >= 1

    def test_validate_with_custom_schema_path(self, tmp_path: Path) -> None:
        """validate_transcript_json should use custom schema when provided."""
        # Create a minimal custom schema that only requires 'custom_field'
        custom_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["custom_field"],
            "properties": {"custom_field": {"type": "string"}},
        }
        schema_path = tmp_path / "custom_schema.json"
        schema_path.write_text(json.dumps(custom_schema), encoding="utf-8")

        # Create a document that matches the custom schema
        valid_doc = {"custom_field": "value"}
        doc_path = tmp_path / "custom_doc.json"
        doc_path.write_text(json.dumps(valid_doc), encoding="utf-8")

        is_valid, errors = validate_transcript_json(doc_path, schema_path=schema_path)

        assert is_valid is True
        assert errors == []

    def test_validate_with_custom_schema_invalid_document(self, tmp_path: Path) -> None:
        """validate_transcript_json should reject invalid doc with custom schema."""
        custom_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["required_field"],
            "properties": {"required_field": {"type": "integer"}},
        }
        schema_path = tmp_path / "custom_schema.json"
        schema_path.write_text(json.dumps(custom_schema), encoding="utf-8")

        # Create a document missing the required field
        invalid_doc = {"other_field": "value"}
        doc_path = tmp_path / "invalid_doc.json"
        doc_path.write_text(json.dumps(invalid_doc), encoding="utf-8")

        is_valid, errors = validate_transcript_json(doc_path, schema_path=schema_path)

        assert is_valid is False
        assert len(errors) >= 1

    def test_validate_nonexistent_transcript_raises_error(self) -> None:
        """validate_transcript_json should raise for missing transcript file."""
        nonexistent_path = Path("/nonexistent/transcript.json")

        with pytest.raises(ConfigurationError) as exc_info:
            validate_transcript_json(nonexistent_path)

        assert "File not found" in str(exc_info.value)

    def test_validate_nonexistent_schema_raises_error(self, tmp_path: Path) -> None:
        """validate_transcript_json should raise for missing schema file."""
        valid_transcript = {
            "schema_version": 2,
            "file": "test.wav",
            "language": "en",
            "segments": [],
        }
        transcript_path = tmp_path / "transcript.json"
        transcript_path.write_text(json.dumps(valid_transcript), encoding="utf-8")

        nonexistent_schema = Path("/nonexistent/schema.json")

        with pytest.raises(ConfigurationError) as exc_info:
            validate_transcript_json(transcript_path, schema_path=nonexistent_schema)

        assert "File not found" in str(exc_info.value)

    def test_validate_invalid_json_in_transcript_raises_error(self, tmp_path: Path) -> None:
        """validate_transcript_json should raise for invalid JSON in transcript."""
        transcript_path = tmp_path / "bad_json.json"
        transcript_path.write_text("{not valid json}", encoding="utf-8")

        with pytest.raises(ConfigurationError) as exc_info:
            validate_transcript_json(transcript_path)

        assert "Invalid JSON" in str(exc_info.value)


# ============================================================================
# Tests for validate_many()
# ============================================================================


class TestValidateMany:
    """Test suite for the validate_many() function."""

    def test_validate_many_all_valid(self, tmp_path: Path) -> None:
        """validate_many should return empty list when all files are valid."""
        valid_transcript = {
            "schema_version": 2,
            "file": "test.wav",
            "language": "en",
            "segments": [],
        }

        paths = []
        for i in range(3):
            path = tmp_path / f"valid_{i}.json"
            path.write_text(json.dumps(valid_transcript), encoding="utf-8")
            paths.append(path)

        failures = validate_many(paths)

        assert failures == []

    def test_validate_many_single_invalid(self, tmp_path: Path) -> None:
        """validate_many should report errors for single invalid file."""
        valid_transcript = {
            "schema_version": 2,
            "file": "test.wav",
            "language": "en",
            "segments": [],
        }
        invalid_transcript = {
            "schema_version": 2,
            "file": "test.wav",
            # Missing 'language' field
            "segments": [],
        }

        valid_path = tmp_path / "valid.json"
        valid_path.write_text(json.dumps(valid_transcript), encoding="utf-8")

        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text(json.dumps(invalid_transcript), encoding="utf-8")

        failures = validate_many([valid_path, invalid_path])

        assert len(failures) >= 1
        assert any(str(invalid_path) in f for f in failures)
        assert not any(str(valid_path) in f for f in failures)

    def test_validate_many_multiple_invalid(self, tmp_path: Path) -> None:
        """validate_many should aggregate errors from multiple invalid files."""
        invalid1 = {
            "schema_version": 2,
            "file": "test.wav",
            # Missing 'language' and 'segments'
        }
        invalid2 = {
            "schema_version": 1,  # Wrong version
            "file": "test.wav",
            "language": "en",
            "segments": [],
        }

        path1 = tmp_path / "invalid1.json"
        path1.write_text(json.dumps(invalid1), encoding="utf-8")

        path2 = tmp_path / "invalid2.json"
        path2.write_text(json.dumps(invalid2), encoding="utf-8")

        failures = validate_many([path1, path2])

        assert len(failures) >= 2  # At least one error per file
        assert any(str(path1) in f for f in failures)
        assert any(str(path2) in f for f in failures)

    def test_validate_many_empty_iterable(self) -> None:
        """validate_many should return empty list for empty input."""
        failures = validate_many([])

        assert failures == []

    def test_validate_many_with_custom_schema(self, tmp_path: Path) -> None:
        """validate_many should use custom schema when provided."""
        custom_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        schema_path = tmp_path / "custom.json"
        schema_path.write_text(json.dumps(custom_schema), encoding="utf-8")

        valid_doc = {"name": "test"}
        invalid_doc = {"other": "field"}

        valid_path = tmp_path / "valid.json"
        valid_path.write_text(json.dumps(valid_doc), encoding="utf-8")

        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text(json.dumps(invalid_doc), encoding="utf-8")

        failures = validate_many([valid_path, invalid_path], schema_path=schema_path)

        assert len(failures) >= 1
        assert any(str(invalid_path) in f for f in failures)
        assert not any(str(valid_path) in f for f in failures)

    def test_validate_many_error_format_includes_path(self, tmp_path: Path) -> None:
        """validate_many errors should be prefixed with file path."""
        invalid_transcript = {
            "schema_version": 2,
            "file": "test.wav",
            "segments": [],
            # Missing 'language'
        }
        path = tmp_path / "test_transcript.json"
        path.write_text(json.dumps(invalid_transcript), encoding="utf-8")

        failures = validate_many([path])

        assert len(failures) >= 1
        # Each error should start with the file path
        for failure in failures:
            assert failure.startswith(str(path) + ": ")

    def test_validate_many_generator_input(self, tmp_path: Path) -> None:
        """validate_many should accept any iterable, including generators."""
        valid_transcript = {
            "schema_version": 2,
            "file": "test.wav",
            "language": "en",
            "segments": [],
        }

        paths = []
        for i in range(2):
            path = tmp_path / f"gen_{i}.json"
            path.write_text(json.dumps(valid_transcript), encoding="utf-8")
            paths.append(path)

        def path_generator() -> Any:
            yield from paths

        failures = validate_many(path_generator())

        assert failures == []


# ============================================================================
# Tests for DEFAULT_SCHEMA_PATH
# ============================================================================


class TestDefaultSchemaPath:
    """Test suite for the DEFAULT_SCHEMA_PATH constant."""

    def test_default_schema_path_exists(self) -> None:
        """DEFAULT_SCHEMA_PATH should point to an existing file."""
        assert DEFAULT_SCHEMA_PATH.exists()
        assert DEFAULT_SCHEMA_PATH.is_file()

    def test_default_schema_path_is_valid_json(self) -> None:
        """DEFAULT_SCHEMA_PATH should contain valid JSON."""
        schema = load_json(DEFAULT_SCHEMA_PATH)

        assert isinstance(schema, dict)
        assert "$schema" in schema or "type" in schema

    def test_default_schema_path_is_transcript_v2_schema(self) -> None:
        """DEFAULT_SCHEMA_PATH should be the transcript-v2 schema."""
        assert "transcript-v2" in DEFAULT_SCHEMA_PATH.name

        schema = load_json(DEFAULT_SCHEMA_PATH)
        assert "schema_version" in str(schema.get("required", []))


# ============================================================================
# Integration tests
# ============================================================================


class TestValidationIntegration:
    """Integration tests combining multiple validation functions."""

    def test_full_validation_workflow(self, tmp_path: Path) -> None:
        """Test complete validation workflow with valid and invalid files."""
        # Create valid transcripts
        valid = {
            "schema_version": 2,
            "file": "audio.wav",
            "language": "en",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "Test"}],
        }
        valid_path = tmp_path / "valid.json"
        valid_path.write_text(json.dumps(valid), encoding="utf-8")

        # Create invalid transcript
        invalid = {
            "schema_version": 2,
            "language": "invalid-lang-code",  # Invalid format
            # Missing 'file' and 'segments'
        }
        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text(json.dumps(invalid), encoding="utf-8")

        # Validate individual files
        is_valid1, errors1 = validate_transcript_json(valid_path)
        is_valid2, errors2 = validate_transcript_json(invalid_path)

        assert is_valid1 is True
        assert errors1 == []
        assert is_valid2 is False
        assert len(errors2) >= 1

        # Validate batch
        failures = validate_many([valid_path, invalid_path])
        assert len(failures) >= 1
        assert all(str(invalid_path) in f for f in failures)

    def test_validation_with_audio_state_segment(self, tmp_path: Path) -> None:
        """Test validation of transcript with audio_state in segment."""
        transcript = {
            "schema_version": 2,
            "file": "enriched.wav",
            "language": "en",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                    "audio_state": {
                        "prosody": {
                            "pitch": {"level": "high", "mean_hz": 245.3},
                            "energy": {"level": "neutral"},
                            "rate": {"level": "neutral"},
                            "pauses": {"count": 0},
                        },
                        "rendering": "[audio: high pitch]",
                        "extraction_status": {
                            "prosody": "success",
                            "emotion_dimensional": "skipped",
                            "emotion_categorical": "skipped",
                            "errors": [],
                        },
                    },
                }
            ],
        }
        transcript_path = tmp_path / "enriched.json"
        transcript_path.write_text(json.dumps(transcript), encoding="utf-8")

        is_valid, errors = validate_transcript_json(transcript_path)

        assert is_valid is True
        assert errors == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
