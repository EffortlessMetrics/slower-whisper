"""
Tests for speaker_id.py module.

Test coverage:
- get_speaker_id() with all input types (None, str, dict, dataclass, object)
- get_speaker_label_or_id() with all input types
- Edge cases (empty strings, None values, missing keys)
- Label preference and fallback behavior
- Type coercion (int IDs, custom __str__)
- Dataclass type vs instance handling
"""

from __future__ import annotations

from dataclasses import dataclass

from transcription.speaker_id import (
    get_speaker_id,
    get_speaker_label_or_id,
)

# ============================================================================
# Tests for get_speaker_id()
# ============================================================================


class TestGetSpeakerIdNoneHandling:
    """Tests for get_speaker_id() with None input."""

    def test_none_returns_none(self):
        """None input should return None."""
        assert get_speaker_id(None) is None


class TestGetSpeakerIdStringHandling:
    """Tests for get_speaker_id() with string inputs."""

    def test_string_returns_string(self):
        """String input should return the same string."""
        assert get_speaker_id("spk_0") == "spk_0"

    def test_empty_string_returns_empty(self):
        """Empty string input should return empty string."""
        assert get_speaker_id("") == ""

    def test_whitespace_string_returns_whitespace(self):
        """Whitespace-only string should be returned as-is."""
        assert get_speaker_id("   ") == "   "

    def test_unicode_string_returns_unicode(self):
        """Unicode string should be handled correctly."""
        assert get_speaker_id("speaker_\u4e2d\u6587") == "speaker_\u4e2d\u6587"


class TestGetSpeakerIdDictHandling:
    """Tests for get_speaker_id() with dict inputs."""

    def test_dict_with_id_extracts_id(self):
        """Dict with 'id' key should extract the ID."""
        assert get_speaker_id({"id": "spk_1"}) == "spk_1"

    def test_dict_with_id_and_other_keys(self):
        """Dict with 'id' and other keys should extract only the ID."""
        result = get_speaker_id({"id": "spk_0", "confidence": 0.9, "label": "Alice"})
        assert result == "spk_0"

    def test_dict_without_id_returns_none(self):
        """Dict without 'id' key should return None."""
        assert get_speaker_id({"label": "Alice"}) is None

    def test_dict_with_id_none_returns_none(self):
        """Dict with id=None should return None."""
        assert get_speaker_id({"id": None}) is None

    def test_dict_with_int_id_converts_to_string(self):
        """Dict with integer ID should convert to string."""
        assert get_speaker_id({"id": 42}) == "42"

    def test_dict_with_float_id_converts_to_string(self):
        """Dict with float ID should convert to string."""
        assert get_speaker_id({"id": 3.14}) == "3.14"

    def test_dict_with_empty_string_id(self):
        """Dict with empty string ID should return empty string."""
        assert get_speaker_id({"id": ""}) == ""

    def test_dict_empty_returns_none(self):
        """Empty dict should return None."""
        assert get_speaker_id({}) is None


class TestGetSpeakerIdDataclassHandling:
    """Tests for get_speaker_id() with dataclass inputs."""

    def test_dataclass_with_speaker_id(self):
        """Dataclass with speaker_id attribute should extract it."""

        @dataclass
        class Speaker:
            speaker_id: str

        assert get_speaker_id(Speaker(speaker_id="spk_2")) == "spk_2"

    def test_dataclass_with_id(self):
        """Dataclass with 'id' attribute should extract it."""

        @dataclass
        class Speaker:
            id: str

        assert get_speaker_id(Speaker(id="spk_3")) == "spk_3"

    def test_dataclass_with_both_speaker_id_and_id(self):
        """Dataclass with both speaker_id and id should prefer speaker_id."""

        @dataclass
        class Speaker:
            speaker_id: str
            id: str

        # Implementation uses: data.get("speaker_id") or data.get("id")
        # So speaker_id is preferred
        result = get_speaker_id(Speaker(speaker_id="primary", id="secondary"))
        assert result == "primary"

    def test_dataclass_with_none_speaker_id_falls_back_to_id(self):
        """Dataclass with None speaker_id should fall back to id."""

        @dataclass
        class Speaker:
            speaker_id: str | None
            id: str

        result = get_speaker_id(Speaker(speaker_id=None, id="fallback_id"))
        assert result == "fallback_id"

    def test_dataclass_with_int_speaker_id(self):
        """Dataclass with integer speaker_id should convert to string."""

        @dataclass
        class Speaker:
            speaker_id: int

        assert get_speaker_id(Speaker(speaker_id=123)) == "123"

    def test_dataclass_without_speaker_id_or_id_returns_none(self):
        """Dataclass without speaker_id or id should return None."""

        @dataclass
        class OtherData:
            name: str
            confidence: float

        assert get_speaker_id(OtherData(name="Alice", confidence=0.9)) is None

    def test_dataclass_type_not_instance_falls_through(self):
        """Dataclass type (not instance) should fall through to str()."""

        @dataclass
        class Speaker:
            id: str

        # is_dataclass returns True for the class itself, but we check not isinstance(speaker, type)
        result = get_speaker_id(Speaker)
        assert result is not None
        # Falls through to str() conversion
        assert "Speaker" in result


class TestGetSpeakerIdObjectHandling:
    """Tests for get_speaker_id() with regular object inputs."""

    def test_object_with_speaker_id_attr(self):
        """Object with speaker_id attribute should extract it."""

        class Speaker:
            speaker_id = "spk_4"

        assert get_speaker_id(Speaker()) == "spk_4"

    def test_object_with_instance_speaker_id_attr(self):
        """Object with instance speaker_id should extract it."""

        class Speaker:
            def __init__(self):
                self.speaker_id = "spk_5"

        assert get_speaker_id(Speaker()) == "spk_5"

    def test_object_with_none_speaker_id_falls_through(self):
        """Object with speaker_id=None should fall through to str()."""

        class Speaker:
            speaker_id = None

        result = get_speaker_id(Speaker())
        # Falls through to str() because speaker_id is None
        assert "Speaker" in result

    def test_object_without_speaker_id(self):
        """Object without speaker_id should fall through to str()."""

        class Speaker:
            name = "Alice"

        result = get_speaker_id(Speaker())
        assert "Speaker" in result  # str(obj) contains class name

    def test_object_with_custom_str(self):
        """Object with custom __str__ should use it in fallback."""

        class CustomSpeaker:
            def __str__(self):
                return "custom_speaker_repr"

        assert get_speaker_id(CustomSpeaker()) == "custom_speaker_repr"


class TestGetSpeakerIdFallbackBehavior:
    """Tests for get_speaker_id() fallback to str() conversion."""

    def test_int_converts_to_string(self):
        """Integer input should convert to string."""
        assert get_speaker_id(123) == "123"

    def test_float_converts_to_string(self):
        """Float input should convert to string."""
        assert get_speaker_id(3.14) == "3.14"

    def test_list_converts_to_string(self):
        """List input should convert to string (fallback behavior)."""
        result = get_speaker_id(["spk_0", "spk_1"])
        assert result == "['spk_0', 'spk_1']"

    def test_tuple_converts_to_string(self):
        """Tuple input should convert to string (fallback behavior)."""
        result = get_speaker_id(("spk_0",))
        assert result == "('spk_0',)"

    def test_bool_converts_to_string(self):
        """Boolean input should convert to string."""
        assert get_speaker_id(True) == "True"
        assert get_speaker_id(False) == "False"


# ============================================================================
# Tests for get_speaker_label_or_id()
# ============================================================================


class TestGetSpeakerLabelOrIdNoneHandling:
    """Tests for get_speaker_label_or_id() with None input."""

    def test_none_returns_default_fallback(self):
        """None input should return default fallback 'unknown'."""
        assert get_speaker_label_or_id(None) == "unknown"

    def test_none_with_custom_fallback(self):
        """None input with custom fallback should return that fallback."""
        assert get_speaker_label_or_id(None, fallback="anon") == "anon"

    def test_none_with_empty_string_fallback(self):
        """None input with empty string fallback should return empty string."""
        assert get_speaker_label_or_id(None, fallback="") == ""


class TestGetSpeakerLabelOrIdStringHandling:
    """Tests for get_speaker_label_or_id() with string inputs."""

    def test_string_returns_string(self):
        """String input should return the same string."""
        assert get_speaker_label_or_id("spk_0") == "spk_0"

    def test_empty_string_returns_empty(self):
        """Empty string input should return empty string (not fallback)."""
        assert get_speaker_label_or_id("") == ""

    def test_whitespace_string_returns_whitespace(self):
        """Whitespace-only string should be returned as-is."""
        assert get_speaker_label_or_id("   ") == "   "


class TestGetSpeakerLabelOrIdDictHandling:
    """Tests for get_speaker_label_or_id() with dict inputs."""

    def test_dict_prefers_label_over_id(self):
        """Dict with both label and id should prefer label."""
        result = get_speaker_label_or_id({"id": "spk_0", "label": "Alice"})
        assert result == "Alice"

    def test_dict_with_only_id(self):
        """Dict with only id should return id."""
        assert get_speaker_label_or_id({"id": "spk_0"}) == "spk_0"

    def test_dict_with_only_label(self):
        """Dict with only label should return label."""
        assert get_speaker_label_or_id({"label": "Bob"}) == "Bob"

    def test_dict_without_label_or_id_returns_fallback(self):
        """Dict without label or id should return fallback."""
        assert get_speaker_label_or_id({"confidence": 0.9}) == "unknown"

    def test_dict_with_empty_label_uses_id(self):
        """Dict with empty string label should fall back to id."""
        result = get_speaker_label_or_id({"id": "spk_0", "label": ""})
        assert result == "spk_0"

    def test_dict_with_none_label_uses_id(self):
        """Dict with label=None should fall back to id."""
        result = get_speaker_label_or_id({"id": "spk_0", "label": None})
        assert result == "spk_0"

    def test_dict_with_none_id_returns_fallback(self):
        """Dict with id=None (and no label) should return fallback."""
        assert get_speaker_label_or_id({"id": None}) == "unknown"

    def test_dict_with_int_id_converts_to_string(self):
        """Dict with integer id should convert to string."""
        assert get_speaker_label_or_id({"id": 42}) == "42"

    def test_dict_with_int_label_converts_to_string(self):
        """Dict with integer label should convert to string."""
        assert get_speaker_label_or_id({"label": 123}) == "123"

    def test_dict_empty_returns_fallback(self):
        """Empty dict should return fallback."""
        assert get_speaker_label_or_id({}) == "unknown"

    def test_dict_with_custom_fallback(self):
        """Dict without id/label with custom fallback."""
        result = get_speaker_label_or_id({"confidence": 0.9}, fallback="anonymous")
        assert result == "anonymous"


class TestGetSpeakerLabelOrIdDataclassHandling:
    """Tests for get_speaker_label_or_id() with dataclass inputs."""

    def test_dataclass_with_id(self):
        """Dataclass with 'id' attribute should extract it."""

        @dataclass
        class Speaker:
            id: str

        assert get_speaker_label_or_id(Speaker(id="spk_1")) == "spk_1"

    def test_dataclass_with_speaker_id(self):
        """Dataclass with speaker_id attribute should extract it."""

        @dataclass
        class Speaker:
            speaker_id: str

        assert get_speaker_label_or_id(Speaker(speaker_id="spk_2")) == "spk_2"

    def test_dataclass_with_both_id_and_speaker_id(self):
        """Dataclass with both id and speaker_id should prefer id."""

        @dataclass
        class Speaker:
            id: str
            speaker_id: str

        # Implementation: data.get("id") or data.get("speaker_id")
        result = get_speaker_label_or_id(Speaker(id="primary", speaker_id="secondary"))
        assert result == "primary"

    def test_dataclass_with_none_id_falls_back_to_speaker_id(self):
        """Dataclass with None id should fall back to speaker_id."""

        @dataclass
        class Speaker:
            id: str | None
            speaker_id: str

        result = get_speaker_label_or_id(Speaker(id=None, speaker_id="fallback"))
        assert result == "fallback"

    def test_dataclass_without_id_or_speaker_id_returns_fallback(self):
        """Dataclass without id or speaker_id should return fallback."""

        @dataclass
        class OtherData:
            name: str

        assert get_speaker_label_or_id(OtherData(name="test")) == "unknown"

    def test_dataclass_type_not_instance_falls_through(self):
        """Dataclass type (not instance) should fall through to str()."""

        @dataclass
        class Speaker:
            id: str

        result = get_speaker_label_or_id(Speaker)
        assert "Speaker" in result


class TestGetSpeakerLabelOrIdObjectHandling:
    """Tests for get_speaker_label_or_id() with regular object inputs."""

    def test_object_with_speaker_id_attr(self):
        """Object with speaker_id attribute should extract it."""

        class Speaker:
            speaker_id = "spk_3"

        assert get_speaker_label_or_id(Speaker()) == "spk_3"

    def test_object_with_instance_speaker_id_attr(self):
        """Object with instance speaker_id should extract it."""

        class Speaker:
            def __init__(self):
                self.speaker_id = "spk_4"

        assert get_speaker_label_or_id(Speaker()) == "spk_4"

    def test_object_without_speaker_id(self):
        """Object without speaker_id should fall through to str()."""

        class Speaker:
            name = "Alice"

        result = get_speaker_label_or_id(Speaker())
        assert "Speaker" in result

    def test_object_with_none_speaker_id_falls_through(self):
        """Object with speaker_id=None should fall through to str()."""

        class Speaker:
            speaker_id = None

        result = get_speaker_label_or_id(Speaker())
        # Falls through to str() because speaker_id is None
        assert "Speaker" in result


class TestGetSpeakerLabelOrIdFallbackBehavior:
    """Tests for get_speaker_label_or_id() fallback and str() conversion."""

    def test_arbitrary_object_converts_to_string(self):
        """Object without speaker attributes should convert via str()."""

        class Custom:
            def __str__(self):
                return "custom_speaker"

        assert get_speaker_label_or_id(Custom()) == "custom_speaker"

    def test_int_converts_to_string(self):
        """Integer input should convert to string."""
        assert get_speaker_label_or_id(123) == "123"

    def test_float_converts_to_string(self):
        """Float input should convert to string."""
        assert get_speaker_label_or_id(3.14) == "3.14"

    def test_list_converts_to_string(self):
        """List input should convert to string (fallback behavior)."""
        result = get_speaker_label_or_id(["spk_0", "spk_1"])
        assert result == "['spk_0', 'spk_1']"


# ============================================================================
# Edge case and integration tests
# ============================================================================


class TestEdgeCases:
    """Edge cases and boundary condition tests."""

    def test_nested_dict_id(self):
        """Dict with nested dict as id should convert to string."""
        result = get_speaker_id({"id": {"nested": "value"}})
        assert result == "{'nested': 'value'}"

    def test_deeply_nested_speaker_object(self):
        """Deeply nested object structures should be handled."""

        @dataclass
        class Inner:
            speaker_id: str

        @dataclass
        class Outer:
            inner: Inner

        # Outer has no speaker_id, so returns None from dataclass handling
        result = get_speaker_id(Outer(inner=Inner(speaker_id="deep")))
        assert result is None

    def test_special_characters_in_id(self):
        """Special characters in IDs should be preserved."""
        special_id = "spk_!@#$%^&*()"
        assert get_speaker_id(special_id) == special_id
        assert get_speaker_id({"id": special_id}) == special_id

    def test_very_long_id(self):
        """Very long IDs should be handled."""
        long_id = "spk_" + "x" * 10000
        assert get_speaker_id(long_id) == long_id

    def test_numeric_string_id(self):
        """Numeric strings should remain as strings."""
        assert get_speaker_id("12345") == "12345"
        assert get_speaker_id({"id": "12345"}) == "12345"

    def test_zero_id(self):
        """Zero as ID should convert to string '0', not be treated as falsy."""
        assert get_speaker_id({"id": 0}) == "0"
        assert get_speaker_label_or_id({"id": 0}) == "0"

    def test_false_id(self):
        """False as ID should convert to string 'False', not be treated as falsy."""
        # Note: In the implementation, False will be converted via str()
        # since bool is not None
        assert get_speaker_id({"id": False}) == "False"

    def test_empty_string_label_not_treated_as_truthy(self):
        """Empty string label should not be used (falsy check)."""
        # label="" is falsy, so should fall through to id
        result = get_speaker_label_or_id({"id": "spk_0", "label": ""})
        assert result == "spk_0"


class TestConsistencyBetweenFunctions:
    """Tests ensuring consistency between get_speaker_id and get_speaker_label_or_id."""

    def test_string_input_same_for_both(self):
        """Both functions should return the same result for string input."""
        test_id = "spk_test"
        assert get_speaker_id(test_id) == get_speaker_label_or_id(test_id)

    def test_dict_with_only_id_same_for_both(self):
        """Dict with only id should give same result for both functions."""
        test_dict = {"id": "spk_0"}
        assert get_speaker_id(test_dict) == get_speaker_label_or_id(test_dict)

    def test_label_or_id_never_returns_none_with_valid_input(self):
        """get_speaker_label_or_id should always return a string (with fallback)."""
        # Unlike get_speaker_id which returns None, get_speaker_label_or_id has fallback
        assert get_speaker_label_or_id({"confidence": 0.9}) == "unknown"
        assert get_speaker_id({"confidence": 0.9}) is None
