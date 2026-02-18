"""Tests for content moderation module."""

from __future__ import annotations

import pytest

from slower_whisper.pipeline.content_moderation import (
    ContentMatch,
    ContentPattern,
    LexiconModerator,
    LexiconModeratorConfig,
    create_moderator_for_call_center,
    create_moderator_for_healthcare,
)


class TestLexiconModerator:
    """Tests for LexiconModerator class."""

    def test_clean_text(self):
        """Clean text is not flagged."""
        moderator = LexiconModerator()
        result = moderator.check("Hello, how can I help you today?")
        assert not result.is_flagged
        assert result.highest_severity is None
        assert result.matches == []
        assert result.suggested_action == "allow"

    def test_empty_text(self):
        """Empty text is not flagged."""
        moderator = LexiconModerator()
        result = moderator.check("")
        assert not result.is_flagged
        assert result.highest_severity is None
        assert result.matches == []

    def test_low_severity_detected(self):
        """Low severity content is detected."""
        moderator = LexiconModerator()
        result = moderator.check("What the heck is going on?")
        assert result.is_flagged
        assert result.highest_severity == "low"
        assert len(result.matches) == 1
        assert result.matches[0].matched_text.lower() == "heck"
        assert result.suggested_action == "allow"

    def test_medium_severity_detected(self):
        """Medium severity content is detected."""
        moderator = LexiconModerator()
        result = moderator.check("This is total bullshit")
        assert result.is_flagged
        assert result.highest_severity == "medium"
        assert result.suggested_action == "warn"

    def test_high_severity_detected(self):
        """High severity content is detected."""
        moderator = LexiconModerator()
        result = moderator.check("What the fuck")
        assert result.is_flagged
        assert result.highest_severity == "high"
        assert result.suggested_action == "warn"

    def test_highest_severity_wins(self):
        """When multiple severities present, highest is reported."""
        moderator = LexiconModerator()
        result = moderator.check("Damn, this is fucking ridiculous")
        assert result.is_flagged
        assert result.highest_severity == "high"
        assert len(result.matches) >= 2

    def test_case_insensitive(self):
        """Matching is case insensitive."""
        moderator = LexiconModerator()
        result1 = moderator.check("DAMN")
        result2 = moderator.check("Damn")
        result3 = moderator.check("damn")
        assert result1.is_flagged
        assert result2.is_flagged
        assert result3.is_flagged

    def test_word_boundaries(self):
        """Patterns respect word boundaries."""
        moderator = LexiconModerator()
        # "dam" should not match "damn" pattern
        result = moderator.check("The dam is breaking")
        # Should not match "dam" as a standalone word
        assert not any(m.matched_text.lower() == "dam" for m in result.matches)


class TestModerationResult:
    """Tests for ModerationResult dataclass."""

    def test_get_matches_by_severity(self):
        """Can filter matches by severity."""
        moderator = LexiconModerator()
        result = moderator.check("Damn, this is such crap")
        low_matches = result.get_matches_by_severity("low")
        assert len(low_matches) >= 1
        for match in low_matches:
            assert match.severity == "low"

    def test_get_matches_by_category(self):
        """Can filter matches by category."""
        moderator = LexiconModerator()
        result = moderator.check("What the heck")
        mild_matches = result.get_matches_by_category("mild_language")
        assert len(mild_matches) >= 1

    def test_categories_found(self):
        """Categories are tracked correctly."""
        moderator = LexiconModerator()
        result = moderator.check("Damn, this is shit")
        assert "mild_language" in result.categories_found or "profanity" in result.categories_found


class TestMasking:
    """Tests for content masking."""

    def test_mask_preserves_first(self):
        """Default masking preserves first character."""
        moderator = LexiconModerator()
        result = moderator.check("What the fuck")
        masked = moderator.mask("What the fuck", result.matches)
        assert "f***" in masked or "f**" in masked

    def test_mask_full_word(self):
        """Can mask entire word."""
        config = LexiconModeratorConfig(
            mask_preserve_first=False,
            mask_preserve_last=False,
        )
        moderator = LexiconModerator(config)
        result = moderator.check("What the fuck")
        masked = moderator.mask("What the fuck", result.matches)
        assert "****" in masked

    def test_mask_preserve_both(self):
        """Can preserve first and last."""
        config = LexiconModeratorConfig(
            mask_preserve_first=True,
            mask_preserve_last=True,
        )
        moderator = LexiconModerator(config)
        result = moderator.check("What the fuck")
        masked = moderator.mask("What the fuck", result.matches)
        assert "f**k" in masked

    def test_custom_mask_character(self):
        """Can use custom mask character."""
        config = LexiconModeratorConfig(mask_character="#")
        moderator = LexiconModerator(config)
        result = moderator.check("What the fuck")
        masked = moderator.mask("What the fuck", result.matches)
        assert "#" in masked
        assert "*" not in masked

    def test_mask_empty_matches(self):
        """Masking with no matches returns original."""
        moderator = LexiconModerator()
        result = moderator.mask("Hello world", [])
        assert result == "Hello world"

    def test_check_and_mask(self):
        """check_and_mask convenience method works."""
        moderator = LexiconModerator()
        result, masked = moderator.check_and_mask("This is shit")
        assert result.is_flagged
        assert "s***" in masked or "s**" in masked


class TestConfiguration:
    """Tests for LexiconModeratorConfig."""

    def test_default_config(self):
        """Default configuration is valid."""
        config = LexiconModeratorConfig()
        assert config.mask_character == "*"
        assert config.mask_preserve_first is True
        assert config.mask_preserve_last is False
        assert config.enable_mild is True
        assert config.enable_profanity is True
        assert config.enable_slurs is True

    def test_invalid_mask_character(self):
        """Invalid mask character raises error."""
        with pytest.raises(ValueError, match="single character"):
            LexiconModeratorConfig(mask_character="**")

    def test_disable_mild(self):
        """Can disable mild language detection."""
        config = LexiconModeratorConfig(enable_mild=False)
        moderator = LexiconModerator(config)
        result = moderator.check("What the heck")
        # "heck" is mild_language, should not be detected
        heck_matches = [m for m in result.matches if m.matched_text.lower() == "heck"]
        assert len(heck_matches) == 0

    def test_disable_profanity(self):
        """Can disable profanity detection."""
        config = LexiconModeratorConfig(enable_profanity=False)
        moderator = LexiconModerator(config)
        result = moderator.check("This is total shit")
        # "shit" is profanity, should not be detected
        shit_matches = [m for m in result.matches if m.matched_text.lower() == "shit"]
        assert len(shit_matches) == 0

    def test_custom_severity_action(self):
        """Can customize severity actions."""
        config = LexiconModeratorConfig(
            severity_actions={
                "low": "warn",
                "medium": "block",
                "high": "block",
                "critical": "block",
            }
        )
        moderator = LexiconModerator(config)
        result = moderator.check("What the heck")
        assert result.suggested_action == "warn"


class TestCustomPatterns:
    """Tests for custom pattern support."""

    def test_add_pattern(self):
        """Can add custom patterns."""
        moderator = LexiconModerator()
        moderator.add_pattern("foobar", "high", "custom")
        result = moderator.check("This is foobar content")
        assert result.is_flagged
        assert any(m.matched_text.lower() == "foobar" for m in result.matches)
        assert "custom" in result.categories_found

    def test_add_patterns_bulk(self):
        """Can add multiple patterns at once."""
        moderator = LexiconModerator()
        moderator.add_patterns(
            [
                ContentPattern("alpha", "low", "greek"),
                ContentPattern("beta", "medium", "greek"),
            ]
        )
        result = moderator.check("alpha and beta testing")
        assert result.is_flagged
        assert len(result.matches) == 2

    def test_custom_pattern_in_config(self):
        """Custom patterns can be set in config."""
        config = LexiconModeratorConfig(
            custom_patterns=[
                ContentPattern("foobar", "high", "custom"),
            ]
        )
        moderator = LexiconModerator(config)
        result = moderator.check("This is foobar")
        assert result.is_flagged

    def test_set_severity_action(self):
        """Can modify severity actions at runtime."""
        moderator = LexiconModerator()
        moderator.set_severity_action("low", "block")
        result = moderator.check("What the heck")
        assert result.suggested_action == "block"


class TestContentPattern:
    """Tests for ContentPattern dataclass."""

    def test_pattern_compilation(self):
        """Patterns are compiled with word boundaries."""
        pattern = ContentPattern("test", "low", "custom")
        assert pattern.compiled is not None
        # Should match "test" but not "testing"
        assert pattern.compiled.search("this is a test")
        assert not pattern.compiled.search("testing")

    def test_pattern_attributes(self):
        """Pattern has correct attributes."""
        pattern = ContentPattern("word", "high", "category")
        assert pattern.pattern == "word"
        assert pattern.severity == "high"
        assert pattern.category == "category"


class TestContentMatch:
    """Tests for ContentMatch dataclass."""

    def test_match_attributes(self):
        """ContentMatch has correct attributes."""
        match = ContentMatch(
            start=10,
            end=14,
            matched_text="test",
            pattern=r"test",
            severity="low",
            category="custom",
        )
        assert match.start == 10
        assert match.end == 14
        assert match.matched_text == "test"
        assert match.pattern == r"test"
        assert match.severity == "low"
        assert match.category == "custom"


class TestPresetModerators:
    """Tests for preset moderator factories."""

    def test_call_center_moderator(self):
        """Call center moderator has expected patterns."""
        moderator = create_moderator_for_call_center()

        # Standard profanity should work
        result = moderator.check("This is bullshit")
        assert result.is_flagged

        # Escalation signals should be detected
        result = moderator.check("I want to speak to a manager")
        assert result.is_flagged
        assert "escalation_signal" in result.categories_found

        # Legal threats should be detected
        result = moderator.check("I'm going to sue you")
        assert result.is_flagged
        assert "legal_threat" in result.categories_found

    def test_healthcare_moderator(self):
        """Healthcare moderator has expected patterns."""
        moderator = create_moderator_for_healthcare()

        # Standard profanity should work
        result = moderator.check("This is bullshit")
        assert result.is_flagged

        # Self-harm indicators should be critical
        result = moderator.check("I want to hurt myself")
        assert result.is_flagged
        assert result.highest_severity == "critical"
        assert "self_harm" in result.categories_found


class TestEdgeCases:
    """Tests for edge cases."""

    def test_multiple_matches_same_word(self):
        """Multiple occurrences of same word are all matched."""
        moderator = LexiconModerator()
        result = moderator.check("damn damn damn")
        assert len(result.matches) == 3

    def test_overlapping_patterns_handled(self):
        """Overlapping patterns don't cause issues."""
        moderator = LexiconModerator()
        # "asshole" and "ass" - asshole should match
        result = moderator.check("You asshole")
        assert result.is_flagged
        # At least one match for asshole
        assert any("asshole" in m.matched_text.lower() for m in result.matches)

    def test_special_characters_in_text(self):
        """Special characters don't break matching."""
        moderator = LexiconModerator()
        result = moderator.check("What the fuck! Really?!?")
        assert result.is_flagged

    def test_unicode_text(self):
        """Unicode text is handled correctly."""
        moderator = LexiconModerator()
        result = moderator.check("This is fine \u2764")
        assert not result.is_flagged

    def test_very_long_text(self):
        """Very long text is handled efficiently."""
        moderator = LexiconModerator()
        long_text = "This is clean content. " * 1000
        result = moderator.check(long_text)
        assert not result.is_flagged

    def test_matches_are_sorted(self):
        """Matches are sorted by position."""
        moderator = LexiconModerator()
        result = moderator.check("damn at start, shit in middle, crap at end")
        positions = [m.start for m in result.matches]
        assert positions == sorted(positions)
