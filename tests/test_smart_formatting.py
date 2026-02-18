"""Tests for smart formatting (ITN-lite) module."""

from __future__ import annotations

from slower_whisper.pipeline.smart_formatting import (
    FormatMatch,
    FormattedText,
    SmartFormatter,
    SmartFormatterConfig,
)


class TestSmartFormatter:
    """Tests for SmartFormatter class."""

    def test_empty_text(self):
        """Empty text returns unchanged."""
        formatter = SmartFormatter()
        result = formatter.format("")
        assert result.original == ""
        assert result.formatted == ""
        assert result.matches == []
        assert not result.has_changes()

    def test_no_changes_needed(self):
        """Text without patterns returns unchanged."""
        formatter = SmartFormatter()
        result = formatter.format("Hello world")
        assert result.original == "Hello world"
        assert result.formatted == "Hello world"
        assert result.matches == []
        assert not result.has_changes()

    def test_render_convenience(self):
        """render() returns only the formatted string."""
        formatter = SmartFormatter()
        result = formatter.render("five pm")
        assert result == "5:00 PM"


class TestCurrencyFormatting:
    """Tests for currency formatting."""

    def test_simple_dollars(self):
        """Simple dollar amounts."""
        formatter = SmartFormatter()
        result = formatter.format("I paid twenty dollars")
        assert result.formatted == "I paid $20.00"
        assert len(result.matches) == 1
        assert result.matches[0].format_type == "currency"

    def test_hundred_dollars(self):
        """Hundred dollar amounts."""
        formatter = SmartFormatter()
        result = formatter.format("two hundred dollars")
        assert result.formatted == "$200.00"

    def test_thousands_of_dollars(self):
        """Thousand dollar amounts."""
        formatter = SmartFormatter()
        result = formatter.format("five thousand dollars")
        assert result.formatted == "$5000.00"

    def test_dollars_and_cents(self):
        """Dollars and cents together."""
        formatter = SmartFormatter()
        result = formatter.format("twenty five dollars and fifty cents")
        assert result.formatted == "$25.50"

    def test_cents_only(self):
        """Cents only."""
        formatter = SmartFormatter()
        result = formatter.format("fifty cents")
        assert result.formatted == "$0.50"

    def test_custom_currency_symbol(self):
        """Custom currency symbol."""
        config = SmartFormatterConfig(currency_symbol="€")
        formatter = SmartFormatter(config)
        result = formatter.format("fifty dollars")
        assert result.formatted == "€50.00"

    def test_zero_decimal_places(self):
        """Zero decimal places for currency."""
        config = SmartFormatterConfig(currency_decimal_places=0)
        formatter = SmartFormatter(config)
        result = formatter.format("fifty dollars")
        assert result.formatted == "$50"

    def test_disabled_currency(self):
        """Currency formatting can be disabled."""
        config = SmartFormatterConfig(enable_currency=False)
        formatter = SmartFormatter(config)
        result = formatter.format("fifty dollars")
        assert result.formatted == "fifty dollars"
        assert not result.has_changes()


class TestDateFormatting:
    """Tests for date formatting."""

    def test_month_ordinal(self):
        """Month followed by ordinal."""
        formatter = SmartFormatter()
        result = formatter.format("january fifteenth")
        assert result.formatted == "January 15"
        assert result.matches[0].format_type == "date"

    def test_month_cardinal(self):
        """Month followed by cardinal number."""
        formatter = SmartFormatter()
        result = formatter.format("january fifteen")
        assert result.formatted == "January 15"

    def test_ordinal_of_month(self):
        """Ordinal of month pattern."""
        formatter = SmartFormatter()
        result = formatter.format("the fifteenth of january")
        assert result.formatted == "January 15"

    def test_compound_day(self):
        """Compound day number like twenty first."""
        formatter = SmartFormatter()
        result = formatter.format("march twenty first")
        assert result.formatted == "March 21"

    def test_hyphenated_ordinal(self):
        """Hyphenated ordinal like twenty-first."""
        formatter = SmartFormatter()
        result = formatter.format("march twenty-first")
        assert result.formatted == "March 21"

    def test_all_months(self):
        """All months are recognized."""
        formatter = SmartFormatter()
        months = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
        for month in months:
            result = formatter.format(f"{month} first")
            assert month.capitalize() in result.formatted

    def test_disabled_dates(self):
        """Date formatting can be disabled."""
        config = SmartFormatterConfig(enable_dates=False)
        formatter = SmartFormatter(config)
        result = formatter.format("january fifteenth")
        assert result.formatted == "january fifteenth"


class TestTimeFormatting:
    """Tests for time formatting."""

    def test_simple_time_pm(self):
        """Simple time with PM."""
        formatter = SmartFormatter()
        result = formatter.format("five pm")
        assert result.formatted == "5:00 PM"
        assert result.matches[0].format_type == "time"

    def test_simple_time_am(self):
        """Simple time with AM."""
        formatter = SmartFormatter()
        result = formatter.format("seven am")
        assert result.formatted == "7:00 AM"

    def test_time_with_minutes(self):
        """Time with minutes."""
        formatter = SmartFormatter()
        result = formatter.format("five thirty pm")
        assert result.formatted == "5:30 PM"

    def test_noon(self):
        """Noon keyword."""
        formatter = SmartFormatter()
        result = formatter.format("meet at noon")
        assert result.formatted == "meet at 12:00 PM"

    def test_midnight(self):
        """Midnight keyword."""
        formatter = SmartFormatter()
        result = formatter.format("at midnight")
        assert result.formatted == "at 12:00 AM"

    def test_24_hour_format(self):
        """24-hour time format."""
        config = SmartFormatterConfig(time_format_24h=True)
        formatter = SmartFormatter(config)
        result = formatter.format("five pm")
        assert result.formatted == "17:00"

    def test_24_hour_noon(self):
        """Noon in 24-hour format."""
        config = SmartFormatterConfig(time_format_24h=True)
        formatter = SmartFormatter(config)
        result = formatter.format("noon")
        assert result.formatted == "12:00"

    def test_24_hour_midnight(self):
        """Midnight in 24-hour format."""
        config = SmartFormatterConfig(time_format_24h=True)
        formatter = SmartFormatter(config)
        result = formatter.format("midnight")
        assert result.formatted == "00:00"

    def test_disabled_times(self):
        """Time formatting can be disabled."""
        config = SmartFormatterConfig(enable_times=False)
        formatter = SmartFormatter(config)
        result = formatter.format("five pm")
        assert result.formatted == "five pm"


class TestPhoneFormatting:
    """Tests for phone number formatting."""

    def test_seven_digit_phone(self):
        """Seven digit phone number (local)."""
        formatter = SmartFormatter()
        result = formatter.format("call five five five one two three four")
        assert result.formatted == "call 555-1234"
        assert result.matches[0].format_type == "phone"

    def test_ten_digit_phone(self):
        """Ten digit phone number with area code."""
        formatter = SmartFormatter()
        result = formatter.format("five five five one two three four five six seven")
        assert result.formatted == "(555) 123-4567"

    def test_eleven_digit_phone(self):
        """Eleven digit phone number with country code."""
        formatter = SmartFormatter()
        result = formatter.format("one eight zero zero five five five one two three four")
        assert result.formatted == "1-800-555-1234"

    def test_oh_for_zero(self):
        """'oh' is recognized as zero."""
        formatter = SmartFormatter()
        result = formatter.format("five oh five five five five one two three four")
        assert "505" in result.formatted

    def test_disabled_phones(self):
        """Phone formatting can be disabled."""
        config = SmartFormatterConfig(enable_phones=False)
        formatter = SmartFormatter(config)
        result = formatter.format("five five five one two three four")
        assert result.formatted == "five five five one two three four"


class TestEmailFormatting:
    """Tests for email formatting."""

    def test_simple_email(self):
        """Simple email address."""
        formatter = SmartFormatter()
        result = formatter.format("email me at john at example dot com")
        assert result.formatted == "email me at john@example.com"
        assert result.matches[0].format_type == "email"

    def test_email_with_dot_local(self):
        """Email with dot in local part."""
        formatter = SmartFormatter()
        result = formatter.format("john dot doe at example dot com")
        assert result.formatted == "john.doe@example.com"

    def test_email_with_subdomain(self):
        """Email with subdomain."""
        formatter = SmartFormatter()
        result = formatter.format("john at mail dot example dot com")
        assert result.formatted == "john@mail.example.com"

    def test_disabled_emails(self):
        """Email formatting can be disabled."""
        config = SmartFormatterConfig(enable_emails=False)
        formatter = SmartFormatter(config)
        result = formatter.format("john at example dot com")
        assert result.formatted == "john at example dot com"


class TestUrlFormatting:
    """Tests for URL formatting."""

    def test_www_url(self):
        """WWW URL."""
        formatter = SmartFormatter()
        result = formatter.format("go to w w w dot example dot com")
        assert result.formatted == "go to www.example.com"
        assert result.matches[0].format_type == "url"

    def test_disabled_urls(self):
        """URL formatting can be disabled."""
        config = SmartFormatterConfig(enable_urls=False)
        formatter = SmartFormatter(config)
        result = formatter.format("w w w dot example dot com")
        assert result.formatted == "w w w dot example dot com"


class TestMultipleMatches:
    """Tests for multiple matches in same text."""

    def test_multiple_currencies(self):
        """Multiple currency amounts."""
        formatter = SmartFormatter()
        result = formatter.format("from fifty dollars to one hundred dollars")
        assert "$50.00" in result.formatted
        assert "$100.00" in result.formatted
        assert len(result.matches) == 2

    def test_mixed_formats(self):
        """Mixed format types."""
        formatter = SmartFormatter()
        result = formatter.format("meeting at five pm costs twenty dollars")
        assert "5:00 PM" in result.formatted
        assert "$20.00" in result.formatted
        assert len(result.matches) == 2

    def test_overlapping_matches_handled(self):
        """Overlapping matches should not cause issues."""
        formatter = SmartFormatter()
        # This should not crash even if patterns might overlap
        result = formatter.format("five five five five five five five")
        assert result is not None


class TestFormatMatch:
    """Tests for FormatMatch dataclass."""

    def test_match_attributes(self):
        """FormatMatch has correct attributes."""
        match = FormatMatch(
            start=0,
            end=10,
            original="five pm",
            formatted="5:00 PM",
            format_type="time",
            confidence=0.95,
        )
        assert match.start == 0
        assert match.end == 10
        assert match.original == "five pm"
        assert match.formatted == "5:00 PM"
        assert match.format_type == "time"
        assert match.confidence == 0.95

    def test_default_confidence(self):
        """Default confidence is 1.0."""
        match = FormatMatch(
            start=0,
            end=10,
            original="test",
            formatted="TEST",
            format_type="currency",
        )
        assert match.confidence == 1.0


class TestFormattedText:
    """Tests for FormattedText dataclass."""

    def test_has_changes_true(self):
        """has_changes returns True when matches exist."""
        result = FormattedText(
            original="five pm",
            formatted="5:00 PM",
            matches=[
                FormatMatch(
                    start=0, end=7, original="five pm", formatted="5:00 PM", format_type="time"
                )
            ],
        )
        assert result.has_changes() is True

    def test_has_changes_false(self):
        """has_changes returns False when no matches."""
        result = FormattedText(
            original="hello",
            formatted="hello",
            matches=[],
        )
        assert result.has_changes() is False


class TestSmartFormatterConfig:
    """Tests for SmartFormatterConfig."""

    def test_default_config(self):
        """Default configuration values."""
        config = SmartFormatterConfig()
        assert config.enable_currency is True
        assert config.enable_dates is True
        assert config.enable_times is True
        assert config.enable_phones is True
        assert config.enable_emails is True
        assert config.enable_urls is True
        assert config.currency_symbol == "$"
        assert config.currency_decimal_places == 2
        assert config.time_format_24h is False

    def test_custom_config(self):
        """Custom configuration values."""
        config = SmartFormatterConfig(
            enable_currency=False,
            currency_symbol="£",
            time_format_24h=True,
        )
        assert config.enable_currency is False
        assert config.currency_symbol == "£"
        assert config.time_format_24h is True


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_case_insensitive(self):
        """Formatting is case insensitive."""
        formatter = SmartFormatter()
        result1 = formatter.format("FIVE PM")
        result2 = formatter.format("Five Pm")
        result3 = formatter.format("five pm")
        # All should produce formatted time
        assert "PM" in result1.formatted or "pm" in result1.formatted.lower()
        assert "PM" in result2.formatted or "pm" in result2.formatted.lower()
        assert "PM" in result3.formatted or "pm" in result3.formatted.lower()

    def test_preserves_surrounding_text(self):
        """Surrounding text is preserved."""
        formatter = SmartFormatter()
        result = formatter.format("The meeting is at five pm tomorrow")
        assert result.formatted.startswith("The meeting is at ")
        assert result.formatted.endswith(" tomorrow")

    def test_preserves_original(self):
        """Original text is always preserved."""
        formatter = SmartFormatter()
        original = "pay fifty dollars at five pm"
        result = formatter.format(original)
        assert result.original == original
        assert result.formatted != original  # Should have changes
