"""Smart Formatting (ITN-lite) for spoken text normalization.

This module provides non-mutating text normalization for LLM context rendering,
converting spoken forms to written forms for improved readability.

Supported conversions:
- Currency: "two hundred dollars" → "$200.00"
- Dates: "january fifteenth" → "January 15"
- Times: "five pm" → "5:00 PM"
- Phone numbers: "five five five one two three four" → "(555) 123-4"
- Emails/URLs: spoken "at/dot" normalization

All conversions preserve the original text and return match information
for traceability and reversibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# Number word mappings
ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

MAGNITUDES = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000,
    "trillion": 1000000000000,
}

# Month names for date parsing
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# Ordinal suffixes
ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "twenty first": 21,
    "twenty-first": 21,
    "twenty second": 22,
    "twenty-second": 22,
    "twenty third": 23,
    "twenty-third": 23,
    "twenty fourth": 24,
    "twenty-fourth": 24,
    "twenty fifth": 25,
    "twenty-fifth": 25,
    "twenty sixth": 26,
    "twenty-sixth": 26,
    "twenty seventh": 27,
    "twenty-seventh": 27,
    "twenty eighth": 28,
    "twenty-eighth": 28,
    "twenty ninth": 29,
    "twenty-ninth": 29,
    "thirtieth": 30,
    "thirty first": 31,
    "thirty-first": 31,
}

FormatType = Literal["currency", "date", "time", "phone", "email", "url", "number"]


@dataclass(slots=True)
class FormatMatch:
    """A single format match within text.

    Attributes:
        start: Start character index in original text.
        end: End character index in original text.
        original: The original spoken text.
        formatted: The normalized written form.
        format_type: Type of formatting applied.
        confidence: Confidence score (0.0-1.0) for the match.
    """

    start: int
    end: int
    original: str
    formatted: str
    format_type: FormatType
    confidence: float = 1.0


@dataclass(slots=True)
class FormattedText:
    """Result of smart formatting operation.

    Attributes:
        original: The original input text (never modified).
        formatted: The text with all normalizations applied.
        matches: List of all format matches found.
    """

    original: str
    formatted: str
    matches: list[FormatMatch] = field(default_factory=list)

    def has_changes(self) -> bool:
        """Return True if any formatting was applied."""
        return len(self.matches) > 0


@dataclass(slots=True)
class SmartFormatterConfig:
    """Configuration for smart formatting behavior.

    Attributes:
        enable_currency: Format currency expressions.
        enable_dates: Format date expressions.
        enable_times: Format time expressions.
        enable_phones: Format phone number expressions.
        enable_emails: Format email address expressions.
        enable_urls: Format URL expressions.
        currency_symbol: Currency symbol to use (default: $).
        currency_decimal_places: Decimal places for currency (default: 2).
        time_format_24h: Use 24-hour time format.
    """

    enable_currency: bool = True
    enable_dates: bool = True
    enable_times: bool = True
    enable_phones: bool = True
    enable_emails: bool = True
    enable_urls: bool = True
    currency_symbol: str = "$"
    currency_decimal_places: int = 2
    time_format_24h: bool = False


def _words_to_number(words: list[str]) -> int | None:
    """Convert a list of number words to an integer.

    Args:
        words: List of number words (e.g., ["two", "hundred", "fifty"]).

    Returns:
        Integer value, or None if parsing fails.
    """
    if not words:
        return None

    result = 0
    current = 0

    for word in words:
        word_lower = word.lower()

        if word_lower in ONES:
            current += ONES[word_lower]
        elif word_lower in TENS:
            current += TENS[word_lower]
        elif word_lower == "hundred":
            if current == 0:
                current = 1
            current *= 100
        elif word_lower == "thousand":
            if current == 0:
                current = 1
            result += current * 1000
            current = 0
        elif word_lower == "million":
            if current == 0:
                current = 1
            result += current * 1000000
            current = 0
        elif word_lower == "billion":
            if current == 0:
                current = 1
            result += current * 1000000000
            current = 0
        elif word_lower == "and":
            continue
        else:
            return None

    return result + current


def _parse_number_phrase(text: str) -> tuple[int | None, int, int]:
    """Parse a number phrase from text.

    Args:
        text: Text to parse.

    Returns:
        Tuple of (value, start_index, end_index), or (None, 0, 0) if no match.
    """
    # Build pattern for number words
    number_words = list(ONES.keys()) + list(TENS.keys()) + list(MAGNITUDES.keys()) + ["and", "a"]
    pattern = r"\b(" + "|".join(number_words) + r")(?:\s+(?:" + "|".join(number_words) + r"))*\b"

    match = re.search(pattern, text.lower())
    if not match:
        return None, 0, 0

    phrase = match.group(0)
    words = phrase.split()

    # Handle "a hundred" -> "one hundred"
    if words and words[0] == "a":
        words[0] = "one"

    value = _words_to_number(words)
    if value is None:
        return None, 0, 0

    return value, match.start(), match.end()


class SmartFormatter:
    """Smart text formatter for spoken-to-written text normalization.

    This formatter applies various normalizations to convert spoken text
    to more readable written forms while preserving the original text
    and tracking all changes.

    Example:
        >>> formatter = SmartFormatter()
        >>> result = formatter.format("I paid two hundred dollars at five pm")
        >>> print(result.formatted)
        "I paid $200.00 at 5:00 PM"
        >>> print(result.original)
        "I paid two hundred dollars at five pm"
    """

    def __init__(self, config: SmartFormatterConfig | None = None):
        """Initialize formatter with optional configuration.

        Args:
            config: Formatter configuration. Uses defaults if not provided.
        """
        self.config = config or SmartFormatterConfig()

    def format(self, text: str) -> FormattedText:
        """Format text with all enabled normalizations.

        Args:
            text: Input text to format.

        Returns:
            FormattedText with original, formatted text, and match details.
        """
        matches: list[FormatMatch] = []

        if self.config.enable_currency:
            matches.extend(self._format_currency(text))

        if self.config.enable_dates:
            matches.extend(self._format_dates(text))

        if self.config.enable_times:
            matches.extend(self._format_times(text))

        if self.config.enable_phones:
            matches.extend(self._format_phones(text))

        if self.config.enable_emails:
            matches.extend(self._format_emails(text))

        if self.config.enable_urls:
            matches.extend(self._format_urls(text))

        # Sort matches by start position (descending) for replacement
        matches.sort(key=lambda m: m.start, reverse=True)

        # Remove overlapping matches (keep first/longest)
        filtered_matches: list[FormatMatch] = []
        for match in matches:
            overlaps = False
            for existing in filtered_matches:
                if not (match.end <= existing.start or match.start >= existing.end):
                    overlaps = True
                    break
            if not overlaps:
                filtered_matches.append(match)

        # Apply replacements
        formatted = text
        for match in filtered_matches:
            formatted = formatted[: match.start] + match.formatted + formatted[match.end :]

        # Sort matches by start position (ascending) for output
        filtered_matches.sort(key=lambda m: m.start)

        return FormattedText(original=text, formatted=formatted, matches=filtered_matches)

    def render(self, text: str) -> str:
        """Convenience method to get only the formatted text.

        Args:
            text: Input text to format.

        Returns:
            Formatted text string.
        """
        return self.format(text).formatted

    def _format_currency(self, text: str) -> list[FormatMatch]:
        """Find and format currency expressions.

        Patterns:
        - "two hundred dollars" → "$200.00"
        - "fifty cents" → "$0.50"
        - "twenty five dollars and fifty cents" → "$25.50"
        """
        matches: list[FormatMatch] = []

        # Pattern for dollars and optional cents
        # e.g., "two hundred dollars and fifty cents"
        dollar_pattern = (
            r"\b((?:(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
            r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
            r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
            r"million|billion|and|a)\s*)+)\s*dollars?"
            r"(?:\s+and\s+((?:(?:one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
            r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s*)+)"
            r"\s*cents?)?"
        )

        for match in re.finditer(dollar_pattern, text, re.IGNORECASE):
            dollar_words = match.group(1).strip().split()
            if dollar_words and dollar_words[0].lower() == "a":
                dollar_words[0] = "one"
            dollar_value = _words_to_number(dollar_words)

            cents_value = 0
            if match.group(2):
                cent_words = match.group(2).strip().split()
                cents_value = _words_to_number(cent_words) or 0

            if dollar_value is not None:
                total = dollar_value + cents_value / 100
                formatted = (
                    f"{self.config.currency_symbol}{total:.{self.config.currency_decimal_places}f}"
                )
                matches.append(
                    FormatMatch(
                        start=match.start(),
                        end=match.end(),
                        original=match.group(0),
                        formatted=formatted,
                        format_type="currency",
                    )
                )

        # Pattern for cents only
        cents_pattern = (
            r"\b((?:(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
            r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
            r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s*)+)\s*cents?\b"
        )

        for match in re.finditer(cents_pattern, text, re.IGNORECASE):
            # Skip if this is part of a dollars and cents match
            is_part_of_dollar = False
            for dm in matches:
                if match.start() >= dm.start and match.end() <= dm.end:
                    is_part_of_dollar = True
                    break
            if is_part_of_dollar:
                continue

            cent_words = match.group(1).strip().split()
            cents_val = _words_to_number(cent_words)
            if cents_val is not None:
                formatted = f"{self.config.currency_symbol}0.{cents_val:02d}"
                matches.append(
                    FormatMatch(
                        start=match.start(),
                        end=match.end(),
                        original=match.group(0),
                        formatted=formatted,
                        format_type="currency",
                    )
                )

        return matches

    def _format_dates(self, text: str) -> list[FormatMatch]:
        """Find and format date expressions.

        Patterns:
        - "january fifteenth" → "January 15"
        - "the fifteenth of january" → "January 15"
        - "january fifteen" → "January 15"
        """
        matches: list[FormatMatch] = []

        # Month + ordinal/number pattern
        month_names = "|".join(MONTHS.keys())
        ordinal_names = "|".join(ORDINALS.keys())
        number_names = "|".join(
            [k for k in ONES.keys() if ONES[k] <= 31]
            + [
                "twenty",
                "thirty",
                "twenty one",
                "twenty-one",
                "twenty two",
                "twenty-two",
                "twenty three",
                "twenty-three",
                "twenty four",
                "twenty-four",
                "twenty five",
                "twenty-five",
                "twenty six",
                "twenty-six",
                "twenty seven",
                "twenty-seven",
                "twenty eight",
                "twenty-eight",
                "twenty nine",
                "twenty-nine",
                "thirty one",
                "thirty-one",
            ]
        )

        # Pattern: "january fifteenth" or "january fifteen"
        pattern1 = rf"\b({month_names})\s+(?:the\s+)?({ordinal_names}|{number_names})\b"

        for match in re.finditer(pattern1, text, re.IGNORECASE):
            month_name = match.group(1).lower()
            day_name = match.group(2).lower().replace("-", " ")

            month_num = MONTHS.get(month_name)
            day_num = ORDINALS.get(day_name) or ONES.get(day_name)

            # Handle compound numbers like "twenty one"
            if day_num is None and " " in day_name:
                parts = day_name.split()
                if len(parts) == 2:
                    tens_val = TENS.get(parts[0], 0)
                    ones_val = ONES.get(parts[1], 0)
                    day_num = tens_val + ones_val

            if month_num and day_num and 1 <= day_num <= 31:
                formatted = f"{MONTH_NAMES[month_num - 1]} {day_num}"
                matches.append(
                    FormatMatch(
                        start=match.start(),
                        end=match.end(),
                        original=match.group(0),
                        formatted=formatted,
                        format_type="date",
                    )
                )

        # Pattern: "the fifteenth of january"
        pattern2 = rf"\b(?:the\s+)?({ordinal_names})\s+of\s+({month_names})\b"

        for match in re.finditer(pattern2, text, re.IGNORECASE):
            day_name = match.group(1).lower().replace("-", " ")
            month_name = match.group(2).lower()

            month_num = MONTHS.get(month_name)
            day_num = ORDINALS.get(day_name)

            if month_num and day_num and 1 <= day_num <= 31:
                formatted = f"{MONTH_NAMES[month_num - 1]} {day_num}"
                matches.append(
                    FormatMatch(
                        start=match.start(),
                        end=match.end(),
                        original=match.group(0),
                        formatted=formatted,
                        format_type="date",
                    )
                )

        return matches

    def _format_times(self, text: str) -> list[FormatMatch]:
        """Find and format time expressions.

        Patterns:
        - "five pm" → "5:00 PM"
        - "five thirty pm" → "5:30 PM"
        - "noon" → "12:00 PM"
        - "midnight" → "12:00 AM"
        """
        matches: list[FormatMatch] = []

        # Hour words (1-12)
        hour_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
        }

        # Minute words
        minute_words = {
            "oh one": 1,
            "oh two": 2,
            "oh three": 3,
            "oh four": 4,
            "oh five": 5,
            "oh six": 6,
            "oh seven": 7,
            "oh eight": 8,
            "oh nine": 9,
            **{k: v for k, v in ONES.items() if v <= 59},
            **{k: v for k, v in TENS.items() if v <= 59},
        }

        hour_pattern = "|".join(hour_words.keys())
        minute_pattern = "|".join(minute_words.keys())

        # Pattern: "five pm" or "five thirty pm" or "five oh five pm"
        time_pattern = (
            rf"\b({hour_pattern})"
            rf"(?:\s+({minute_pattern}|{minute_pattern}\s+{minute_pattern}))?"
            rf"\s*(a\.?m\.?|p\.?m\.?|am|pm|oclock|o'clock)\b"
        )

        for match in re.finditer(time_pattern, text, re.IGNORECASE):
            hour_word = match.group(1).lower()
            minute_word = match.group(2).lower() if match.group(2) else None
            period = match.group(3).lower().replace(".", "").replace("'", "")

            hour = hour_words.get(hour_word, 0)
            minute = 0

            if minute_word:
                minute_word = minute_word.strip()
                if minute_word in minute_words:
                    minute = minute_words[minute_word]
                else:
                    # Handle compound minutes like "thirty five"
                    parts = minute_word.split()
                    if len(parts) == 2:
                        tens_val = TENS.get(parts[0], minute_words.get(parts[0], 0))
                        ones_val = ONES.get(parts[1], 0)
                        minute = tens_val + ones_val

            if period in ("oclock", "o'clock"):
                # Assume context determines AM/PM, default to blank
                period_str = ""
            else:
                period_str = " AM" if "a" in period else " PM"

            if self.config.time_format_24h and period_str:
                if period_str == " PM" and hour != 12:
                    hour += 12
                elif period_str == " AM" and hour == 12:
                    hour = 0
                formatted = f"{hour:02d}:{minute:02d}"
            else:
                formatted = f"{hour}:{minute:02d}{period_str}"

            matches.append(
                FormatMatch(
                    start=match.start(),
                    end=match.end(),
                    original=match.group(0),
                    formatted=formatted,
                    format_type="time",
                )
            )

        # Special cases: noon and midnight
        for match in re.finditer(r"\bnoon\b", text, re.IGNORECASE):
            formatted = "12:00" if self.config.time_format_24h else "12:00 PM"
            matches.append(
                FormatMatch(
                    start=match.start(),
                    end=match.end(),
                    original=match.group(0),
                    formatted=formatted,
                    format_type="time",
                )
            )

        for match in re.finditer(r"\bmidnight\b", text, re.IGNORECASE):
            formatted = "00:00" if self.config.time_format_24h else "12:00 AM"
            matches.append(
                FormatMatch(
                    start=match.start(),
                    end=match.end(),
                    original=match.group(0),
                    formatted=formatted,
                    format_type="time",
                )
            )

        return matches

    def _format_phones(self, text: str) -> list[FormatMatch]:
        """Find and format phone number expressions.

        Patterns:
        - "five five five one two three four" → "(555) 123-4" (partial)
        - "five five five one two three four five six seven" → "(555) 123-4567"
        - "one eight hundred five five five one two three four" → "1-800-555-1234"
        """
        matches: list[FormatMatch] = []

        # Single digit words
        digit_words = {
            "zero": "0",
            "oh": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
        }

        digit_pattern = "|".join(digit_words.keys())

        # Pattern for 7+ consecutive digit words
        phone_pattern = rf"\b((?:{digit_pattern})(?:\s+(?:{digit_pattern})){{6,}})\b"

        for match in re.finditer(phone_pattern, text, re.IGNORECASE):
            words = match.group(1).lower().split()
            digits = "".join(digit_words.get(w, "") for w in words)

            if len(digits) >= 7:
                # Format based on length
                if len(digits) == 7:
                    # Local number: 123-4567
                    formatted = f"{digits[:3]}-{digits[3:]}"
                elif len(digits) == 10:
                    # Area code: (555) 123-4567
                    formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
                elif len(digits) == 11 and digits[0] == "1":
                    # Country code: 1-555-123-4567
                    formatted = f"1-{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
                else:
                    # Generic: just group in threes
                    parts = [digits[i : i + 3] for i in range(0, len(digits), 3)]
                    formatted = "-".join(parts)

                matches.append(
                    FormatMatch(
                        start=match.start(),
                        end=match.end(),
                        original=match.group(0),
                        formatted=formatted,
                        format_type="phone",
                    )
                )

        return matches

    def _format_emails(self, text: str) -> list[FormatMatch]:
        """Find and format email address expressions.

        Patterns:
        - "john at example dot com" → "john@example.com"
        - "john dot doe at example dot com" → "john.doe@example.com"
        """
        matches: list[FormatMatch] = []

        # Pattern for spoken email
        email_pattern = r"\b([\w]+(?:\s+dot\s+[\w]+)*)\s+at\s+([\w]+(?:\s+dot\s+[\w]+)+)\b"

        for match in re.finditer(email_pattern, text, re.IGNORECASE):
            local_part = match.group(1).lower()
            domain_part = match.group(2).lower()

            # Replace "dot" with "."
            local_part = re.sub(r"\s+dot\s+", ".", local_part)
            domain_part = re.sub(r"\s+dot\s+", ".", domain_part)

            formatted = f"{local_part}@{domain_part}"

            matches.append(
                FormatMatch(
                    start=match.start(),
                    end=match.end(),
                    original=match.group(0),
                    formatted=formatted,
                    format_type="email",
                )
            )

        return matches

    def _format_urls(self, text: str) -> list[FormatMatch]:
        """Find and format URL expressions.

        Patterns:
        - "w w w dot example dot com" → "www.example.com"
        - "h t t p s colon slash slash example dot com" → "https://example.com"
        """
        matches: list[FormatMatch] = []

        # Pattern for spoken www URLs
        www_pattern = r"\b(w\s+w\s+w|www)\s+dot\s+([\w]+(?:\s+dot\s+[\w]+)+)\b"

        for match in re.finditer(www_pattern, text, re.IGNORECASE):
            domain_part = match.group(2).lower()
            domain_part = re.sub(r"\s+dot\s+", ".", domain_part)
            formatted = f"www.{domain_part}"

            matches.append(
                FormatMatch(
                    start=match.start(),
                    end=match.end(),
                    original=match.group(0),
                    formatted=formatted,
                    format_type="url",
                )
            )

        # Pattern for spoken protocol URLs
        protocol_pattern = (
            r"\b(h\s*t\s*t\s*p\s*s?|https?)\s+colon\s+(?:slash\s+slash|forward\s+slash\s+forward\s+slash)"
            r"\s+([\w]+(?:\s+dot\s+[\w]+)+(?:\s+slash\s+[\w]+)*)\b"
        )

        for match in re.finditer(protocol_pattern, text, re.IGNORECASE):
            protocol = match.group(1).lower().replace(" ", "")
            path_part = match.group(2).lower()
            path_part = re.sub(r"\s+dot\s+", ".", path_part)
            path_part = re.sub(r"\s+slash\s+", "/", path_part)
            formatted = f"{protocol}://{path_part}"

            matches.append(
                FormatMatch(
                    start=match.start(),
                    end=match.end(),
                    original=match.group(0),
                    formatted=formatted,
                    format_type="url",
                )
            )

        return matches


# Export public API
__all__ = [
    "SmartFormatter",
    "SmartFormatterConfig",
    "FormattedText",
    "FormatMatch",
    "FormatType",
]
