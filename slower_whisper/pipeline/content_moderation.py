"""Content moderation for spoken text.

This module provides lexicon-based profanity and toxicity filtering,
with configurable word lists and severity levels.

Features:
- Configurable word lists (default English profanity)
- Severity levels: low/medium/high/critical
- Actions: allow/warn/mask/block
- Custom pattern support for domain-specific terms
- Non-destructive: original text always preserved
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

SeverityLevel = Literal["low", "medium", "high", "critical"]
ModerationAction = Literal["allow", "warn", "mask", "block"]


@dataclass(slots=True)
class ContentMatch:
    """A content moderation match within text.

    Attributes:
        start: Start character index in original text.
        end: End character index in original text.
        matched_text: The actual text that matched.
        pattern: The pattern that triggered the match.
        severity: Severity level of the match.
        category: Category of the match (e.g., "profanity", "slur", "threat").
    """

    start: int
    end: int
    matched_text: str
    pattern: str
    severity: SeverityLevel
    category: str


@dataclass(slots=True)
class ModerationResult:
    """Result of content moderation check.

    Attributes:
        original: The original input text (never modified).
        is_flagged: Whether any content was flagged.
        highest_severity: The highest severity level found, or None if clean.
        matches: List of all content matches found.
        suggested_action: Recommended action based on highest severity.
        categories_found: Set of unique categories detected.
    """

    original: str
    is_flagged: bool
    highest_severity: SeverityLevel | None
    matches: list[ContentMatch]
    suggested_action: ModerationAction
    categories_found: set[str] = field(default_factory=set)

    def get_matches_by_severity(self, severity: SeverityLevel) -> list[ContentMatch]:
        """Get all matches of a specific severity level."""
        return [m for m in self.matches if m.severity == severity]

    def get_matches_by_category(self, category: str) -> list[ContentMatch]:
        """Get all matches in a specific category."""
        return [m for m in self.matches if m.category == category]


@dataclass(slots=True)
class ContentPattern:
    """A content moderation pattern.

    Attributes:
        pattern: Regex pattern string.
        severity: Severity level for matches.
        category: Category for matches.
        compiled: Compiled regex pattern.
    """

    pattern: str
    severity: SeverityLevel
    category: str
    compiled: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self):
        """Compile the regex pattern."""
        # Word boundary pattern, case insensitive
        self.compiled = re.compile(rf"\b{self.pattern}\b", re.IGNORECASE)


# Default English profanity patterns
# Severity levels:
# - low: mild language, generally acceptable in casual contexts
# - medium: moderate profanity, inappropriate in professional contexts
# - high: strong profanity, offensive to many
# - critical: slurs, hate speech, threats - always unacceptable
DEFAULT_PATTERNS: list[ContentPattern] = [
    # Low severity - mild expressions
    ContentPattern(r"darn", "low", "mild_language"),
    ContentPattern(r"dang", "low", "mild_language"),
    ContentPattern(r"heck", "low", "mild_language"),
    ContentPattern(r"crap", "low", "mild_language"),
    ContentPattern(r"damn", "low", "mild_language"),
    ContentPattern(r"hell", "low", "mild_language"),
    # Medium severity - moderate profanity
    ContentPattern(r"ass", "medium", "profanity"),
    ContentPattern(r"bastard", "medium", "profanity"),
    ContentPattern(r"bitch", "medium", "profanity"),
    ContentPattern(r"piss", "medium", "profanity"),
    ContentPattern(r"shit", "medium", "profanity"),
    ContentPattern(r"bullshit", "medium", "profanity"),
    # High severity - strong profanity
    ContentPattern(r"fuck", "high", "profanity"),
    ContentPattern(r"fucking", "high", "profanity"),
    ContentPattern(r"motherfucker", "high", "profanity"),
    ContentPattern(r"fucker", "high", "profanity"),
    ContentPattern(r"cock", "high", "profanity"),
    ContentPattern(r"dick", "high", "profanity"),
    ContentPattern(r"pussy", "high", "profanity"),
    ContentPattern(r"asshole", "high", "profanity"),
    # Critical severity - highly offensive content
    # Note: These are intentionally not enumerated in detail.
    # Organizations should configure their own critical patterns
    # based on their specific needs and regional considerations.
]

# Severity to action mapping (default policy)
DEFAULT_SEVERITY_ACTIONS: dict[SeverityLevel, ModerationAction] = {
    "low": "allow",
    "medium": "warn",
    "high": "warn",
    "critical": "block",
}

# Severity ordering for comparison
SEVERITY_ORDER: dict[SeverityLevel, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


@dataclass(slots=True)
class LexiconModeratorConfig:
    """Configuration for lexicon-based content moderation.

    Attributes:
        patterns: List of content patterns to match.
        severity_actions: Mapping of severity levels to actions.
        default_action: Default action when no severity matches.
        mask_character: Character used for masking (default: *).
        mask_preserve_first: Preserve first character when masking.
        mask_preserve_last: Preserve last character when masking.
        enable_mild: Enable detection of mild language (low severity).
        enable_profanity: Enable detection of profanity.
        enable_slurs: Enable detection of slurs (if configured).
        custom_patterns: Additional custom patterns to add.
    """

    patterns: list[ContentPattern] = field(default_factory=lambda: list(DEFAULT_PATTERNS))
    severity_actions: dict[SeverityLevel, ModerationAction] = field(
        default_factory=lambda: dict(DEFAULT_SEVERITY_ACTIONS)
    )
    default_action: ModerationAction = "allow"
    mask_character: str = "*"
    mask_preserve_first: bool = True
    mask_preserve_last: bool = False
    enable_mild: bool = True
    enable_profanity: bool = True
    enable_slurs: bool = True
    custom_patterns: list[ContentPattern] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration."""
        if len(self.mask_character) != 1:
            raise ValueError("mask_character must be a single character")

    def get_active_patterns(self) -> list[ContentPattern]:
        """Get patterns that are currently enabled."""
        patterns = []

        for p in self.patterns + self.custom_patterns:
            if p.category == "mild_language" and not self.enable_mild:
                continue
            if p.category == "profanity" and not self.enable_profanity:
                continue
            if p.category == "slur" and not self.enable_slurs:
                continue
            patterns.append(p)

        return patterns


class LexiconModerator:
    """Lexicon-based content moderator.

    Performs pattern-based content moderation with configurable
    severity levels and actions.

    Example:
        >>> moderator = LexiconModerator()
        >>> result = moderator.check("This is some damn good work")
        >>> print(result.is_flagged)
        True
        >>> print(result.highest_severity)
        "low"
        >>> print(result.suggested_action)
        "allow"

        >>> # Mask flagged content
        >>> masked = moderator.mask("What the fuck", result.matches)
        >>> print(masked)
        "What the f***"
    """

    def __init__(self, config: LexiconModeratorConfig | None = None):
        """Initialize moderator with optional configuration.

        Args:
            config: Moderator configuration. Uses defaults if not provided.
        """
        self.config = config or LexiconModeratorConfig()

    def check(self, text: str) -> ModerationResult:
        """Check text for content violations.

        Args:
            text: Input text to check.

        Returns:
            ModerationResult with matches and recommended action.
        """
        matches: list[ContentMatch] = []
        categories_found: set[str] = set()
        highest_severity: SeverityLevel | None = None
        highest_severity_order = 0

        for pattern in self.config.get_active_patterns():
            for match in pattern.compiled.finditer(text):
                content_match = ContentMatch(
                    start=match.start(),
                    end=match.end(),
                    matched_text=match.group(0),
                    pattern=pattern.pattern,
                    severity=pattern.severity,
                    category=pattern.category,
                )
                matches.append(content_match)
                categories_found.add(pattern.category)

                # Track highest severity
                severity_order = SEVERITY_ORDER.get(pattern.severity, 0)
                if severity_order > highest_severity_order:
                    highest_severity_order = severity_order
                    highest_severity = pattern.severity

        # Determine suggested action
        if highest_severity is None:
            suggested_action = self.config.default_action
        else:
            suggested_action = self.config.severity_actions.get(
                highest_severity, self.config.default_action
            )

        # Sort matches by position
        matches.sort(key=lambda m: m.start)

        return ModerationResult(
            original=text,
            is_flagged=len(matches) > 0,
            highest_severity=highest_severity,
            matches=matches,
            suggested_action=suggested_action,
            categories_found=categories_found,
        )

    def mask(self, text: str, matches: list[ContentMatch]) -> str:
        """Mask matched content in text.

        Args:
            text: Original text to mask.
            matches: List of content matches to mask.

        Returns:
            Text with matched content masked.
        """
        if not matches:
            return text

        # Sort by position (descending) for safe replacement
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)

        result = text
        for match in sorted_matches:
            original = match.matched_text
            length = len(original)

            if length <= 2:
                # Very short words: mask entirely
                masked = self.config.mask_character * length
            else:
                # Determine what to preserve
                preserve_first = self.config.mask_preserve_first
                preserve_last = self.config.mask_preserve_last

                if preserve_first and preserve_last:
                    # f***k
                    masked = original[0] + self.config.mask_character * (length - 2) + original[-1]
                elif preserve_first:
                    # f***
                    masked = original[0] + self.config.mask_character * (length - 1)
                elif preserve_last:
                    # ***k
                    masked = self.config.mask_character * (length - 1) + original[-1]
                else:
                    # ****
                    masked = self.config.mask_character * length

            result = result[: match.start] + masked + result[match.end :]

        return result

    def check_and_mask(self, text: str) -> tuple[ModerationResult, str]:
        """Check text and return both result and masked text.

        Convenience method that performs both check and mask operations.

        Args:
            text: Input text to check and mask.

        Returns:
            Tuple of (ModerationResult, masked_text).
        """
        result = self.check(text)
        masked = self.mask(text, result.matches) if result.is_flagged else text
        return result, masked

    def add_pattern(
        self,
        pattern: str,
        severity: SeverityLevel,
        category: str,
    ) -> None:
        """Add a custom pattern to the moderator.

        Args:
            pattern: Regex pattern string (word boundaries added automatically).
            severity: Severity level for matches.
            category: Category for matches.
        """
        self.config.custom_patterns.append(
            ContentPattern(pattern=pattern, severity=severity, category=category)
        )

    def add_patterns(self, patterns: list[ContentPattern]) -> None:
        """Add multiple custom patterns.

        Args:
            patterns: List of ContentPattern objects to add.
        """
        self.config.custom_patterns.extend(patterns)

    def set_severity_action(self, severity: SeverityLevel, action: ModerationAction) -> None:
        """Set the action for a severity level.

        Args:
            severity: Severity level to configure.
            action: Action to take for this severity.
        """
        self.config.severity_actions[severity] = action


def create_moderator_for_call_center() -> LexiconModerator:
    """Create a moderator configured for call center use.

    Pre-configured for customer service contexts with:
    - All profanity detection enabled
    - Warn action for medium/high severity
    - Block action for critical severity
    - Additional threat-related patterns

    Returns:
        Configured LexiconModerator instance.
    """
    config = LexiconModeratorConfig(
        severity_actions={
            "low": "allow",
            "medium": "warn",
            "high": "warn",
            "critical": "block",
        },
        custom_patterns=[
            ContentPattern(r"kill\s+you", "critical", "threat"),
            ContentPattern(r"sue\s+you", "high", "legal_threat"),
            ContentPattern(r"lawyer", "medium", "legal_mention"),
            ContentPattern(r"cancel.*account", "low", "escalation_signal"),
            ContentPattern(r"speak.*manager", "low", "escalation_signal"),
            ContentPattern(r"supervisor", "low", "escalation_signal"),
        ],
    )
    return LexiconModerator(config)


def create_moderator_for_healthcare() -> LexiconModerator:
    """Create a moderator configured for healthcare use.

    Pre-configured for medical contexts with:
    - Standard profanity detection
    - Additional patterns for self-harm indicators
    - Higher sensitivity for concerning content

    Returns:
        Configured LexiconModerator instance.
    """
    config = LexiconModeratorConfig(
        severity_actions={
            "low": "allow",
            "medium": "warn",
            "high": "warn",
            "critical": "block",
        },
        custom_patterns=[
            ContentPattern(r"hurt\s+myself", "critical", "self_harm"),
            ContentPattern(r"end\s+it\s+all", "critical", "self_harm"),
            ContentPattern(r"want\s+to\s+die", "critical", "self_harm"),
            ContentPattern(r"kill\s+myself", "critical", "self_harm"),
            ContentPattern(r"suicide", "critical", "self_harm"),
            ContentPattern(r"overdose", "high", "substance"),
        ],
    )
    return LexiconModerator(config)


# Export public API
__all__ = [
    "LexiconModerator",
    "LexiconModeratorConfig",
    "ModerationResult",
    "ContentMatch",
    "ContentPattern",
    "SeverityLevel",
    "ModerationAction",
    "create_moderator_for_call_center",
    "create_moderator_for_healthcare",
]
