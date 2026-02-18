"""Unified safety processing layer for transcript content.

This module provides a unified safety processor that combines:
- PII detection and redaction
- Content moderation (profanity, toxicity)
- Smart formatting (ITN-lite)

The safety layer processes text and returns both the processed text and
detailed safety state information that can be attached to segments.

All processing is non-destructive - original text is always preserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from .content_moderation import (
    LexiconModerator,
    LexiconModeratorConfig,
    ModerationResult,
)
from .safety_config import SafetyConfig
from .smart_formatting import FormattedText, SmartFormatter, SmartFormatterConfig

logger = logging.getLogger(__name__)

# Severity ordering for comparison
SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass(slots=True)
class PIIMatch:
    """A PII detection match within text.

    Attributes:
        start: Start character index in original text.
        end: End character index in original text.
        matched_text: The actual text that matched.
        pii_type: Type of PII detected (e.g., "email", "phone", "name").
        confidence: Detection confidence (0.0-1.0).
    """

    start: int
    end: int
    matched_text: str
    pii_type: str
    confidence: float = 1.0


@dataclass(slots=True)
class SafetyProcessingResult:
    """Result of safety layer processing.

    Attributes:
        original_text: The original input text (never modified).
        processed_text: The text after all processing applied.
        has_pii: Whether PII was detected.
        has_flagged_content: Whether moderation flags were raised.
        has_formatting_changes: Whether smart formatting was applied.

        pii_matches: List of PII detections.
        moderation_result: Content moderation result (if enabled).
        formatting_result: Smart formatting result (if enabled).

        overall_action: Recommended overall action based on all detections.
        safety_state: Dictionary suitable for attaching to segment.audio_state.
    """

    original_text: str
    processed_text: str
    has_pii: bool = False
    has_flagged_content: bool = False
    has_formatting_changes: bool = False

    pii_matches: list[PIIMatch] = field(default_factory=list)
    moderation_result: ModerationResult | None = None
    formatting_result: FormattedText | None = None

    overall_action: Literal["allow", "warn", "mask", "block"] = "allow"
    safety_state: dict[str, Any] = field(default_factory=dict)

    def to_safety_state(self) -> dict[str, Any]:
        """Generate safety_state dict for segment attachment.

        Returns:
            Dictionary with safety processing results suitable for
            attaching to Segment.audio_state["safety"].
        """
        state: dict[str, Any] = {
            "processed": True,
            "action": self.overall_action,
        }

        if self.has_pii:
            state["pii"] = {
                "detected": True,
                "count": len(self.pii_matches),
                "types": list({m.pii_type for m in self.pii_matches}),
            }

        if self.has_flagged_content and self.moderation_result:
            state["moderation"] = {
                "flagged": True,
                "highest_severity": self.moderation_result.highest_severity,
                "categories": list(self.moderation_result.categories_found),
                "match_count": len(self.moderation_result.matches),
            }

        if self.has_formatting_changes and self.formatting_result:
            state["formatting"] = {
                "applied": True,
                "change_count": len(self.formatting_result.matches),
                "types": list({m.format_type for m in self.formatting_result.matches}),
            }

        return state


@dataclass(slots=True)
class SafetyAlertPayload:
    """Payload for safety alert callbacks.

    Attributes:
        segment_id: ID of the segment that triggered the alert.
        segment_start: Start time of the segment.
        segment_end: End time of the segment.
        alert_type: Type of alert ("pii", "moderation", "combined").
        severity: Overall severity level.
        action: Recommended action.
        details: Additional details about the alert.
    """

    segment_id: int
    segment_start: float
    segment_end: float
    alert_type: Literal["pii", "moderation", "combined"]
    severity: str
    action: Literal["allow", "warn", "mask", "block"]
    details: dict[str, Any] = field(default_factory=dict)


class SafetyProcessor:
    """Unified safety processing layer.

    Combines PII detection, content moderation, and smart formatting
    into a single processing pipeline with configurable behavior.

    Example:
        >>> config = SafetyConfig(
        ...     enabled=True,
        ...     enable_content_moderation=True,
        ...     enable_smart_formatting=True,
        ... )
        >>> processor = SafetyProcessor(config)
        >>> result = processor.process("Meeting at five pm, that's bullshit")
        >>> print(result.processed_text)
        "Meeting at 5:00 PM, that's b*******"
        >>> print(result.overall_action)
        "warn"
    """

    def __init__(self, config: SafetyConfig):
        """Initialize safety processor with configuration.

        Args:
            config: Safety configuration controlling processing behavior.
        """
        self.config = config
        self._moderator: LexiconModerator | None = None
        self._formatter: SmartFormatter | None = None
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize processing components based on configuration."""
        if self.config.enable_content_moderation:
            mod_config = None
            if self.config.moderation_config:
                mod_config = LexiconModeratorConfig(**self.config.moderation_config)
            self._moderator = LexiconModerator(mod_config)

        if self.config.enable_smart_formatting:
            fmt_config = None
            if self.config.smart_formatting_config:
                fmt_config = SmartFormatterConfig(**self.config.smart_formatting_config)
            self._formatter = SmartFormatter(fmt_config)

    def process(self, text: str) -> SafetyProcessingResult:
        """Process text through the safety pipeline.

        Args:
            text: Input text to process.

        Returns:
            SafetyProcessingResult with processing details and modified text.
        """
        if not self.config.is_active():
            return SafetyProcessingResult(
                original_text=text,
                processed_text=text,
            )

        result = SafetyProcessingResult(
            original_text=text,
            processed_text=text,
        )

        # Track the current text state for progressive processing
        current_text = text
        actions_needed: list[Literal["allow", "warn", "mask", "block"]] = []

        # Step 1: PII Detection
        if self.config.enable_pii_detection:
            pii_matches = self._detect_pii(current_text)
            if pii_matches:
                result.has_pii = True
                result.pii_matches = pii_matches
                actions_needed.append(self.config.pii_action)

                if self.config.pii_action == "mask":
                    current_text = self._mask_pii(current_text, pii_matches)

        # Step 2: Content Moderation
        if self.config.enable_content_moderation and self._moderator:
            mod_result = self._moderator.check(current_text)

            # Filter by severity threshold
            threshold_order = SEVERITY_ORDER.get(self.config.moderation_severity_threshold, 2)
            relevant_matches = [
                m
                for m in mod_result.matches
                if SEVERITY_ORDER.get(m.severity, 0) >= threshold_order
            ]

            if relevant_matches:
                result.has_flagged_content = True
                result.moderation_result = mod_result
                actions_needed.append(self.config.content_action)

                if self.config.content_action == "mask":
                    current_text = self._moderator.mask(current_text, relevant_matches)

        # Step 3: Smart Formatting
        if self.config.enable_smart_formatting and self._formatter:
            fmt_result = self._formatter.format(current_text)
            if fmt_result.has_changes():
                result.has_formatting_changes = True
                result.formatting_result = fmt_result
                current_text = fmt_result.formatted

        # Determine overall action (highest severity wins)
        action_order = {"allow": 0, "warn": 1, "mask": 2, "block": 3}
        if actions_needed:
            result.overall_action = max(actions_needed, key=lambda a: action_order.get(a, 0))
        else:
            result.overall_action = "allow"

        result.processed_text = current_text
        result.safety_state = result.to_safety_state()

        return result

    def _detect_pii(self, text: str) -> list[PIIMatch]:
        """Detect PII in text.

        Currently uses simple regex patterns. Can be extended with more
        sophisticated NER-based detection.

        Args:
            text: Text to scan for PII.

        Returns:
            List of PIIMatch objects.
        """
        import re

        matches: list[PIIMatch] = []
        allowed_types = self.config.pii_types

        # Email pattern
        if allowed_types is None or "email" in allowed_types:
            for m in re.finditer(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
                matches.append(
                    PIIMatch(
                        start=m.start(),
                        end=m.end(),
                        matched_text=m.group(),
                        pii_type="email",
                    )
                )

        # Phone pattern (US-style)
        if allowed_types is None or "phone" in allowed_types:
            for m in re.finditer(
                r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text
            ):
                matches.append(
                    PIIMatch(
                        start=m.start(),
                        end=m.end(),
                        matched_text=m.group(),
                        pii_type="phone",
                    )
                )

        # SSN pattern
        if allowed_types is None or "ssn" in allowed_types:
            for m in re.finditer(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", text):
                matches.append(
                    PIIMatch(
                        start=m.start(),
                        end=m.end(),
                        matched_text=m.group(),
                        pii_type="ssn",
                    )
                )

        # Credit card pattern (basic)
        if allowed_types is None or "credit_card" in allowed_types:
            for m in re.finditer(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b", text):
                matches.append(
                    PIIMatch(
                        start=m.start(),
                        end=m.end(),
                        matched_text=m.group(),
                        pii_type="credit_card",
                    )
                )

        # Sort by position
        matches.sort(key=lambda m: m.start)

        return matches

    def _mask_pii(self, text: str, matches: list[PIIMatch]) -> str:
        """Mask PII matches in text.

        Args:
            text: Original text.
            matches: PII matches to mask.

        Returns:
            Text with PII masked.
        """
        if not matches:
            return text

        # Sort by position (descending) for safe replacement
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)

        result = text
        for match in sorted_matches:
            # Create type-specific mask
            mask_map = {
                "email": "[EMAIL]",
                "phone": "[PHONE]",
                "ssn": "[SSN]",
                "credit_card": "[CARD]",
                "name": "[NAME]",
            }
            mask = mask_map.get(match.pii_type, "[REDACTED]")
            result = result[: match.start] + mask + result[match.end :]

        return result

    def create_alert(
        self,
        segment_id: int,
        segment_start: float,
        segment_end: float,
        result: SafetyProcessingResult,
    ) -> SafetyAlertPayload | None:
        """Create a safety alert payload from processing result.

        Args:
            segment_id: ID of the processed segment.
            segment_start: Start time of the segment.
            segment_end: End time of the segment.
            result: Safety processing result.

        Returns:
            SafetyAlertPayload if an alert should be emitted, None otherwise.
        """
        if not self.config.emit_alerts:
            return None

        should_alert = False
        alert_type: Literal["pii", "moderation", "combined"] = "combined"
        details: dict[str, Any] = {}
        severity = "low"

        # Check PII alerts
        if result.has_pii and self.config.alert_on_pii:
            should_alert = True
            alert_type = "pii"
            details["pii_types"] = list({m.pii_type for m in result.pii_matches})
            details["pii_count"] = len(result.pii_matches)
            severity = "high"  # PII is always high severity

        # Check moderation alerts
        if result.has_flagged_content and self.config.alert_on_moderation:
            if should_alert:
                alert_type = "combined"
            else:
                should_alert = True
                alert_type = "moderation"

            if result.moderation_result:
                details["categories"] = list(result.moderation_result.categories_found)
                details["moderation_severity"] = result.moderation_result.highest_severity
                if result.moderation_result.highest_severity:
                    mod_severity_order = SEVERITY_ORDER.get(
                        result.moderation_result.highest_severity, 1
                    )
                    if mod_severity_order > SEVERITY_ORDER.get(severity, 1):
                        severity = result.moderation_result.highest_severity

        if not should_alert:
            return None

        return SafetyAlertPayload(
            segment_id=segment_id,
            segment_start=segment_start,
            segment_end=segment_end,
            alert_type=alert_type,
            severity=severity,
            action=result.overall_action,
            details=details,
        )


# Factory function for common use cases
def create_safety_processor(
    preset: Literal["call_center", "healthcare", "minimal"] | None = None,
    **overrides: Any,
) -> SafetyProcessor:
    """Create a safety processor with optional preset configuration.

    Args:
        preset: Optional preset name ("call_center", "healthcare", "minimal").
        **overrides: Configuration overrides to apply on top of preset.

    Returns:
        Configured SafetyProcessor instance.
    """
    from .safety_config import (
        safety_config_for_call_center,
        safety_config_for_healthcare,
        safety_config_minimal,
    )

    if preset == "call_center":
        config = safety_config_for_call_center()
    elif preset == "healthcare":
        config = safety_config_for_healthcare()
    elif preset == "minimal":
        config = safety_config_minimal()
    else:
        config = SafetyConfig()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return SafetyProcessor(config)


__all__ = [
    "SafetyProcessor",
    "SafetyProcessingResult",
    "SafetyAlertPayload",
    "PIIMatch",
    "create_safety_processor",
]
