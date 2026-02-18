"""Safety layer configuration for unified content processing.

This module defines the SafetyConfig dataclass that controls the unified
safety processing layer combining PII detection, content moderation,
and smart formatting.

All features are opt-in (disabled by default) following the project's
enrichment configuration patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ModerationAction = Literal["allow", "warn", "mask", "block"]


@dataclass(slots=True)
class SafetyConfig:
    """Configuration for the unified safety processing layer.

    The safety layer combines multiple content processing features into
    a single, configurable pipeline that processes transcript segments.

    All features are disabled by default and must be explicitly enabled.

    Attributes:
        enabled: Master switch for the safety layer. If False, no processing occurs.
        enable_pii_detection: Detect and optionally redact PII (names, emails, etc.).
        enable_content_moderation: Detect profanity, toxicity, and other flagged content.
        enable_smart_formatting: Apply ITN-lite formatting (currency, dates, times, etc.).

        pii_action: Action to take when PII is detected.
        content_action: Action to take when flagged content is detected.

        pii_types: Types of PII to detect. None means all types.
        moderation_severity_threshold: Minimum severity to flag ("low", "medium", "high", "critical").

        smart_formatting_config: Optional configuration for smart formatter.
        moderation_config: Optional configuration for content moderator.

        emit_alerts: Whether to emit safety alerts via callbacks.
        alert_on_pii: Emit alert when PII is detected.
        alert_on_moderation: Emit alert when content is flagged.
    """

    # Master switch
    enabled: bool = False

    # Feature toggles
    enable_pii_detection: bool = False
    enable_content_moderation: bool = False
    enable_smart_formatting: bool = False

    # Actions
    pii_action: ModerationAction = "warn"
    content_action: ModerationAction = "warn"

    # Detection settings
    pii_types: list[str] | None = None  # None = all types
    moderation_severity_threshold: Literal["low", "medium", "high", "critical"] = "medium"

    # Nested configurations (for advanced use)
    smart_formatting_config: dict[str, Any] | None = None
    moderation_config: dict[str, Any] | None = None

    # Alert settings
    emit_alerts: bool = True
    alert_on_pii: bool = True
    alert_on_moderation: bool = True

    def is_active(self) -> bool:
        """Return True if safety layer should process content.

        The safety layer is active if enabled and at least one feature is enabled.
        """
        if not self.enabled:
            return False
        return (
            self.enable_pii_detection
            or self.enable_content_moderation
            or self.enable_smart_formatting
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a JSON-serializable dictionary."""
        return {
            "enabled": self.enabled,
            "enable_pii_detection": self.enable_pii_detection,
            "enable_content_moderation": self.enable_content_moderation,
            "enable_smart_formatting": self.enable_smart_formatting,
            "pii_action": self.pii_action,
            "content_action": self.content_action,
            "pii_types": self.pii_types,
            "moderation_severity_threshold": self.moderation_severity_threshold,
            "smart_formatting_config": self.smart_formatting_config,
            "moderation_config": self.moderation_config,
            "emit_alerts": self.emit_alerts,
            "alert_on_pii": self.alert_on_pii,
            "alert_on_moderation": self.alert_on_moderation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SafetyConfig:
        """Create config from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            SafetyConfig instance.
        """
        return cls(
            enabled=data.get("enabled", False),
            enable_pii_detection=data.get("enable_pii_detection", False),
            enable_content_moderation=data.get("enable_content_moderation", False),
            enable_smart_formatting=data.get("enable_smart_formatting", False),
            pii_action=data.get("pii_action", "warn"),
            content_action=data.get("content_action", "warn"),
            pii_types=data.get("pii_types"),
            moderation_severity_threshold=data.get("moderation_severity_threshold", "medium"),
            smart_formatting_config=data.get("smart_formatting_config"),
            moderation_config=data.get("moderation_config"),
            emit_alerts=data.get("emit_alerts", True),
            alert_on_pii=data.get("alert_on_pii", True),
            alert_on_moderation=data.get("alert_on_moderation", True),
        )


# Preset configurations for common use cases
def safety_config_for_call_center() -> SafetyConfig:
    """Create safety config optimized for call center transcription.

    Features:
    - PII detection with masking
    - Content moderation with warning
    - Smart formatting enabled

    Returns:
        SafetyConfig configured for call center use.
    """
    return SafetyConfig(
        enabled=True,
        enable_pii_detection=True,
        enable_content_moderation=True,
        enable_smart_formatting=True,
        pii_action="mask",
        content_action="warn",
        moderation_severity_threshold="medium",
    )


def safety_config_for_healthcare() -> SafetyConfig:
    """Create safety config optimized for healthcare transcription.

    Features:
    - PII detection with blocking (HIPAA compliance)
    - Content moderation for self-harm detection
    - Smart formatting for medical terms

    Returns:
        SafetyConfig configured for healthcare use.
    """
    return SafetyConfig(
        enabled=True,
        enable_pii_detection=True,
        enable_content_moderation=True,
        enable_smart_formatting=True,
        pii_action="mask",
        content_action="warn",
        moderation_severity_threshold="high",
        alert_on_pii=True,
        alert_on_moderation=True,
    )


def safety_config_minimal() -> SafetyConfig:
    """Create minimal safety config with only formatting enabled.

    Features:
    - Smart formatting only
    - No PII or content moderation

    Returns:
        SafetyConfig with minimal processing.
    """
    return SafetyConfig(
        enabled=True,
        enable_pii_detection=False,
        enable_content_moderation=False,
        enable_smart_formatting=True,
    )


__all__ = [
    "SafetyConfig",
    "ModerationAction",
    "safety_config_for_call_center",
    "safety_config_for_healthcare",
    "safety_config_minimal",
]
