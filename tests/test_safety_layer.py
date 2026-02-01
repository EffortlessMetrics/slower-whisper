"""Tests for safety layer and streaming integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transcription.safety_config import (
    SafetyConfig,
    safety_config_for_call_center,
    safety_config_for_healthcare,
    safety_config_minimal,
)
from transcription.safety_layer import (
    PIIMatch,
    SafetyAlertPayload,
    SafetyProcessingResult,
    SafetyProcessor,
    create_safety_processor,
)
from transcription.streaming_safety import (
    StreamingSafetyProcessor,
    StreamingSafetyState,
)


class TestSafetyConfig:
    """Tests for SafetyConfig."""

    def test_default_disabled(self):
        """Default config has all features disabled."""
        config = SafetyConfig()
        assert not config.enabled
        assert not config.enable_pii_detection
        assert not config.enable_content_moderation
        assert not config.enable_smart_formatting

    def test_is_active_when_disabled(self):
        """is_active returns False when not enabled."""
        config = SafetyConfig()
        assert not config.is_active()

    def test_is_active_when_enabled_but_no_features(self):
        """is_active returns False when enabled but no features."""
        config = SafetyConfig(enabled=True)
        assert not config.is_active()

    def test_is_active_with_features(self):
        """is_active returns True when enabled with features."""
        config = SafetyConfig(enabled=True, enable_content_moderation=True)
        assert config.is_active()

    def test_to_dict_and_from_dict(self):
        """Config round-trips through dict."""
        original = SafetyConfig(
            enabled=True,
            enable_pii_detection=True,
            pii_action="mask",
        )
        data = original.to_dict()
        restored = SafetyConfig.from_dict(data)
        assert restored.enabled == original.enabled
        assert restored.enable_pii_detection == original.enable_pii_detection
        assert restored.pii_action == original.pii_action


class TestSafetyConfigPresets:
    """Tests for safety config preset functions."""

    def test_call_center_preset(self):
        """Call center preset has expected settings."""
        config = safety_config_for_call_center()
        assert config.enabled
        assert config.enable_pii_detection
        assert config.enable_content_moderation
        assert config.enable_smart_formatting
        assert config.pii_action == "mask"

    def test_healthcare_preset(self):
        """Healthcare preset has expected settings."""
        config = safety_config_for_healthcare()
        assert config.enabled
        assert config.enable_pii_detection
        assert config.enable_content_moderation
        assert config.moderation_severity_threshold == "high"

    def test_minimal_preset(self):
        """Minimal preset has only formatting."""
        config = safety_config_minimal()
        assert config.enabled
        assert not config.enable_pii_detection
        assert not config.enable_content_moderation
        assert config.enable_smart_formatting


class TestSafetyProcessor:
    """Tests for SafetyProcessor."""

    def test_inactive_processor_passthrough(self):
        """Inactive processor passes through text unchanged."""
        config = SafetyConfig()  # Not enabled
        processor = SafetyProcessor(config)
        result = processor.process("Test text")
        assert result.original_text == "Test text"
        assert result.processed_text == "Test text"
        assert not result.has_pii
        assert not result.has_flagged_content

    def test_pii_detection_email(self):
        """Detects email addresses as PII."""
        config = SafetyConfig(enabled=True, enable_pii_detection=True)
        processor = SafetyProcessor(config)
        result = processor.process("Contact me at john@example.com")
        assert result.has_pii
        assert len(result.pii_matches) == 1
        assert result.pii_matches[0].pii_type == "email"

    def test_pii_detection_phone(self):
        """Detects phone numbers as PII."""
        config = SafetyConfig(enabled=True, enable_pii_detection=True)
        processor = SafetyProcessor(config)
        result = processor.process("Call me at 555-123-4567")
        assert result.has_pii
        assert any(m.pii_type == "phone" for m in result.pii_matches)

    def test_pii_masking(self):
        """PII is masked when action is mask."""
        config = SafetyConfig(
            enabled=True,
            enable_pii_detection=True,
            pii_action="mask",
        )
        processor = SafetyProcessor(config)
        result = processor.process("Email: john@example.com")
        assert "[EMAIL]" in result.processed_text
        assert "john@example.com" not in result.processed_text

    def test_pii_type_filtering(self):
        """Can filter PII detection by type."""
        config = SafetyConfig(
            enabled=True,
            enable_pii_detection=True,
            pii_types=["email"],  # Only detect emails
        )
        processor = SafetyProcessor(config)
        result = processor.process("Email: john@example.com Phone: 555-123-4567")
        assert len(result.pii_matches) == 1
        assert result.pii_matches[0].pii_type == "email"

    def test_content_moderation(self):
        """Content moderation detects profanity."""
        config = SafetyConfig(enabled=True, enable_content_moderation=True)
        processor = SafetyProcessor(config)
        result = processor.process("This is bullshit")
        assert result.has_flagged_content
        assert result.moderation_result is not None
        assert result.moderation_result.is_flagged

    def test_content_moderation_masking(self):
        """Content is masked when action is mask."""
        config = SafetyConfig(
            enabled=True,
            enable_content_moderation=True,
            content_action="mask",
        )
        processor = SafetyProcessor(config)
        result = processor.process("What the fuck")
        assert "f***" in result.processed_text or "****" in result.processed_text

    def test_content_moderation_severity_threshold(self):
        """Respects severity threshold."""
        # High threshold - should not flag "damn" (low severity)
        config = SafetyConfig(
            enabled=True,
            enable_content_moderation=True,
            moderation_severity_threshold="high",
        )
        processor = SafetyProcessor(config)
        result = processor.process("Well damn")
        # The moderation result exists but shouldn't trigger action
        assert result.overall_action == "allow"

    def test_smart_formatting(self):
        """Smart formatting is applied."""
        config = SafetyConfig(enabled=True, enable_smart_formatting=True)
        processor = SafetyProcessor(config)
        result = processor.process("Meeting at five pm")
        assert result.has_formatting_changes
        assert "5:00 PM" in result.processed_text

    def test_combined_processing(self):
        """Multiple processors work together."""
        config = SafetyConfig(
            enabled=True,
            enable_pii_detection=True,
            enable_content_moderation=True,
            enable_smart_formatting=True,
            pii_action="mask",
        )
        processor = SafetyProcessor(config)
        result = processor.process("Email john@example.com at five pm, this is bullshit")
        # Check all features triggered
        assert result.has_pii
        assert result.has_flagged_content
        assert result.has_formatting_changes
        # Check text was processed
        assert "[EMAIL]" in result.processed_text
        assert "5:00 PM" in result.processed_text

    def test_overall_action_highest_wins(self):
        """Overall action is highest severity."""
        config = SafetyConfig(
            enabled=True,
            enable_pii_detection=True,
            enable_content_moderation=True,
            pii_action="block",
            content_action="warn",
        )
        processor = SafetyProcessor(config)
        result = processor.process("Email john@example.com, this is crap")
        assert result.overall_action == "block"


class TestSafetyProcessingResult:
    """Tests for SafetyProcessingResult."""

    def test_to_safety_state(self):
        """to_safety_state generates correct structure."""
        result = SafetyProcessingResult(
            original_text="test",
            processed_text="test",
            has_pii=True,
            pii_matches=[
                PIIMatch(0, 4, "test", "email", 1.0),
            ],
        )
        state = result.to_safety_state()
        assert state["processed"] is True
        assert "pii" in state
        assert state["pii"]["detected"] is True
        assert state["pii"]["count"] == 1


class TestSafetyAlertPayload:
    """Tests for SafetyAlertPayload."""

    def test_alert_payload_attributes(self):
        """Alert payload has correct attributes."""
        payload = SafetyAlertPayload(
            segment_id=1,
            segment_start=0.0,
            segment_end=5.0,
            alert_type="pii",
            severity="high",
            action="mask",
            details={"pii_types": ["email"]},
        )
        assert payload.segment_id == 1
        assert payload.alert_type == "pii"
        assert payload.severity == "high"
        assert payload.details["pii_types"] == ["email"]


class TestCreateSafetyProcessor:
    """Tests for create_safety_processor factory."""

    def test_default_processor(self):
        """Creates processor with default config."""
        processor = create_safety_processor()
        assert processor.config.enabled is False

    def test_call_center_preset(self):
        """Creates processor with call_center preset."""
        processor = create_safety_processor(preset="call_center")
        assert processor.config.enabled is True
        assert processor.config.enable_pii_detection is True

    def test_with_overrides(self):
        """Creates processor with overrides."""
        processor = create_safety_processor(
            preset="call_center",
            pii_action="block",
        )
        assert processor.config.pii_action == "block"


# Mock StreamSegment for testing
@dataclass
class MockStreamSegment:
    """Mock StreamSegment for testing."""

    segment_id: str
    start: float
    end: float
    text: str
    speaker_id: str = "spk_0"
    audio_state: dict[str, Any] | None = None


class TestStreamingSafetyProcessor:
    """Tests for StreamingSafetyProcessor."""

    def test_inactive_passthrough(self):
        """Inactive processor passes through."""
        config = SafetyConfig()
        processor = StreamingSafetyProcessor(config)
        segment = MockStreamSegment(segment_id="seg-0", start=0.0, end=5.0, text="Hello world")
        result = processor.process_segment(segment)
        assert result.original_text == "Hello world"
        assert result.processed_text == "Hello world"

    def test_processes_segment(self):
        """Processes segment content."""
        config = SafetyConfig(enabled=True, enable_smart_formatting=True)
        processor = StreamingSafetyProcessor(config)
        segment = MockStreamSegment(
            segment_id="seg-0", start=0.0, end=5.0, text="Meeting at five pm"
        )
        result = processor.process_segment(segment)
        assert "5:00 PM" in result.processed_text

    def test_attaches_safety_state(self):
        """Attaches safety state to segment."""
        config = SafetyConfig(enabled=True, enable_smart_formatting=True)
        processor = StreamingSafetyProcessor(config)
        segment = MockStreamSegment(
            segment_id="seg-0", start=0.0, end=5.0, text="Meeting at five pm"
        )
        processor.process_segment(segment)
        assert segment.audio_state is not None
        assert "safety" in segment.audio_state
        assert segment.audio_state["safety"]["processed"] is True

    def test_cumulative_state(self):
        """Tracks cumulative statistics."""
        config = SafetyConfig(
            enabled=True,
            enable_pii_detection=True,
            enable_content_moderation=True,
        )
        processor = StreamingSafetyProcessor(config)

        seg1 = MockStreamSegment(
            segment_id="seg-0", start=0.0, end=5.0, text="Email: john@example.com"
        )
        seg2 = MockStreamSegment(segment_id="seg-1", start=5.0, end=10.0, text="This is bullshit")

        processor.process_segment(seg1)
        processor.process_segment(seg2)

        state = processor.state
        assert state.total_segments_processed == 2
        assert state.total_pii_detections == 1
        assert state.total_moderation_flags >= 1

    def test_reset_state(self):
        """Can reset cumulative state."""
        config = SafetyConfig(enabled=True, enable_pii_detection=True)
        processor = StreamingSafetyProcessor(config)

        segment = MockStreamSegment(
            segment_id="seg-0", start=0.0, end=5.0, text="Email: john@example.com"
        )
        processor.process_segment(segment)
        assert processor.state.total_segments_processed == 1

        processor.reset_state()
        assert processor.state.total_segments_processed == 0

    def test_process_text(self):
        """Can process raw text without segment."""
        config = SafetyConfig(enabled=True, enable_smart_formatting=True)
        processor = StreamingSafetyProcessor(config)
        result = processor.process_text("five pm")
        assert "5:00 PM" in result.processed_text

    def test_get_summary(self):
        """Can get session summary."""
        config = SafetyConfig(enabled=True)
        processor = StreamingSafetyProcessor(config)
        summary = processor.get_summary()
        assert "config" in summary
        assert "state" in summary


class TestStreamingSafetyState:
    """Tests for StreamingSafetyState."""

    def test_default_state(self):
        """Default state is empty."""
        state = StreamingSafetyState()
        assert state.total_segments_processed == 0
        assert state.total_pii_detections == 0

    def test_to_dict(self):
        """State converts to dict."""
        state = StreamingSafetyState(
            total_segments_processed=10,
            total_pii_detections=5,
            unique_pii_types={"email", "phone"},
        )
        data = state.to_dict()
        assert data["total_segments_processed"] == 10
        assert data["total_pii_detections"] == 5
        assert set(data["unique_pii_types"]) == {"email", "phone"}


class TestCallbackIntegration:
    """Tests for callback integration."""

    def test_emits_safety_alert(self):
        """Emits safety alert callback."""
        alerts: list[SafetyAlertPayload] = []

        class TestCallbacks:
            def on_safety_alert(self, payload: SafetyAlertPayload) -> None:
                alerts.append(payload)

        config = SafetyConfig(
            enabled=True,
            enable_pii_detection=True,
            emit_alerts=True,
            alert_on_pii=True,
        )
        callbacks = TestCallbacks()
        processor = StreamingSafetyProcessor(config, callbacks)

        segment = MockStreamSegment(
            segment_id="seg-0", start=0.0, end=5.0, text="Email: john@example.com"
        )
        processor.process_segment(segment)

        assert len(alerts) == 1
        assert alerts[0].alert_type == "pii"

    def test_no_alert_when_disabled(self):
        """No alert when emit_alerts is False."""
        alerts: list[SafetyAlertPayload] = []

        class TestCallbacks:
            def on_safety_alert(self, payload: SafetyAlertPayload) -> None:
                alerts.append(payload)

        config = SafetyConfig(
            enabled=True,
            enable_pii_detection=True,
            emit_alerts=False,
        )
        callbacks = TestCallbacks()
        processor = StreamingSafetyProcessor(config, callbacks)

        segment = MockStreamSegment(
            segment_id="seg-0", start=0.0, end=5.0, text="Email: john@example.com"
        )
        processor.process_segment(segment)

        assert len(alerts) == 0
