"""Tests for role inference module."""

from __future__ import annotations

from transcription.role_inference import (
    RoleAssignment,
    RoleInferenceConfig,
    RoleInferrer,
    infer_roles,
)


class TestRoleInferenceConfig:
    """Tests for RoleInferenceConfig."""

    def test_default_config(self):
        """Default config has expected values."""
        config = RoleInferenceConfig()
        assert config.enabled is True
        assert config.context == "call_center"
        assert config.min_confidence == 0.3
        assert config.use_turn_patterns is True

    def test_custom_config(self):
        """Custom config values work."""
        config = RoleInferenceConfig(
            context="meeting",
            min_confidence=0.5,
            use_turn_patterns=False,
        )
        assert config.context == "meeting"
        assert config.min_confidence == 0.5
        assert config.use_turn_patterns is False


class TestRoleAssignment:
    """Tests for RoleAssignment dataclass."""

    def test_to_dict_and_from_dict(self):
        """RoleAssignment round-trips through dict."""
        original = RoleAssignment(
            speaker_id="spk_0",
            role="agent",
            confidence=0.85,
            evidence=["greeting: 'how can I help'"],
            original_label="Speaker 1",
        )
        data = original.to_dict()
        restored = RoleAssignment.from_dict(data)

        assert restored.speaker_id == original.speaker_id
        assert restored.role == original.role
        assert restored.confidence == original.confidence
        assert restored.evidence == original.evidence
        assert restored.original_label == original.original_label


class TestRoleInferrer:
    """Tests for RoleInferrer class."""

    def test_disabled_returns_empty(self):
        """Disabled inferrer returns empty dict."""
        config = RoleInferenceConfig(enabled=False)
        inferrer = RoleInferrer(config)
        turns = [{"speaker_id": "spk_0", "text": "Hello"}]
        assignments = inferrer.infer_roles(turns)
        assert assignments == {}

    def test_empty_turns(self):
        """Empty turns list returns empty dict."""
        inferrer = RoleInferrer()
        assignments = inferrer.infer_roles([])
        assert assignments == {}

    def test_agent_greeting_detection(self):
        """Detects agent from greeting phrases."""
        inferrer = RoleInferrer()
        turns = [
            {"speaker_id": "spk_0", "text": "Thank you for calling, how can I help you today?"},
        ]
        assignments = inferrer.infer_roles(turns)

        assert "spk_0" in assignments
        assert assignments["spk_0"].role == "agent"
        assert assignments["spk_0"].confidence > 0.5
        assert any("how can I help" in e for e in assignments["spk_0"].evidence)

    def test_customer_intent_detection(self):
        """Detects customer from intent phrases."""
        inferrer = RoleInferrer()
        turns = [
            {"speaker_id": "spk_0", "text": "I'm calling about my account, I have a problem."},
        ]
        assignments = inferrer.infer_roles(turns)

        assert "spk_0" in assignments
        assert assignments["spk_0"].role == "customer"
        assert assignments["spk_0"].confidence > 0.5

    def test_two_speaker_call_center(self):
        """Correctly identifies agent and customer in call."""
        inferrer = RoleInferrer()
        turns = [
            {"speaker_id": "spk_0", "text": "Thank you for calling support, how can I help you?"},
            {"speaker_id": "spk_1", "text": "Hi, I'm calling about my order. I haven't received it."},
            {"speaker_id": "spk_0", "text": "I'd be happy to help. Let me check your order status."},
            {"speaker_id": "spk_1", "text": "Thank you. My account number is 12345."},
            {"speaker_id": "spk_0", "text": "I can see your order here. Is there anything else?"},
        ]
        assignments = inferrer.infer_roles(turns)

        assert "spk_0" in assignments
        assert "spk_1" in assignments
        assert assignments["spk_0"].role == "agent"
        assert assignments["spk_1"].role == "customer"

    def test_facilitator_detection(self):
        """Detects facilitator from moderation phrases."""
        config = RoleInferenceConfig(context="meeting")
        inferrer = RoleInferrer(config)
        turns = [
            {"speaker_id": "spk_0", "text": "Welcome everyone. Let's move on to the next agenda item."},
            {"speaker_id": "spk_1", "text": "I have an update on the project."},
        ]
        assignments = inferrer.infer_roles(turns)

        assert "spk_0" in assignments
        assert assignments["spk_0"].role == "facilitator"

    def test_low_confidence_returns_unknown(self):
        """Low confidence results in unknown role."""
        config = RoleInferenceConfig(min_confidence=0.9)
        inferrer = RoleInferrer(config)
        turns = [
            {"speaker_id": "spk_0", "text": "Hello there."},
        ]
        assignments = inferrer.infer_roles(turns)

        assert "spk_0" in assignments
        assert assignments["spk_0"].role == "unknown"

    def test_first_speaker_heuristic(self):
        """First speaker gets agent bonus in call center context."""
        inferrer = RoleInferrer()
        turns = [
            {"speaker_id": "spk_0", "text": "Hello, good morning."},
            {"speaker_id": "spk_1", "text": "Hi, I need some help."},
        ]
        assignments = inferrer.infer_roles(turns)

        # spk_0 should have agent bias from being first speaker
        # spk_1 has "I need" customer trigger
        assert "spk_0" in assignments
        assert "spk_1" in assignments

    def test_preserves_original_label(self):
        """Original label is preserved in assignment."""
        inferrer = RoleInferrer()
        turns = [
            {"speaker_id": "spk_0", "text": "Thank you for calling, how can I help?"},
        ]
        speakers = [
            {"id": "spk_0", "label": "Speaker 1"},
        ]
        assignments = inferrer.infer_roles(turns, speakers)

        assert assignments["spk_0"].original_label == "Speaker 1"

    def test_update_speakers(self):
        """Can update speaker list with inferred roles."""
        inferrer = RoleInferrer()
        turns = [
            {"speaker_id": "spk_0", "text": "Thank you for calling, how can I help?"},
            {"speaker_id": "spk_1", "text": "I'm calling about my account."},
        ]
        speakers = [
            {"id": "spk_0", "label": "Speaker 1"},
            {"id": "spk_1", "label": "Speaker 2"},
        ]

        assignments = inferrer.infer_roles(turns, speakers)
        updated = inferrer.update_speakers(speakers, assignments)

        assert updated[0]["label"] == "agent"
        assert updated[0]["_original_label"] == "Speaker 1"
        assert updated[0]["role_confidence"] > 0.5

        assert updated[1]["label"] == "customer"
        assert updated[1]["_original_label"] == "Speaker 2"

    def test_custom_triggers(self):
        """Custom triggers are applied."""
        config = RoleInferenceConfig(
            custom_agent_triggers=[
                (r"\bcompany greeting\b", 0.9, "custom greeting"),
            ],
        )
        inferrer = RoleInferrer(config)
        turns = [
            {"speaker_id": "spk_0", "text": "Company greeting to you!"},
        ]
        assignments = inferrer.infer_roles(turns)

        assert assignments["spk_0"].role == "agent"
        assert any("custom greeting" in e for e in assignments["spk_0"].evidence)


class TestInferRolesFunction:
    """Tests for convenience function."""

    def test_infer_roles_basic(self):
        """Convenience function works."""
        turns = [
            {"speaker_id": "spk_0", "text": "How can I help you today?"},
        ]
        assignments = infer_roles(turns)

        assert "spk_0" in assignments
        assert assignments["spk_0"].role == "agent"

    def test_infer_roles_with_config(self):
        """Convenience function accepts config kwargs."""
        turns = [
            {"speaker_id": "spk_0", "text": "Hello"},
        ]
        assignments = infer_roles(turns, min_confidence=0.9)

        # Should be unknown due to high confidence threshold
        assert assignments["spk_0"].role == "unknown"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_missing_speaker_id(self):
        """Handles missing speaker_id gracefully."""
        inferrer = RoleInferrer()
        turns = [
            {"text": "Hello there"},  # No speaker_id
            {"speaker_id": "spk_0", "text": "How can I help?"},
        ]
        assignments = inferrer.infer_roles(turns)
        assert "spk_0" in assignments

    def test_empty_text(self):
        """Handles empty text gracefully."""
        inferrer = RoleInferrer()
        turns = [
            {"speaker_id": "spk_0", "text": ""},
            {"speaker_id": "spk_0", "text": "How can I help?"},
        ]
        assignments = inferrer.infer_roles(turns)
        assert "spk_0" in assignments

    def test_single_turn_low_evidence(self):
        """Single turn with ambiguous text."""
        inferrer = RoleInferrer()
        turns = [
            {"speaker_id": "spk_0", "text": "Okay, sounds good."},
        ]
        assignments = inferrer.infer_roles(turns)
        # Should have low confidence (0.5 or less due to first speaker heuristic)
        assert assignments["spk_0"].confidence <= 0.5

    def test_question_counting(self):
        """Questions are counted correctly."""
        inferrer = RoleInferrer()
        # Verify internal method
        assert inferrer._count_questions("What? How? Why?") == 3
        assert inferrer._count_questions("Hello there.") == 0

    def test_case_insensitive(self):
        """Pattern matching is case insensitive."""
        inferrer = RoleInferrer()
        turns1 = [{"speaker_id": "spk_0", "text": "HOW CAN I HELP YOU?"}]
        turns2 = [{"speaker_id": "spk_0", "text": "how can i help you?"}]
        turns3 = [{"speaker_id": "spk_0", "text": "How Can I Help You?"}]

        a1 = inferrer.infer_roles(turns1)
        a2 = inferrer.infer_roles(turns2)
        a3 = inferrer.infer_roles(turns3)

        assert a1["spk_0"].role == "agent"
        assert a2["spk_0"].role == "agent"
        assert a3["spk_0"].role == "agent"

    def test_multiple_triggers_accumulate(self):
        """Multiple triggers accumulate evidence."""
        inferrer = RoleInferrer()
        turns = [
            {"speaker_id": "spk_0", "text": "Thank you for calling. How can I help? I'd be happy to assist. Let me check that for you."},
        ]
        assignments = inferrer.infer_roles(turns)

        # Should have multiple evidence items
        assert len(assignments["spk_0"].evidence) > 3
        assert assignments["spk_0"].confidence > 0.7


class TestSpeakerKeyVariants:
    """Tests for speaker_id vs speaker key handling."""

    def test_speaker_key_fallback(self):
        """Falls back to 'speaker' key if 'speaker_id' missing."""
        inferrer = RoleInferrer()
        turns = [
            {"speaker": "spk_0", "text": "How can I help you?"},
        ]
        assignments = inferrer.infer_roles(turns)

        assert "spk_0" in assignments
        assert assignments["spk_0"].role == "agent"
