"""Tests for turn-taking policy module."""

from __future__ import annotations

import pytest

from transcription.turn_taking_policy import (
    AGGRESSIVE_POLICY,
    BALANCED_POLICY,
    CONSERVATIVE_POLICY,
    EndOfTurnEvaluation,
    EndOfTurnSignal,
    ExtendedEndOfTurnHintPayload,
    TurnTakingEvaluator,
    TurnTakingPolicy,
    get_policy,
)


class TestTurnTakingPolicy:
    """Tests for TurnTakingPolicy dataclass."""

    def test_default_policy(self):
        """Default policy has balanced settings."""
        policy = TurnTakingPolicy()
        assert policy.name == "balanced"
        assert policy.silence_threshold_ms == 700
        assert policy.confidence_threshold == 0.75
        assert policy.enable_prosody is True

    def test_to_dict_and_from_dict(self):
        """Policy round-trips through dict."""
        original = TurnTakingPolicy(
            name="custom",
            silence_threshold_ms=500,
            confidence_threshold=0.8,
        )
        data = original.to_dict()
        restored = TurnTakingPolicy.from_dict(data)

        assert restored.name == original.name
        assert restored.silence_threshold_ms == original.silence_threshold_ms
        assert restored.confidence_threshold == original.confidence_threshold


class TestPresetPolicies:
    """Tests for preset policy configurations."""

    def test_aggressive_policy(self):
        """Aggressive policy has fast settings."""
        assert AGGRESSIVE_POLICY.name == "aggressive"
        assert AGGRESSIVE_POLICY.silence_threshold_ms == 300
        assert AGGRESSIVE_POLICY.confidence_threshold == 0.6
        assert AGGRESSIVE_POLICY.enable_prosody is False

    def test_balanced_policy(self):
        """Balanced policy has moderate settings."""
        assert BALANCED_POLICY.name == "balanced"
        assert BALANCED_POLICY.silence_threshold_ms == 700
        assert BALANCED_POLICY.confidence_threshold == 0.75
        assert BALANCED_POLICY.enable_prosody is True

    def test_conservative_policy(self):
        """Conservative policy has slow settings."""
        assert CONSERVATIVE_POLICY.name == "conservative"
        assert CONSERVATIVE_POLICY.silence_threshold_ms == 1200
        assert CONSERVATIVE_POLICY.confidence_threshold == 0.85
        assert CONSERVATIVE_POLICY.enable_prosody is True

    def test_get_policy_aggressive(self):
        """Can get aggressive policy by name."""
        policy = get_policy("aggressive")
        assert policy == AGGRESSIVE_POLICY

    def test_get_policy_balanced(self):
        """Can get balanced policy by name."""
        policy = get_policy("balanced")
        assert policy == BALANCED_POLICY

    def test_get_policy_conservative(self):
        """Can get conservative policy by name."""
        policy = get_policy("conservative")
        assert policy == CONSERVATIVE_POLICY

    def test_get_policy_invalid(self):
        """Invalid policy name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown policy"):
            get_policy("invalid")


class TestTurnTakingEvaluator:
    """Tests for TurnTakingEvaluator class."""

    def test_too_short_turn(self):
        """Turn too short for policy returns not end."""
        evaluator = TurnTakingEvaluator(BALANCED_POLICY)
        result = evaluator.evaluate(
            text="Hi",
            silence_duration_ms=800,
            turn_duration_ms=200,  # Less than min_turn_duration_ms
        )
        assert result.is_end_of_turn is False
        assert result.confidence == 0.0

    def test_silence_threshold_detection(self):
        """Detects end based on silence threshold."""
        evaluator = TurnTakingEvaluator(BALANCED_POLICY)
        result = evaluator.evaluate(
            text="Hello there.",
            silence_duration_ms=800,  # Exceeds 700ms threshold
            turn_duration_ms=2000,
        )
        assert "SILENCE_THRESHOLD" in result.reason_codes
        assert result.confidence > 0.5

    def test_terminal_punctuation_detection(self):
        """Detects terminal punctuation."""
        evaluator = TurnTakingEvaluator(BALANCED_POLICY)
        result = evaluator.evaluate(
            text="This is a complete sentence.",
            silence_duration_ms=100,
            turn_duration_ms=2000,
        )
        assert "TERMINAL_PUNCT" in result.reason_codes

    def test_question_detection(self):
        """Detects questions."""
        evaluator = TurnTakingEvaluator(BALANCED_POLICY)
        result = evaluator.evaluate(
            text="How can I help you?",
            silence_duration_ms=100,
            turn_duration_ms=2000,
        )
        assert "TERMINAL_PUNCT" in result.reason_codes
        assert "QUESTION_DETECTED" in result.reason_codes

    def test_falling_intonation(self):
        """Detects falling intonation when prosody enabled."""
        evaluator = TurnTakingEvaluator(BALANCED_POLICY)
        result = evaluator.evaluate(
            text="I understand",
            silence_duration_ms=500,
            turn_duration_ms=2000,
            prosody_falling=True,
        )
        assert "FALLING_INTONATION" in result.reason_codes

    def test_prosody_disabled_ignores_intonation(self):
        """Ignores prosody when disabled."""
        evaluator = TurnTakingEvaluator(AGGRESSIVE_POLICY)
        result = evaluator.evaluate(
            text="I understand",
            silence_duration_ms=100,
            turn_duration_ms=2000,
            prosody_falling=True,
        )
        assert "FALLING_INTONATION" not in result.reason_codes

    def test_max_silence_forces_end(self):
        """Long silence forces turn end regardless of confidence."""
        evaluator = TurnTakingEvaluator(BALANCED_POLICY)
        result = evaluator.evaluate(
            text="",  # Empty text
            silence_duration_ms=3500,  # Exceeds max (3000ms)
            turn_duration_ms=5000,
        )
        assert result.is_end_of_turn is True
        assert "LONG_PAUSE" in result.reason_codes

    def test_complete_sentence_heuristic(self):
        """Detects complete sentences."""
        evaluator = TurnTakingEvaluator(BALANCED_POLICY)
        result = evaluator.evaluate(
            text="I would like to order a pizza",
            silence_duration_ms=0,
            turn_duration_ms=3000,
        )
        assert "COMPLETE_SENTENCE" in result.reason_codes

    def test_incomplete_sentence_not_detected(self):
        """Incomplete sentences not marked as complete."""
        evaluator = TurnTakingEvaluator(BALANCED_POLICY)
        result = evaluator.evaluate(
            text="I would like to",  # Ends with "to"
            silence_duration_ms=0,
            turn_duration_ms=2000,
        )
        assert "COMPLETE_SENTENCE" not in result.reason_codes


class TestPolicyComparison:
    """Tests comparing different policy behaviors."""

    def test_aggressive_detects_sooner(self):
        """Aggressive policy detects turn end sooner."""
        aggressive_eval = TurnTakingEvaluator(AGGRESSIVE_POLICY)
        conservative_eval = TurnTakingEvaluator(CONSERVATIVE_POLICY)

        # Short silence, some punctuation
        text = "Okay."
        silence_ms = 400  # Between aggressive (300) and conservative (1200)

        aggressive_result = aggressive_eval.evaluate(
            text=text,
            silence_duration_ms=silence_ms,
            turn_duration_ms=1000,
        )
        conservative_result = conservative_eval.evaluate(
            text=text,
            silence_duration_ms=silence_ms,
            turn_duration_ms=1000,
        )

        # Aggressive should have higher confidence or detect end
        assert aggressive_result.confidence >= conservative_result.confidence

    def test_conservative_needs_more_evidence(self):
        """Conservative policy requires more evidence."""
        balanced_eval = TurnTakingEvaluator(BALANCED_POLICY)
        conservative_eval = TurnTakingEvaluator(CONSERVATIVE_POLICY)

        result_balanced = balanced_eval.evaluate(
            text="Hello.",
            silence_duration_ms=750,
            turn_duration_ms=1500,
        )
        result_conservative = conservative_eval.evaluate(
            text="Hello.",
            silence_duration_ms=750,
            turn_duration_ms=1500,
        )

        # Balanced might detect end, conservative should not
        if result_balanced.is_end_of_turn:
            assert result_conservative.confidence < CONSERVATIVE_POLICY.confidence_threshold


class TestEndOfTurnEvaluation:
    """Tests for EndOfTurnEvaluation dataclass."""

    def test_to_dict(self):
        """Evaluation converts to dict correctly."""
        evaluation = EndOfTurnEvaluation(
            is_end_of_turn=True,
            confidence=0.85,
            reason_codes=["SILENCE_THRESHOLD", "TERMINAL_PUNCT"],
            signals=[
                EndOfTurnSignal(
                    code="SILENCE_THRESHOLD",
                    strength=0.7,
                    description="Silence detected",
                ),
            ],
            silence_duration_ms=800.0,
            policy_name="balanced",
        )

        data = evaluation.to_dict()

        assert data["is_end_of_turn"] is True
        assert data["confidence"] == 0.85
        assert "SILENCE_THRESHOLD" in data["reason_codes"]
        assert len(data["signals"]) == 1
        assert data["signals"][0]["code"] == "SILENCE_THRESHOLD"


class TestExtendedEndOfTurnHintPayload:
    """Tests for ExtendedEndOfTurnHintPayload."""

    def test_payload_attributes(self):
        """Payload has correct attributes."""
        payload = ExtendedEndOfTurnHintPayload(
            confidence=0.9,
            silence_duration=0.8,
            terminal_punctuation=True,
            partial_text="How can I help?",
            reason_codes=["TERMINAL_PUNCT", "QUESTION_DETECTED"],
            policy_name="balanced",
        )

        assert payload.confidence == 0.9
        assert payload.silence_duration == 0.8
        assert payload.terminal_punctuation is True
        assert "QUESTION_DETECTED" in payload.reason_codes

    def test_to_dict(self):
        """Payload converts to dict."""
        payload = ExtendedEndOfTurnHintPayload(
            confidence=0.85,
            silence_duration=0.5,
            terminal_punctuation=False,
            partial_text="Hello",
            reason_codes=["SILENCE_THRESHOLD"],
            policy_name="aggressive",
        )

        data = payload.to_dict()

        assert data["confidence"] == 0.85
        assert data["policy_name"] == "aggressive"
        assert data["reason_codes"] == ["SILENCE_THRESHOLD"]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self):
        """Handles empty text."""
        evaluator = TurnTakingEvaluator(BALANCED_POLICY)
        result = evaluator.evaluate(
            text="",
            silence_duration_ms=800,
            turn_duration_ms=1000,
        )
        # Should still detect silence
        assert "SILENCE_THRESHOLD" in result.reason_codes

    def test_whitespace_only_text(self):
        """Handles whitespace-only text."""
        evaluator = TurnTakingEvaluator(BALANCED_POLICY)
        result = evaluator.evaluate(
            text="   ",
            silence_duration_ms=800,
            turn_duration_ms=1000,
        )
        assert result is not None

    def test_very_long_text(self):
        """Handles very long text."""
        evaluator = TurnTakingEvaluator(BALANCED_POLICY)
        long_text = "This is a sentence. " * 100
        result = evaluator.evaluate(
            text=long_text,
            silence_duration_ms=500,
            turn_duration_ms=60000,
        )
        assert result is not None
        assert "TERMINAL_PUNCT" in result.reason_codes

    def test_custom_policy(self):
        """Can use fully custom policy."""
        custom = TurnTakingPolicy(
            name="custom",
            silence_threshold_ms=1000,
            confidence_threshold=0.5,
            enable_prosody=False,
            enable_punctuation=False,
            enable_sentence_completion=False,
        )
        evaluator = TurnTakingEvaluator(custom)

        result = evaluator.evaluate(
            text="Hello?",  # Question mark won't be detected
            silence_duration_ms=1100,
            turn_duration_ms=2000,
        )

        # Only silence should be detected
        assert "SILENCE_THRESHOLD" in result.reason_codes
        assert "TERMINAL_PUNCT" not in result.reason_codes

    def test_weights_affect_confidence(self):
        """Policy weights affect confidence calculation."""
        high_silence_weight = TurnTakingPolicy(
            name="custom",
            silence_threshold_ms=500,
            silence_weight=0.9,
            punctuation_weight=0.05,
            prosody_weight=0.05,
        )

        high_punct_weight = TurnTakingPolicy(
            name="custom",
            silence_threshold_ms=500,
            silence_weight=0.05,
            punctuation_weight=0.9,
            prosody_weight=0.05,
        )

        eval_silence = TurnTakingEvaluator(high_silence_weight)
        eval_punct = TurnTakingEvaluator(high_punct_weight)

        # Same input with silence but no punctuation
        result_silence = eval_silence.evaluate(
            text="Hello",  # No terminal punctuation
            silence_duration_ms=600,
            turn_duration_ms=2000,
        )

        result_punct = eval_punct.evaluate(
            text="Hello",
            silence_duration_ms=600,
            turn_duration_ms=2000,
        )

        # High silence weight should have higher confidence
        assert result_silence.confidence > result_punct.confidence
