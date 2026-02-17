"""Turn-taking policy profiles for END_OF_TURN_HINT behavior.

This module provides configurable turn-taking policies that control
when END_OF_TURN_HINT events are emitted during streaming transcription.

Three preset profiles:
- Aggressive: Fast response, lower confidence (300ms silence, 0.6 confidence)
- Balanced: Default behavior (700ms silence, 0.75 confidence)
- Conservative: High accuracy (1200ms silence, 0.85 confidence)

Each policy specifies silence thresholds, confidence requirements,
and which signals contribute to turn-end detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PolicyName = Literal["aggressive", "balanced", "conservative", "custom"]
ReasonCode = Literal[
    "SILENCE_THRESHOLD",
    "TERMINAL_PUNCT",
    "FALLING_INTONATION",
    "COMPLETE_SENTENCE",
    "QUESTION_DETECTED",
    "LONG_PAUSE",
]


@dataclass(slots=True)
class TurnTakingPolicy:
    """Configuration for turn-taking behavior.

    Attributes:
        name: Policy name for identification.
        silence_threshold_ms: Minimum silence duration to consider turn end (ms).
        confidence_threshold: Minimum confidence score to emit hint (0.0-1.0).
        enable_prosody: Use prosodic features (falling intonation).
        enable_punctuation: Use terminal punctuation detection.
        enable_sentence_completion: Use sentence completion heuristics.
        min_turn_duration_ms: Minimum turn duration before considering end.
        max_silence_for_continuation_ms: Maximum silence before forcing end.
        prosody_weight: Weight for prosodic signals in confidence calculation.
        punctuation_weight: Weight for punctuation signals.
        silence_weight: Weight for silence duration signals.
    """

    name: PolicyName = "balanced"
    silence_threshold_ms: int = 700
    confidence_threshold: float = 0.75
    enable_prosody: bool = True
    enable_punctuation: bool = True
    enable_sentence_completion: bool = True
    min_turn_duration_ms: int = 500
    max_silence_for_continuation_ms: int = 3000
    prosody_weight: float = 0.25
    punctuation_weight: float = 0.35
    silence_weight: float = 0.40

    def to_dict(self) -> dict[str, Any]:
        """Convert policy to dictionary."""
        return {
            "name": self.name,
            "silence_threshold_ms": self.silence_threshold_ms,
            "confidence_threshold": self.confidence_threshold,
            "enable_prosody": self.enable_prosody,
            "enable_punctuation": self.enable_punctuation,
            "enable_sentence_completion": self.enable_sentence_completion,
            "min_turn_duration_ms": self.min_turn_duration_ms,
            "max_silence_for_continuation_ms": self.max_silence_for_continuation_ms,
            "prosody_weight": self.prosody_weight,
            "punctuation_weight": self.punctuation_weight,
            "silence_weight": self.silence_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnTakingPolicy:
        """Create policy from dictionary."""
        return cls(
            name=data.get("name", "custom"),
            silence_threshold_ms=data.get("silence_threshold_ms", 700),
            confidence_threshold=data.get("confidence_threshold", 0.75),
            enable_prosody=data.get("enable_prosody", True),
            enable_punctuation=data.get("enable_punctuation", True),
            enable_sentence_completion=data.get("enable_sentence_completion", True),
            min_turn_duration_ms=data.get("min_turn_duration_ms", 500),
            max_silence_for_continuation_ms=data.get("max_silence_for_continuation_ms", 3000),
            prosody_weight=data.get("prosody_weight", 0.25),
            punctuation_weight=data.get("punctuation_weight", 0.35),
            silence_weight=data.get("silence_weight", 0.40),
        )


# Preset policies
AGGRESSIVE_POLICY = TurnTakingPolicy(
    name="aggressive",
    silence_threshold_ms=300,
    confidence_threshold=0.6,
    enable_prosody=False,  # Skip prosody for speed
    enable_punctuation=True,
    enable_sentence_completion=False,  # Skip for speed
    min_turn_duration_ms=300,
    max_silence_for_continuation_ms=1500,
    prosody_weight=0.0,
    punctuation_weight=0.40,
    silence_weight=0.60,
)

BALANCED_POLICY = TurnTakingPolicy(
    name="balanced",
    silence_threshold_ms=700,
    confidence_threshold=0.75,
    enable_prosody=True,
    enable_punctuation=True,
    enable_sentence_completion=True,
    min_turn_duration_ms=500,
    max_silence_for_continuation_ms=3000,
    prosody_weight=0.25,
    punctuation_weight=0.35,
    silence_weight=0.40,
)

CONSERVATIVE_POLICY = TurnTakingPolicy(
    name="conservative",
    silence_threshold_ms=1200,
    confidence_threshold=0.85,
    enable_prosody=True,
    enable_punctuation=True,
    enable_sentence_completion=True,
    min_turn_duration_ms=800,
    max_silence_for_continuation_ms=5000,
    prosody_weight=0.30,
    punctuation_weight=0.30,
    silence_weight=0.40,
)


def get_policy(name: PolicyName | str) -> TurnTakingPolicy:
    """Get a turn-taking policy by name.

    Args:
        name: Policy name ("aggressive", "balanced", "conservative").

    Returns:
        TurnTakingPolicy instance.

    Raises:
        ValueError: If policy name is unknown.
    """
    policies = {
        "aggressive": AGGRESSIVE_POLICY,
        "balanced": BALANCED_POLICY,
        "conservative": CONSERVATIVE_POLICY,
    }

    if name in policies:
        return policies[name]

    raise ValueError(f"Unknown policy: {name}. Available: {list(policies.keys())}")


@dataclass(slots=True)
class EndOfTurnSignal:
    """Signal contributing to end-of-turn detection.

    Attributes:
        code: Reason code for this signal.
        strength: Signal strength (0.0-1.0).
        description: Human-readable description.
    """

    code: ReasonCode
    strength: float
    description: str


@dataclass(slots=True)
class EndOfTurnEvaluation:
    """Result of turn-end evaluation.

    Attributes:
        is_end_of_turn: Whether turn end is detected.
        confidence: Confidence score (0.0-1.0).
        reason_codes: List of reason codes that contributed.
        signals: Detailed signals that contributed.
        silence_duration_ms: Current silence duration.
        policy_name: Name of policy used.
    """

    is_end_of_turn: bool
    confidence: float
    reason_codes: list[ReasonCode]
    signals: list[EndOfTurnSignal] = field(default_factory=list)
    silence_duration_ms: float = 0.0
    policy_name: str = "balanced"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_end_of_turn": self.is_end_of_turn,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "signals": [
                {"code": s.code, "strength": s.strength, "description": s.description}
                for s in self.signals
            ],
            "silence_duration_ms": self.silence_duration_ms,
            "policy_name": self.policy_name,
        }


class TurnTakingEvaluator:
    """Evaluates turn-end conditions based on policy.

    This class applies a turn-taking policy to evaluate whether
    a turn has ended, computing confidence scores based on
    silence duration, punctuation, and prosodic features.

    Example:
        >>> policy = get_policy("balanced")
        >>> evaluator = TurnTakingEvaluator(policy)
        >>> result = evaluator.evaluate(
        ...     text="How can I help you?",
        ...     silence_duration_ms=800,
        ...     turn_duration_ms=2000,
        ... )
        >>> print(result.is_end_of_turn)
        True
        >>> print(result.reason_codes)
        ["TERMINAL_PUNCT", "SILENCE_THRESHOLD", "QUESTION_DETECTED"]
    """

    def __init__(self, policy: TurnTakingPolicy | None = None):
        """Initialize evaluator with policy.

        Args:
            policy: Turn-taking policy. Uses BALANCED_POLICY if None.
        """
        self.policy = policy or BALANCED_POLICY

    def evaluate(
        self,
        text: str,
        silence_duration_ms: float,
        turn_duration_ms: float,
        prosody_falling: bool | None = None,
        prosody_flat: bool | None = None,
    ) -> EndOfTurnEvaluation:
        """Evaluate whether turn has ended.

        Args:
            text: Current text in the turn.
            silence_duration_ms: Duration of trailing silence (ms).
            turn_duration_ms: Total turn duration so far (ms).
            prosody_falling: True if prosody indicates falling intonation.
            prosody_flat: True if prosody indicates flat/level intonation.

        Returns:
            EndOfTurnEvaluation with detection result and confidence.
        """
        signals: list[EndOfTurnSignal] = []
        reason_codes: list[ReasonCode] = []

        # Check minimum turn duration
        if turn_duration_ms < self.policy.min_turn_duration_ms:
            return EndOfTurnEvaluation(
                is_end_of_turn=False,
                confidence=0.0,
                reason_codes=[],
                signals=[],
                silence_duration_ms=silence_duration_ms,
                policy_name=self.policy.name,
            )

        # Evaluate silence
        silence_signal = self._evaluate_silence(silence_duration_ms)
        if silence_signal:
            signals.append(silence_signal)
            reason_codes.append(silence_signal.code)

        # Evaluate punctuation
        if self.policy.enable_punctuation:
            punct_signals = self._evaluate_punctuation(text)
            for sig in punct_signals:
                signals.append(sig)
                if sig.code not in reason_codes:
                    reason_codes.append(sig.code)

        # Evaluate prosody
        if self.policy.enable_prosody and prosody_falling is not None:
            prosody_signal = self._evaluate_prosody(prosody_falling, prosody_flat)
            if prosody_signal:
                signals.append(prosody_signal)
                reason_codes.append(prosody_signal.code)

        # Evaluate sentence completion
        if self.policy.enable_sentence_completion:
            sentence_signal = self._evaluate_sentence_completion(text)
            if sentence_signal:
                signals.append(sentence_signal)
                if sentence_signal.code not in reason_codes:
                    reason_codes.append(sentence_signal.code)

        # Calculate weighted confidence
        confidence = self._calculate_confidence(signals)

        # Determine if turn has ended
        is_end = (
            confidence >= self.policy.confidence_threshold
            or silence_duration_ms >= self.policy.max_silence_for_continuation_ms
        )

        # Add LONG_PAUSE if forced by max silence
        if (
            silence_duration_ms >= self.policy.max_silence_for_continuation_ms
            and "LONG_PAUSE" not in reason_codes
        ):
            reason_codes.append("LONG_PAUSE")

        return EndOfTurnEvaluation(
            is_end_of_turn=is_end,
            confidence=confidence,
            reason_codes=reason_codes,
            signals=signals,
            silence_duration_ms=silence_duration_ms,
            policy_name=self.policy.name,
        )

    def _evaluate_silence(self, silence_duration_ms: float) -> EndOfTurnSignal | None:
        """Evaluate silence duration signal.

        Args:
            silence_duration_ms: Duration of silence in milliseconds.

        Returns:
            EndOfTurnSignal if silence meets threshold.
        """
        if silence_duration_ms >= self.policy.silence_threshold_ms:
            # Calculate strength based on how far past threshold
            base_strength = 0.6
            extra_strength = min(
                0.4,
                (silence_duration_ms - self.policy.silence_threshold_ms)
                / self.policy.silence_threshold_ms
                * 0.4,
            )
            strength = base_strength + extra_strength

            return EndOfTurnSignal(
                code="SILENCE_THRESHOLD",
                strength=strength,
                description=f"Silence of {silence_duration_ms:.0f}ms exceeds threshold",
            )

        return None

    def _evaluate_punctuation(self, text: str) -> list[EndOfTurnSignal]:
        """Evaluate punctuation-based signals.

        Args:
            text: Text to evaluate.

        Returns:
            List of punctuation-related signals.
        """
        signals: list[EndOfTurnSignal] = []
        text = text.strip()

        if not text:
            return signals

        # Check terminal punctuation
        if text.endswith((".", "!", ";")):
            signals.append(
                EndOfTurnSignal(
                    code="TERMINAL_PUNCT",
                    strength=0.8,
                    description="Terminal punctuation detected",
                )
            )

        # Check question mark
        if text.endswith("?"):
            signals.append(
                EndOfTurnSignal(
                    code="TERMINAL_PUNCT",
                    strength=0.9,
                    description="Question mark detected",
                )
            )
            signals.append(
                EndOfTurnSignal(
                    code="QUESTION_DETECTED",
                    strength=0.7,
                    description="Question detected",
                )
            )

        return signals

    def _evaluate_prosody(
        self,
        prosody_falling: bool,
        prosody_flat: bool | None,
    ) -> EndOfTurnSignal | None:
        """Evaluate prosodic signals.

        Args:
            prosody_falling: True if falling intonation detected.
            prosody_flat: True if flat intonation detected.

        Returns:
            EndOfTurnSignal if prosody indicates turn end.
        """
        if prosody_falling:
            return EndOfTurnSignal(
                code="FALLING_INTONATION",
                strength=0.7,
                description="Falling intonation detected",
            )

        return None

    def _evaluate_sentence_completion(self, text: str) -> EndOfTurnSignal | None:
        """Evaluate sentence completion heuristics.

        Simple heuristic: sentences typically have subject-verb structure
        and reasonable length.

        Args:
            text: Text to evaluate.

        Returns:
            EndOfTurnSignal if text appears to be complete sentence.
        """
        text = text.strip()
        words = text.split()

        # Very short utterances are likely incomplete
        if len(words) < 3:
            return None

        # Check for common incomplete patterns
        incomplete_endings = (
            "the",
            "a",
            "an",
            "and",
            "but",
            "or",
            "to",
            "of",
            "for",
            "with",
            "is",
            "are",
            "was",
            "were",
            "will",
            "would",
            "could",
            "should",
            "that",
            "which",
            "who",
        )

        last_word = words[-1].lower().rstrip(".,!?;:")
        if last_word in incomplete_endings:
            return None

        # If ends with common sentence-ending patterns
        if len(words) >= 4:
            return EndOfTurnSignal(
                code="COMPLETE_SENTENCE",
                strength=0.5,
                description="Sentence appears complete",
            )

        return None

    def _calculate_confidence(self, signals: list[EndOfTurnSignal]) -> float:
        """Calculate weighted confidence from signals.

        Args:
            signals: List of detected signals.

        Returns:
            Confidence score (0.0-1.0).
        """
        if not signals:
            return 0.0

        # Group signals by category
        silence_strength = 0.0
        punct_strength = 0.0
        prosody_strength = 0.0

        for signal in signals:
            if signal.code == "SILENCE_THRESHOLD" or signal.code == "LONG_PAUSE":
                silence_strength = max(silence_strength, signal.strength)
            elif signal.code in ("TERMINAL_PUNCT", "QUESTION_DETECTED", "COMPLETE_SENTENCE"):
                punct_strength = max(punct_strength, signal.strength)
            elif signal.code == "FALLING_INTONATION":
                prosody_strength = max(prosody_strength, signal.strength)

        # Calculate weighted average
        total_weight = (
            self.policy.silence_weight + self.policy.punctuation_weight + self.policy.prosody_weight
        )

        if total_weight == 0:
            return 0.0

        weighted_sum = (
            silence_strength * self.policy.silence_weight
            + punct_strength * self.policy.punctuation_weight
            + prosody_strength * self.policy.prosody_weight
        )

        return min(1.0, weighted_sum / total_weight)


@dataclass(slots=True)
class ExtendedEndOfTurnHintPayload:
    """Extended payload for end-of-turn hints with reason codes.

    This extends the base EndOfTurnHintPayload with additional
    policy-specific information.

    Attributes:
        confidence: Confidence score (0.0-1.0).
        silence_duration: Duration of trailing silence (seconds).
        terminal_punctuation: Whether terminal punctuation detected.
        partial_text: The text that triggered the hint.
        reason_codes: List of reasons for the hint.
        policy_name: Name of the policy used.
        signals: Detailed signal information.
    """

    confidence: float
    silence_duration: float
    terminal_punctuation: bool
    partial_text: str
    reason_codes: list[ReasonCode] = field(default_factory=list)
    policy_name: str = "balanced"
    signals: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "confidence": self.confidence,
            "silence_duration": self.silence_duration,
            "terminal_punctuation": self.terminal_punctuation,
            "partial_text": self.partial_text,
            "reason_codes": list(self.reason_codes),
            "policy_name": self.policy_name,
            "signals": list(self.signals),
        }


__all__ = [
    "TurnTakingPolicy",
    "TurnTakingEvaluator",
    "EndOfTurnEvaluation",
    "EndOfTurnSignal",
    "ExtendedEndOfTurnHintPayload",
    "PolicyName",
    "ReasonCode",
    "get_policy",
    "AGGRESSIVE_POLICY",
    "BALANCED_POLICY",
    "CONSERVATIVE_POLICY",
]
