"""Role inference for speaker classification.

This module provides speaker role classification (agent/customer/facilitator)
based on phrase triggers and turn patterns for call center and meeting
transcription use cases.

Features:
- Phrase triggers: "how can I help", "I'm calling about", etc.
- Turn patterns: who initiates, question density, response timing
- Probabilistic assignment with confidence scores
- Reversible labeling (original labels preserved)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

RoleType = Literal["agent", "customer", "facilitator", "participant", "unknown"]


@dataclass(slots=True)
class RoleAssignment:
    """Assignment of a role to a speaker.

    Attributes:
        speaker_id: The speaker identifier (e.g., "spk_0").
        role: The inferred role.
        confidence: Confidence score (0.0-1.0).
        evidence: List of evidence supporting the assignment.
        original_label: The original speaker label before assignment.
    """

    speaker_id: str
    role: RoleType
    confidence: float
    evidence: list[str] = field(default_factory=list)
    original_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "speaker_id": self.speaker_id,
            "role": self.role,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "original_label": self.original_label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoleAssignment:
        """Create from dictionary."""
        return cls(
            speaker_id=data.get("speaker_id", ""),
            role=data.get("role", "unknown"),
            confidence=data.get("confidence", 0.0),
            evidence=list(data.get("evidence", [])),
            original_label=data.get("original_label"),
        )


@dataclass(slots=True)
class SpeakerScores:
    """Accumulated scores for role inference.

    Attributes:
        agent_score: Accumulated evidence for agent role.
        customer_score: Accumulated evidence for customer role.
        facilitator_score: Accumulated evidence for facilitator role.
        total_turns: Number of turns from this speaker.
        questions_asked: Number of questions asked.
        first_turn_index: Index of first turn (for initiation detection).
        evidence: List of evidence strings.
    """

    agent_score: float = 0.0
    customer_score: float = 0.0
    facilitator_score: float = 0.0
    total_turns: int = 0
    questions_asked: int = 0
    first_turn_index: int = -1
    evidence: list[str] = field(default_factory=list)


# Agent trigger phrases (customer service representatives)
AGENT_TRIGGERS: list[tuple[re.Pattern[str], float, str]] = [
    (re.compile(r"\bhow can I help\b", re.IGNORECASE), 0.8, "greeting: 'how can I help'"),
    (re.compile(r"\bhow may I help\b", re.IGNORECASE), 0.8, "greeting: 'how may I help'"),
    (re.compile(r"\bhow may I assist\b", re.IGNORECASE), 0.8, "greeting: 'how may I assist'"),
    (re.compile(r"\bwhat can I do for you\b", re.IGNORECASE), 0.7, "greeting: 'what can I do for you'"),
    (re.compile(r"\bthank you for calling\b", re.IGNORECASE), 0.9, "greeting: 'thank you for calling'"),
    (re.compile(r"\bthank you for contacting\b", re.IGNORECASE), 0.9, "greeting: 'thank you for contacting'"),
    (re.compile(r"\bmy name is .* and I'll be\b", re.IGNORECASE), 0.85, "introduction with role"),
    (re.compile(r"\blet me check\b", re.IGNORECASE), 0.5, "service phrase: 'let me check'"),
    (re.compile(r"\blet me look that up\b", re.IGNORECASE), 0.5, "service phrase: 'let me look that up'"),
    (re.compile(r"\blet me verify\b", re.IGNORECASE), 0.5, "service phrase: 'let me verify'"),
    (re.compile(r"\bI can help you with that\b", re.IGNORECASE), 0.7, "service phrase: 'I can help'"),
    (re.compile(r"\bI'd be happy to\b", re.IGNORECASE), 0.4, "service phrase: 'I'd be happy to'"),
    (re.compile(r"\bfor security purposes\b", re.IGNORECASE), 0.6, "verification: 'for security purposes'"),
    (re.compile(r"\bcan I have your account\b", re.IGNORECASE), 0.7, "verification: request account"),
    (re.compile(r"\bverify your identity\b", re.IGNORECASE), 0.7, "verification: verify identity"),
    (re.compile(r"\bis there anything else\b", re.IGNORECASE), 0.6, "closing: 'anything else'"),
    (re.compile(r"\bwill there be anything else\b", re.IGNORECASE), 0.6, "closing: 'anything else'"),
    (re.compile(r"\bhave a great day\b", re.IGNORECASE), 0.3, "closing: 'have a great day'"),
]

# Customer trigger phrases
CUSTOMER_TRIGGERS: list[tuple[re.Pattern[str], float, str]] = [
    (re.compile(r"\bI'm calling about\b", re.IGNORECASE), 0.9, "intent: 'I'm calling about'"),
    (re.compile(r"\bI'm calling to\b", re.IGNORECASE), 0.8, "intent: 'I'm calling to'"),
    (re.compile(r"\bI need to\b", re.IGNORECASE), 0.4, "need: 'I need to'"),
    (re.compile(r"\bI need help with\b", re.IGNORECASE), 0.7, "need: 'I need help with'"),
    (re.compile(r"\bI want to\b", re.IGNORECASE), 0.3, "intent: 'I want to'"),
    (re.compile(r"\bcan you help me\b", re.IGNORECASE), 0.7, "request: 'can you help me'"),
    (re.compile(r"\bI have a question\b", re.IGNORECASE), 0.6, "inquiry: 'I have a question'"),
    (re.compile(r"\bI have a problem\b", re.IGNORECASE), 0.7, "problem: 'I have a problem'"),
    (re.compile(r"\bmy account\b", re.IGNORECASE), 0.4, "ownership: 'my account'"),
    (re.compile(r"\bmy order\b", re.IGNORECASE), 0.5, "ownership: 'my order'"),
    (re.compile(r"\bI received\b", re.IGNORECASE), 0.3, "experience: 'I received'"),
    (re.compile(r"\bI was charged\b", re.IGNORECASE), 0.6, "billing: 'I was charged'"),
    (re.compile(r"\bwhy am I\b", re.IGNORECASE), 0.5, "inquiry: 'why am I'"),
    (re.compile(r"\bwhy was I\b", re.IGNORECASE), 0.5, "inquiry: 'why was I'"),
    (re.compile(r"\bI didn't receive\b", re.IGNORECASE), 0.6, "problem: 'I didn't receive'"),
    (re.compile(r"\bI haven't received\b", re.IGNORECASE), 0.6, "problem: 'I haven't received'"),
]

# Facilitator trigger phrases (meeting moderators, interviewers)
FACILITATOR_TRIGGERS: list[tuple[re.Pattern[str], float, str]] = [
    (re.compile(r"\blet's move on to\b", re.IGNORECASE), 0.7, "moderation: 'let's move on to'"),
    (re.compile(r"\blet's discuss\b", re.IGNORECASE), 0.5, "moderation: 'let's discuss'"),
    (re.compile(r"\blet's talk about\b", re.IGNORECASE), 0.4, "moderation: 'let's talk about'"),
    (re.compile(r"\bnext agenda item\b", re.IGNORECASE), 0.8, "moderation: 'next agenda item'"),
    (re.compile(r"\bmoving on\b", re.IGNORECASE), 0.4, "moderation: 'moving on'"),
    (re.compile(r"\bwould anyone like to\b", re.IGNORECASE), 0.6, "facilitation: 'would anyone like to'"),
    (re.compile(r"\bdoes anyone have\b", re.IGNORECASE), 0.5, "facilitation: 'does anyone have'"),
    (re.compile(r"\blet's go around\b", re.IGNORECASE), 0.7, "facilitation: 'let's go around'"),
    (re.compile(r"\bto summarize\b", re.IGNORECASE), 0.5, "summary: 'to summarize'"),
    (re.compile(r"\bin summary\b", re.IGNORECASE), 0.5, "summary: 'in summary'"),
    (re.compile(r"\bwelcome everyone\b", re.IGNORECASE), 0.7, "opening: 'welcome everyone'"),
    (re.compile(r"\bthank you all for\b", re.IGNORECASE), 0.5, "opening: 'thank you all for'"),
]


@dataclass(slots=True)
class RoleInferenceConfig:
    """Configuration for role inference.

    Attributes:
        enabled: Whether role inference is enabled.
        context: Context type for inference ("call_center", "meeting", "interview").
        min_confidence: Minimum confidence threshold for assignment.
        use_turn_patterns: Use turn-taking patterns for inference.
        use_question_density: Use question density for inference.
        use_initiation: Use conversation initiation for inference.
        custom_agent_triggers: Additional agent trigger patterns.
        custom_customer_triggers: Additional customer trigger patterns.
    """

    enabled: bool = True
    context: Literal["call_center", "meeting", "interview", "general"] = "call_center"
    min_confidence: float = 0.3
    use_turn_patterns: bool = True
    use_question_density: bool = True
    use_initiation: bool = True
    custom_agent_triggers: list[tuple[str, float, str]] = field(default_factory=list)
    custom_customer_triggers: list[tuple[str, float, str]] = field(default_factory=list)


class RoleInferrer:
    """Speaker role inference engine.

    Analyzes transcript turns to infer speaker roles based on:
    - Phrase triggers (greeting patterns, service language)
    - Turn patterns (who responds vs initiates)
    - Question density (agents ask more verification questions)

    Example:
        >>> inferrer = RoleInferrer()
        >>> turns = [
        ...     {"speaker_id": "spk_0", "text": "Thank you for calling, how can I help?"},
        ...     {"speaker_id": "spk_1", "text": "I'm calling about my account"},
        ... ]
        >>> assignments = inferrer.infer_roles(turns)
        >>> print(assignments["spk_0"].role)
        "agent"
        >>> print(assignments["spk_1"].role)
        "customer"
    """

    def __init__(self, config: RoleInferenceConfig | None = None):
        """Initialize role inferrer.

        Args:
            config: Configuration for role inference.
        """
        self.config = config or RoleInferenceConfig()
        self._compiled_custom_triggers: dict[str, list[tuple[re.Pattern[str], float, str]]] = {}
        self._compile_custom_triggers()

    def _compile_custom_triggers(self) -> None:
        """Compile custom trigger patterns."""
        self._compiled_custom_triggers["agent"] = [
            (re.compile(pattern, re.IGNORECASE), score, evidence)
            for pattern, score, evidence in self.config.custom_agent_triggers
        ]
        self._compiled_custom_triggers["customer"] = [
            (re.compile(pattern, re.IGNORECASE), score, evidence)
            for pattern, score, evidence in self.config.custom_customer_triggers
        ]

    def infer_roles(
        self,
        turns: list[dict[str, Any]],
        speakers: list[dict[str, Any]] | None = None,
    ) -> dict[str, RoleAssignment]:
        """Infer roles for all speakers based on turns.

        Args:
            turns: List of turn dictionaries with speaker_id and text.
            speakers: Optional list of speaker metadata for label preservation.

        Returns:
            Dictionary mapping speaker_id to RoleAssignment.
        """
        if not self.config.enabled:
            return {}

        # Collect scores for each speaker
        speaker_scores: dict[str, SpeakerScores] = {}

        for turn_idx, turn in enumerate(turns):
            speaker_id = turn.get("speaker_id") or turn.get("speaker") or ""
            text = turn.get("text", "")

            if not speaker_id:
                continue

            if speaker_id not in speaker_scores:
                speaker_scores[speaker_id] = SpeakerScores()

            scores = speaker_scores[speaker_id]
            scores.total_turns += 1

            # Track first turn for initiation detection
            if scores.first_turn_index < 0:
                scores.first_turn_index = turn_idx

            # Count questions
            if self._count_questions(text) > 0:
                scores.questions_asked += self._count_questions(text)

            # Apply phrase triggers
            self._apply_triggers(text, scores)

        # Apply turn pattern heuristics
        if self.config.use_turn_patterns:
            self._apply_turn_patterns(speaker_scores, turns)

        # Apply initiation heuristics
        if self.config.use_initiation:
            self._apply_initiation_heuristics(speaker_scores, turns)

        # Get original labels if speakers provided
        original_labels: dict[str, str | None] = {}
        if speakers:
            for speaker in speakers:
                sid = speaker.get("id", "")
                original_labels[sid] = speaker.get("label")

        # Convert scores to assignments
        assignments: dict[str, RoleAssignment] = {}
        for speaker_id, scores in speaker_scores.items():
            assignment = self._scores_to_assignment(speaker_id, scores)
            assignment.original_label = original_labels.get(speaker_id)

            # Only include if meets confidence threshold
            if assignment.confidence >= self.config.min_confidence:
                assignments[speaker_id] = assignment
            else:
                # Assign unknown with low confidence
                assignments[speaker_id] = RoleAssignment(
                    speaker_id=speaker_id,
                    role="unknown",
                    confidence=assignment.confidence,
                    evidence=assignment.evidence,
                    original_label=original_labels.get(speaker_id),
                )

        return assignments

    def _apply_triggers(self, text: str, scores: SpeakerScores) -> None:
        """Apply phrase triggers to update scores.

        Args:
            text: Text to analyze.
            scores: Speaker scores to update.
        """
        # Agent triggers
        for pattern, score, evidence in AGENT_TRIGGERS:
            if pattern.search(text):
                scores.agent_score += score
                scores.evidence.append(f"agent:{evidence}")

        # Custom agent triggers
        for pattern, score, evidence in self._compiled_custom_triggers.get("agent", []):
            if pattern.search(text):
                scores.agent_score += score
                scores.evidence.append(f"agent:{evidence}")

        # Customer triggers
        for pattern, score, evidence in CUSTOMER_TRIGGERS:
            if pattern.search(text):
                scores.customer_score += score
                scores.evidence.append(f"customer:{evidence}")

        # Custom customer triggers
        for pattern, score, evidence in self._compiled_custom_triggers.get("customer", []):
            if pattern.search(text):
                scores.customer_score += score
                scores.evidence.append(f"customer:{evidence}")

        # Facilitator triggers
        for pattern, score, evidence in FACILITATOR_TRIGGERS:
            if pattern.search(text):
                scores.facilitator_score += score
                scores.evidence.append(f"facilitator:{evidence}")

    def _apply_turn_patterns(
        self,
        speaker_scores: dict[str, SpeakerScores],
        turns: list[dict[str, Any]],
    ) -> None:
        """Apply turn-taking pattern heuristics.

        In call center context:
        - Agents typically respond more than initiate
        - Agents ask verification questions early
        - Customers explain problems at length

        Args:
            speaker_scores: Speaker scores to update.
            turns: List of turns for analysis.
        """
        if len(turns) < 3:
            return

        # Analyze response patterns
        for i, turn in enumerate(turns[1:], 1):
            speaker_id = turn.get("speaker_id", "")
            prev_speaker = turns[i - 1].get("speaker_id", "")

            if speaker_id not in speaker_scores:
                continue

            # Check if this is a response (different speaker than previous)
            if speaker_id != prev_speaker:
                # Quick responses might indicate agent (ready to help)
                # Long responses might indicate customer (explaining problem)
                text = turn.get("text", "")
                if len(text) < 50:
                    # Short response - slight agent bias
                    speaker_scores[speaker_id].agent_score += 0.1
                elif len(text) > 200:
                    # Long response - slight customer bias
                    speaker_scores[speaker_id].customer_score += 0.1

    def _apply_initiation_heuristics(
        self,
        speaker_scores: dict[str, SpeakerScores],
        turns: list[dict[str, Any]],
    ) -> None:
        """Apply conversation initiation heuristics.

        In call center context:
        - Agents typically speak first (greeting)
        - In meetings, facilitators typically speak first

        Args:
            speaker_scores: Speaker scores to update.
            turns: List of turns for analysis.
        """
        if not turns:
            return

        # First speaker analysis
        first_speaker = turns[0].get("speaker_id", "")
        if first_speaker in speaker_scores:
            if self.config.context == "call_center":
                # In call centers, first speaker is likely agent
                speaker_scores[first_speaker].agent_score += 0.3
                speaker_scores[first_speaker].evidence.append("pattern:first_speaker")
            elif self.config.context == "meeting":
                # In meetings, first speaker is likely facilitator
                speaker_scores[first_speaker].facilitator_score += 0.3
                speaker_scores[first_speaker].evidence.append("pattern:first_speaker")

    def _count_questions(self, text: str) -> int:
        """Count questions in text.

        Args:
            text: Text to analyze.

        Returns:
            Number of questions detected.
        """
        # Simple heuristic: count question marks
        return text.count("?")

    def _scores_to_assignment(
        self,
        speaker_id: str,
        scores: SpeakerScores,
    ) -> RoleAssignment:
        """Convert accumulated scores to role assignment.

        Args:
            speaker_id: Speaker identifier.
            scores: Accumulated scores.

        Returns:
            RoleAssignment with determined role and confidence.
        """
        # Normalize scores
        total = scores.agent_score + scores.customer_score + scores.facilitator_score

        if total == 0:
            return RoleAssignment(
                speaker_id=speaker_id,
                role="unknown",
                confidence=0.0,
                evidence=list(scores.evidence),
            )

        agent_prob = scores.agent_score / total
        customer_prob = scores.customer_score / total
        facilitator_prob = scores.facilitator_score / total

        # Determine winning role
        max_prob = max(agent_prob, customer_prob, facilitator_prob)

        if max_prob == agent_prob:
            role: RoleType = "agent"
            confidence = agent_prob
        elif max_prob == customer_prob:
            role = "customer"
            confidence = customer_prob
        else:
            role = "facilitator"
            confidence = facilitator_prob

        # Adjust confidence based on evidence strength
        if len(scores.evidence) < 2:
            confidence *= 0.5  # Reduce confidence with little evidence

        return RoleAssignment(
            speaker_id=speaker_id,
            role=role,
            confidence=min(confidence, 1.0),
            evidence=list(scores.evidence),
        )

    def update_speakers(
        self,
        speakers: list[dict[str, Any]],
        assignments: dict[str, RoleAssignment],
        preserve_original: bool = True,
    ) -> list[dict[str, Any]]:
        """Update speaker list with inferred roles.

        Args:
            speakers: Original speaker list.
            assignments: Role assignments from infer_roles().
            preserve_original: If True, store original label in _original_label.

        Returns:
            Updated speaker list with role labels.
        """
        updated = []
        for speaker in speakers:
            speaker_copy = dict(speaker)
            sid = speaker.get("id", "")

            if sid in assignments:
                assignment = assignments[sid]
                if preserve_original and "label" in speaker_copy:
                    speaker_copy["_original_label"] = speaker_copy["label"]
                speaker_copy["label"] = assignment.role
                speaker_copy["role_confidence"] = assignment.confidence
                speaker_copy["role_evidence"] = assignment.evidence

            updated.append(speaker_copy)

        return updated


def infer_roles(turns: list[dict[str, Any]], **config_kwargs: Any) -> dict[str, RoleAssignment]:
    """Convenience function for role inference.

    Args:
        turns: List of turn dictionaries.
        **config_kwargs: Configuration options passed to RoleInferenceConfig.

    Returns:
        Dictionary mapping speaker_id to RoleAssignment.
    """
    config = RoleInferenceConfig(**config_kwargs)
    inferrer = RoleInferrer(config)
    return inferrer.infer_roles(turns)


__all__ = [
    "RoleInferrer",
    "RoleInferenceConfig",
    "RoleAssignment",
    "RoleType",
    "infer_roles",
    "AGENT_TRIGGERS",
    "CUSTOMER_TRIGGERS",
    "FACILITATOR_TRIGGERS",
]
