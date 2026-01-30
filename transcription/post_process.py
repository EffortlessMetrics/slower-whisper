"""Unified post-processing orchestration for transcript enrichment.

This module provides a unified configuration and orchestration layer for
running all post-ASR processors on finalized segments and turns:

- Safety processing (PII + moderation + formatting)
- Role inference (agent/customer/facilitator)
- Topic segmentation
- Turn-taking policy evaluation
- Environment classification
- Extended prosody analysis

The PostProcessor runs enabled processors in dependency order and
attaches results to segments/turns without mutating the original text.

Example:
    >>> config = PostProcessConfig(
    ...     enable_safety=True,
    ...     enable_roles=True,
    ...     enable_topics=True,
    ... )
    >>> processor = PostProcessor(config)
    >>> # Process a segment
    >>> result = processor.process_segment(segment_ctx)
    >>> # Process accumulated turns
    >>> processor.process_turn(turn_dict)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .environment_classifier import (
    AudioHealthMetrics,
    EnvironmentClassifier,
    EnvironmentClassifierConfig,
    EnvironmentState,
)
from .role_inference import RoleAssignment, RoleInferenceConfig, RoleInferrer
from .safety_config import SafetyConfig
from .safety_layer import SafetyProcessingResult, SafetyProcessor
from .streaming_callbacks import EndOfTurnHintPayload, RoleAssignedPayload
from .topic_segmentation import (
    StreamingTopicSegmenter,
    TopicBoundaryPayload,
    TopicChunk,
    TopicSegmentationConfig,
)
from .turn_taking_policy import (
    EndOfTurnEvaluation,
    TurnTakingEvaluator,
    TurnTakingPolicy,
    get_policy,
)

if TYPE_CHECKING:
    import numpy as np

    from .enrichment_config import EnrichmentConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SegmentContext:
    """Context for processing a single segment.

    Provides all information needed by segment processors.

    Attributes:
        session_id: Unique session identifier.
        segment_id: Segment identifier within session.
        speaker_id: Speaker identifier (if available).
        start: Start time in seconds.
        end: End time in seconds.
        text: Segment text content.
        words: Optional word-level timing information.
        audio_state: Optional existing audio state (prosody/emotion).
        transcript_state: Optional global transcript state.
    """

    session_id: str
    segment_id: str
    speaker_id: str | None
    start: float
    end: float
    text: str
    words: list[dict[str, Any]] | None = None
    audio_state: dict[str, Any] | None = None
    transcript_state: dict[str, Any] | None = None


@runtime_checkable
class SegmentProcessor(Protocol):
    """Protocol for segment processors.

    Implement this to create custom segment processors that can be
    registered with the PostProcessor.
    """

    def process(self, ctx: SegmentContext) -> dict[str, Any]:
        """Process a segment and return results.

        Args:
            ctx: Segment context with all available information.

        Returns:
            Dictionary with processor-specific results.
        """
        ...


@dataclass(slots=True)
class PostProcessConfig:
    """Unified configuration for post-processing.

    Controls which post-processors are enabled and their configurations.
    All features are disabled by default.

    Attributes:
        enabled: Master switch for post-processing.
        enable_safety: Run safety processing (PII + moderation + formatting).
        enable_roles: Run role inference on speakers.
        enable_topics: Run topic segmentation.
        enable_turn_taking: Run turn-taking policy evaluation.
        enable_environment: Run environment classification.
        enable_prosody_extended: Run extended prosody analysis.

        safety_config: Configuration for safety processing.
        role_config: Configuration for role inference.
        topic_config: Configuration for topic segmentation.
        turn_taking_policy: Turn-taking policy name or custom policy.
        environment_config: Configuration for environment classification.

        role_decision_turns: Number of turns before role decision.
        role_decision_seconds: Seconds before role decision.
    """

    enabled: bool = True

    # Feature toggles
    enable_safety: bool = False
    enable_roles: bool = False
    enable_topics: bool = False
    enable_turn_taking: bool = False
    enable_environment: bool = False
    enable_prosody_extended: bool = False

    # Nested configurations
    safety_config: SafetyConfig | None = None
    role_config: RoleInferenceConfig | None = None
    topic_config: TopicSegmentationConfig | None = None
    turn_taking_policy: str | TurnTakingPolicy = "balanced"
    environment_config: EnvironmentClassifierConfig | None = None

    # Role inference timing
    role_decision_turns: int = 5
    role_decision_seconds: float = 30.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "enable_safety": self.enable_safety,
            "enable_roles": self.enable_roles,
            "enable_topics": self.enable_topics,
            "enable_turn_taking": self.enable_turn_taking,
            "enable_environment": self.enable_environment,
            "enable_prosody_extended": self.enable_prosody_extended,
            "role_decision_turns": self.role_decision_turns,
            "role_decision_seconds": self.role_decision_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PostProcessConfig:
        """Create from dictionary."""
        config = cls(
            enabled=data.get("enabled", True),
            enable_safety=data.get("enable_safety", False),
            enable_roles=data.get("enable_roles", False),
            enable_topics=data.get("enable_topics", False),
            enable_turn_taking=data.get("enable_turn_taking", False),
            enable_environment=data.get("enable_environment", False),
            enable_prosody_extended=data.get("enable_prosody_extended", False),
            role_decision_turns=data.get("role_decision_turns", 5),
            role_decision_seconds=data.get("role_decision_seconds", 30.0),
        )

        if "safety_config" in data and data["safety_config"]:
            config.safety_config = SafetyConfig.from_dict(data["safety_config"])

        return config

    @classmethod
    def from_enrichment_config(cls, config: EnrichmentConfig) -> PostProcessConfig | None:
        """Create PostProcessConfig from EnrichmentConfig.

        This is a convenience method that delegates to EnrichmentConfig.to_post_process_config().
        It provides an alternative entry point for creating PostProcessConfig from enrichment settings.

        Args:
            config: EnrichmentConfig instance with enrichment settings.

        Returns:
            PostProcessConfig if any post-processing features are enabled,
            None otherwise.
        """
        return config.to_post_process_config()


@dataclass(slots=True)
class PostProcessResult:
    """Result of post-processing a segment.

    Attributes:
        safety: Safety processing result (if enabled).
        environment: Environment classification (if enabled).
        prosody_extended: Extended prosody analysis (if enabled).
        end_of_turn: Turn-taking evaluation (if enabled).
    """

    safety: SafetyProcessingResult | None = None
    environment: EnvironmentState | None = None
    prosody_extended: dict[str, Any] | None = None
    end_of_turn: EndOfTurnEvaluation | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for segment audio_state."""
        result: dict[str, Any] = {}

        if self.safety:
            result["safety"] = self.safety.to_safety_state()

        if self.environment:
            result["environment"] = self.environment.to_dict()

        if self.prosody_extended:
            result["prosody_extended"] = self.prosody_extended

        if self.end_of_turn:
            result["turn_taking"] = self.end_of_turn.to_dict()

        return result


@dataclass(slots=True)
class TurnProcessResult:
    """Result of post-processing a turn.

    Attributes:
        topic_boundary: Topic boundary detected (if any).
        keywords: Keywords extracted from turn.
    """

    topic_boundary: TopicBoundaryPayload | None = None
    keywords: list[str] = field(default_factory=list)


class PostProcessor:
    """Unified post-processor orchestrating all enrichment processors.

    Runs enabled processors in dependency order:
    1. Safety (PII + moderation + formatting)
    2. Environment classification
    3. Extended prosody
    4. Turn-taking evaluation

    Turn-level processors (roles, topics) run separately via process_turn().

    Example:
        >>> config = PostProcessConfig(enable_safety=True)
        >>> processor = PostProcessor(config)
        >>>
        >>> # Process segments
        >>> ctx = SegmentContext(
        ...     session_id="sess_1",
        ...     segment_id="seg_0",
        ...     speaker_id="spk_0",
        ...     start=0.0,
        ...     end=2.5,
        ...     text="Hello, how can I help you today?",
        ... )
        >>> result = processor.process_segment(ctx)
        >>>
        >>> # Process turns for roles/topics
        >>> turn = {"id": "turn_0", "speaker_id": "spk_0", "text": "...", "start": 0, "end": 5}
        >>> processor.process_turn(turn)
        >>> roles = processor.get_role_assignments()
    """

    def __init__(
        self,
        config: PostProcessConfig | None = None,
        on_safety_alert: Any | None = None,
        on_role_assigned: Any | None = None,
        on_topic_boundary: Any | None = None,
        on_end_of_turn_hint: Any | None = None,
    ):
        """Initialize post-processor.

        Args:
            config: Post-processing configuration.
            on_safety_alert: Callback for safety alerts.
            on_role_assigned: Callback for role assignments.
            on_topic_boundary: Callback for topic boundaries.
            on_end_of_turn_hint: Callback for end-of-turn hints.
        """
        self.config = config or PostProcessConfig()
        self._on_safety_alert = on_safety_alert
        self._on_role_assigned = on_role_assigned
        self._on_topic_boundary = on_topic_boundary
        self._on_end_of_turn_hint = on_end_of_turn_hint

        # Initialize processors based on config
        self._safety_processor: SafetyProcessor | None = None
        self._role_inferrer: RoleInferrer | None = None
        self._topic_segmenter: StreamingTopicSegmenter | None = None
        self._turn_taking_evaluator: TurnTakingEvaluator | None = None
        self._environment_classifier: EnvironmentClassifier | None = None

        # State tracking
        self._turns: list[dict[str, Any]] = []
        self._role_assignments: dict[str, RoleAssignment] = {}
        self._roles_decided = False
        self._start_time: float | None = None
        self._last_end_time: float = 0.0
        self._finalized = False

        self._initialize_processors()

    def _initialize_processors(self) -> None:
        """Initialize enabled processors."""
        if not self.config.enabled:
            return

        if self.config.enable_safety:
            safety_config = self.config.safety_config or SafetyConfig(enabled=True)
            # Ensure safety is actually enabled
            if not safety_config.enabled:
                safety_config.enabled = True
            self._safety_processor = SafetyProcessor(safety_config)

        if self.config.enable_roles:
            role_config = self.config.role_config or RoleInferenceConfig()
            self._role_inferrer = RoleInferrer(role_config)

        if self.config.enable_topics:
            topic_config = self.config.topic_config or TopicSegmentationConfig()
            self._topic_segmenter = StreamingTopicSegmenter(
                topic_config,
                on_boundary=self._handle_topic_boundary,
            )

        if self.config.enable_turn_taking:
            if isinstance(self.config.turn_taking_policy, str):
                policy = get_policy(self.config.turn_taking_policy)
            else:
                policy = self.config.turn_taking_policy
            self._turn_taking_evaluator = TurnTakingEvaluator(policy)

        if self.config.enable_environment:
            env_config = self.config.environment_config or EnvironmentClassifierConfig()
            self._environment_classifier = EnvironmentClassifier(env_config)

    def process_segment(
        self,
        ctx: SegmentContext,
        audio: np.ndarray | None = None,
        sr: int = 16000,
        silence_duration_ms: float = 0.0,
        turn_duration_ms: float = 0.0,
    ) -> PostProcessResult:
        """Process a finalized segment through enabled processors.

        Args:
            ctx: Segment context with text and metadata.
            audio: Optional audio array for prosody extraction.
            sr: Sample rate for audio.
            silence_duration_ms: Trailing silence for turn-taking.
            turn_duration_ms: Current turn duration for turn-taking.

        Returns:
            PostProcessResult with all processor outputs.
        """
        if not self.config.enabled:
            return PostProcessResult()

        result = PostProcessResult()

        # Track start time for role decision timing
        if self._start_time is None:
            self._start_time = ctx.start

        # Track last end time for finalize
        self._last_end_time = max(self._last_end_time, ctx.end)

        # 1. Safety processing
        if self._safety_processor:
            try:
                result.safety = self._safety_processor.process(ctx.text)

                # Emit alert if needed
                if self._on_safety_alert and result.safety:
                    alert = self._safety_processor.create_alert(
                        segment_id=hash(ctx.segment_id) % 10000,
                        segment_start=ctx.start,
                        segment_end=ctx.end,
                        result=result.safety,
                    )
                    if alert:
                        self._invoke_callback_safely(self._on_safety_alert, alert)
            except Exception as e:
                logger.warning("Safety processing failed: %s", e)

        # 2. Environment classification
        if self._environment_classifier and ctx.audio_state:
            try:
                # Extract metrics from audio_state if available
                audio_health = ctx.audio_state.get("audio_health", {})
                if audio_health:
                    metrics = AudioHealthMetrics.from_audio_health_snapshot(audio_health)
                    result.environment = self._environment_classifier.classify(metrics)
            except Exception as e:
                logger.warning("Environment classification failed: %s", e)

        # 3. Extended prosody
        if self.config.enable_prosody_extended and audio is not None:
            try:
                from .prosody_extended import extract_prosody_extended

                prosody_state = extract_prosody_extended(audio, sr)
                result.prosody_extended = prosody_state.to_dict()
            except Exception as e:
                logger.warning("Extended prosody extraction failed: %s", e)

        # 4. Turn-taking evaluation
        if self._turn_taking_evaluator:
            try:
                # Get prosody falling/flat from audio_state or prosody_extended
                prosody_falling = None
                prosody_flat = None

                if result.prosody_extended:
                    boundary_tone = result.prosody_extended.get("boundary_tone", {})
                    tone = boundary_tone.get("tone", "unknown")
                    prosody_falling = tone == "falling"
                    prosody_flat = tone == "flat"
                elif ctx.audio_state:
                    prosody = ctx.audio_state.get("prosody_extended", {})
                    boundary_tone = prosody.get("boundary_tone", {})
                    tone = boundary_tone.get("tone", "unknown")
                    prosody_falling = tone == "falling"
                    prosody_flat = tone == "flat"

                result.end_of_turn = self._turn_taking_evaluator.evaluate(
                    text=ctx.text,
                    silence_duration_ms=silence_duration_ms,
                    turn_duration_ms=turn_duration_ms,
                    prosody_falling=prosody_falling,
                    prosody_flat=prosody_flat,
                )

                # Emit on_end_of_turn_hint callback if turn-end detected
                if result.end_of_turn.is_end_of_turn and self._on_end_of_turn_hint:
                    # Convert reason codes to lowercase snake_case for API consistency
                    reason_codes_mapped = self._map_reason_codes(result.end_of_turn.reason_codes)

                    # Check for terminal punctuation in reason codes
                    has_terminal_punct = any(
                        code in result.end_of_turn.reason_codes
                        for code in ("TERMINAL_PUNCT", "QUESTION_DETECTED")
                    )

                    payload = EndOfTurnHintPayload(
                        confidence=result.end_of_turn.confidence,
                        silence_duration=result.end_of_turn.silence_duration_ms / 1000.0,
                        terminal_punctuation=has_terminal_punct,
                        partial_text=ctx.text,
                        reason_codes=reason_codes_mapped,
                        silence_duration_ms=result.end_of_turn.silence_duration_ms,
                        policy_name=result.end_of_turn.policy_name,
                    )
                    self._invoke_callback_safely(self._on_end_of_turn_hint, payload)
            except Exception as e:
                logger.warning("Turn-taking evaluation failed: %s", e)

        return result

    def _map_reason_codes(self, reason_codes: Sequence[str]) -> list[str]:
        """Map internal reason codes to API-friendly lowercase snake_case.

        Args:
            reason_codes: List of internal reason codes (e.g., "TERMINAL_PUNCT").

        Returns:
            List of API-friendly reason codes (e.g., "terminal_punctuation").
        """
        code_map = {
            "TERMINAL_PUNCT": "terminal_punctuation",
            "SILENCE_THRESHOLD": "silence_threshold",
            "FALLING_INTONATION": "boundary_tone",
            "COMPLETE_SENTENCE": "complete_sentence",
            "QUESTION_DETECTED": "question_detected",
            "LONG_PAUSE": "long_pause",
        }
        return [code_map.get(code, code.lower()) for code in reason_codes]

    def process_turn(self, turn: dict[str, Any]) -> TurnProcessResult:
        """Process a completed turn for topic/role analysis.

        Args:
            turn: Turn dictionary with id, speaker_id, text, start, end.

        Returns:
            TurnProcessResult with any boundaries detected.
        """
        if not self.config.enabled:
            return TurnProcessResult()

        result = TurnProcessResult()
        self._turns.append(turn)

        # Track last end time for finalize
        turn_end = turn.get("end", 0.0)
        self._last_end_time = max(self._last_end_time, turn_end)

        # Topic segmentation
        if self._topic_segmenter:
            try:
                boundary = self._topic_segmenter.add_turn(turn)
                result.topic_boundary = boundary
            except Exception as e:
                logger.warning("Topic segmentation failed: %s", e)

        # Role inference (check if decision time)
        if self._role_inferrer and not self._roles_decided:
            self._check_role_decision(turn)

        return result

    def _check_role_decision(self, turn: dict[str, Any]) -> None:
        """Check if role decision should be made.

        Args:
            turn: Latest turn added.
        """
        if self._roles_decided:
            return

        turn_count = len(self._turns)
        elapsed_sec = turn.get("end", 0.0) - (self._start_time or 0.0)

        triggered_by_turns = turn_count >= self.config.role_decision_turns
        triggered_by_time = elapsed_sec >= self.config.role_decision_seconds
        should_decide = triggered_by_turns or triggered_by_time

        if should_decide and self._role_inferrer:
            try:
                self._role_assignments = self._role_inferrer.infer_roles(self._turns)
                self._roles_decided = True

                # Emit callback with typed payload
                if self._on_role_assigned and self._role_assignments:
                    # Determine trigger reason
                    trigger = "turn_count" if triggered_by_turns else "elapsed_time"
                    timestamp = turn.get("end", 0.0)

                    # Convert RoleAssignment objects to dicts
                    assignments_dict = {
                        k: v.to_dict() for k, v in self._role_assignments.items()
                    }

                    payload = RoleAssignedPayload(
                        assignments=assignments_dict,
                        timestamp=timestamp,
                        trigger=trigger,
                    )
                    self._invoke_callback_safely(self._on_role_assigned, payload)
            except Exception as e:
                logger.warning("Role inference failed: %s", e)

    def _handle_topic_boundary(self, payload: TopicBoundaryPayload) -> None:
        """Handle topic boundary detection.

        Args:
            payload: Topic boundary payload.
        """
        if self._on_topic_boundary:
            self._invoke_callback_safely(self._on_topic_boundary, payload)

    def _invoke_callback_safely(self, callback: Any, *args: Any) -> None:
        """Invoke a callback safely, catching exceptions.

        Args:
            callback: Callback function.
            *args: Arguments to pass.
        """
        try:
            callback(*args)
        except Exception as e:
            logger.warning("Callback raised exception: %s", e)

    def get_role_assignments(self) -> dict[str, RoleAssignment]:
        """Get current role assignments.

        Returns:
            Dictionary mapping speaker_id to RoleAssignment.
        """
        # Force role decision if not yet made but we have turns
        if not self._roles_decided and self._role_inferrer and self._turns:
            self._role_assignments = self._role_inferrer.infer_roles(self._turns)
            self._roles_decided = True

        return self._role_assignments

    def get_topics(self) -> list[TopicChunk]:
        """Get finalized topic chunks.

        Returns:
            List of TopicChunk objects.
        """
        if self._topic_segmenter:
            return self._topic_segmenter.finalize()
        return []

    def finalize(self) -> None:
        """Finalize post-processing and close any open state.

        Must be called after all segments/turns have been processed to:
        - Close the current topic chunk with proper end_time
        - Trigger final role inference if not yet decided
        - Emit final callbacks

        This method is idempotent - calling it multiple times has no effect
        after the first call.
        """
        if self._finalized:
            return

        self._finalized = True

        # Close open topic chunk
        if self._topic_segmenter is not None:
            closed_topic = self._topic_segmenter.close_current_topic(
                end_time=self._last_end_time if self._last_end_time > 0 else None
            )
            # Emit callback for the closed topic if one was actually closed
            if closed_topic is not None and self._on_topic_boundary:
                # Create a boundary payload indicating finalization
                from .topic_segmentation import TopicBoundaryPayload

                payload = TopicBoundaryPayload(
                    previous_topic_id=closed_topic.id,
                    new_topic_id="",  # No new topic - this is the end
                    boundary_turn_id=closed_topic.turn_ids[-1] if closed_topic.turn_ids else "",
                    boundary_time=self._last_end_time,
                    similarity_score=0.0,  # Not a similarity-based boundary
                    keywords_previous=closed_topic.keywords,
                    keywords_new=[],
                )
                self._invoke_callback_safely(self._on_topic_boundary, payload)

        # Final role inference if not done yet
        if self._role_inferrer is not None and not self._roles_decided and self._turns:
            try:
                self._role_assignments = self._role_inferrer.infer_roles(self._turns)
                self._roles_decided = True

                # Emit callback with "finalize" trigger
                if self._on_role_assigned and self._role_assignments:
                    assignments_dict = {
                        k: v.to_dict() for k, v in self._role_assignments.items()
                    }
                    role_payload = RoleAssignedPayload(
                        assignments=assignments_dict,
                        timestamp=self._last_end_time,
                        trigger="finalize",
                    )
                    self._invoke_callback_safely(self._on_role_assigned, role_payload)
            except Exception as e:
                logger.warning("Final role inference failed: %s", e)

    def reset(self) -> None:
        """Reset processor state for new session."""
        self._turns = []
        self._role_assignments = {}
        self._roles_decided = False
        self._start_time = None
        self._last_end_time = 0.0
        self._finalized = False

        if self._topic_segmenter:
            self._topic_segmenter.reset()


# Preset configurations
def post_process_config_for_call_center() -> PostProcessConfig:
    """Create post-process config optimized for call centers.

    Enables:
    - Safety processing with PII masking
    - Role inference for agent/customer detection
    - Turn-taking with balanced policy

    Returns:
        Configured PostProcessConfig.
    """
    from .safety_config import safety_config_for_call_center

    return PostProcessConfig(
        enabled=True,
        enable_safety=True,
        enable_roles=True,
        enable_turn_taking=True,
        safety_config=safety_config_for_call_center(),
        role_config=RoleInferenceConfig(context="call_center"),
        turn_taking_policy="balanced",
    )


def post_process_config_for_meetings() -> PostProcessConfig:
    """Create post-process config optimized for meetings.

    Enables:
    - Topic segmentation for chapters
    - Role inference for facilitator detection
    - Environment classification

    Returns:
        Configured PostProcessConfig.
    """
    return PostProcessConfig(
        enabled=True,
        enable_topics=True,
        enable_roles=True,
        enable_environment=True,
        role_config=RoleInferenceConfig(context="meeting"),
        topic_config=TopicSegmentationConfig(
            min_topic_duration_sec=60.0,
            max_topic_duration_sec=600.0,
        ),
    )


def post_process_config_minimal() -> PostProcessConfig:
    """Create minimal post-process config.

    Enables only smart formatting (via safety layer).

    Returns:
        Configured PostProcessConfig.
    """
    from .safety_config import safety_config_minimal

    return PostProcessConfig(
        enabled=True,
        enable_safety=True,
        safety_config=safety_config_minimal(),
    )


__all__ = [
    "PostProcessConfig",
    "PostProcessor",
    "PostProcessResult",
    "TurnProcessResult",
    "SegmentContext",
    "SegmentProcessor",
    "post_process_config_for_call_center",
    "post_process_config_for_meetings",
    "post_process_config_minimal",
]
