"""Callback protocols for streaming event handling (v2.1.0).

This module defines the callback interface for streaming enrichment sessions,
enabling real-time event handling for downstream consumers.

Key features:
- Protocol-based interface for type-safe callbacks
- Sync callbacks (async optional/stretch goal)
- Error isolation: callback exceptions never kill the pipeline
- Supports segment finalized, speaker turn, semantic update, and error events
- v2.1: Conversation physics, audio health, VAD, barge-in, end-of-turn hints,
  corrections, and commitments

See docs/STREAMING_ARCHITECTURE.md for usage patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .topic_segmentation import TopicBoundaryPayload

if TYPE_CHECKING:
    from .audio_health import AudioHealthSnapshot
    from .conversation_physics import ConversationPhysicsSnapshot
    from .safety_layer import SafetyAlertPayload
    from .streaming import StreamSegment
    from .streaming_semantic import CommitmentEntry, CorrectionEvent, SemanticUpdatePayload

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamingError:
    """Error context for callback error handling.

    Attributes:
        exception: The exception that occurred.
        context: Human-readable context about where the error occurred.
        segment_start: Start time of segment being processed (if applicable).
        segment_end: End time of segment being processed (if applicable).
        recoverable: Whether the pipeline can continue after this error.
    """

    exception: Exception
    context: str
    segment_start: float | None = None
    segment_end: float | None = None
    recoverable: bool = True


@dataclass(slots=True)
class VADActivityPayload:
    """Payload for VAD (Voice Activity Detection) state change events.

    Attributes:
        energy_level: Current audio energy level (RMS or dB scale).
        is_speech: Whether the current frame contains speech.
        silence_duration_sec: Duration of continuous silence in seconds.
            Reset to 0 when speech is detected.
    """

    energy_level: float
    is_speech: bool
    silence_duration_sec: float


@dataclass(slots=True)
class BargeInPayload:
    """Payload for barge-in detection events.

    Barge-in occurs when user speech is detected while TTS audio is
    being played back. This enables interrupt-driven conversation flow.

    Attributes:
        energy: Audio energy level that triggered barge-in detection.
        tts_elapsed_sec: Time elapsed since TTS playback started, in seconds.
    """

    energy: float
    tts_elapsed_sec: float


@dataclass(slots=True)
class EndOfTurnHintPayload:
    """Payload for end-of-turn prediction events.

    End-of-turn hints are predictive signals that the user may have
    finished speaking, enabling faster response times.

    Attributes:
        confidence: Confidence score (0.0-1.0) that the turn has ended.
        silence_duration: Duration of trailing silence in seconds.
        terminal_punctuation: Whether terminal punctuation was detected
            (period, question mark, exclamation mark).
        partial_text: The partial transcript that triggered the hint.
        reason_codes: List of reason codes explaining why hint was triggered.
            Possible values: "terminal_punctuation", "silence_threshold",
            "boundary_tone", "complete_sentence", "question_detected", "long_pause".
        silence_duration_ms: Silence duration in milliseconds (for precision).
        policy_name: Name of the turn-taking policy that was applied.
    """

    confidence: float
    silence_duration: float
    terminal_punctuation: bool
    partial_text: str
    reason_codes: list[str] = field(default_factory=list)
    silence_duration_ms: float = 0.0
    policy_name: str = "balanced"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "confidence": self.confidence,
            "silence_duration": self.silence_duration,
            "terminal_punctuation": self.terminal_punctuation,
            "partial_text": self.partial_text,
            "reason_codes": list(self.reason_codes),
            "silence_duration_ms": self.silence_duration_ms,
            "policy_name": self.policy_name,
        }


@dataclass(slots=True)
class RoleAssignedPayload:
    """Payload for role assignment callback.

    Emitted when speaker roles are inferred after sufficient turns
    have been processed (typically after role_decision_turns or
    role_decision_seconds threshold is reached).

    Attributes:
        assignments: Dictionary mapping speaker_id to RoleAssignment dict.
            Each assignment contains: speaker_id, role, confidence, evidence,
            original_label.
        timestamp: When roles were decided (end time of triggering turn).
        trigger: What caused the role decision:
            - "turn_count": Reached role_decision_turns threshold
            - "elapsed_time": Reached role_decision_seconds threshold
            - "finalize": Forced decision at session end
    """

    assignments: dict[str, Any]  # speaker_id -> RoleAssignment.to_dict()
    timestamp: float
    trigger: str  # "turn_count" | "elapsed_time" | "finalize"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "assignments": self.assignments,
            "timestamp": self.timestamp,
            "trigger": self.trigger,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoleAssignedPayload:
        """Create from dictionary."""
        return cls(
            assignments=data.get("assignments", {}),
            timestamp=data.get("timestamp", 0.0),
            trigger=data.get("trigger", "unknown"),
        )


@runtime_checkable
class StreamCallbacks(Protocol):
    """Callback interface for streaming enrichment events.

    Implement this protocol to receive real-time notifications as
    segments are finalized, speaker turns are detected, and semantic
    annotations are computed.

    All callbacks are optional - implement only what you need. Unimplemented
    methods default to no-op behavior.

    Important: Callback implementations must not raise exceptions. If a
    callback raises, it will be caught, logged, and on_error will be
    invoked. The streaming pipeline will continue regardless.

    Example:
        >>> class MyCallbacks:
        ...     def on_segment_finalized(self, segment):
        ...         print(f"Finalized: {segment.text}")
        ...         self.db.insert(segment)
        ...
        ...     def on_error(self, error):
        ...         logging.error(f"Callback error: {error.exception}")
        ...
        >>> session = StreamingEnrichmentSession(
        ...     wav_path="audio.wav",
        ...     config=config,
        ...     callbacks=MyCallbacks()
        ... )
    """

    def on_segment_finalized(self, segment: StreamSegment) -> None:
        """Called when a segment is finalized with enrichment complete.

        This is the primary callback for consuming enriched segments.
        The segment will have audio_state populated if enrichment was
        enabled and succeeded.

        Args:
            segment: The finalized segment with optional audio_state.
                     Guaranteed to have start, end, text, and speaker_id.

        Example:
            >>> def on_segment_finalized(self, segment):
            ...     print(f"[{segment.start:.2f}s] {segment.text}")
            ...     if segment.audio_state:
            ...         print(f"  Audio: {segment.audio_state['rendering']}")
        """
        ...

    def on_speaker_turn(self, turn: dict) -> None:
        """Called when a speaker turn is detected.

        A turn is a contiguous sequence of segments from the same speaker.
        This callback fires when a turn boundary is detected (speaker change
        or long pause).

        Args:
            turn: Turn dictionary with keys:
                - id: Turn identifier (e.g., "turn_0")
                - speaker_id: Speaker identifier
                - start: Turn start time in seconds
                - end: Turn end time in seconds
                - segment_ids: List of segment IDs in this turn
                - text: Concatenated text from all segments
        """
        ...

    def on_semantic_update(self, payload: SemanticUpdatePayload) -> None:
        """Called when semantic annotations are computed for a turn.

        Semantic updates include keywords, risk tags, and action items
        extracted from the turn text.

        Args:
            payload: SemanticUpdatePayload with:
                - turn: The annotated Turn object
                - keywords: List of extracted keywords
                - risk_tags: List of detected risk flags
                - actions: List of action items
                - question_count: Number of questions in the turn
                - context_size: Current context window size (turn count)
        """
        ...

    def on_error(self, error: StreamingError) -> None:
        """Called when an error occurs during streaming.

        This includes enrichment failures, callback exceptions, and
        other recoverable errors. The streaming pipeline continues
        after recoverable errors.

        Args:
            error: StreamingError with exception details and context.

        Example:
            >>> def on_error(self, error):
            ...     if not error.recoverable:
            ...         self.alert_ops_team(error)
            ...     self.metrics.increment("streaming_errors")
        """
        ...

    def on_physics_update(self, snapshot: ConversationPhysicsSnapshot) -> None:
        """Called when conversation physics are updated.

        Conversation physics track the dynamics of the conversation,
        including speaking rates, turn-taking patterns, and overlap
        statistics.

        Args:
            snapshot: ConversationPhysicsSnapshot containing:
                - speaker_stats: Per-speaker statistics (talk time, word count, etc.)
                - turn_count: Total number of turns so far
                - overlap_ratio: Ratio of overlapping speech
                - avg_turn_duration: Average turn duration in seconds
                - conversation_pace: Words per minute across all speakers
                - last_update_ts: Timestamp of this snapshot
        """
        ...

    def on_audio_health(self, snapshot: AudioHealthSnapshot) -> None:
        """Called when audio health is computed for a chunk.

        Audio health metrics help detect issues like clipping, low
        signal levels, excessive noise, or encoding artifacts.

        Args:
            snapshot: AudioHealthSnapshot containing:
                - clipping_ratio: Ratio of samples at max amplitude
                - noise_floor_db: Estimated noise floor in decibels
                - snr_db: Signal-to-noise ratio in decibels
                - silence_ratio: Ratio of silence in the chunk
                - peak_db: Peak amplitude in decibels
                - rms_db: RMS amplitude in decibels
                - issues: List of detected issues (e.g., "clipping", "low_signal")
        """
        ...

    def on_vad_activity(self, payload: VADActivityPayload) -> None:
        """Called when VAD state changes.

        Voice Activity Detection (VAD) events indicate transitions
        between speech and silence, enabling responsive UI and
        processing logic.

        Args:
            payload: VADActivityPayload containing:
                - energy_level: Current audio energy level (RMS or dB scale)
                - is_speech: Whether the current frame contains speech
                - silence_duration_sec: Duration of continuous silence in seconds
        """
        ...

    def on_barge_in(self, payload: BargeInPayload) -> None:
        """Called when barge-in is detected during TTS playback.

        Barge-in detection allows the system to interrupt TTS playback
        when the user starts speaking, enabling natural conversational
        interruptions.

        Args:
            payload: BargeInPayload containing:
                - energy: Audio energy level that triggered barge-in
                - tts_elapsed_sec: Time since TTS playback started (seconds)
        """
        ...

    def on_end_of_turn_hint(self, payload: EndOfTurnHintPayload) -> None:
        """Called when end-of-turn is predicted.

        End-of-turn hints are predictive signals based on silence
        duration, terminal punctuation, and other prosodic cues.
        These enable faster response times by anticipating when
        the user has finished speaking.

        Args:
            payload: EndOfTurnHintPayload containing:
                - confidence: Confidence score (0.0-1.0) that turn ended
                - silence_duration: Duration of trailing silence (seconds)
                - terminal_punctuation: Whether terminal punctuation detected
                - partial_text: The partial transcript that triggered the hint
        """
        ...

    def on_correction(self, correction: CorrectionEvent) -> None:
        """Called when a correction is detected.

        Corrections occur when the ASR model revises previous output,
        typically due to more context becoming available. Tracking
        corrections is useful for understanding ASR stability and
        implementing correction-aware UIs.

        Args:
            correction: CorrectionEvent containing:
                - segment_id: ID of the corrected segment
                - original_text: The text before correction
                - corrected_text: The text after correction
                - correction_type: Type of correction (e.g., "substitution", "deletion")
                - confidence_delta: Change in confidence score
                - timestamp: When the correction was detected
        """
        ...

    def on_commitment(self, commitment: CommitmentEntry) -> None:
        """Called when a new commitment is extracted.

        Commitments are actionable items extracted from the conversation,
        such as promises, deadlines, or follow-up tasks.

        Args:
            commitment: CommitmentEntry containing:
                - id: Unique commitment identifier
                - speaker_id: Who made the commitment
                - text: The commitment text
                - commitment_type: Type (e.g., "promise", "deadline", "action")
                - confidence: Extraction confidence score
                - source_turn_id: The turn this was extracted from
                - timestamp: When the commitment was detected
        """
        ...

    def on_safety_alert(self, payload: SafetyAlertPayload) -> None:
        """Called when a safety alert is triggered.

        Safety alerts are generated when the safety layer detects
        content that requires attention, such as PII or flagged content.

        Args:
            payload: SafetyAlertPayload containing:
                - segment_id: ID of the segment that triggered the alert
                - segment_start: Start time of the segment
                - segment_end: End time of the segment
                - alert_type: Type of alert ("pii", "moderation", "combined")
                - severity: Overall severity level
                - action: Recommended action
                - details: Additional details about the alert
        """
        ...

    def on_role_assigned(self, payload: RoleAssignedPayload) -> None:
        """Called when speaker roles are assigned.

        Role assignments are computed after enough turns have been
        processed (typically 5 turns or 30 seconds).

        Args:
            payload: RoleAssignedPayload containing:
                - assignments: Dictionary mapping speaker_id to RoleAssignment dict
                  with keys: speaker_id, role, confidence, evidence, original_label
                - timestamp: When roles were decided
                - trigger: What caused decision ("turn_count", "elapsed_time", "finalize")
        """
        ...

    def on_topic_boundary(self, payload: TopicBoundaryPayload) -> None:
        """Called when a topic boundary is detected.

        Topic boundaries are detected when vocabulary shifts significantly
        between rolling windows of turns.

        Args:
            payload: TopicBoundaryPayload containing:
                - previous_topic_id: ID of the topic that ended
                - new_topic_id: ID of the new topic starting
                - boundary_turn_id: Turn ID where boundary was detected
                - boundary_time: Time of the boundary in seconds
                - similarity_score: Similarity score that triggered boundary
                - keywords_previous: Keywords from previous topic
                - keywords_new: Keywords from new topic
        """
        ...


class NoOpCallbacks:
    """Default no-op callback implementation.

    Used internally when no callbacks are provided. All methods are no-ops.
    """

    def on_segment_finalized(self, segment: StreamSegment) -> None:
        pass

    def on_speaker_turn(self, turn: dict) -> None:
        pass

    def on_semantic_update(self, payload: SemanticUpdatePayload) -> None:
        pass

    def on_error(self, error: StreamingError) -> None:
        pass

    def on_physics_update(self, snapshot: ConversationPhysicsSnapshot) -> None:
        pass

    def on_audio_health(self, snapshot: AudioHealthSnapshot) -> None:
        pass

    def on_vad_activity(self, payload: VADActivityPayload) -> None:
        pass

    def on_barge_in(self, payload: BargeInPayload) -> None:
        pass

    def on_end_of_turn_hint(self, payload: EndOfTurnHintPayload) -> None:
        pass

    def on_correction(self, correction: CorrectionEvent) -> None:
        pass

    def on_commitment(self, commitment: CommitmentEntry) -> None:
        pass

    def on_safety_alert(self, payload: SafetyAlertPayload) -> None:
        pass

    def on_role_assigned(self, payload: RoleAssignedPayload) -> None:
        pass

    def on_topic_boundary(self, payload: TopicBoundaryPayload) -> None:
        pass


def invoke_callback_safely(
    callbacks: StreamCallbacks | object | None,
    method_name: str,
    *args,
    **kwargs,
) -> bool:
    """Invoke a callback method safely, catching and logging any exceptions.

    This function ensures that callback exceptions never crash the streaming
    pipeline. If a callback raises, it's logged and on_error is invoked
    (if available and not the failing method).

    Args:
        callbacks: The callbacks object (may be None).
        method_name: Name of the method to invoke (e.g., "on_segment_finalized").
        *args: Positional arguments to pass to the callback.
        **kwargs: Keyword arguments to pass to the callback.

    Returns:
        True if the callback was invoked successfully, False if it failed
        or callbacks was None.
    """
    if callbacks is None:
        return False

    method = getattr(callbacks, method_name, None)
    if method is None:
        return False

    try:
        method(*args, **kwargs)
        return True
    except Exception as e:
        logger.warning(
            "Callback %s raised exception: %s",
            method_name,
            e,
            exc_info=True,
        )

        # Try to invoke on_error if this wasn't already an on_error call
        if method_name != "on_error":
            error = StreamingError(
                exception=e,
                context=f"Exception in callback {method_name}",
                recoverable=True,
            )
            try:
                on_error = getattr(callbacks, "on_error", None)
                if on_error is not None:
                    on_error(error)
            except Exception as e2:
                logger.error(
                    "on_error callback also raised: %s",
                    e2,
                    exc_info=True,
                )

        return False


# Export public API
__all__ = [
    "StreamCallbacks",
    "StreamingError",
    "VADActivityPayload",
    "BargeInPayload",
    "EndOfTurnHintPayload",
    "RoleAssignedPayload",
    "TopicBoundaryPayload",
    "NoOpCallbacks",
    "invoke_callback_safely",
]
