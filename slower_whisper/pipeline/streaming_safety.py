"""Streaming integration for safety processing layer.

This module provides streaming-aware safety processing that integrates
with the existing streaming callback system. It processes FINALIZED
segments only and emits safety alerts via callbacks.

Usage:
    processor = StreamingSafetyProcessor(safety_config, callbacks)
    # Called automatically on segment finalization
    processor.process_segment(segment)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .safety_config import SafetyConfig
from .safety_layer import SafetyAlertPayload, SafetyProcessingResult, SafetyProcessor
from .streaming_callbacks import StreamingError, invoke_callback_safely

if TYPE_CHECKING:
    from .streaming import StreamSegment
    from .streaming_callbacks import StreamCallbacks

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamingSafetyState:
    """Accumulated safety state across a streaming session.

    Attributes:
        total_segments_processed: Number of segments processed.
        total_pii_detections: Total PII matches found.
        total_moderation_flags: Total moderation flags raised.
        total_formatting_changes: Total formatting changes applied.
        alerts_emitted: Number of safety alerts emitted.
        segments_blocked: Number of segments where action was "block".
        unique_pii_types: Set of unique PII types detected.
        unique_moderation_categories: Set of unique moderation categories.
    """

    total_segments_processed: int = 0
    total_pii_detections: int = 0
    total_moderation_flags: int = 0
    total_formatting_changes: int = 0
    alerts_emitted: int = 0
    segments_blocked: int = 0
    unique_pii_types: set[str] = field(default_factory=set)
    unique_moderation_categories: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "total_segments_processed": self.total_segments_processed,
            "total_pii_detections": self.total_pii_detections,
            "total_moderation_flags": self.total_moderation_flags,
            "total_formatting_changes": self.total_formatting_changes,
            "alerts_emitted": self.alerts_emitted,
            "segments_blocked": self.segments_blocked,
            "unique_pii_types": list(self.unique_pii_types),
            "unique_moderation_categories": list(self.unique_moderation_categories),
        }


class StreamingSafetyProcessor:
    """Streaming-aware safety processor.

    Integrates with the streaming pipeline to process FINALIZED segments
    and emit safety alerts via the callback system.

    The processor:
    - Processes only finalized segments (not partial)
    - Attaches safety_state to segment.audio_state
    - Emits on_safety_alert callbacks when alerts are triggered
    - Tracks cumulative safety statistics

    Example:
        >>> config = SafetyConfig(enabled=True, enable_content_moderation=True)
        >>> processor = StreamingSafetyProcessor(config, callbacks)
        >>> # Process each finalized segment
        >>> result = processor.process_segment(segment)
        >>> print(result.overall_action)
        "allow"
    """

    def __init__(
        self,
        config: SafetyConfig,
        callbacks: StreamCallbacks | object | None = None,
    ):
        """Initialize streaming safety processor.

        Args:
            config: Safety configuration.
            callbacks: Optional callbacks object for alert emission.
        """
        self.config = config
        self.callbacks = callbacks
        self._processor = SafetyProcessor(config)
        self._state = StreamingSafetyState()

    @property
    def state(self) -> StreamingSafetyState:
        """Get current cumulative safety state."""
        return self._state

    def process_segment(
        self,
        segment: StreamSegment,
        attach_state: bool = True,
    ) -> SafetyProcessingResult:
        """Process a finalized segment through the safety pipeline.

        Args:
            segment: The finalized StreamSegment to process.
            attach_state: If True, attach safety_state to segment.audio_state.

        Returns:
            SafetyProcessingResult with processing details.
        """
        if not self.config.is_active():
            return SafetyProcessingResult(
                original_text=segment.text,
                processed_text=segment.text,
            )

        segment_id = segment.segment_id or "unknown"

        try:
            result = self._processor.process(segment.text)
        except Exception as e:
            logger.warning("Safety processing failed for segment %s: %s", segment_id, e)
            self._emit_error(e, segment)
            return SafetyProcessingResult(
                original_text=segment.text,
                processed_text=segment.text,
            )

        # Update cumulative state
        self._state.total_segments_processed += 1

        if result.has_pii:
            self._state.total_pii_detections += len(result.pii_matches)
            for match in result.pii_matches:
                self._state.unique_pii_types.add(match.pii_type)

        if result.has_flagged_content and result.moderation_result:
            self._state.total_moderation_flags += len(result.moderation_result.matches)
            self._state.unique_moderation_categories.update(
                result.moderation_result.categories_found
            )

        if result.has_formatting_changes and result.formatting_result:
            self._state.total_formatting_changes += len(result.formatting_result.matches)

        if result.overall_action == "block":
            self._state.segments_blocked += 1

        # Attach safety state to segment
        if attach_state:
            self._attach_safety_state(segment, result)

        # Emit alert if needed
        # Use hash of segment_id as integer for alert (or 0 if not available)
        alert_segment_id = hash(segment_id) % 10000 if segment_id != "unknown" else 0
        alert = self._processor.create_alert(
            segment_id=alert_segment_id,
            segment_start=segment.start,
            segment_end=segment.end,
            result=result,
        )

        if alert:
            self._emit_alert(alert)
            self._state.alerts_emitted += 1

        return result

    def process_text(self, text: str) -> SafetyProcessingResult:
        """Process arbitrary text through the safety pipeline.

        Convenience method for processing text without a segment context.

        Args:
            text: Text to process.

        Returns:
            SafetyProcessingResult with processing details.
        """
        return self._processor.process(text)

    def _attach_safety_state(
        self,
        segment: StreamSegment,
        result: SafetyProcessingResult,
    ) -> None:
        """Attach safety state to segment's audio_state.

        Args:
            segment: Segment to update.
            result: Safety processing result.
        """
        # Initialize audio_state if needed
        if segment.audio_state is None:
            segment.audio_state = {}

        # Attach safety state
        segment.audio_state["safety"] = result.to_safety_state()

        # Update text if modified
        if result.processed_text != result.original_text:
            # Store original and update current
            segment.audio_state["safety"]["original_text"] = result.original_text
            # Note: We don't modify segment.text to preserve original
            # The processed text is available in the safety state

    def _emit_alert(self, alert: SafetyAlertPayload) -> None:
        """Emit safety alert via callbacks.

        Args:
            alert: Alert payload to emit.
        """
        invoke_callback_safely(
            self.callbacks,
            "on_safety_alert",
            alert,
        )

    def _emit_error(self, exception: Exception, segment: StreamSegment) -> None:
        """Emit error via callbacks.

        Args:
            exception: The exception that occurred.
            segment: The segment being processed.
        """
        segment_id = segment.segment_id or "unknown"
        error = StreamingError(
            exception=exception,
            context=f"Safety processing failed for segment {segment_id}",
            segment_start=segment.start,
            segment_end=segment.end,
            recoverable=True,
        )
        invoke_callback_safely(self.callbacks, "on_error", error)

    def reset_state(self) -> None:
        """Reset cumulative safety state.

        Call this when starting a new streaming session.
        """
        self._state = StreamingSafetyState()

    def get_summary(self) -> dict[str, Any]:
        """Get summary of safety processing for the session.

        Returns:
            Dictionary with cumulative statistics.
        """
        return {
            "config": self.config.to_dict(),
            "state": self._state.to_dict(),
        }


def extend_callbacks_with_safety(callbacks_class: type[Any]) -> type[Any]:
    """Decorator to add on_safety_alert method to a callbacks class.

    This is a helper for extending existing callback implementations
    with safety alert support.

    Args:
        callbacks_class: Class to extend.

    Returns:
        Extended class with on_safety_alert method.

    Example:
        >>> @extend_callbacks_with_safety
        ... class MyCallbacks:
        ...     def on_segment_finalized(self, segment):
        ...         pass
    """

    def on_safety_alert(self: Any, payload: SafetyAlertPayload) -> None:
        """Handle safety alert. Override in subclass."""
        pass

    if not hasattr(callbacks_class, "on_safety_alert"):
        callbacks_class.on_safety_alert = on_safety_alert

    return callbacks_class


__all__ = [
    "StreamingSafetyProcessor",
    "StreamingSafetyState",
    "extend_callbacks_with_safety",
]
