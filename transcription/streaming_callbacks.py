"""Callback protocols for streaming event handling (v1.9.0).

This module defines the callback interface for streaming enrichment sessions,
enabling real-time event handling for downstream consumers.

Key features:
- Protocol-based interface for type-safe callbacks
- Sync callbacks (async optional/stretch goal)
- Error isolation: callback exceptions never kill the pipeline
- Supports segment finalized, speaker turn, semantic update, and error events

See docs/STREAMING_ARCHITECTURE.md for usage patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .streaming import StreamSegment
    from .streaming_semantic import SemanticUpdatePayload

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
    "NoOpCallbacks",
    "invoke_callback_safely",
]
