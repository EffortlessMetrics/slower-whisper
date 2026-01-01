"""
Streaming enrichment session for real-time audio feature extraction.

This module extends the base StreamingSession with audio enrichment capabilities,
enabling real-time prosody and emotion extraction for streaming transcription.
It wraps the base session and enriches segments as they're finalized.

Key features:
- Reuses AudioSegmentExtractor for efficient audio access
- Graceful error handling with extraction_status tracking
- Optional prosody/emotion/categorical emotion extraction
- Supports reset for session reuse
- Type-safe with comprehensive docstrings

See docs/STREAMING_ARCHITECTURE.md for architecture details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio_enrichment import _enrich_segment_with_extractor
from .audio_utils import AudioSegmentExtractor
from .models import Segment
from .streaming import (
    StreamChunk,
    StreamConfig,
    StreamEvent,
    StreamEventType,
    StreamingSession,
    StreamSegment,
)
from .streaming_callbacks import (
    StreamCallbacks,
    StreamingError,
    invoke_callback_safely,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamingEnrichmentConfig:
    """
    Configuration for streaming audio enrichment.

    Controls which audio features to extract in real-time as segments
    are finalized. All enrichment is optional and defaults to disabled
    for minimal latency.

    Attributes:
        base_config: Configuration for base StreamingSession (gap detection, etc.)
        enable_prosody: Extract prosodic features (pitch, energy, rate, pauses).
        enable_emotion: Extract dimensional emotion features (valence, arousal).
        enable_categorical_emotion: Extract categorical emotion labels.
        speaker_baseline: Optional baseline statistics for speaker-relative normalization.
                         If None, absolute thresholds are used.
    """

    base_config: StreamConfig = field(default_factory=StreamConfig)
    enable_prosody: bool = False
    enable_emotion: bool = False
    enable_categorical_emotion: bool = False
    speaker_baseline: dict[str, Any] | None = None


class StreamingEnrichmentSession:
    """
    Streaming session with real-time audio enrichment.

    Wraps the base StreamingSession and enriches finalized segments with
    audio features (prosody, emotion) as they arrive. Designed for low-latency
    real-time applications where you need both transcription and audio analysis.

    The session maintains a single AudioSegmentExtractor for efficient audio
    access across all segments. Audio features are only extracted for FINAL
    segments to avoid wasted computation on changing partials.

    Example:
        >>> from pathlib import Path
        >>> from transcription.streaming_enrich import (
        ...     StreamingEnrichmentSession,
        ...     StreamingEnrichmentConfig
        ... )
        >>> from transcription.streaming import StreamChunk, StreamConfig
        >>>
        >>> # Configure enrichment
        >>> config = StreamingEnrichmentConfig(
        ...     base_config=StreamConfig(max_gap_sec=1.0),
        ...     enable_prosody=True,
        ...     enable_emotion=True
        ... )
        >>>
        >>> # Initialize session with audio file
        >>> wav_path = Path("normalized_audio.wav")
        >>> session = StreamingEnrichmentSession(wav_path, config)
        >>>
        >>> # Process chunks
        >>> chunk = {"start": 0.0, "end": 2.5, "text": "Hello world", "speaker_id": "spk_0"}
        >>> events = session.ingest_chunk(chunk)
        >>>
        >>> # Check enriched segment
        >>> for event in events:
        ...     if event.type == StreamEventType.FINAL_SEGMENT:
        ...         audio_state = event.segment.audio_state
        ...         if audio_state:
        ...             print(audio_state["rendering"])
        >>>
        >>> # Finalize stream
        >>> final_events = session.end_of_stream()

    Callbacks (v1.9.0):
        The session supports optional callbacks for real-time event handling:

        >>> class MyCallbacks:
        ...     def on_segment_finalized(self, segment):
        ...         print(f"Finalized: {segment.text}")
        ...
        >>> session = StreamingEnrichmentSession(
        ...     wav_path,
        ...     config=config,
        ...     callbacks=MyCallbacks()
        ... )

        See streaming_callbacks.StreamCallbacks for the full interface.

    Performance notes:
        - Prosody extraction: ~5-20ms per segment
        - Emotion extraction: ~50-200ms per segment (GPU-accelerated)
        - Total latency budget: ~60-220ms for full enrichment
        - For real-time streaming (<500ms latency), consider disabling emotion
    """

    def __init__(
        self,
        wav_path: Path | str,
        config: StreamingEnrichmentConfig | None = None,
        callbacks: StreamCallbacks | object | None = None,
    ) -> None:
        """
        Initialize streaming enrichment session.

        Args:
            wav_path: Path to normalized 16kHz mono WAV file for audio extraction.
            config: Enrichment configuration. Defaults to no enrichment.
            callbacks: Optional callbacks for real-time event handling.
                      Callback exceptions are caught and logged; they never
                      crash the streaming pipeline.

        Raises:
            FileNotFoundError: If wav_path does not exist.
            RuntimeError: If audio file cannot be opened.
        """
        self.config = config or StreamingEnrichmentConfig()
        self._callbacks = callbacks
        self.wav_path = Path(wav_path)

        # Initialize base streaming session
        self._base_session = StreamingSession(self.config.base_config)

        # Initialize audio extractor for efficient segment access
        try:
            self._extractor = AudioSegmentExtractor(self.wav_path)
            logger.info(
                "Initialized audio extractor: %s (duration=%.2fs, sr=%dHz)",
                self.wav_path.name,
                self._extractor.duration_seconds,
                self._extractor.sample_rate,
            )
        except Exception as e:
            logger.error("Failed to initialize audio extractor: %s", e, exc_info=True)
            raise

        # Track session state
        self._chunk_count = 0
        self._segment_count = 0
        self._enrichment_errors = 0

        # Track speaker turns for on_speaker_turn callback
        self._current_turn_segments: list[StreamSegment] = []
        self._current_turn_speaker: str | None = None
        self._turn_counter = 0

    def ingest_chunk(self, chunk: StreamChunk) -> list[StreamEvent]:
        """
        Ingest a streaming chunk and return enriched events.

        This method wraps the base session's ingest_chunk() and enriches
        any FINAL segments with audio features. PARTIAL segments are passed
        through without enrichment to minimize latency.

        Args:
            chunk: Input chunk with start, end, text, and optional speaker_id.

        Returns:
            List of StreamEvent objects. Each FINAL_SEGMENT event will have
            an enriched segment with audio_state populated (if enrichment is
            enabled and succeeds). PARTIAL_SEGMENT events are unenriched.

        Raises:
            ValueError: If chunk violates monotonic time ordering.

        Example:
            >>> chunk = {"start": 0.0, "end": 1.5, "text": "Hello", "speaker_id": None}
            >>> events = session.ingest_chunk(chunk)
            >>> assert len(events) == 1
            >>> assert events[0].type == StreamEventType.PARTIAL_SEGMENT
        """
        self._chunk_count += 1

        # Get base events from underlying session
        base_events = self._base_session.ingest_chunk(chunk)

        # Enrich FINAL segments, pass through PARTIAL segments
        enriched_events: list[StreamEvent] = []
        for event in base_events:
            if event.type == StreamEventType.FINAL_SEGMENT:
                enriched_segment = self._enrich_stream_segment(event.segment)
                enriched_events.append(
                    StreamEvent(type=StreamEventType.FINAL_SEGMENT, segment=enriched_segment)
                )
                self._segment_count += 1
                logger.debug(
                    "Finalized and enriched segment %d: [%.2fs - %.2fs] '%s'",
                    self._segment_count,
                    enriched_segment.start,
                    enriched_segment.end,
                    enriched_segment.text[:50],
                )

                # Track speaker turns and detect boundaries
                self._track_speaker_turn(enriched_segment)

                # Invoke callback for finalized segment
                invoke_callback_safely(
                    self._callbacks,
                    "on_segment_finalized",
                    enriched_segment,
                )
            else:
                # PARTIAL segments passed through without enrichment
                enriched_events.append(event)

        return enriched_events

    def end_of_stream(self) -> list[StreamEvent]:
        """
        Finalize the stream and enrich any remaining segments.

        Flushes the base session, enriching any partial segment that becomes
        final. After this call, the session is cleared and ready for reuse
        (call reset() to reinitialize if needed).

        Returns:
            List of FINAL_SEGMENT events with enriched audio_state.

        Example:
            >>> final_events = session.end_of_stream()
            >>> for event in final_events:
            ...     assert event.type == StreamEventType.FINAL_SEGMENT
            ...     if event.segment.audio_state:
            ...         print(event.segment.audio_state["rendering"])
        """
        # Get final segment from base session
        base_events = self._base_session.end_of_stream()

        # Enrich all final segments
        enriched_events: list[StreamEvent] = []
        for event in base_events:
            enriched_segment = self._enrich_stream_segment(event.segment)
            enriched_events.append(
                StreamEvent(type=StreamEventType.FINAL_SEGMENT, segment=enriched_segment)
            )
            self._segment_count += 1

            # Invoke callback for finalized segment
            invoke_callback_safely(
                self._callbacks,
                "on_segment_finalized",
                enriched_segment,
            )

        # Finalize any remaining turn
        self._finalize_current_turn()

        logger.info(
            "Stream ended: %d chunks ingested, %d segments finalized, %d enrichment errors",
            self._chunk_count,
            self._segment_count,
            self._enrichment_errors,
        )

        return enriched_events

    def reset(self) -> None:
        """
        Reset session state for reuse.

        Clears the base session and resets counters. The AudioSegmentExtractor
        is preserved for continued use with the same audio file.

        This is useful for processing multiple streams from the same audio
        file without recreating the session.

        Example:
            >>> session.reset()
            >>> # Process new stream with same audio file
            >>> events = session.ingest_chunk(first_chunk)
        """
        self._base_session = StreamingSession(self.config.base_config)
        self._chunk_count = 0
        self._segment_count = 0
        self._enrichment_errors = 0
        self._current_turn_segments = []
        self._current_turn_speaker = None
        self._turn_counter = 0
        logger.debug("Session reset")

    def _enrich_stream_segment(self, segment: StreamSegment) -> StreamSegment:
        """
        Enrich a finalized stream segment with audio features.

        Internal helper that converts a StreamSegment to a temporary Segment,
        extracts audio features using the reusable AudioSegmentExtractor, and
        returns an enriched StreamSegment.

        Args:
            segment: Finalized stream segment to enrich.

        Returns:
            New StreamSegment with audio_state populated. If enrichment fails
            or is disabled, audio_state will be None or contain error status.
        """
        # Skip enrichment if all features disabled
        if not (
            self.config.enable_prosody
            or self.config.enable_emotion
            or self.config.enable_categorical_emotion
        ):
            return segment

        try:
            # Convert StreamSegment to temporary Segment for enrichment
            temp_segment = Segment(
                id=self._segment_count,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker={"id": segment.speaker_id} if segment.speaker_id else None,
            )

            # Extract audio features using shared extractor
            audio_state = _enrich_segment_with_extractor(
                extractor=self._extractor,
                wav_path=self.wav_path,
                segment=temp_segment,
                enable_prosody=self.config.enable_prosody,
                enable_emotion=self.config.enable_emotion,
                enable_categorical_emotion=self.config.enable_categorical_emotion,
                speaker_baseline=self.config.speaker_baseline,
            )

            # Check for errors
            extraction_status = audio_state.get("extraction_status", {})
            errors = extraction_status.get("errors", [])
            if errors:
                self._enrichment_errors += 1
                logger.warning(
                    "Enrichment completed with errors for segment [%.2fs - %.2fs]: %s",
                    segment.start,
                    segment.end,
                    errors,
                )

                # Invoke on_error callback for enrichment errors
                error = StreamingError(
                    exception=RuntimeError("; ".join(errors)),
                    context="Enrichment completed with partial errors",
                    segment_start=segment.start,
                    segment_end=segment.end,
                    recoverable=True,
                )
                invoke_callback_safely(self._callbacks, "on_error", error)

            # Return new StreamSegment with audio_state
            return StreamSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker_id=segment.speaker_id,
                audio_state=audio_state,
            )

        except Exception as e:
            # Graceful degradation on enrichment failure
            self._enrichment_errors += 1
            error_msg = f"Failed to enrich segment: {e}"
            logger.error(error_msg, exc_info=True)

            # Invoke on_error callback for enrichment failure
            error = StreamingError(
                exception=e,
                context="Failed to enrich segment",
                segment_start=segment.start,
                segment_end=segment.end,
                recoverable=True,
            )
            invoke_callback_safely(self._callbacks, "on_error", error)

            # Return segment with error audio_state
            return StreamSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker_id=segment.speaker_id,
                audio_state={
                    "prosody": None,
                    "emotion": None,
                    "rendering": "[audio: neutral]",
                    "extraction_status": {
                        "prosody": "failed",
                        "emotion_dimensional": "failed",
                        "emotion_categorical": "failed",
                        "errors": [error_msg],
                    },
                },
            )

    def _track_speaker_turn(self, segment: StreamSegment) -> None:
        """
        Track speaker turns and invoke on_speaker_turn callback when a turn boundary is detected.

        A turn boundary occurs when:
        1. Speaker changes (different speaker_id)
        2. First segment with a speaker (starting a new turn)

        Args:
            segment: The finalized segment to track.
        """
        # Check if we have a speaker change
        speaker_changed = segment.speaker_id != self._current_turn_speaker

        if speaker_changed:
            # Finalize the current turn before starting a new one
            self._finalize_current_turn()

            # Start new turn with this segment
            self._current_turn_speaker = segment.speaker_id
            self._current_turn_segments = [segment]
        else:
            # Continue current turn
            self._current_turn_segments.append(segment)

    def _finalize_current_turn(self) -> None:
        """
        Finalize the current turn and invoke on_speaker_turn callback.

        Builds a turn dict from accumulated segments and calls the callback
        if there are any segments in the current turn.
        """
        if not self._current_turn_segments:
            return

        # Build turn dict
        turn_dict = self._build_turn_dict(
            turn_id=self._turn_counter,
            segments=self._current_turn_segments,
            speaker_id=self._current_turn_speaker,
        )

        # Increment turn counter
        self._turn_counter += 1

        # Invoke callback
        invoke_callback_safely(
            self._callbacks,
            "on_speaker_turn",
            turn_dict,
        )

        logger.debug(
            "Finalized turn %s: speaker=%s, segments=%d, duration=%.2fs",
            turn_dict["id"],
            turn_dict["speaker_id"],
            len(turn_dict["segment_ids"]),
            turn_dict["end"] - turn_dict["start"],
        )

        # Clear turn buffer
        self._current_turn_segments = []
        self._current_turn_speaker = None

    def _build_turn_dict(
        self,
        turn_id: int,
        segments: list[StreamSegment],
        speaker_id: str | None,
    ) -> dict[str, Any]:
        """
        Build a turn dictionary from accumulated segments.

        Args:
            turn_id: Sequential turn identifier.
            segments: List of segments in this turn.
            speaker_id: Speaker ID for this turn.

        Returns:
            Turn dict with keys: id, speaker_id, start, end, segment_ids, text.
        """
        if not segments:
            raise ValueError("Cannot build turn from empty segments list")

        # Extract segment IDs (use segment count as proxy since StreamSegment doesn't have id)
        # We'll use the index in the overall stream as a rough segment ID
        segment_ids = list(range(self._segment_count - len(segments), self._segment_count))

        # Aggregate text from all segments
        texts = [seg.text.strip() for seg in segments if seg.text.strip()]

        # Time bounds
        start = segments[0].start
        end = segments[-1].end

        return {
            "id": f"turn_{turn_id}",
            "speaker_id": speaker_id or "unknown",
            "start": start,
            "end": end,
            "segment_ids": segment_ids,
            "text": " ".join(texts),
        }

    def get_stats(self) -> dict[str, Any]:
        """
        Get session statistics for monitoring.

        Returns:
            Dictionary with keys:
                - chunk_count: Number of chunks ingested
                - segment_count: Number of segments finalized
                - enrichment_errors: Number of enrichment failures
                - audio_duration_sec: Total audio file duration
                - audio_sample_rate: Audio sample rate

        Example:
            >>> stats = session.get_stats()
            >>> print(f"Processed {stats['chunk_count']} chunks")
            >>> print(f"Finalized {stats['segment_count']} segments")
            >>> print(f"Error rate: {stats['enrichment_errors'] / stats['segment_count']:.1%}")
        """
        return {
            "chunk_count": self._chunk_count,
            "segment_count": self._segment_count,
            "enrichment_errors": self._enrichment_errors,
            "audio_duration_sec": self._extractor.duration_seconds,
            "audio_sample_rate": self._extractor.sample_rate,
        }


# Export public API
__all__ = [
    "StreamingEnrichmentConfig",
    "StreamingEnrichmentSession",
    # Re-export callback types for convenience
    "StreamCallbacks",
    "StreamingError",
]
