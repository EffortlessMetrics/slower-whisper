"""Server-Sent Events helpers for streaming transcription."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from .config import TranscriptionConfig
from .service_serialization import _segment_to_dict

logger = logging.getLogger(__name__)


class SSEStreamingSession:
    """
    Manages state for an SSE streaming transcription session.

    Uses the same event envelope format as WebSocket streaming for
    consistency across the API.

    Attributes:
        stream_id: Unique identifier for this stream (sse-{uuid4})
        _event_id_counter: Monotonically increasing event ID
        _segment_seq: Segment sequence counter
    """

    def __init__(self) -> None:
        """Initialize a new SSE streaming session."""
        self.stream_id = f"sse-{uuid.uuid4()}"
        self._event_id_counter = 0
        self._segment_seq = 0
        self._start_time = time.time()
        self._chunks_processed = 0
        self._segments_partial = 0
        self._segments_finalized = 0
        self._errors = 0

    def _next_event_id(self) -> int:
        """Generate next monotonically increasing event ID."""
        self._event_id_counter += 1
        return self._event_id_counter

    def _next_segment_id(self) -> str:
        """Generate next segment ID."""
        seg_id = f"seg-{self._segment_seq}"
        self._segment_seq += 1
        return seg_id

    def _server_timestamp(self) -> int:
        """Get current server timestamp in milliseconds."""
        return int(time.time() * 1000)

    def create_envelope(
        self,
        event_type: str,
        payload: dict[str, Any],
        segment_id: str | None = None,
        ts_audio_start: float | None = None,
        ts_audio_end: float | None = None,
    ) -> dict[str, Any]:
        """Create an event envelope with current metadata."""
        result: dict[str, Any] = {
            "event_id": self._next_event_id(),
            "stream_id": self.stream_id,
            "type": event_type,
            "ts_server": self._server_timestamp(),
            "payload": payload,
        }
        if segment_id is not None:
            result["segment_id"] = segment_id
        if ts_audio_start is not None:
            result["ts_audio_start"] = ts_audio_start
        if ts_audio_end is not None:
            result["ts_audio_end"] = ts_audio_end
        return result

    def get_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        duration = time.time() - self._start_time
        return {
            "segments_partial": self._segments_partial,
            "segments_finalized": self._segments_finalized,
            "errors": self._errors,
            "duration_sec": round(duration, 3),
        }


async def _generate_sse_transcription(
    audio_path: Path,
    config: TranscriptionConfig,
    include_words: bool,
) -> AsyncGenerator[str, None]:
    """Generate SSE events during transcription using event envelope format.

    Uses a thread pool to run the synchronous transcription engine while
    yielding events as segments are produced. Events follow the same
    envelope structure as WebSocket streaming for API consistency.

    Event types emitted:
    - PARTIAL: Intermediate segment (emitted during transcription progress)
    - FINALIZED: Final segment with complete transcription
    - SESSION_ENDED: Stream complete with statistics
    - ERROR: Error occurred during processing

    Args:
        audio_path: Path to the normalized audio file
        config: Transcription configuration
        include_words: Whether to include word-level timestamps

    Yields:
        SSE-formatted event strings: "data: {event_envelope_json}\\n\\n"
    """
    import asyncio
    import queue
    import threading

    from .asr_engine import TranscriptionEngine
    from .config import AsrConfig
    from .diarization_orchestrator import _maybe_run_diarization
    from .transcription_helpers import _maybe_build_chunks

    # Create session for tracking state
    session = SSEStreamingSession()

    # Create a queue to communicate between threads
    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def run_transcription() -> None:
        """Run transcription in a background thread, pushing events to the queue."""
        try:
            # Create ASR configuration
            asr_cfg = AsrConfig(
                model_name=config.model,
                device=config.device,
                compute_type=config.compute_type,
                vad_min_silence_ms=config.vad_min_silence_ms,
                beam_size=config.beam_size,
                language=config.language,
                task=config.task,
                word_timestamps=config.word_timestamps,
            )

            # Initialize the transcription engine
            engine = TranscriptionEngine(asr_cfg)

            # Check if audio file exists
            if not audio_path.exists():
                session._errors += 1
                event_queue.put(
                    session.create_envelope(
                        "ERROR",
                        {
                            "code": "file_not_found",
                            "message": "Audio file not found",
                            "recoverable": False,
                        },
                    )
                )
                event_queue.put(None)
                return

            # Perform transcription using the engine
            logger.info("SSE: Starting transcription for %s", audio_path.name)

            transcript = engine.transcribe_file(audio_path)

            # Emit segments - first as PARTIAL, then final one as FINALIZED
            total_segments = len(transcript.segments)
            for idx, segment in enumerate(transcript.segments):
                seg_dict = _segment_to_dict(segment, include_words=include_words)
                segment_id = session._next_segment_id()
                is_last = idx == total_segments - 1

                # Emit PARTIAL for intermediate segments during processing
                if not is_last:
                    session._segments_partial += 1
                    event_queue.put(
                        session.create_envelope(
                            "PARTIAL",
                            {"segment": seg_dict},
                            segment_id=segment_id,
                            ts_audio_start=segment.start,
                            ts_audio_end=segment.end,
                        )
                    )
                else:
                    # Last segment gets FINALIZED
                    session._segments_finalized += 1
                    event_queue.put(
                        session.create_envelope(
                            "FINALIZED",
                            {"segment": seg_dict},
                            segment_id=segment_id,
                            ts_audio_start=segment.start,
                            ts_audio_end=segment.end,
                        )
                    )

            # Run diarization if enabled
            if config.enable_diarization:
                transcript = _maybe_run_diarization(transcript, audio_path, config)
                transcript = _maybe_build_chunks(transcript, config)

                # Re-emit segments with speaker info as FINALIZED updates
                for segment in transcript.segments:
                    if segment.speaker:
                        seg_dict = _segment_to_dict(segment, include_words=include_words)
                        segment_id = session._next_segment_id()
                        session._segments_finalized += 1
                        event_queue.put(
                            session.create_envelope(
                                "FINALIZED",
                                {"segment": seg_dict},
                                segment_id=segment_id,
                                ts_audio_start=segment.start,
                                ts_audio_end=segment.end,
                            )
                        )

            # Emit SESSION_ENDED event with metadata and statistics
            event_queue.put(
                session.create_envelope(
                    "SESSION_ENDED",
                    {
                        "stats": session.get_stats(),
                        "total_segments": len(transcript.segments),
                        "language": transcript.language,
                        "file_name": transcript.file_name,
                        "speakers": transcript.speakers,
                        "meta": transcript.meta,
                    },
                )
            )

        except Exception as e:
            logger.exception("SSE transcription error: %s", e)
            session._errors += 1
            event_queue.put(
                session.create_envelope(
                    "ERROR",
                    {
                        "code": "transcription_error",
                        "message": "Transcription failed",
                        "recoverable": False,
                    },
                )
            )

        finally:
            # Signal end of stream
            event_queue.put(None)

    # Start transcription in background thread
    thread = threading.Thread(target=run_transcription, daemon=True)
    thread.start()

    # Yield SSE events as they arrive
    try:
        while True:
            try:
                # Use asyncio.to_thread for non-blocking queue access
                event = await asyncio.to_thread(event_queue.get, timeout=0.1)
                if event is None:
                    break
                # SSE format: "data: {json}\n\n"
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # No event yet, continue polling
                continue
            except Exception:
                # Queue access error, break the loop
                break
    finally:
        # Ensure thread cleanup
        thread.join(timeout=1.0)
