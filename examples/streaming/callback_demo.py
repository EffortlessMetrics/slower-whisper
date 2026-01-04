"""
Streaming callbacks demo: real-time event handling with StreamCallbacks (v1.9.0).

This example demonstrates how to use the StreamCallbacks protocol for real-time
event handling during streaming enrichment. Callbacks enable reactive patterns
like:

- Logging/monitoring segment progress
- Persisting segments to a database as they finalize
- Triggering alerts on errors or specific semantic patterns
- Updating UI in real-time

**Key Design Points**:

- Callbacks are optional - implement only the methods you need
- Callback exceptions are caught and logged; they never crash the pipeline
- on_error is invoked for both enrichment errors AND callback exceptions
- All callbacks receive structured data (dataclasses/typed dicts)

**The Callback Protocol**:

    class StreamCallbacks(Protocol):
        def on_segment_finalized(self, segment: StreamSegment) -> None: ...
        def on_speaker_turn(self, turn: dict) -> None: ...
        def on_semantic_update(self, payload: SemanticUpdatePayload) -> None: ...
        def on_error(self, error: StreamingError) -> None: ...

**Usage**:

    # Basic demo (uses sample transcript)
    uv run python examples/streaming/callback_demo.py

    # With custom audio file
    uv run python examples/streaming/callback_demo.py --audio path/to/audio.wav

    # Enable prosody enrichment (requires enrich-prosody extra)
    uv run python examples/streaming/callback_demo.py --enable-prosody

**Architecture**:

    Audio + Transcript → StreamingEnrichmentSession → Callbacks → Display
                              |                           |
                         enriches segments          on_segment_finalized
                         detects errors             on_error
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from transcription import (
    SemanticUpdatePayload,
    StreamChunk,
    StreamConfig,
    StreamingEnrichmentConfig,
    StreamingEnrichmentSession,
    StreamingError,
    Transcript,
    load_transcript,
)
from transcription.streaming import StreamEventType, StreamSegment

# ===========================
# Example Callback Implementation
# ===========================


@dataclass
class LoggingCallbacks:
    """
    Example callback implementation that logs all events to console.

    This demonstrates the callback pattern: implement only the methods you need.
    All methods are optional - unimplemented methods default to no-op behavior.

    Attributes:
        verbose: If True, prints detailed segment info including audio_state.
        segment_count: Running count of finalized segments.
        error_count: Running count of errors encountered.
        segments: List of all finalized segments (for post-processing).
    """

    verbose: bool = False
    segment_count: int = field(default=0, init=False)
    error_count: int = field(default=0, init=False)
    segments: list[StreamSegment] = field(default_factory=list, init=False)

    def on_segment_finalized(self, segment: StreamSegment) -> None:
        """Called when a segment is finalized with enrichment complete.

        This is the primary callback for consuming enriched segments.
        The segment will have audio_state populated if enrichment was
        enabled and succeeded.

        Args:
            segment: The finalized segment with optional audio_state.
        """
        self.segment_count += 1
        self.segments.append(segment)

        # Format timestamp range
        time_range = f"{segment.start:>6.2f}s - {segment.end:>6.2f}s"
        speaker = segment.speaker_id or "unknown"

        print(f"\n[FINALIZED #{self.segment_count}] {time_range} | speaker={speaker}")
        print(f'  Text: "{segment.text}"')

        # Show audio_state if present and verbose
        if segment.audio_state and self.verbose:
            rendering = segment.audio_state.get("rendering", "[no rendering]")
            print(f"  Audio: {rendering}")

            # Show extraction status
            status = segment.audio_state.get("extraction_status", {})
            prosody_status = status.get("prosody", "n/a")
            emotion_status = status.get("emotion_dimensional", "n/a")
            print(f"  Status: prosody={prosody_status}, emotion={emotion_status}")

        elif segment.audio_state:
            # Brief rendering in non-verbose mode
            rendering = segment.audio_state.get("rendering", "")
            if rendering and rendering != "[audio: neutral]":
                print(f"  Audio: {rendering}")

    def on_speaker_turn(self, turn: dict) -> None:
        """Called when a speaker turn is detected.

        A turn is a contiguous sequence of segments from the same speaker.
        This callback fires when a turn boundary is detected.

        Args:
            turn: Turn dictionary with speaker_id, start, end, text, etc.
        """
        turn_id = turn.get("id", "unknown")
        speaker = turn.get("speaker_id", "unknown")
        text_preview = turn.get("text", "")[:50]

        print(f"\n[TURN] {turn_id} | speaker={speaker}")
        print(f'  Preview: "{text_preview}..."')

    def on_semantic_update(self, payload: SemanticUpdatePayload) -> None:
        """Called when semantic annotations are computed for a turn.

        Semantic updates include keywords, risk tags, and action items.

        Args:
            payload: SemanticUpdatePayload with turn, keywords, risk_tags, etc.
        """
        turn_id = payload.turn.id
        keywords = payload.keywords
        risk_tags = payload.risk_tags
        actions = payload.actions

        print(f"\n[SEMANTIC] {turn_id}")
        if keywords:
            print(f"  Keywords: {keywords}")
        if risk_tags:
            print(f"  Risk tags: {risk_tags}")
        if actions:
            action_texts = [a.get("text", str(a)) for a in actions]
            print(f"  Actions: {action_texts}")

    def on_error(self, error: StreamingError) -> None:
        """Called when an error occurs during streaming.

        This includes enrichment failures, callback exceptions, and
        other recoverable errors. The streaming pipeline continues
        after recoverable errors.

        Args:
            error: StreamingError with exception details and context.
        """
        self.error_count += 1

        severity = "RECOVERABLE" if error.recoverable else "FATAL"
        print(f"\n[ERROR - {severity}] {error.context}")
        print(f"  Exception: {error.exception}")

        if error.segment_start is not None and error.segment_end is not None:
            print(f"  Segment: {error.segment_start:.2f}s - {error.segment_end:.2f}s")

    def print_summary(self) -> None:
        """Print summary statistics at the end of the stream."""
        print("\n" + "=" * 60)
        print("CALLBACK SUMMARY")
        print("=" * 60)
        print(f"Segments finalized: {self.segment_count}")
        print(f"Errors encountered: {self.error_count}")

        if self.segments:
            total_duration = self.segments[-1].end - self.segments[0].start
            print(f"Total duration: {total_duration:.2f}s")

            # Count segments with audio enrichment
            enriched = sum(1 for s in self.segments if s.audio_state)
            print(f"Segments with audio_state: {enriched}/{self.segment_count}")


# ===========================
# Transcript Replay Helpers
# ===========================


def transcript_to_stream_chunks(
    transcript: Transcript,
    chunk_size: int = 3,
) -> list[StreamChunk]:
    """
    Convert transcript segments into stream chunks for replay.

    This simulates ASR output by breaking transcript segments into
    smaller word-group chunks.

    Args:
        transcript: Source transcript to replay.
        chunk_size: Number of words per chunk.

    Returns:
        List of StreamChunk dictionaries.
    """
    chunks: list[StreamChunk] = []
    cursor = 0.0

    for segment in transcript.segments:
        words = segment.text.split()
        if not words:
            continue

        # Split into word groups
        parts = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
        duration = max(segment.end - segment.start, 0.6)
        step = duration / len(parts)

        # Extract speaker ID
        speaker_dict = segment.speaker if isinstance(segment.speaker, dict) else {}
        speaker_id = speaker_dict.get("id")

        for text in parts:
            start = cursor
            end = start + step
            chunks.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "speaker_id": speaker_id,
                }
            )
            cursor = end

        # Small gap between segments
        cursor += 0.1

    return chunks


# ===========================
# Main Demo
# ===========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming callbacks demo: real-time event handling (v1.9.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic demo with sample transcript
  %(prog)s

  # With custom audio and transcript
  %(prog)s --audio path/to/audio.wav --transcript path/to/transcript.json

  # Enable prosody enrichment
  %(prog)s --audio audio.wav --enable-prosody

  # Verbose mode (show full audio_state)
  %(prog)s --audio audio.wav --enable-prosody --verbose
        """,
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Path to normalized WAV file (16kHz mono). Required for enrichment.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=Path("whisper_json/sample.json"),
        help="Path to transcript JSON file to replay (default: whisper_json/sample.json)",
    )
    parser.add_argument(
        "--enable-prosody",
        action="store_true",
        help="Enable prosody feature extraction (requires audio file)",
    )
    parser.add_argument(
        "--enable-emotion",
        action="store_true",
        help="Enable emotion feature extraction (requires audio file + torch)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed audio_state for each segment",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Number of words per simulated ASR chunk (default: 3)",
    )
    args = parser.parse_args()

    # Validation
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")

    # Check if audio file is required but not provided
    if (args.enable_prosody or args.enable_emotion) and not args.audio:
        parser.error("--audio is required when using --enable-prosody or --enable-emotion")

    # Check if audio file exists
    if args.audio and not args.audio.exists():
        print(f"Error: Audio file not found: {args.audio}")
        print("\nTo run without audio enrichment, omit the --audio flag:")
        print(f"  python {sys.argv[0]}")
        sys.exit(1)

    # Check if transcript exists
    if not args.transcript.exists():
        print(f"Error: Transcript file not found: {args.transcript}")
        print("\nPlease provide a valid transcript JSON file.")
        print("You can create one by running the transcription pipeline first.")
        sys.exit(1)

    # Print header
    print("=" * 60)
    print("STREAMING CALLBACKS DEMO (v1.9.0)")
    print("=" * 60)
    print(f"\nTranscript: {args.transcript}")
    if args.audio:
        print(f"Audio: {args.audio}")
        print(f"Prosody: {'enabled' if args.enable_prosody else 'disabled'}")
        print(f"Emotion: {'enabled' if args.enable_emotion else 'disabled'}")
    else:
        print("Audio: not provided (running without enrichment)")
    print(f"Verbose: {args.verbose}")
    print()

    # Load transcript for replay
    transcript = load_transcript(args.transcript)
    print(f"Loaded transcript with {len(transcript.segments)} segments")

    # Create callback handler
    callbacks = LoggingCallbacks(verbose=args.verbose)

    # Create enrichment configuration
    config = StreamingEnrichmentConfig(
        base_config=StreamConfig(max_gap_sec=0.75),
        enable_prosody=args.enable_prosody and args.audio is not None,
        enable_emotion=args.enable_emotion and args.audio is not None,
    )

    # Convert transcript to stream chunks
    chunks = transcript_to_stream_chunks(transcript, args.chunk_size)
    print(f"Generated {len(chunks)} simulated ASR chunks")
    print("\nStreaming with callbacks...")
    print("-" * 60)

    if args.audio:
        # Full enrichment mode with audio file
        session = StreamingEnrichmentSession(
            wav_path=args.audio,
            config=config,
            callbacks=callbacks,
        )

        # Process chunks through session
        for chunk in chunks:
            events = session.ingest_chunk(chunk)
            # Events are handled by callbacks; we could also process them here
            for event in events:
                if event.type == StreamEventType.PARTIAL_SEGMENT:
                    # Optionally show partial progress
                    pass

        # Finalize stream
        session.end_of_stream()

        # Print session stats
        stats = session.get_stats()
        print("\n" + "-" * 60)
        print(
            f"Session stats: {stats['chunk_count']} chunks, "
            f"{stats['segment_count']} segments, "
            f"{stats['enrichment_errors']} errors"
        )
    else:
        # Demo mode without audio file - simulate callback invocations
        print("\n[INFO] Running in demo mode without audio enrichment.")
        print("[INFO] Callbacks will be invoked with unenriched segments.\n")

        # Import StreamingSession for basic operation
        from transcription import StreamingSession

        session = StreamingSession(StreamConfig(max_gap_sec=0.75))

        for chunk in chunks:
            events = session.ingest_chunk(chunk)
            for event in events:
                if event.type == StreamEventType.FINAL_SEGMENT:
                    # Manually invoke callback for demo
                    callbacks.on_segment_finalized(event.segment)

        # Finalize and invoke callbacks for any remaining segment
        for event in session.end_of_stream():
            callbacks.on_segment_finalized(event.segment)

    # Print summary
    callbacks.print_summary()


if __name__ == "__main__":
    main()
