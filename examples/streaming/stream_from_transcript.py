"""
Replay an existing transcript as streaming chunks.

This demonstrates how `StreamingSession` stitches incoming post-ASR chunks into
partial/final segment events without dealing with raw audio.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from slower_whisper.pipeline import (
    StreamChunk,
    StreamConfig,
    StreamEvent,
    StreamingSession,
    Transcript,
    load_transcript,
)


def transcript_to_stream_chunks(
    transcript: Transcript, gap_sec: float, chunk_size: int
) -> Iterable[StreamChunk]:
    """
    Convert transcript segments into smaller stream chunks with artificial gaps.

    Each segment text is broken into `chunk_size` word groups so that a single
    segment produces multiple partial updates before it finalizes.
    """

    cursor = 0.0
    for segment in transcript.segments:
        words = segment.text.split()
        if not words:
            continue

        parts = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
        duration = max(segment.end - segment.start, 0.6)
        step = duration / len(parts)
        speaker_id = (segment.speaker or {}).get("id")

        for text in parts:
            start = cursor
            end = start + step
            yield {
                "start": start,
                "end": end,
                "text": text,
                "speaker_id": speaker_id,
            }
            cursor = end

        cursor += gap_sec


def print_event(event: StreamEvent) -> None:
    """Pretty-print a streaming event."""

    segment = event.segment
    label = "FINAL" if event.type.value == "final_segment" else "PARTIAL"
    speaker = segment.speaker_id or "unknown"
    print(
        f"[{label:<7}] {segment.start:>6.2f}-{segment.end:>6.2f} | speaker={speaker:<8} | {segment.text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a transcript as streaming chunks to demonstrate StreamingSession"
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=Path("whisper_json/sample.json"),
        help="Path to a transcript JSON file (schema v2)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Number of words per emitted stream chunk",
    )
    parser.add_argument(
        "--gap-sec",
        type=float,
        default=0.25,
        help="Gap to insert between segments when replaying",
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=0.75,
        help="Maximum gap that still stitches chunks into the same segment",
    )
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")

    transcript = load_transcript(args.transcript)
    session = StreamingSession(StreamConfig(max_gap_sec=args.max_gap_sec))

    for chunk in transcript_to_stream_chunks(transcript, args.gap_sec, args.chunk_size):
        for event in session.ingest_chunk(chunk):
            print_event(event)

    for event in session.end_of_stream():
        print_event(event)


if __name__ == "__main__":
    main()
