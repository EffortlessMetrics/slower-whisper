"""
Live streaming semantics demo: streaming + semantic annotation + mock LLM.

This example demonstrates a complete streaming conversation pipeline with:

1. **Streaming replay**: Simulates real-time ASR by replaying existing transcript
   as small chunks with artificial gaps (reuses StreamingSession)

2. **Semantic annotation**: Extracts risk tags (churn_risk, escalation, pricing)
   and action items from finalized segments using KeywordSemanticAnnotator

3. **Mock LLM integration**: Demonstrates how downstream systems can consume
   semantically-enriched segments with a simple mock LLM client

4. **Summary statistics**: Displays aggregate stats at the end

**Key Design Points**:

- Semantic annotation runs ONLY on finalized segments (not partials)
- Mock LLM generates template responses without API keys/network calls
- Clean separation: streaming → semantic → LLM consumption
- Production-ready pattern: replace MockLLMClient with real API client

**Usage**:

    # Basic demo (no LLM responses)
    uv run python examples/streaming/live_semantics_demo.py whisper_json/sample.json

    # With mock LLM responses
    uv run python examples/streaming/live_semantics_demo.py \\
        whisper_json/sample.json --enable-llm

    # Faster replay with larger chunks
    uv run python examples/streaming/live_semantics_demo.py \\
        whisper_json/sample.json --chunk-size 5 --speed 2.0

**Architecture**:

    Audio (replay) → StreamingSession → Semantic Annotator → Mock LLM → Display
                         |                     |                  |
                     partials            final segments    optional insights
                      + finals
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from transcription import (
    KeywordSemanticAnnotator,
    Segment,
    StreamChunk,
    StreamConfig,
    StreamEvent,
    StreamingSession,
    Transcript,
    load_transcript,
)
from transcription.streaming import StreamEventType

# ===========================
# Mock LLM Client
# ===========================


@dataclass(slots=True)
class MockLLMClient:
    """
    Placeholder LLM client that returns template responses.

    In production, replace this with a real API client (OpenAI, Anthropic, etc.)
    that consumes semantically-enriched segments and generates actionable insights.
    """

    enabled: bool = False

    def analyze_segment(self, segment: Segment, semantic: dict) -> str | None:
        """
        Generate mock LLM response for a finalized segment with semantic tags.

        Args:
            segment: Finalized segment from streaming session
            semantic: Semantic annotations dict with keywords, risk_tags, actions

        Returns:
            Mock insight string or None if LLM disabled
        """
        if not self.enabled:
            return None

        risk_tags = semantic.get("risk_tags", [])
        keywords = semantic.get("keywords", [])
        text = segment.text.strip()

        # Template-based responses based on risk tags
        if "churn_risk" in risk_tags:
            return f"Customer expressing cancellation intent. Priority: HIGH. Keywords: {keywords}"
        elif "escalation" in risk_tags:
            return f"Escalation requested. Requires supervisor attention. Keywords: {keywords}"
        elif "pricing" in risk_tags:
            return f"Pricing concern detected. Consider discount offer. Keywords: {keywords}"
        elif semantic.get("actions"):
            action = semantic["actions"][0]
            return f"Action item: {action.get('pattern', 'commitment')} detected"
        else:
            # Generic sentiment response for segments without specific risks
            if len(text) > 50:
                return "Standard response pattern. No immediate action required."
            return None


# ===========================
# Semantic Annotation Helper
# ===========================


def create_segment_from_stream(stream_segment, segment_id: int) -> Segment:
    """
    Convert a StreamSegment to a Segment for semantic annotation.

    Args:
        stream_segment: StreamSegment from streaming session
        segment_id: Sequential segment ID

    Returns:
        Segment object compatible with KeywordSemanticAnnotator
    """
    speaker_dict = {"id": stream_segment.speaker_id} if stream_segment.speaker_id else None
    return Segment(
        id=segment_id,
        start=stream_segment.start,
        end=stream_segment.end,
        text=stream_segment.text,
        speaker=speaker_dict,
    )


def annotate_segment(segment: Segment, annotator: KeywordSemanticAnnotator) -> dict:
    """
    Extract semantic annotations from a single finalized segment.

    Args:
        segment: Segment to annotate
        annotator: Pre-configured semantic annotator

    Returns:
        Semantic annotations dict with keywords, risk_tags, actions
    """
    # Create temporary single-segment transcript for annotation
    temp_transcript = Transcript(
        file_name="stream",
        language="en",
        segments=[segment],
        meta={},
    )

    # Run semantic annotator
    annotated = annotator.annotate(temp_transcript)

    # Extract semantic dict from annotations
    annotations = getattr(annotated, "annotations", {}) or {}
    return annotations.get("semantic", {}) or {}


# ===========================
# Streaming Replay
# ===========================


def transcript_to_stream_chunks(
    transcript: Transcript, chunk_size: int, speed: float
) -> Iterable[tuple[StreamChunk, float]]:
    """
    Convert transcript segments into stream chunks with timing information.

    Args:
        transcript: Source transcript to replay
        chunk_size: Number of words per chunk
        speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x faster)

    Yields:
        Tuples of (chunk, delay_sec) where delay_sec is the time to wait
        before emitting the next chunk
    """
    cursor = 0.0

    for segment in transcript.segments:
        words = segment.text.split()
        if not words:
            continue

        # Split segment text into chunk_size word groups
        parts = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
        duration = max(segment.end - segment.start, 0.6)
        step = duration / len(parts)

        # Extract speaker ID from segment
        speaker_dict = segment.speaker if isinstance(segment.speaker, dict) else {}
        speaker_id = speaker_dict.get("id")

        for i, text in enumerate(parts):
            start = cursor
            end = start + step
            chunk: StreamChunk = {
                "start": start,
                "end": end,
                "text": text,
                "speaker_id": speaker_id,
            }

            # Calculate delay based on speed (simulate streaming latency)
            delay = (step / speed) if i < len(parts) - 1 else 0.0

            yield chunk, delay
            cursor = end

        # Add gap between segments
        cursor += 0.25 / speed


# ===========================
# Display Formatting
# ===========================


def format_semantic_tags(semantic: dict) -> str:
    """Format semantic annotations for display."""
    risk_tags = semantic.get("risk_tags", [])
    keywords = semantic.get("keywords", [])

    parts = []
    if risk_tags:
        parts.append(f"[{', '.join(risk_tags)}]")
    if keywords:
        parts.append(f"keywords: {keywords}")

    return " ".join(parts) if parts else "no tags"


def print_event(
    event: StreamEvent,
    semantic: dict | None = None,
    llm_response: str | None = None,
    show_partials: bool = True,
) -> None:
    """
    Pretty-print a streaming event with optional semantic/LLM data.

    Args:
        event: Stream event (partial or final)
        semantic: Optional semantic annotations for finals
        llm_response: Optional LLM insight for finals
        show_partials: Whether to print partial events
    """
    segment = event.segment
    is_final = event.type == StreamEventType.FINAL_SEGMENT

    # Skip partials if requested
    if not is_final and not show_partials:
        return

    label = "FINAL" if is_final else "PARTIAL"
    speaker = segment.speaker_id or "unknown"

    # Base segment info
    print(
        f'\n[{label:<7}] {segment.start:>6.2f}-{segment.end:>6.2f} | {speaker:<8} | "{segment.text}"'
    )

    # Add semantic info for finals
    if is_final and semantic:
        print(f"  → semantic: {format_semantic_tags(semantic)}")

    # Add LLM response for finals
    if is_final and llm_response:
        print(f"  → LLM: {llm_response}")


def print_summary(stats: dict) -> None:
    """Print summary statistics at the end of the stream."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nTotal segments: {stats['total_segments']}")
    print(f"Total partials: {stats['total_partials']}")

    if stats["speaker_counts"]:
        print("\nSpeaker distribution:")
        for speaker, count in sorted(stats["speaker_counts"].items()):
            print(f"  {speaker}: {count} segments")

    if stats["risk_counts"]:
        print("\nRisk tag distribution:")
        for risk, count in sorted(stats["risk_counts"].items(), key=lambda x: -x[1]):
            print(f"  {risk}: {count} occurrences")

    if stats["keywords"]:
        print("\nTop keywords:")
        for keyword, count in stats["keywords"].most_common(10):
            print(f"  {keyword}: {count}")

    if stats["actions_count"] > 0:
        print(f"\nAction items detected: {stats['actions_count']}")

    print()


# ===========================
# Main Demo
# ===========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live streaming semantics demo: streaming + annotation + mock LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic demo
  %(prog)s whisper_json/sample.json

  # With mock LLM responses
  %(prog)s whisper_json/sample.json --enable-llm

  # Fast replay with larger chunks
  %(prog)s whisper_json/sample.json --chunk-size 5 --speed 2.0

  # Hide partial segments (finals only)
  %(prog)s whisper_json/sample.json --no-partials
        """,
    )
    parser.add_argument(
        "transcript",
        type=Path,
        nargs="?",
        default=Path("whisper_json/sample.json"),
        help="Path to transcript JSON file (schema v2)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Number of words per stream chunk (default: 3)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0 = real-time)",
    )
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable mock LLM responses for finalized segments",
    )
    parser.add_argument(
        "--no-partials",
        action="store_true",
        help="Hide partial segments (show finals only)",
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=0.75,
        help="Maximum gap to stitch chunks into same segment (default: 0.75)",
    )
    args = parser.parse_args()

    # Validation
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    if args.speed <= 0:
        parser.error("--speed must be > 0")
    if not args.transcript.exists():
        parser.error(f"Transcript file not found: {args.transcript}")

    print("=" * 80)
    print("LIVE STREAMING SEMANTICS DEMO")
    print("=" * 80)
    print(f"\nSource: {args.transcript}")
    print(f"Config: chunk_size={args.chunk_size}, speed={args.speed}x, llm={args.enable_llm}")
    print(f"Stream: max_gap={args.max_gap_sec}s\n")

    # Initialize components
    transcript = load_transcript(args.transcript)
    session = StreamingSession(StreamConfig(max_gap_sec=args.max_gap_sec))
    annotator = KeywordSemanticAnnotator()
    llm_client = MockLLMClient(enabled=args.enable_llm)

    # Statistics tracking
    stats = {
        "total_segments": 0,
        "total_partials": 0,
        "speaker_counts": Counter(),
        "risk_counts": Counter(),
        "keywords": Counter(),
        "actions_count": 0,
    }

    segment_id_counter = 0

    # Stream chunks with timing
    for chunk, delay in transcript_to_stream_chunks(transcript, args.chunk_size, args.speed):
        # Ingest chunk into streaming session
        for event in session.ingest_chunk(chunk):
            stats["total_partials"] += 1

            if event.type == StreamEventType.FINAL_SEGMENT:
                # Final segment: run semantic annotation
                stats["total_segments"] += 1
                segment_id_counter += 1

                # Convert stream segment to Segment for annotation
                segment = create_segment_from_stream(event.segment, segment_id_counter)

                # Run semantic annotation
                semantic = annotate_segment(segment, annotator)

                # Update statistics
                if event.segment.speaker_id:
                    stats["speaker_counts"][event.segment.speaker_id] += 1

                for risk in semantic.get("risk_tags", []):
                    stats["risk_counts"][risk] += 1

                for keyword in semantic.get("keywords", []):
                    stats["keywords"][keyword] += 1

                stats["actions_count"] += len(semantic.get("actions", []))

                # Get LLM response
                llm_response = llm_client.analyze_segment(segment, semantic)

                # Display final segment with annotations
                print_event(event, semantic, llm_response, show_partials=not args.no_partials)
            else:
                # Partial segment: display without annotations
                print_event(event, show_partials=not args.no_partials)

        # Simulate streaming delay
        if delay > 0:
            time.sleep(delay)

    # End of stream: finalize any remaining partial
    for event in session.end_of_stream():
        if event.type == StreamEventType.FINAL_SEGMENT:
            stats["total_segments"] += 1
            segment_id_counter += 1

            segment = create_segment_from_stream(event.segment, segment_id_counter)
            semantic = annotate_segment(segment, annotator)

            if event.segment.speaker_id:
                stats["speaker_counts"][event.segment.speaker_id] += 1

            for risk in semantic.get("risk_tags", []):
                stats["risk_counts"][risk] += 1

            for keyword in semantic.get("keywords", []):
                stats["keywords"][keyword] += 1

            stats["actions_count"] += len(semantic.get("actions", []))

            llm_response = llm_client.analyze_segment(segment, semantic)
            print_event(event, semantic, llm_response, show_partials=not args.no_partials)

    # Print summary
    print_summary(stats)


if __name__ == "__main__":
    main()
