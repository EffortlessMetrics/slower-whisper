#!/usr/bin/env python3
"""
Chunking Example

This script demonstrates how to segment transcripts into RAG-friendly chunks
while respecting speaker turns.

It uses turn-aware chunking to:
1. Preserve speaker continuity (avoid splitting mid-turn)
2. Detect conversation patterns (rapid turn-taking, overlaps)
3. Maintain optimal chunk sizes for LLM context windows

Usage:
    python chunking_example.py
"""

from typing import Any

# Import from the public API
from transcription import ChunkingConfig, build_chunks
from transcription.models import Chunk, Segment, Transcript, Turn


def create_demo_transcript() -> Transcript:
    """Create a synthetic transcript with conversation patterns.

    Simulates a podcast/interview with:
    - Long monologues (needs splitting by duration)
    - Rapid back-and-forth (needs grouping)
    - Overlapping speech
    """
    segments = []
    turns: list[Turn | dict[str, Any]] = []

    current_time = 0.0

    # 1. Host introduction (Long monologue)
    host_text = "Welcome to the show. " * 20  # ~100 words
    seg = Segment(
        id=0, start=current_time, end=current_time + 30.0, text=host_text, speaker={"id": "host"}
    )
    segments.append(seg)
    turns.append(
        Turn(
            id="turn_0",
            speaker_id="host",
            segment_ids=[0],
            start=current_time,
            end=current_time + 30.0,
            text=host_text,
            metadata={},
        )
    )
    current_time += 31.0

    # 2. Rapid back-and-forth (Interview)
    for i in range(1, 6):
        # Host asks, Guest answers quickly
        segments.append(
            Segment(
                id=len(segments),
                start=current_time,
                end=current_time + 2.0,
                text=f"Question {i}?",
                speaker={"id": "host"},
            )
        )
        turns.append(
            Turn(
                id=f"turn_{len(turns)}",
                speaker_id="host",
                segment_ids=[len(segments) - 1],
                start=current_time,
                end=current_time + 2.0,
                text=f"Question {i}?",
                metadata={},
            )
        )
        current_time += 2.2  # Short gap

        segments.append(
            Segment(
                id=len(segments),
                start=current_time,
                end=current_time + 2.0,
                text=f"Answer {i}.",
                speaker={"id": "guest"},
            )
        )
        turns.append(
            Turn(
                id=f"turn_{len(turns)}",
                speaker_id="guest",
                segment_ids=[len(segments) - 1],
                start=current_time,
                end=current_time + 2.0,
                text=f"Answer {i}.",
                metadata={},
            )
        )
        current_time += 2.2

    return Transcript(file_name="demo_podcast.wav", language="en", segments=segments, turns=turns)


def print_chunk_analysis(chunks: list[Chunk]):
    """Print analysis of generated chunks."""
    print(f"\nGenerated {len(chunks)} chunks:\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk.id}")
        print(f"  Time:   {chunk.start:.1f}s - {chunk.end:.1f}s ({chunk.end - chunk.start:.1f}s)")
        print(f"  Turns:  {len(chunk.turn_ids)} turns ({', '.join(chunk.speaker_ids)})")
        print(f"  Tokens: ~{chunk.token_count_estimate}")

        # Highlight turn-aware features
        flags = []
        if chunk.crosses_turn_boundary:
            flags.append(f"Crosses {chunk.turn_boundary_count} boundaries")
        if chunk.has_rapid_turn_taking:
            flags.append("RAPID TURN-TAKING")
        if chunk.has_overlapping_speech:
            flags.append("OVERLAPPING SPEECH")

        if flags:
            print(f"  Flags:  [{', '.join(flags)}]")

        print(
            f"  Text:   {chunk.text[:60]}..." if len(chunk.text) > 60 else f"  Text:   {chunk.text}"
        )
        print("-" * 50)


def main():
    print("=" * 60)
    print("Turn-Aware Chunking Demo")
    print("=" * 60)

    # 1. Create synthetic data
    print("Generating synthetic transcript...")
    transcript = create_demo_transcript()
    print(
        f"Created transcript with {len(transcript.segments)} segments, {len(transcript.turns)} turns."
    )

    # 2. Configure chunking
    # We want chunks around 30s, but willing to stretch to 45s to keep turns together.
    # turn_affinity=0.8 means we strongly prefer splitting at turn boundaries.
    config = ChunkingConfig(
        target_duration_s=20.0,  # Smaller target for demo
        max_duration_s=40.0,
        turn_affinity=0.8,  # Strong preference for turn boundaries
        cross_turn_penalty=1.0,
        min_turn_gap_s=0.5,  # Gap threshold for rapid turn taking
    )

    print("\nConfiguration:")
    print(f"  Target Duration: {config.target_duration_s}s")
    print(f"  Turn Affinity:   {config.turn_affinity} (0.0-1.0)")
    print(f"  Turn Gap:        {config.min_turn_gap_s}s")

    # 3. Apply chunking
    print("\nApplying chunking...")
    chunks = build_chunks(transcript, config)

    # 4. Show results
    print_chunk_analysis(chunks)

    print("\nUse Cases:")
    print("1. RAG: Index these chunks with metadata (speaker_ids, flags).")
    print("2. Summarization: Summarize chunks individually, then aggregate.")
    print("3. Analytics: Analyze 'rapid turn-taking' chunks for engagement.")


if __name__ == "__main__":
    main()
