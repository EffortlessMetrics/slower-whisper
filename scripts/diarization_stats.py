#!/usr/bin/env python3
"""
Generate human-readable diarization statistics from a transcript JSON.

Usage:
    uv run python scripts/diarization_stats.py whisper_json/audio.json
"""

import json
import sys
from pathlib import Path
from typing import Any


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def analyze_diarization(transcript: dict[str, Any]) -> None:
    """Print human-readable diarization statistics."""
    file_name = transcript.get("file_name", "unknown")
    meta = transcript.get("meta", {})
    diarization = meta.get("diarization", {})
    speakers = transcript.get("speakers", [])
    turns = transcript.get("turns", [])
    segments = transcript.get("segments", [])

    print(f"\n{'=' * 60}")
    print(f"Diarization Statistics: {file_name}")
    print(f"{'=' * 60}\n")

    # Overall status
    print("=== Diarization Status ===")
    status = diarization.get("status", "unknown")
    requested = diarization.get("requested", False)
    backend = diarization.get("backend", "none")

    print(f"Status:    {status}")
    print(f"Requested: {requested}")
    print(f"Backend:   {backend}")

    if status == "failed":
        error_type = diarization.get("error_type", "unknown")
        error_msg = diarization.get("error_message", "")
        print("\n❌ Diarization failed!")
        print(f"Error type: {error_type}")
        print(f"Error: {error_msg}")
        return

    if not requested or status != "success":
        print("\n⚠️  Diarization not enabled or failed")
        return

    print()

    # Speaker statistics
    print("=== Speakers ===")
    if not speakers:
        print("No speakers detected")
    else:
        print(f"Total speakers: {len(speakers)}\n")

        for i, speaker in enumerate(speakers, 1):
            speaker_id = speaker.get("id", "unknown")
            first_seen = speaker.get("first_seen", 0.0)
            last_seen = speaker.get("last_seen", 0.0)
            total_time = speaker.get("total_speech_time", 0.0)
            num_segments = speaker.get("num_segments", 0)

            print(f"{i}. {speaker_id}")
            print(f"   First seen:  {format_time(first_seen)}")
            print(f"   Last seen:   {format_time(last_seen)}")
            print(f"   Speech time: {format_time(total_time)} ({total_time:.1f}s)")
            print(f"   Segments:    {num_segments}")

            # Calculate talk time percentage
            total_duration = max(seg.get("end", 0) for seg in segments) if segments else 0
            if total_duration > 0:
                talk_pct = (total_time / total_duration) * 100
                print(f"   Talk %:      {talk_pct:.1f}%")
            print()

    # Turn statistics
    print("=== Turn Structure ===")
    if not turns:
        print("No turns detected")
    else:
        print(f"Total turns: {len(turns)}\n")

        # Turn alternation pattern
        turn_speakers = [t.get("speaker_id", "?") for t in turns[:20]]
        print(f"First 20 turns: {' → '.join(turn_speakers)}")
        print()

        # Count turns per speaker
        from collections import Counter

        turn_counts = Counter(t.get("speaker_id") for t in turns)
        print("Turns per speaker:")
        for speaker_id, count in turn_counts.most_common():
            print(f"  {speaker_id}: {count}")
        print()

        # Average turn duration
        turn_durations = [t.get("end", 0) - t.get("start", 0) for t in turns]
        avg_duration = sum(turn_durations) / len(turn_durations) if turn_durations else 0
        print(f"Average turn duration: {avg_duration:.1f}s")
        print()

    # Segment-level statistics
    print("=== Segment-Level ===")
    total_segments = len(segments)
    labeled_segments = sum(1 for seg in segments if seg.get("speaker"))

    print(f"Total segments:   {total_segments}")
    print(f"Labeled segments: {labeled_segments}")
    if total_segments > 0:
        labeled_pct = (labeled_segments / total_segments) * 100
        print(f"Labeled %:        {labeled_pct:.1f}%")
    print()

    # Show first few segments with speakers
    print("=== First 10 Segments (with speakers) ===")
    for i, seg in enumerate(segments[:10], 1):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        speaker_obj = seg.get("speaker")
        speaker_id = speaker_obj.get("id") if speaker_obj else None
        confidence = speaker_obj.get("confidence", 0.0) if speaker_obj else 0.0
        text = seg.get("text", "").strip()[:60]  # Truncate long text

        speaker_str = f"{speaker_id} ({confidence:.2f})" if speaker_id else "no speaker"
        print(f"{i:2d}. [{format_time(start)}-{format_time(end)}] {speaker_str:20s} {text}")
    print()

    # Quality checks
    print("=== Quality Checks ===")
    checks = []

    # Check 1: Reasonable speaker count
    if len(speakers) == 0:
        checks.append("❌ No speakers detected")
    elif len(speakers) <= 4:
        checks.append(f"✓ Speaker count reasonable ({len(speakers)})")
    else:
        checks.append(
            f"⚠️  Many speakers detected ({len(speakers)}) - may indicate oversegmentation"
        )

    # Check 2: Most segments labeled
    if labeled_pct >= 90:
        checks.append(f"✓ Most segments labeled ({labeled_pct:.1f}%)")
    elif labeled_pct >= 50:
        checks.append(f"⚠️  Some segments unlabeled ({labeled_pct:.1f}%)")
    else:
        checks.append(f"❌ Many segments unlabeled ({labeled_pct:.1f}%)")

    # Check 3: Turn alternation
    if len(turns) >= 2:
        # Check if turns alternate (not same speaker twice in a row)
        same_speaker_runs = 0
        for i in range(1, len(turns)):
            if turns[i].get("speaker_id") == turns[i - 1].get("speaker_id"):
                same_speaker_runs += 1

        if same_speaker_runs == 0:
            checks.append("✓ Perfect turn alternation")
        elif same_speaker_runs <= len(turns) * 0.1:
            checks.append(f"✓ Good turn alternation ({same_speaker_runs} repeats)")
        else:
            checks.append(f"⚠️  Some same-speaker turns ({same_speaker_runs}/{len(turns)})")

    for check in checks:
        print(check)

    print(f"\n{'=' * 60}\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/diarization_stats.py whisper_json/audio.json")
        sys.exit(1)

    json_path = Path(sys.argv[1])

    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    try:
        with open(json_path) as f:
            transcript = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_path}: {e}")
        sys.exit(1)

    analyze_diarization(transcript)


if __name__ == "__main__":
    main()
