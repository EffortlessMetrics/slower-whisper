"""Utilities for dogfooding (testing real-world use cases before release).

This module provides reusable functions for:
- Checking model cache status
- Computing diarization quality metrics
- Orchestrating dogfood workflows

These are used by both standalone scripts and automated workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import load_transcript
from .cache import CachePaths
from .samples import get_samples_cache_dir


def _get_speaker_id(speaker: Any) -> str | None:
    """Extract a speaker ID string from segment.speaker."""
    if speaker is None:
        return None
    if isinstance(speaker, str):
        return speaker
    if isinstance(speaker, dict):
        raw_id = speaker.get("id")
        return str(raw_id) if raw_id is not None else None
    return str(speaker)


def _get_cache_size(path: Path) -> int:
    """Compute total size of files under a directory."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def get_model_cache_status() -> dict[str, Any]:
    """Return detailed model cache status (paths, sizes, what's missing).

    Returns:
        Dict with structure:
        {
            "root": str,
            "caches": {
                "hf": {"path": str, "bytes": int, "formatted": str},
                "whisper": {...},
                "emotion": {...},
                "diarization": {...},
                "samples": {...},
            },
            "total_bytes": int,
            "total_formatted": str,
        }
    """
    paths = CachePaths.from_env().ensure_dirs()
    samples_dir = get_samples_cache_dir()

    cache_locations = {
        "hf": paths.hf_home,
        "torch": paths.torch_home,
        "whisper": paths.whisper_root,
        "emotion": paths.emotion_root,
        "diarization": paths.diarization_root,
        "samples": samples_dir,
    }

    caches = {}
    total = 0
    for name, path in cache_locations.items():
        size = _get_cache_size(path)
        total += size
        caches[name] = {"path": str(path), "bytes": size, "formatted": _format_size(size)}

    return {
        "root": str(paths.root),
        "caches": caches,
        "total_bytes": total,
        "total_formatted": _format_size(total),
    }


def compute_diarization_stats(json_path: Path) -> dict[str, Any]:
    """Compute diarization quality metrics from a transcript JSON.

    Returns:
        Dict with structure:
        {
            "file_name": str,
            "duration": float,
            "num_segments": int,
            "num_speakers": int,
            "speaker_stats": [
                {"id": str, "speech_time": float, "talk_percentage": float, "num_segments": int},
                ...
            ],
            "labeling_coverage": {
                "labeled_segments": int,
                "unlabeled_segments": int,
                "coverage_percentage": float,
            },
            "turn_stats": {
                "num_turns": int,
                "turn_sequence": list[str],  # first 20 turns
                "repeated_speakers": int,  # consecutive turns by same speaker
            },
            "quality_checks": {
                "speaker_count_reasonable": bool,  # 1-10 speakers
                "good_labeling_coverage": bool,  # >80% labeled
                "good_turn_alternation": bool,  # <20% repeated
            },
        }
    """
    transcript = load_transcript(json_path)

    # Basic metadata
    duration = transcript.segments[-1].end if transcript.segments else 0.0
    num_segments = len(transcript.segments)

    # Speaker analysis
    speaker_times: dict[str, float] = {}
    speaker_counts: dict[str, int] = {}
    labeled_count = 0

    for seg in transcript.segments:
        speaker_id = _get_speaker_id(seg.speaker)
        if speaker_id:
            labeled_count += 1
            duration_seg = seg.end - seg.start
            speaker_times[speaker_id] = speaker_times.get(speaker_id, 0.0) + duration_seg
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

    num_speakers = len(speaker_times)
    total_speech_time = sum(speaker_times.values())

    speaker_stats = []
    for speaker_id in sorted(speaker_times.keys()):
        speech_time = speaker_times[speaker_id]
        talk_pct = (speech_time / total_speech_time * 100) if total_speech_time > 0 else 0.0
        speaker_stats.append(
            {
                "id": speaker_id,
                "speech_time": round(speech_time, 2),
                "talk_percentage": round(talk_pct, 1),
                "num_segments": speaker_counts[speaker_id],
            }
        )

    # Labeling coverage
    unlabeled_count = num_segments - labeled_count
    coverage_pct = (labeled_count / num_segments * 100) if num_segments > 0 else 0.0

    # Turn structure analysis
    turns: list[str] = []
    current_speaker: str | None = None
    repeated_speakers = 0

    for seg in transcript.segments:
        speaker_id = _get_speaker_id(seg.speaker)
        if speaker_id is None:
            continue
        if speaker_id == current_speaker:
            repeated_speakers += 1
            continue

        turns.append(speaker_id)
        current_speaker = speaker_id

    turn_sequence = turns[:20]  # first 20 turns for preview
    num_turns = len(turns)

    # Quality checks
    speaker_count_reasonable = 1 <= num_speakers <= 10
    good_labeling_coverage = coverage_pct >= 80.0
    good_turn_alternation = (repeated_speakers / num_turns * 100) < 20.0 if num_turns > 0 else True

    return {
        "file_name": transcript.file_name,
        "duration": round(duration, 2),
        "num_segments": num_segments,
        "num_speakers": num_speakers,
        "speaker_stats": speaker_stats,
        "labeling_coverage": {
            "labeled_segments": labeled_count,
            "unlabeled_segments": unlabeled_count,
            "coverage_percentage": round(coverage_pct, 1),
        },
        "turn_stats": {
            "num_turns": num_turns,
            "turn_sequence": turn_sequence,
            "repeated_speakers": repeated_speakers,
        },
        "quality_checks": {
            "speaker_count_reasonable": speaker_count_reasonable,
            "good_labeling_coverage": good_labeling_coverage,
            "good_turn_alternation": good_turn_alternation,
        },
    }


def print_cache_status(status: dict[str, Any]) -> None:
    """Pretty-print cache status dict (for CLI use)."""
    print("slower-whisper cache locations:")
    print(f"  Root:         {status['root']}")
    print()

    for name, info in status["caches"].items():
        label = name.upper() if len(name) <= 2 else name.capitalize()
        print(f"  {label:13} {info['path']}")
        print(f"                ({info['formatted']})")

    print()
    print(f"  Total:        {status['total_formatted']}")


def print_diarization_stats(stats: dict[str, Any]) -> None:
    """Pretty-print diarization stats dict (for CLI use)."""
    print(f"=== Diarization Stats: {stats['file_name']} ===")
    print(f"Duration: {stats['duration']}s")
    print(f"Segments: {stats['num_segments']}")
    print()

    print("=== Speakers ===")
    print(f"Total speakers: {stats['num_speakers']}")
    for i, speaker in enumerate(stats["speaker_stats"], 1):
        print(
            f"{i}. {speaker['id']} - "
            f"Speech time: {speaker['speech_time']}s, "
            f"Talk %: {speaker['talk_percentage']}%, "
            f"Segments: {speaker['num_segments']}"
        )
    print()

    print("=== Labeling Coverage ===")
    cov = stats["labeling_coverage"]
    print(f"Labeled:   {cov['labeled_segments']}/{stats['num_segments']} segments")
    print(f"Coverage:  {cov['coverage_percentage']}%")
    print()

    print("=== Turn Structure ===")
    turn = stats["turn_stats"]
    print(f"Total turns: {turn['num_turns']}")
    if turn["turn_sequence"]:
        seq_str = " → ".join(turn["turn_sequence"])
        print(f"First {len(turn['turn_sequence'])} turns: {seq_str}")
        if turn["num_turns"] > 20:
            print(f"  (+ {turn['num_turns'] - 20} more turns)")
    print(f"Repeated speakers: {turn['repeated_speakers']} consecutive turns")
    print()

    print("=== Quality Checks ===")
    checks = stats["quality_checks"]
    print(
        f"{'✓' if checks['speaker_count_reasonable'] else '✗'} "
        f"Speaker count reasonable ({stats['num_speakers']})"
    )
    print(
        f"{'✓' if checks['good_labeling_coverage'] else '✗'} "
        f"Good labeling coverage ({cov['coverage_percentage']}%)"
    )
    print(
        f"{'✓' if checks['good_turn_alternation'] else '✗'} "
        f"Good turn alternation ({turn['repeated_speakers']} repeats)"
    )


def save_dogfood_results(
    output_path: Path,
    sample_name: str,
    stats: dict[str, Any],
    cache_status: dict[str, Any] | None = None,
    llm_output: str | None = None,
) -> None:
    """Save structured dogfood results to JSON for tracking.

    Args:
        output_path: Where to save results (e.g., dogfood_results/mini-diarization-20250118.json)
        sample_name: Sample dataset name
        stats: Diarization stats from compute_diarization_stats()
        cache_status: Optional cache status from get_model_cache_status()
        llm_output: Optional LLM analysis output
    """
    from datetime import datetime

    result = {
        "timestamp": datetime.now().isoformat(),
        "sample_name": sample_name,
        "diarization_stats": stats,
    }

    if cache_status:
        result["cache_status"] = cache_status

    if llm_output:
        result["llm_output"] = llm_output

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to: {output_path}")
