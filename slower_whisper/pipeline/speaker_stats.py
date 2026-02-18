"""Compute per-speaker aggregates (v1.2 scaffolding).

Produces a list of speaker-stat dicts suitable for inclusion on the
Transcript object as `transcript.speaker_stats`.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from statistics import median
from typing import Any

from .models import (
    ProsodySummary,
    SentimentSummary,
    SpeakerStats,
    Transcript,
)
from .speaker_id import get_speaker_id
from .turn_helpers import turn_to_dict

logger = logging.getLogger(__name__)


def _collect_segment_durations_by_speaker(transcript: Transcript) -> dict[str, list[float]]:
    """Collect segment durations grouped by speaker ID.

    Args:
        transcript: Transcript with segments containing speaker info.

    Returns:
        Dict mapping speaker ID to list of segment durations (seconds).
        Unknown speakers assigned to "spk_0".
    """
    durations: dict[str, list[float]] = defaultdict(list)
    for seg in transcript.segments:
        speaker_id = get_speaker_id(seg.speaker) or "spk_0"
        durations[speaker_id].append(max(seg.end - seg.start, 0.0))
    return durations


def _collect_prosody_by_speaker(transcript: Transcript) -> dict[str, dict[str, list[float]]]:
    """Collect prosody metrics (pitch, energy) grouped by speaker ID.

    Extracts pitch (mean_hz) and energy (db_rms) from segment audio_state.

    Args:
        transcript: Transcript with enriched audio_state containing prosody.

    Returns:
        Nested dict: speaker_id -> metric_name -> list of values.
        E.g. {"spk_0": {"pitch": [245.3, 220.1], "energy": [-8.2, -10.1]}}
    """
    by_speaker: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for seg in transcript.segments:
        speaker_id = get_speaker_id(seg.speaker) or "spk_0"
        audio_state: dict[str, Any] = seg.audio_state or {}
        prosody = audio_state.get("prosody") or {}
        if not isinstance(prosody, dict):
            continue
        # Prosody fields are nested: prosody.pitch.mean_hz, prosody.energy.db_rms
        pitch_data = prosody.get("pitch") or {}
        if isinstance(pitch_data, dict):
            pitch = pitch_data.get("mean_hz")
            if isinstance(pitch, float | int):
                by_speaker[speaker_id]["pitch"].append(float(pitch))
        energy_data = prosody.get("energy") or {}
        if isinstance(energy_data, dict):
            energy = energy_data.get("db_rms")
            if isinstance(energy, float | int):
                by_speaker[speaker_id]["energy"].append(float(energy))
    return by_speaker


def _collect_sentiment_by_speaker(transcript: Transcript) -> dict[str, Counter]:
    """Collect sentiment level counts grouped by speaker ID.

    Extracts valence level (positive/neutral/negative) from emotion data.

    Args:
        transcript: Transcript with enriched audio_state containing emotion.

    Returns:
        Dict mapping speaker ID to Counter of sentiment levels.
        E.g. {"spk_0": Counter({"positive": 5, "neutral": 3})}
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for seg in transcript.segments:
        speaker_id = get_speaker_id(seg.speaker) or "spk_0"
        audio_state: dict[str, Any] = seg.audio_state or {}
        emotion = audio_state.get("emotion") or {}
        if not isinstance(emotion, dict):
            continue
        # Note: EmotionState has valence directly, not nested under "dimensional"
        valence = emotion.get("valence") or {}
        if not isinstance(valence, dict):
            continue
        level = (valence.get("level") or "").lower()
        if level in ("positive", "neutral", "negative"):
            counts[speaker_id][level] += 1
    return counts


def compute_speaker_stats(transcript: Transcript) -> list[dict]:
    """Compute per-speaker aggregate statistics for a transcript.

    Analyzes all segments and turns in the transcript to produce comprehensive
    statistics for each speaker, including talk time, turn counts, interruption
    patterns, prosody summaries, and sentiment distributions.

    The result is intentionally simple and deterministic; it can be extended
    later or replaced with ML-powered summaries. Statistics are computed from
    segment-level data and turn metadata when available.

    Args:
        transcript: A Transcript object containing segments with speaker
            assignments. For richer statistics, the transcript should also
            have turns populated (via turn detection) and audio_state enriched
            (via prosody and emotion extraction).

    Returns:
        A list of dicts, one per speaker, each containing:
            - speaker_id (str): The speaker identifier (e.g., "spk_0", "spk_1")
            - total_talk_time (float): Total speaking duration in seconds
            - num_turns (int): Number of conversation turns
            - avg_turn_duration (float): Average turn length in seconds
            - interruptions_initiated (int): Times this speaker interrupted others
            - interruptions_received (int): Times this speaker was interrupted
            - question_turns (int): Turns containing at least one question
            - prosody_summary (dict): Median pitch (Hz) and energy (dB) values
            - sentiment_summary (dict): Distribution of positive/neutral/negative

        The transcript's `speaker_stats` attribute is also updated with this
        list as a side effect.

    Example:
        >>> from transcription.models import Transcript, Segment
        >>> transcript = Transcript(
        ...     file_name="meeting.wav",
        ...     segments=[
        ...         Segment(id=0, start=0.0, end=2.0, text="Hello", speaker={"id": "spk_0"}),
        ...         Segment(id=1, start=2.0, end=5.0, text="Hi there", speaker={"id": "spk_1"}),
        ...     ]
        ... )
        >>> stats = compute_speaker_stats(transcript)
        >>> len(stats)
        2
        >>> stats[0]["speaker_id"]
        'spk_0'
        >>> stats[0]["total_talk_time"]
        2.0

    Note:
        - Segments without speaker info are assigned to "spk_0"
        - Prosody values are None if audio_state is not enriched
        - Sentiment percentages sum to 1.0 for each speaker
        - Interruption counts require turn metadata with `interruption_started_here`
    """
    durations = _collect_segment_durations_by_speaker(transcript)
    prosody_raw = _collect_prosody_by_speaker(transcript)
    sentiment_counts = _collect_sentiment_by_speaker(transcript)

    # Group turns by speaker if present
    turns_by_speaker: dict[str, list[dict]] = defaultdict(list)
    if transcript.turns:
        for t in transcript.turns:
            t_dict = turn_to_dict(t)
            sid = t_dict.get("speaker_id") or "spk_0"
            turns_by_speaker[sid].append(t_dict)

    # Count interruptions based on turn.metadata.interruption_started_here
    initiated: Counter = Counter()
    received: Counter = Counter()
    if transcript.turns:
        for idx, t in enumerate(transcript.turns):
            t_dict = turn_to_dict(t)
            meta = t_dict.get("metadata") or {}
            if idx > 0 and meta.get("interruption_started_here"):
                prev = turn_to_dict(transcript.turns[idx - 1])
                if prev.get("speaker_id") != t_dict.get("speaker_id"):
                    initiated[t_dict.get("speaker_id")] += 1
                    received[prev.get("speaker_id")] += 1

    stats_out: list[dict] = []

    for speaker_id, seg_durations in durations.items():
        total_talk = sum(seg_durations)
        speaker_turns = turns_by_speaker.get(speaker_id, [])
        num_turns = len(speaker_turns)
        avg_turn_duration = total_talk / num_turns if num_turns > 0 else 0.0

        prosody = prosody_raw.get(speaker_id, {})
        pitch_values = prosody.get("pitch") or []
        energy_values = prosody.get("energy") or []

        prosody_summary = ProsodySummary(
            pitch_median_hz=median(pitch_values) if pitch_values else None,
            energy_median_db=median(energy_values) if energy_values else None,
        )

        counts = sentiment_counts.get(speaker_id, Counter())
        total_sentiments = sum(counts.values()) or 1
        sentiment_summary = SentimentSummary(
            positive=counts["positive"] / total_sentiments,
            neutral=counts["neutral"] / total_sentiments,
            negative=counts["negative"] / total_sentiments,
        )

        question_turns = sum(
            1 for t in speaker_turns if (t.get("metadata") or {}).get("question_count", 0) > 0
        )

        s = SpeakerStats(
            speaker_id=speaker_id,
            total_talk_time=total_talk,
            num_turns=num_turns,
            avg_turn_duration=avg_turn_duration,
            interruptions_initiated=initiated[speaker_id],
            interruptions_received=received[speaker_id],
            question_turns=question_turns,
            prosody_summary=prosody_summary,
            sentiment_summary=sentiment_summary,
        )
        stats_out.append(s.to_dict())

    # Attach to transcript for convenience
    transcript.speaker_stats = stats_out  # type: ignore[assignment]
    return stats_out
