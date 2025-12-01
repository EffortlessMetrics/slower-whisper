"""Compute per-speaker aggregates (v1.2 scaffolding).

Produces a list of speaker-stat dicts suitable for inclusion on the
Transcript object as `transcript.speaker_stats`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any

from .models import (
    ProsodySummary,
    SentimentSummary,
    SpeakerStats,
    Transcript,
)
from .turn_helpers import turn_to_dict


def _collect_segment_durations_by_speaker(transcript: Transcript) -> dict[str, list[float]]:
    durations: dict[str, list[float]] = defaultdict(list)
    for seg in transcript.segments:
        speaker_id = None
        if seg.speaker:
            speaker_id = (
                seg.speaker.get("id") if isinstance(seg.speaker, dict) else str(seg.speaker)
            )
        speaker_id = speaker_id or "spk_0"
        durations[speaker_id].append(max(seg.end - seg.start, 0.0))
    return durations


def _collect_prosody_by_speaker(transcript: Transcript) -> dict[str, dict[str, list[float]]]:
    by_speaker: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for seg in transcript.segments:
        speaker_id = None
        if seg.speaker:
            speaker_id = (
                seg.speaker.get("id") if isinstance(seg.speaker, dict) else str(seg.speaker)
            )
        speaker_id = speaker_id or "spk_0"
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
    counts: dict[str, Counter] = defaultdict(Counter)
    for seg in transcript.segments:
        speaker_id = None
        if seg.speaker:
            speaker_id = (
                seg.speaker.get("id") if isinstance(seg.speaker, dict) else str(seg.speaker)
            )
        speaker_id = speaker_id or "spk_0"
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
    """Compute a list of speaker stat dicts for the transcript.

    The result is intentionally simple and deterministic; it can be
    extended later or replaced with ML-powered summaries.
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
