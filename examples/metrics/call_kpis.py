"""
Compute simple call KPIs from a slower-whisper transcript.

Usage:
    python examples/metrics/call_kpis.py path/to/transcript.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transcription import load_transcript
from transcription.speaker_stats import compute_speaker_stats
from transcription.turns import build_turns


def _ensure_stats(transcript):
    """Populate turns and speaker_stats if absent."""
    if not transcript.turns:
        transcript = build_turns(transcript)
    if not transcript.speaker_stats:
        compute_speaker_stats(transcript)
    return transcript


def _field(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _talk_ratios(transcript):
    total_time = sum(seg.end - seg.start for seg in transcript.segments)
    ratios = []
    for stat in transcript.speaker_stats or []:
        talk = float(_field(stat, "total_talk_time", 0.0) or 0.0)
        speaker_id = _field(stat, "speaker_id")
        ratios.append((speaker_id, talk / total_time if total_time > 0 else 0.0))
    return ratios


def _longest_monologue(transcript):
    if not transcript.turns:
        return None

    def _duration(turn):
        end = float(_field(turn, "end", 0.0) or 0.0)
        start = float(_field(turn, "start", 0.0) or 0.0)
        return end - start

    longest = max(transcript.turns, key=_duration)
    duration = _duration(longest)
    speaker_id = _field(longest, "speaker_id")
    text = _field(longest, "text", "")
    return speaker_id, duration, text


def _question_rates(transcript):
    rates = []
    for stat in transcript.speaker_stats or []:
        speaker_id = _field(stat, "speaker_id")
        questions = int(_field(stat, "question_turns", 0) or 0)
        num_turns = int(_field(stat, "num_turns", 0) or 0)
        rates.append((speaker_id, questions, questions / num_turns if num_turns else 0.0))
    return rates


def _interruptions(transcript):
    interruptions = []
    for stat in transcript.speaker_stats or []:
        speaker_id = _field(stat, "speaker_id")
        initiated = int(_field(stat, "interruptions_initiated", 0) or 0)
        received = int(_field(stat, "interruptions_received", 0) or 0)
        interruptions.append((speaker_id, initiated, received))
    return interruptions


def _sentiment_summary(transcript):
    sentiments = []
    for stat in transcript.speaker_stats or []:
        speaker_id = _field(stat, "speaker_id")
        sentiment = _field(stat, "sentiment_summary", {}) or {}
        positive = float(_field(sentiment, "positive", 0.0) or 0.0)
        negative = float(_field(sentiment, "negative", 0.0) or 0.0)
        sentiments.append((speaker_id, positive, negative))
    return sentiments


def render_report(transcript):
    transcript = _ensure_stats(transcript)

    print(f"# KPIs for {transcript.file_name}")

    print("\nTalk ratio (per speaker):")
    for speaker_id, ratio in _talk_ratios(transcript):
        print(f"- {speaker_id}: {ratio * 100:.1f}% of total talk time")

    longest = _longest_monologue(transcript)
    if longest:
        speaker_id, duration, text = longest
        snippet = (text or "").split()
        snippet = " ".join(snippet[:18]) + ("..." if len(snippet) > 18 else "")
        print(f"\nLongest monologue: {duration:.1f}s by {speaker_id}: {snippet}")

    print("\nQuestion rate:")
    for speaker_id, questions, rate in _question_rates(transcript):
        print(f"- {speaker_id}: {questions} question turns ({rate * 100:.1f}% of turns)")

    print("\nInterruptions:")
    for speaker_id, initiated, received in _interruptions(transcript):
        print(f"- {speaker_id}: started {initiated}, received {received}")

    sentiments = _sentiment_summary(transcript)
    if sentiments:
        print("\nSentiment (rough signal):")
        for speaker_id, pos, neg in sentiments:
            print(f"- {speaker_id}: positive={pos:.2f}, negative={neg:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute simple KPIs from a transcript.")
    parser.add_argument("transcript", type=Path, help="Path to transcript JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    transcript = load_transcript(args.transcript)
    render_report(transcript)
