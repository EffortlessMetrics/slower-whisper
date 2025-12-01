"""Integration-style tests for speaker analytics wiring (turn metadata, speaker_stats)."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from transcription import EnrichmentConfig, enrich_transcript
from transcription.models import Segment, Transcript

pytestmark = pytest.mark.integration


def _write_silent_wav(path: Path, duration_sec: float = 0.5) -> None:
    """Create a tiny silent WAV so enrich_transcript() passes file checks."""
    sample_rate = 16000
    num_frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\0\0" * num_frames)


def test_enrich_transcript_populates_analytics(monkeypatch, tmp_path):
    """Speaker analytics should run when enabled in EnrichmentConfig."""
    audio_path = tmp_path / "sample.wav"
    _write_silent_wav(audio_path)

    transcript = Transcript(
        file_name="sample.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=1.0, text="Hello?", speaker={"id": "spk_0"}),
            Segment(id=1, start=1.0, end=2.0, text="Thanks.", speaker={"id": "spk_1"}),
        ],
    )

    def _fake_enrich_transcript_audio(**_kwargs):
        for seg in transcript.segments:
            seg.audio_state = {"prosody": {"pause_before_ms": 120.0}}
        return transcript

    monkeypatch.setattr(
        "transcription.audio_enrichment.enrich_transcript_audio", _fake_enrich_transcript_audio
    )

    cfg = EnrichmentConfig(
        enable_prosody=False,
        enable_emotion=False,
        enable_categorical_emotion=False,
        enable_turn_metadata=True,
        enable_speaker_stats=True,
        skip_existing=False,
    )

    enriched = enrich_transcript(transcript, audio_path, cfg)

    assert enriched.turns
    assert all((t.get("metadata") or {}) for t in enriched.turns)  # type: ignore[arg-type]
    assert enriched.speaker_stats is not None
    assert len(enriched.speaker_stats) == 2


def test_enrich_transcript_can_disable_speaker_stats(monkeypatch, tmp_path):
    """Speaker stats should be omitted when disabled, while turn metadata remains."""
    audio_path = tmp_path / "sample.wav"
    _write_silent_wav(audio_path)

    transcript = Transcript(
        file_name="sample.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=1.0, text="Hello there", speaker={"id": "spk_0"}),
            Segment(id=1, start=1.0, end=2.0, text="Another line", speaker={"id": "spk_0"}),
        ],
    )

    monkeypatch.setattr(
        "transcription.audio_enrichment.enrich_transcript_audio",
        lambda **_kwargs: transcript,
    )

    cfg = EnrichmentConfig(
        enable_turn_metadata=True,
        enable_speaker_stats=False,
        skip_existing=False,
    )

    enriched = enrich_transcript(transcript, audio_path, cfg)

    assert enriched.turns
    assert all((t.get("metadata") or {}) for t in enriched.turns)  # type: ignore[arg-type]
    assert enriched.speaker_stats is None
