"""Smoke tests: JSON write/read round-trip fidelity.

Creates a realistic Transcript, writes to JSON, loads it back, and verifies
full fidelity of all fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcription.models import SCHEMA_VERSION, Segment, Transcript, Word
from transcription.writers import load_transcript_from_json, write_json


@pytest.mark.smoke
@pytest.mark.timeout(10)
class TestJsonRoundTrip:
    """JSON write → load round-trip preserves all data."""

    def _make_transcript(self) -> Transcript:
        """Create a realistic Transcript with segments and words."""
        words = [
            Word(word="Hello", start=0.0, end=0.3, probability=0.95),
            Word(word="world", start=0.35, end=0.7, probability=0.92),
        ]
        segments = [
            Segment(
                id=0,
                start=0.0,
                end=2.5,
                text="Hello world, this is a test.",
                speaker={"id": "spk_0", "confidence": 0.87},
                tone="neutral",
                audio_state={"pitch_mean": 120.5, "energy_db": -22.3},
                words=words,
            ),
            Segment(
                id=1,
                start=2.8,
                end=5.1,
                text="Second segment with more text.",
                speaker={"id": "spk_1", "confidence": 0.91},
            ),
        ]
        return Transcript(
            file_name="test_audio.wav",
            language="en",
            segments=segments,
            meta={"model": "tiny", "device": "cpu"},
        )

    def test_round_trip_preserves_segments(self, tmp_path: Path) -> None:
        """Write then load should preserve segment count and text."""
        transcript = self._make_transcript()
        json_path = tmp_path / "output.json"

        write_json(transcript, json_path)
        loaded = load_transcript_from_json(json_path)

        assert len(loaded.segments) == len(transcript.segments)
        for orig, loaded_seg in zip(transcript.segments, loaded.segments, strict=True):
            assert loaded_seg.id == orig.id
            assert loaded_seg.text == orig.text
            assert loaded_seg.start == pytest.approx(orig.start)
            assert loaded_seg.end == pytest.approx(orig.end)

    def test_round_trip_preserves_metadata(self, tmp_path: Path) -> None:
        """File name, language, and meta should survive round-trip."""
        transcript = self._make_transcript()
        json_path = tmp_path / "output.json"

        write_json(transcript, json_path)
        loaded = load_transcript_from_json(json_path)

        assert loaded.file_name == transcript.file_name
        assert loaded.language == transcript.language
        assert loaded.meta is not None
        assert loaded.meta["model"] == "tiny"

    def test_round_trip_preserves_words(self, tmp_path: Path) -> None:
        """Word-level timestamps should survive round-trip."""
        transcript = self._make_transcript()
        json_path = tmp_path / "output.json"

        write_json(transcript, json_path)
        loaded = load_transcript_from_json(json_path)

        # First segment has words
        orig_words = transcript.segments[0].words
        loaded_words = loaded.segments[0].words
        assert orig_words is not None
        assert loaded_words is not None
        assert len(loaded_words) == len(orig_words)
        for ow, lw in zip(orig_words, loaded_words, strict=True):
            assert lw.word == ow.word
            assert lw.start == pytest.approx(ow.start)
            assert lw.end == pytest.approx(ow.end)
            assert lw.probability == pytest.approx(ow.probability)

    def test_round_trip_preserves_speaker_and_audio_state(self, tmp_path: Path) -> None:
        """Speaker info and audio_state should survive round-trip."""
        transcript = self._make_transcript()
        json_path = tmp_path / "output.json"

        write_json(transcript, json_path)
        loaded = load_transcript_from_json(json_path)

        # First segment has speaker and audio_state
        assert loaded.segments[0].speaker == transcript.segments[0].speaker
        assert loaded.segments[0].audio_state == transcript.segments[0].audio_state

    def test_schema_version_in_output(self, tmp_path: Path) -> None:
        """JSON output should contain the current schema version."""
        import json

        transcript = self._make_transcript()
        json_path = tmp_path / "output.json"

        write_json(transcript, json_path)
        data = json.loads(json_path.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
