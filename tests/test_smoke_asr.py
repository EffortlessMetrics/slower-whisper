"""Smoke tests: real ASR engine with tiny model.

These tests exercise the actual faster-whisper model on real audio.
They require SLOWER_WHISPER_TEST_REAL=1 and are excluded from default test runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slower_whisper.pipeline.asr_engine import TranscriptionEngine
from slower_whisper.pipeline.legacy_config import AsrConfig

AUDIO_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "asr" / "audio"
CALL_CENTER_WAV = AUDIO_DIR / "call_center_narrowband.wav"


@pytest.fixture(scope="module")
def engine() -> TranscriptionEngine:
    """Load a real tiny model once per module."""
    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8")
    return TranscriptionEngine(cfg)


@pytest.mark.smoke
@pytest.mark.timeout(60)
class TestRealAsrEngine:
    """Tests that exercise the real ASR engine with a tiny model."""

    def test_asr_engine_loads_real_model(self, engine: TranscriptionEngine) -> None:
        """Engine should load a real faster-whisper model, not the dummy fallback."""
        assert not engine.using_dummy, "Expected real model but got DummyWhisperModel"

    def test_asr_engine_transcribes_speech(self, engine: TranscriptionEngine) -> None:
        """Transcribing call center audio should produce text with expected keywords."""
        assert CALL_CENTER_WAV.exists(), f"Missing fixture: {CALL_CENTER_WAV}"
        transcript = engine.transcribe_file(CALL_CENTER_WAV)
        full_text = transcript.full_text.lower()
        # The TTS audio is about a support call; at least one keyword should appear
        found = [
            kw for kw in ["support", "password", "email", "account", "help"] if kw in full_text
        ]
        assert found, f"Expected at least one keyword in transcript, got: {full_text[:300]}"

    def test_asr_engine_returns_valid_segments(self, engine: TranscriptionEngine) -> None:
        """Segments should have non-empty text and valid timestamps."""
        assert CALL_CENTER_WAV.exists()
        transcript = engine.transcribe_file(CALL_CENTER_WAV)
        assert len(transcript.segments) > 0, "Expected at least one segment"
        for seg in transcript.segments:
            assert seg.text.strip(), f"Segment {seg.id} has empty text"
            assert seg.start >= 0, f"Segment {seg.id} has negative start"
            assert seg.end > seg.start, f"Segment {seg.id} end <= start"
        assert transcript.language == "en", f"Expected 'en', got '{transcript.language}'"

    def test_word_timestamps(self, engine: TranscriptionEngine) -> None:
        """Word timestamps should produce words with valid timing when enabled."""
        assert CALL_CENTER_WAV.exists()
        cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8", word_timestamps=True)
        wt_engine = TranscriptionEngine(cfg)
        transcript = wt_engine.transcribe_file(CALL_CENTER_WAV)
        # At least one segment should have words
        segments_with_words = [s for s in transcript.segments if s.words]
        assert segments_with_words, "Expected at least one segment with word timestamps"
        for seg in segments_with_words:
            for word in seg.words:
                assert word.word.strip(), "Word text is empty"
                assert word.start >= 0, "Word has negative start"
                assert word.end >= word.start, "Word end < start"
