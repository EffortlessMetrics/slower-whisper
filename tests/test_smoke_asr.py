"""Smoke tests: real ASR engine with the tiny faster-whisper model."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import transcription.asr_engine as asr_engine
from transcription.asr_engine import TranscriptionEngine
from transcription.legacy_config import AsrConfig

AUDIO_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "asr" / "audio"
CALL_CENTER_WAV = AUDIO_DIR / "call_center_narrowband.wav"
REAL_ASR_ENABLED = os.environ.get("SLOWER_WHISPER_TEST_REAL") == "1"

pytestmark = pytest.mark.skipif(
    not REAL_ASR_ENABLED,
    reason="set SLOWER_WHISPER_TEST_REAL=1 to run real-model ASR smoke tests",
)


@pytest.fixture(scope="module")
def engine() -> TranscriptionEngine:
    """Load a real tiny model once per module."""
    cfg = AsrConfig(model_name="tiny", device="cpu", compute_type="int8")
    return TranscriptionEngine(cfg)


@pytest.mark.smoke
@pytest.mark.timeout(60)
class TestRealAsrEngine:
    """Exercise a real model rather than an injected test backend."""

    def test_asr_engine_loads_real_model(
        self,
        engine: TranscriptionEngine,
    ) -> None:
        assert not hasattr(asr_engine, "DummyWhisperModel")
        assert engine.model_load_attempts[-1] == {
            "device": "cpu",
            "compute_type": "int8",
            "outcome": "selected",
            "reason_code": "ok",
        }

    def test_asr_engine_transcribes_speech(
        self,
        engine: TranscriptionEngine,
    ) -> None:
        assert CALL_CENTER_WAV.exists(), f"Missing fixture: {CALL_CENTER_WAV}"
        transcript = engine.transcribe_file(CALL_CENTER_WAV)
        full_text = transcript.full_text.lower()
        found = [
            keyword
            for keyword in ["support", "password", "email", "account", "help"]
            if keyword in full_text
        ]
        assert found, f"Expected at least one keyword in real transcript, got: {full_text[:300]}"
        assert transcript.meta["asr_backend"] == "faster-whisper"
        assert "asr_fallback_reason" not in transcript.meta

    def test_asr_engine_returns_valid_segments(
        self,
        engine: TranscriptionEngine,
    ) -> None:
        assert CALL_CENTER_WAV.exists()
        transcript = engine.transcribe_file(CALL_CENTER_WAV)
        assert transcript.segments
        for item in transcript.segments:
            assert item.text.strip()
            assert item.start >= 0
            assert item.end > item.start
        assert transcript.language == "en"

    def test_word_timestamps(self) -> None:
        assert CALL_CENTER_WAV.exists()
        cfg = AsrConfig(
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            word_timestamps=True,
        )
        transcript = TranscriptionEngine(cfg).transcribe_file(CALL_CENTER_WAV)
        segments_with_words = [item for item in transcript.segments if item.words]
        assert segments_with_words
        for item in segments_with_words:
            assert item.words is not None
            for word in item.words:
                assert word.word.strip()
                assert word.start >= 0
                assert word.end >= word.start
