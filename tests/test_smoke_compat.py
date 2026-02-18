"""Smoke tests: slower_whisper drop-in compatibility layer.

Verifies that slower_whisper.WhisperModel works as a drop-in replacement
for faster_whisper.WhisperModel with real model loading and transcription.
"""

from __future__ import annotations

from pathlib import Path

import pytest

AUDIO_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "asr" / "audio"
CALL_CENTER_WAV = AUDIO_DIR / "call_center_narrowband.wav"


@pytest.mark.smoke
@pytest.mark.timeout(60)
class TestSlowerWhisperCompat:
    """Tests the slower_whisper compatibility shim with real models."""

    def test_whisper_model_transcribes(self) -> None:
        """WhisperModel.transcribe() should return (segments, info) tuple."""
        from slower_whisper import WhisperModel

        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(CALL_CENTER_WAV))

        # info should have language
        assert info.language == "en", f"Expected 'en', got '{info.language}'"

        # Materialize segments (they may be a generator)
        seg_list = list(segments)
        assert len(seg_list) > 0, "Expected at least one segment"

        for seg in seg_list:
            # Attribute access
            assert seg.text.strip(), f"Segment {seg.id} has empty text"
            assert seg.start >= 0
            assert seg.end > seg.start

    def test_segment_tuple_unpacking(self) -> None:
        """Segments should support tuple-style unpacking for backwards compat."""
        from slower_whisper import WhisperModel

        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(CALL_CENTER_WAV))

        seg_list = list(segments)
        assert seg_list, "Need at least one segment for unpacking test"

        # Tuple unpacking should work (id, seek, start, end, text, tokens, ...)
        first = seg_list[0]
        seg_id, seek, start, end, text, *rest = first
        assert isinstance(seg_id, int)
        assert isinstance(start, float)
        assert isinstance(end, float)
        assert isinstance(text, str)
        assert text.strip()

    def test_last_transcript_available(self) -> None:
        """model.last_transcript should be available after transcription."""
        from slower_whisper import WhisperModel

        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(CALL_CENTER_WAV))
        # Must consume the generator to trigger transcription
        list(segments)

        transcript = model.last_transcript
        assert transcript is not None, "last_transcript should be set after transcribe()"
        assert transcript.language == "en"
        assert len(transcript.segments) > 0
