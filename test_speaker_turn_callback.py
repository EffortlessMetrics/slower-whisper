#!/usr/bin/env env python
"""Quick test to verify on_speaker_turn callback implementation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from transcription.streaming import StreamChunk
from transcription.streaming_enrich import (
    StreamingEnrichmentConfig,
    StreamingEnrichmentSession,
)


def _chunk(start: float, end: float, text: str, speaker: str | None = None) -> StreamChunk:
    """Create a StreamChunk for testing."""
    return {"start": start, "end": end, "text": text, "speaker_id": speaker}


def _create_mock_extractor(duration: float = 60.0, sample_rate: int = 16000) -> MagicMock:
    """Create a mock AudioSegmentExtractor."""
    mock = MagicMock()
    mock.duration_seconds = duration
    mock.sample_rate = sample_rate
    mock.total_frames = int(duration * sample_rate)
    mock.wav_path = Path("/tmp/fake_audio.wav")
    return mock


class RecordingCallbacks:
    """Callbacks that record all calls."""

    def __init__(self) -> None:
        self.finalized_segments = []
        self.speaker_turns = []
        self.errors = []

    def on_segment_finalized(self, segment) -> None:
        self.finalized_segments.append(segment)

    def on_speaker_turn(self, turn: dict) -> None:
        self.speaker_turns.append(turn)
        print(f"Turn detected: {turn}")

    def on_error(self, error) -> None:
        self.errors.append(error)


def test_speaker_turn_detection():
    """Test that on_speaker_turn is called when speaker changes."""
    callbacks = RecordingCallbacks()
    config = StreamingEnrichmentConfig(
        enable_prosody=False,
        enable_emotion=False,
        enable_categorical_emotion=False,
    )
    mock_extractor = _create_mock_extractor()

    with patch(
        "transcription.streaming_enrich.AudioSegmentExtractor",
        return_value=mock_extractor,
    ):
        session = StreamingEnrichmentSession(
            wav_path=Path("/tmp/fake.wav"),
            config=config,
            callbacks=callbacks,
        )

        # Speaker A speaks
        session.ingest_chunk(_chunk(0.0, 1.0, "Hello from speaker A", "spk_A"))
        session.ingest_chunk(_chunk(1.1, 2.0, "More from speaker A", "spk_A"))

        # No turn finalized yet (still on speaker A)
        assert len(callbacks.speaker_turns) == 0

        # Speaker B speaks (triggers turn finalization for speaker A)
        session.ingest_chunk(_chunk(2.1, 3.0, "Hello from speaker B", "spk_B"))

        # Now we should have 1 turn (speaker A's turn)
        assert len(callbacks.speaker_turns) == 1
        turn_a = callbacks.speaker_turns[0]
        assert turn_a["speaker_id"] == "spk_A"
        assert turn_a["id"] == "turn_0"
        assert turn_a["start"] == 0.0
        assert turn_a["end"] == 2.0
        assert "Hello from speaker A More from speaker A" in turn_a["text"]

        # Speaker A speaks again (triggers turn finalization for speaker B)
        session.ingest_chunk(_chunk(3.1, 4.0, "Back to speaker A", "spk_A"))

        # Now we should have 2 turns
        assert len(callbacks.speaker_turns) == 2
        turn_b = callbacks.speaker_turns[1]
        assert turn_b["speaker_id"] == "spk_B"
        assert turn_b["id"] == "turn_1"
        assert turn_b["start"] == 2.1
        assert turn_b["end"] == 3.0
        assert "Hello from speaker B" in turn_b["text"]

        # End of stream finalizes the last turn (speaker A's second turn)
        session.end_of_stream()

        assert len(callbacks.speaker_turns) == 3
        turn_a2 = callbacks.speaker_turns[2]
        assert turn_a2["speaker_id"] == "spk_A"
        assert turn_a2["id"] == "turn_2"
        assert turn_a2["start"] == 3.1
        assert turn_a2["end"] == 4.0
        assert "Back to speaker A" in turn_a2["text"]

        print(f"\nTest passed! Detected {len(callbacks.speaker_turns)} turns:")
        for turn in callbacks.speaker_turns:
            print(
                f"  - {turn['id']}: {turn['speaker_id']} [{turn['start']:.1f}s-{turn['end']:.1f}s]"
            )


if __name__ == "__main__":
    test_speaker_turn_detection()
    print("\nAll tests passed!")
