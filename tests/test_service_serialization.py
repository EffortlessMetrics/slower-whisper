"""REST transcript serialization for supported optional producer fields."""

from __future__ import annotations

import json

from transcription.models import Transcript
from transcription.service_serialization import _transcript_to_dict


def test_supported_optional_fields_are_json_serializable() -> None:
    transcript = Transcript(
        file_name="clip.wav",
        language="en",
        segments=[],
        annotations={
            "topics": ["alpha", "beta"],
            "confidence": 0.9,
            "reviewed": True,
        },
        speakers=[
            {
                "speaker_id": "SPEAKER_00",
                "label": "Alice",
            }
        ],
        turns=[
            {
                "speaker_id": "SPEAKER_00",
                "start": 0.0,
                "end": 1.25,
                "text": "hello",
            }
        ],
        speaker_stats=[
            {
                "speaker_id": "SPEAKER_00",
                "duration": 1.25,
                "turn_count": 1,
            }
        ],
        chunks=[
            {
                "chunk_id": "chunk-0001",
                "segment_ids": [0],
                "summary": "hello",
            }
        ],
    )

    payload = _transcript_to_dict(transcript, include_words=True)
    decoded = json.loads(json.dumps(payload, allow_nan=False))

    assert decoded["annotations"] == transcript.annotations
    assert decoded["speakers"] == transcript.speakers
    assert decoded["turns"] == transcript.turns
    assert decoded["speaker_stats"] == transcript.speaker_stats
    assert decoded["chunks"] == transcript.chunks
