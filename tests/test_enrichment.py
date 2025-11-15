"""Tests for enrichment modules."""

import json
from pathlib import Path
from transcription.models import Transcript, Segment
from transcription.enrichment.tone import ToneAnalyzer, ToneConfig
from transcription.enrichment.analytics.indexer import generate_index
from transcription.writers import write_json


def test_tone_analyzer_mock(tmp_path):
    """Test tone analyzer with mock provider."""
    # Create config with mock provider (no API calls)
    config = ToneConfig(api_provider="mock")
    analyzer = ToneAnalyzer(config)

    # Create a sample transcript
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=2.0, text="Hello world."),
            Segment(id=1, start=2.0, end=4.0, text="How are you?"),
        ],
    )

    # Annotate
    enriched = analyzer.annotate(transcript)

    # Check segments have tone
    assert enriched.segments[0].tone == "neutral"  # Mock assigns neutral
    assert enriched.segments[1].tone == "neutral"

    # Check metadata
    assert "enrichments" in enriched.meta
    assert enriched.meta["enrichments"]["tone_version"] == "1.0"
    assert enriched.meta["enrichments"]["tone_provider"] == "mock"


def test_index_generation(tmp_path):
    """Test transcript index generation."""
    # Create a sample transcript JSON
    json_dir = tmp_path / "whisper_json"
    json_dir.mkdir()

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=2.0, text="Test segment.", tone="neutral", speaker="SPEAKER_00"),
        ],
        meta={
            "audio_duration_sec": 2.0,
            "generated_at": "2025-01-01T00:00:00Z",
            "model_name": "large-v3",
            "enrichments": {
                "tone_version": "1.0",
                "speaker_version": "1.0",
            },
        },
    )

    json_path = json_dir / "test.json"
    write_json(transcript, json_path)

    # Generate JSON index
    index_path = tmp_path / "index.json"
    generate_index(json_dir, index_path, format="json")

    # Verify index
    with open(index_path, "r") as f:
        index = json.load(f)

    assert index["total_files"] == 1
    assert index["total_duration_sec"] == 2.0
    assert len(index["transcripts"]) == 1

    t = index["transcripts"][0]
    assert t["file_name"] == "test.wav"
    assert t["language"] == "en"
    assert t["segment_count"] == 1
    assert t["has_tone"] is True
    assert t["has_speakers"] is True


def test_index_generation_csv(tmp_path):
    """Test CSV index generation."""
    # Create a sample transcript JSON
    json_dir = tmp_path / "whisper_json"
    json_dir.mkdir()

    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=2.0, text="Test segment."),
        ],
        meta={
            "audio_duration_sec": 2.0,
            "generated_at": "2025-01-01T00:00:00Z",
            "model_name": "large-v3",
        },
    )

    json_path = json_dir / "test.json"
    write_json(transcript, json_path)

    # Generate CSV index
    csv_path = tmp_path / "index.csv"
    generate_index(json_dir, csv_path, format="csv")

    # Verify CSV exists and has content
    assert csv_path.exists()
    content = csv_path.read_text()

    # Check header
    assert "file_name" in content
    assert "duration_sec" in content
    assert "language" in content

    # Check data
    assert "test.wav" in content
    assert "en" in content
