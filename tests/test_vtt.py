"""Tests for VTT writer."""

from pathlib import Path
from transcription.models import Transcript, Segment
from transcription.writers import write_vtt, _fmt_vtt_ts


def test_vtt_timestamp_formatting():
    """Test VTT timestamp formatting."""
    # Test basic formatting
    assert _fmt_vtt_ts(0.0) == "00:00:00.000"
    assert _fmt_vtt_ts(1.5) == "00:00:01.500"
    assert _fmt_vtt_ts(61.234) == "00:01:01.234"
    assert _fmt_vtt_ts(3661.789) == "01:01:01.789"
    assert _fmt_vtt_ts(3725.5) == "01:02:05.500"


def test_vtt_writer(tmp_path):
    """Test VTT file generation."""
    # Create a sample transcript
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=2.5, text="Hello world.", speaker="SPEAKER_00", tone="neutral"),
            Segment(id=1, start=2.5, end=5.0, text="How are you?", speaker="SPEAKER_01", tone="questioning"),
            Segment(id=2, start=5.0, end=8.5, text="I'm doing great!", speaker="SPEAKER_00", tone="positive"),
        ],
    )

    # Write VTT
    vtt_path = tmp_path / "test.vtt"
    write_vtt(transcript, vtt_path)

    # Read and verify
    content = vtt_path.read_text(encoding="utf-8")
    lines = content.strip().split("\n")

    # Check header
    assert lines[0] == "WEBVTT"

    # Check content includes segments
    assert "Hello world." in content
    assert "How are you?" in content
    assert "I'm doing great!" in content

    # Check timestamps are in VTT format (with dots, not commas)
    assert "00:00:00.000 --> 00:00:02.500" in content
    assert "00:00:02.500 --> 00:00:05.000" in content
    assert "00:00:05.000 --> 00:00:08.500" in content

    # Check speaker and tone annotations in cue IDs
    assert "[SPEAKER_00]" in content
    assert "[SPEAKER_01]" in content
    assert "(neutral)" in content
    assert "(questioning)" in content
    assert "(positive)" in content


def test_vtt_writer_without_enrichment(tmp_path):
    """Test VTT generation for transcripts without speaker/tone."""
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=[
            Segment(id=0, start=0.0, end=2.0, text="Simple test."),
        ],
    )

    vtt_path = tmp_path / "test.vtt"
    write_vtt(transcript, vtt_path)

    content = vtt_path.read_text(encoding="utf-8")

    # Should still work without speaker/tone
    assert "WEBVTT" in content
    assert "Simple test." in content
    assert "00:00:00.000 --> 00:00:02.000" in content
