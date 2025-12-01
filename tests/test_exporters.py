from pathlib import Path

import pytest

from transcription.exporters import export_csv, export_html, export_textgrid, export_vtt
from transcription.models import Segment, Transcript, Turn
from transcription.validation import validate_transcript_json


def _sample_transcript() -> Transcript:
    segments = [
        Segment(id=0, start=0.0, end=1.2, text="Hello there", speaker={"id": "spk_0"}),
        Segment(id=1, start=1.3, end=2.0, text="General Kenobi", speaker={"id": "spk_1"}),
    ]
    turns = [
        Turn(
            id="turn_0",
            speaker_id="spk_0",
            segment_ids=[0],
            start=0.0,
            end=1.2,
            text="Hello there",
            metadata={},
        ),
        Turn(
            id="turn_1",
            speaker_id="spk_1",
            segment_ids=[1],
            start=1.3,
            end=2.0,
            text="General Kenobi",
            metadata={},
        ),
    ]
    return Transcript(
        file_name="sample.wav",
        language="en",
        segments=segments,
        turns=turns,
        meta={"pipeline_version": "test"},
    )


def test_export_formats(tmp_path: Path) -> None:
    transcript = _sample_transcript()

    csv_path = tmp_path / "out.csv"
    export_csv(transcript, csv_path, unit="segments")
    assert csv_path.read_text(encoding="utf-8").startswith("id,start,end,speaker,text")

    html_path = tmp_path / "out.html"
    export_html(transcript, html_path, unit="turns")
    html_text = html_path.read_text(encoding="utf-8")
    assert "Hello there" in html_text and "General Kenobi" in html_text
    assert "turn" in html_text

    vtt_path = tmp_path / "out.vtt"
    export_vtt(transcript, vtt_path, unit="segments")
    vtt_text = vtt_path.read_text(encoding="utf-8")
    assert vtt_text.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.200" in vtt_text

    tg_path = tmp_path / "out.TextGrid"
    export_textgrid(transcript, tg_path, unit="segments")
    tg_text = tg_path.read_text(encoding="utf-8")
    assert 'class = "IntervalTier"' in tg_text
    assert "intervals: size = 2" in tg_text


def test_validate_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    json_path = tmp_path / "transcript.json"
    json_path.write_text(
        """
        {
          "schema_version": 2,
          "file": "sample.wav",
          "language": "en",
          "segments": [
            {"id": 0, "start": 0.0, "end": 1.2, "text": "Hello there"},
            {"id": 1, "start": 1.3, "end": 2.0, "text": "General Kenobi"}
          ],
          "turns": [
            {"id": "turn_0", "speaker_id": "spk_0", "start": 0.0, "end": 1.2, "segment_ids": [0], "text": "Hello there"},
            {"id": "turn_1", "speaker_id": "spk_1", "start": 1.3, "end": 2.0, "segment_ids": [1], "text": "General Kenobi"}
          ]
        }
        """,
        encoding="utf-8",
    )

    ok, errors = validate_transcript_json(json_path)
    assert ok
    assert errors == []


def test_xss_protection(tmp_path: Path) -> None:
    """Test that speaker names and text are properly escaped in all export formats."""
    # Create transcript with XSS attack vectors in speaker and text fields
    malicious_speaker = '<script>alert("XSS")</script>'
    malicious_text = "<img src=x onerror=\"alert('XSS')\">"

    segments = [
        Segment(
            id=0,
            start=0.0,
            end=1.0,
            text=malicious_text,
            speaker={"id": malicious_speaker},
        )
    ]
    transcript = Transcript(
        file_name="test.wav",
        language="en",
        segments=segments,
        meta={},
    )

    # Test HTML export - should escape both speaker and text
    html_path = tmp_path / "test.html"
    export_html(transcript, html_path)
    html_content = html_path.read_text(encoding="utf-8")

    # Verify speaker name is escaped
    assert "&lt;script&gt;" in html_content
    assert '<script>alert("XSS")</script>' not in html_content

    # Verify text is escaped
    assert "&lt;img" in html_content
    assert "<img src=x onerror=\"alert('XSS')\">" not in html_content

    # Test VTT export - should escape both speaker and text
    vtt_path = tmp_path / "test.vtt"
    export_vtt(transcript, vtt_path)
    vtt_content = vtt_path.read_text(encoding="utf-8")

    # Verify speaker name is escaped
    assert "&lt;script&gt;" in vtt_content
    assert '<script>alert("XSS")</script>' not in vtt_content

    # Verify text is escaped
    assert "&lt;img" in vtt_content
    assert "<img src=x onerror=\"alert('XSS')\">" not in vtt_content

    # Test TextGrid export - quotes should be escaped
    tg_path = tmp_path / "test.TextGrid"
    export_textgrid(transcript, tg_path)
    tg_content = tg_path.read_text(encoding="utf-8")

    # TextGrid uses Praat-style escaping ('' for quotes), but HTML tags should still be present
    # since TextGrid is not an HTML format - this is expected behavior
    # Check for escaped versions since TextGrid replaces " with '' (double quotes only)
    malicious_speaker_escaped = malicious_speaker.replace('"', "''")
    malicious_text_escaped = malicious_text.replace('"', "''")
    assert malicious_speaker_escaped in tg_content
    assert malicious_text_escaped in tg_content
