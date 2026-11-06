"""WebVTT payloads must not become file-level cue boundaries."""

import csv
import html
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from transcription.exporters import export_csv, export_vtt
from transcription.models import Segment, Transcript, Turn


def _transcript(text: str, speaker: str, unit: str) -> Transcript:
    transcript = Transcript(
        file_name="synthetic.wav",
        language="en",
        segments=[
            Segment(id=0, start=1.0, end=2.0, text=text, speaker={"label": speaker}),
            Segment(id=1, start=3.0, end=4.0, text="following cue", speaker={"id": "next"}),
        ],
    )
    if unit != "segments":
        turns = [
            Turn(
                id=f"turn_{index}",
                speaker_id=label,
                segment_ids=[index],
                start=start,
                end=end,
                text=content,
            )
            for index, label, start, end, content in [
                (0, speaker, 1.0, 2.0, text),
                (1, "next", 3.0, 4.0, "following cue"),
            ]
        ]
        transcript.turns = [turn.to_dict() for turn in turns] if unit == "dict_turns" else turns
    return transcript


@pytest.mark.parametrize("unit", ["segments", "turns", "dict_turns"])
@pytest.mark.parametrize(
    "text",
    [
        "first paragraph\n\nsecond paragraph",
        "first\r\n\r\nsecond",
        "first\r\rsecond",
        "first\nsecond",
        "first\nsecond\nthird",
        "first\n \t\nsecond",
        "first\n",
        "\nfirst",
        "\n\n  padded text  \n\n",
        "  padded text  ",
        "\t \n\r\n",
        "",
        "first\n\n00:00:07.000 --> 00:00:08.000\nsecond",
        "first\n00:00:07.000 --> 00:00:08.000\nsecond",
        'naïve 日本語 &amp; &#10; <b>literal</b> "don\'t"',
    ],
)
def test_vtt_preserves_text_within_its_cue(tmp_path: Path, text: str, unit: str) -> None:
    transcript = _transcript(text, "speaker", unit)
    original = deepcopy(transcript)
    path = tmp_path / "nested" / "transcript.vtt"

    export_vtt(transcript, path, unit="segments" if unit == "segments" else "turns")

    blocks = path.read_text(encoding="utf-8").split("\n\n")
    assert len(blocks) == 4  # header, two complete cues, final terminator
    assert blocks[0] == "WEBVTT"
    timing, payload = blocks[1].split("\n", 1)
    assert timing == "00:00:01.000 --> 00:00:02.000"
    assert html.unescape(payload) == f"speaker: {text}"
    assert "<" not in payload  # literal tags must not become cue markup
    assert "\r" not in payload
    assert blocks[2] == "00:00:03.000 --> 00:00:04.000\nnext: following cue"
    assert blocks[3] == ""
    assert transcript == original


@pytest.mark.parametrize("unit", ["segments", "turns", "dict_turns"])
def test_vtt_speaker_cannot_terminate_or_inject_cues(tmp_path: Path, unit: str) -> None:
    speaker = 'O\'Brien "Q" <v someone> &amp;\r\n\r\n00:00:07.000 --> 00:00:08.000'
    transcript = _transcript("content", speaker, unit)
    path = tmp_path / "transcript.vtt"

    export_vtt(transcript, path, unit="segments" if unit == "segments" else "turns")

    blocks = path.read_text(encoding="utf-8").split("\n\n")
    assert len(blocks) == 4
    timing, payload = blocks[1].split("\n", 1)
    assert timing == "00:00:01.000 --> 00:00:02.000"
    assert html.unescape(payload) == f"{speaker}: content"
    assert "<" not in payload and "\r" not in payload


def test_csv_retains_existing_text_normalization(tmp_path: Path) -> None:
    path = tmp_path / "transcript.csv"
    export_csv(_transcript("  padded text  ", "speaker", "segments"), path)
    with path.open(encoding="utf-8", newline="") as source:
        assert next(csv.DictReader(source))["text"] == "padded text"


def test_vtt_empty_transcript(tmp_path: Path) -> None:
    path = tmp_path / "empty.vtt"
    export_vtt(Transcript(file_name="silent.wav", language="en", segments=[]), path)
    assert path.read_bytes() == b"WEBVTT\n\n"


@pytest.mark.parametrize("unit", ["segments", "turns", "dict_turns"])
def test_vtt_single_line_feeds_survive_ffmpeg_conversion(tmp_path: Path, unit: str) -> None:
    path = tmp_path / "transcript.vtt"
    output = tmp_path / "transcript.srt"
    export_vtt(
        _transcript("first\nsecond\nthird", "speaker", unit),
        path,
        unit="segments" if unit == "segments" else "turns",
    )
    assert "speaker: first\nsecond\nthird\n\n" in path.read_text(encoding="utf-8")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is required for subtitle conversion compatibility")
    subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(path), str(output)],
        check=True,
        capture_output=True,
        timeout=15,
    )
    assert output.read_text(encoding="utf-8") == (
        "1\n00:00:01,000 --> 00:00:02,000\nspeaker: first\nsecond\nthird\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nnext: following cue\n\n"
    )
