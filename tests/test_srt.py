from pathlib import Path

import pytest

from transcription.models import Segment, Transcript
from transcription.writers import _fmt_srt_ts, write_srt


def test_fmt_srt_ts_basic() -> None:
    assert _fmt_srt_ts(0.0) == "00:00:00,000"
    # 1 hour, 2 minutes, 3.456 seconds
    t = 1 * 3600 + 2 * 60 + 3.456
    assert _fmt_srt_ts(t) == "01:02:03,456"


@pytest.mark.parametrize("offset_ms", [0, 59_000, 3_599_000, 359_999_000])
def test_fmt_srt_ts_preserves_integer_milliseconds(offset_ms: int) -> None:
    for total_ms in range(offset_ms, offset_ms + 10_001):
        # Build the expected clock directly from integer input, without float math.
        seconds, ms = divmod(total_ms, 1000)
        expected = f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02},{ms:03}"
        assert _fmt_srt_ts(total_ms / 1000) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0004, "00:00:00,000"),
        (0.0006, "00:00:00,001"),
        (1.001, "00:00:01,001"),
        (1.9994, "00:00:01,999"),
        (1.9996, "00:00:02,000"),
        (59.9996, "00:01:00,000"),
        (3599.9996, "01:00:00,000"),
        (359999.9996, "100:00:00,000"),
    ],
)
def test_fmt_srt_ts_rounds_before_splitting_clock(seconds: float, expected: str) -> None:
    assert _fmt_srt_ts(seconds) == expected


def test_write_srt_quantizes_segment_boundaries(tmp_path: Path) -> None:
    transcript = Transcript(
        file_name="synthetic.wav",
        language="en",
        segments=[
            Segment(id=0, start=1.001, end=1.9996, text="first"),
            Segment(id=1, start=59.9996, end=3600.0006, text="second"),
        ],
    )
    output = tmp_path / "transcript.srt"

    write_srt(transcript, output)

    assert output.read_text(encoding="utf-8") == (
        "1\n00:00:01,001 --> 00:00:02,000\nfirst\n\n2\n00:01:00,000 --> 01:00:00,001\nsecond\n\n"
    )
    assert transcript.segments[0].start == 1.001
    assert transcript.segments[0].end == 1.9996
