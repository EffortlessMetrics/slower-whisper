import pytest

from transcription.models import Segment, Transcript
from transcription.writers import _fmt_srt_ts, write_srt


def test_fmt_srt_ts_basic() -> None:
    assert _fmt_srt_ts(0.0) == "00:00:00,000"
    # 1 hour, 2 minutes, 3.456 seconds
    t = 1 * 3600 + 2 * 60 + 3.456
    assert _fmt_srt_ts(t) == "01:02:03,456"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (1.001, "00:00:01,001"),
        (1.2344, "00:00:01,234"),
        (1.2346, "00:00:01,235"),
        (59.9996, "00:01:00,000"),
        (3599.9996, "01:00:00,000"),
    ],
)
def test_fmt_srt_ts_rounds_milliseconds(seconds: float, expected: str) -> None:
    assert _fmt_srt_ts(seconds) == expected


def test_write_srt_rounds_segment_boundaries(tmp_path) -> None:
    transcript = Transcript(
        file_name="synthetic.wav",
        language="en",
        segments=[Segment(id=0, start=1.001, end=59.9996, text="Example")],
    )
    output = tmp_path / "example.srt"
    write_srt(transcript, output)
    assert output.read_text(encoding="utf-8") == ("1\n00:00:01,001 --> 00:01:00,000\nExample\n\n")
