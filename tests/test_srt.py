from slower_whisper.pipeline.writers import _fmt_srt_ts


def test_fmt_srt_ts_basic() -> None:
    assert _fmt_srt_ts(0.0) == "00:00:00,000"
    # 1 hour, 2 minutes, 3.456 seconds
    t = 1 * 3600 + 2 * 60 + 3.456
    assert _fmt_srt_ts(t) == "01:02:03,456"
