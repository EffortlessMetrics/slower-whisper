"""Tests for audio normalization utilities."""

import os
from pathlib import Path

from transcription import audio_io
from transcription.config import Paths


def test_normalize_all_refreshes_when_source_is_newer(tmp_path, monkeypatch):
    """normalize_all should rerun ffmpeg when the source file is newer than the output."""
    paths = Paths(root=tmp_path)
    audio_io.ensure_dirs(paths)

    # Pretend ffmpeg is available and stub out the subprocess call.
    monkeypatch.setattr(audio_io, "ffmpeg_available", lambda: True)

    calls: list[list[str]] = []

    class FakeResult:
        """Fake subprocess.CompletedProcess for testing."""

        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        dst = Path(cmd[-1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(f"run-{len(calls)}", encoding="utf-8")
        return FakeResult()

    monkeypatch.setattr(audio_io.subprocess, "run", fake_run)

    src = paths.raw_dir / "clip.mp3"
    src.write_text("v1", encoding="utf-8")

    audio_io.normalize_all(paths)
    dst = paths.norm_dir / "clip.wav"
    assert dst.read_text(encoding="utf-8") == "run-1"
    assert len(calls) == 1

    # Make the source newer and ensure we refresh the normalized file.
    src.write_text("v2", encoding="utf-8")
    newer_time = dst.stat().st_mtime + 10
    os.utime(src, times=(newer_time, newer_time))

    audio_io.normalize_all(paths)
    assert dst.read_text(encoding="utf-8") == "run-2"
    assert len(calls) == 2

    # When timestamps match, no additional normalization should happen.
    synced_time = dst.stat().st_mtime
    os.utime(src, times=(synced_time, synced_time))
    audio_io.normalize_all(paths)
    assert len(calls) == 2
