"""Tests for audio normalization utilities."""

import os
from pathlib import Path

import pytest

from transcription import audio_io
from transcription.config import Paths


def test_normalize_all_refreshes_when_source_is_newer(tmp_path, monkeypatch):
    """normalize_all should rerun ffmpeg when the source file is newer than the output."""
    paths = Paths(root=tmp_path)
    audio_io.ensure_dirs(paths)

    # Pretend ffmpeg is available and stub out the subprocess call.
    monkeypatch.setattr(audio_io, "ffmpeg_available", lambda: True)

    normalization_calls: list[list[str]] = []

    class FakeVersionResult:
        """Fake subprocess result for version check."""

        returncode = 0
        stdout = "ffmpeg version 6.1.1 Copyright (c) 2000-2024"
        stderr = ""

    class FakeNormResult:
        """Fake subprocess.CompletedProcess for normalization."""

        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        # Handle version check calls separately
        if cmd == ["ffmpeg", "-version"]:
            return FakeVersionResult()

        # Track normalization calls
        normalization_calls.append(list(cmd))
        dst = Path(cmd[-1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(f"run-{len(normalization_calls)}", encoding="utf-8")
        return FakeNormResult()

    monkeypatch.setattr(audio_io.subprocess, "run", fake_run)

    src = paths.raw_dir / "clip.mp3"
    src.write_text("v1", encoding="utf-8")

    audio_io.normalize_all(paths)
    dst = paths.norm_dir / "clip.wav"
    assert dst.read_text(encoding="utf-8") == "run-1"
    assert len(normalization_calls) == 1

    # Make the source newer and ensure we refresh the normalized file.
    src.write_text("v2", encoding="utf-8")
    newer_time = dst.stat().st_mtime + 10
    os.utime(src, times=(newer_time, newer_time))

    audio_io.normalize_all(paths)
    assert dst.read_text(encoding="utf-8") == "run-2"
    assert len(normalization_calls) == 2

    # When timestamps match, no additional normalization should happen.
    synced_time = dst.stat().st_mtime
    os.utime(src, times=(synced_time, synced_time))
    audio_io.normalize_all(paths)
    assert len(normalization_calls) == 2


class TestFFmpegErrorHandling:
    """Tests for improved ffmpeg error messages."""

    def test_ffmpeg_not_found_error_has_install_instructions(self, monkeypatch):
        """FFmpegNotFoundError should include installation instructions."""
        monkeypatch.setattr(audio_io, "ffmpeg_available", lambda: False)

        with pytest.raises(audio_io.FFmpegNotFoundError) as exc_info:
            audio_io.check_ffmpeg_installation()

        error_msg = str(exc_info.value)
        # Should include installation instructions for common platforms
        assert "apt" in error_msg.lower() or "apt install ffmpeg" in error_msg
        assert "brew" in error_msg.lower() or "brew install ffmpeg" in error_msg
        assert "choco" in error_msg.lower() or "choco install ffmpeg" in error_msg
        # Should include verification steps
        assert "ffmpeg -version" in error_msg
        # Should include documentation link
        assert "ffmpeg.org" in error_msg

    def test_ffmpeg_not_found_error_class_hierarchy(self):
        """FFmpegNotFoundError should be a RuntimeError subclass."""
        assert issubclass(audio_io.FFmpegNotFoundError, RuntimeError)

    def test_ffmpeg_error_class_stores_details(self):
        """FFmpegError should store returncode and stderr."""
        err = audio_io.FFmpegError("Test error", returncode=1, stderr="Invalid data found")
        assert err.returncode == 1
        assert err.stderr == "Invalid data found"
        assert "exit code: 1" in str(err)
        assert "Invalid data found" in str(err)

    def test_get_ffmpeg_version_returns_none_when_unavailable(self, monkeypatch):
        """get_ffmpeg_version should return None if ffmpeg is not available."""
        monkeypatch.setattr(audio_io, "ffmpeg_available", lambda: False)
        assert audio_io.get_ffmpeg_version() is None

    def test_get_ffmpeg_version_parses_version_string(self, monkeypatch):
        """get_ffmpeg_version should parse version from ffmpeg output."""
        monkeypatch.setattr(audio_io, "ffmpeg_available", lambda: True)

        class FakeResult:
            returncode = 0
            stdout = "ffmpeg version 6.1.1 Copyright (c) 2000-2024"
            stderr = ""

        monkeypatch.setattr(
            audio_io.subprocess,
            "run",
            lambda *args, **kwargs: FakeResult(),
        )
        assert audio_io.get_ffmpeg_version() == "6.1.1"

    def test_normalize_all_raises_ffmpeg_not_found_error(self, tmp_path, monkeypatch):
        """normalize_all should raise FFmpegNotFoundError when ffmpeg is missing."""
        paths = Paths(root=tmp_path)
        audio_io.ensure_dirs(paths)

        monkeypatch.setattr(audio_io, "ffmpeg_available", lambda: False)

        with pytest.raises(audio_io.FFmpegNotFoundError):
            audio_io.normalize_all(paths)
