"""Tests for audio normalization utilities."""

import os
from pathlib import Path

import pytest

from slower_whisper.pipeline import audio_io
from slower_whisper.pipeline.config import Paths


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


class TestPathValidation:
    """Tests for path safety validation."""

    def test_validate_path_safety_rejects_forbidden_chars(self):
        """Should raise ValueError if path contains shell metacharacters."""
        unsafe_paths = [
            "file|pipe.wav",
            "file;cmd.wav",
            "$(cmd).wav",
            "file&.wav",
            "`cmd`.wav",
            "file>.wav",
            "file<.wav",
        ]
        for p in unsafe_paths:
            with pytest.raises(ValueError, match="Invalid characters in path"):
                audio_io._validate_path_safety(p)

    def test_validate_path_safety_rejects_leading_dash(self):
        """Should raise ValueError if path starts with - (option injection)."""
        with pytest.raises(ValueError, match="Path cannot start with '-'"):
            audio_io._validate_path_safety("-input.wav")

    def test_validate_path_safety_accepts_safe_paths(self):
        """Should accept standard filenames and paths."""
        safe_paths = [
            "file.wav",
            "/tmp/file.wav",
            "dir/file-123.wav",
            "file_name.wav",
            "./-file.wav",  # prefixed is safe
        ]
        for p in safe_paths:
            audio_io._validate_path_safety(p)

    def test_normalize_single_validates_paths(self, tmp_path, monkeypatch):
        """normalize_single should validate paths before running ffmpeg."""
        src = tmp_path / "src.wav"
        src.touch()
        dst = tmp_path / "dst.wav"

        monkeypatch.setattr(audio_io, "ffmpeg_available", lambda: True)

        # Mock subprocess to avoid actual execution
        monkeypatch.setattr(
            audio_io.subprocess,
            "run",
            lambda *args, **kwargs: None,
        )

        # Unsafe source
        with pytest.raises(ValueError, match="Invalid characters"):
            audio_io.normalize_single(Path("src;rm.wav"), dst)

        # Unsafe dest
        with pytest.raises(ValueError, match="Invalid characters"):
            audio_io.normalize_single(src, Path("dst|.wav"))

        # Leading dash
        with pytest.raises(ValueError, match="Path cannot start with '-'"):
            audio_io.normalize_single(Path("-src.wav"), dst)

    def test_normalize_all_skips_unsafe_paths_and_continues(self, tmp_path, monkeypatch):
        """normalize_all should skip files with unsafe names and continue processing others."""
        paths = Paths(root=tmp_path)
        audio_io.ensure_dirs(paths)

        monkeypatch.setattr(audio_io, "ffmpeg_available", lambda: True)

        class FakeVersionResult:
            returncode = 0
            stdout = "ffmpeg version 6.1.1 Copyright (c) 2000-2024"
            stderr = ""

        class FakeNormResult:
            returncode = 0
            stdout = ""
            stderr = ""

        processed_files: list[str] = []

        def fake_run(cmd, **kwargs):
            if cmd == ["ffmpeg", "-version"]:
                return FakeVersionResult()
            # Track which files were actually processed
            src_path = cmd[3]  # -i argument
            processed_files.append(src_path)
            dst = Path(cmd[-1])
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("normalized", encoding="utf-8")
            return FakeNormResult()

        monkeypatch.setattr(audio_io.subprocess, "run", fake_run)

        # Create a mix of safe and unsafe files
        safe_file = paths.raw_dir / "safe.mp3"
        safe_file.write_text("audio", encoding="utf-8")
        unsafe_file = paths.raw_dir / "unsafe;rm.mp3"  # Contains shell metachar
        unsafe_file.write_text("audio", encoding="utf-8")

        # normalize_all should not raise, should process safe file, skip unsafe
        audio_io.normalize_all(paths)

        # Only safe file should have been processed
        assert len(processed_files) == 1
        assert "safe.mp3" in processed_files[0]

        # Safe file should have output, unsafe should not
        assert (paths.norm_dir / "safe.wav").exists()
        assert not (paths.norm_dir / "unsafe;rm.wav").exists()
