"""Tests for path sanitization and security helpers in audio_io."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from transcription.audio_io import (
    ensure_within_dir,
    get_safe_audio_extension,
    sanitize_filename,
    unique_path,
    validate_path_safety,
)


class TestSanitizeFilename:
    """Tests for sanitize_filename helper."""

    def test_normal_filename_unchanged(self) -> None:
        """Normal filenames pass through with minimal changes."""
        stem, suffix = sanitize_filename("recording", ".wav")
        assert stem == "recording"
        assert suffix == ".wav"

    def test_spaces_replaced_with_underscores(self) -> None:
        """Spaces and special chars become underscores, trailing stripped."""
        stem, suffix = sanitize_filename("my recording (1)", ".mp3")
        # Trailing underscore from ')' is stripped
        assert stem == "my_recording__1"
        assert suffix == ".mp3"

    def test_dotdot_normalized_away(self) -> None:
        """Pathological '..' stem is normalized to default."""
        stem, suffix = sanitize_filename("..", "")
        # ".." -> "__" after replacement, then stripped -> "" -> "audio_file"
        assert stem == "audio_file"
        assert suffix == ""

    def test_single_dot_normalized_away(self) -> None:
        """Single '.' stem is normalized to default."""
        stem, suffix = sanitize_filename(".", "")
        # "." -> "_" after replacement, then stripped -> "" -> "audio_file"
        assert stem == "audio_file"

    def test_hidden_file_dots_stripped(self) -> None:
        """Leading dots that would create hidden files are stripped."""
        stem, suffix = sanitize_filename(".hidden", ".wav")
        # ".hidden" -> "_hidden" after replacement, then strip "._-" from edges
        assert stem == "hidden"
        assert suffix == ".wav"

    def test_traversal_attempt_sanitized(self) -> None:
        """Directory traversal paths are sanitized."""
        stem, suffix = sanitize_filename("../../../etc/passwd", ".txt")
        # "../../../etc/passwd" -> "______etc_passwd" -> strip -> "etc_passwd"
        assert stem == "etc_passwd"
        assert suffix == ".txt"

    def test_underscore_only_becomes_default(self) -> None:
        """Filenames that reduce to only underscores become default."""
        stem, suffix = sanitize_filename("___", ".wav")
        assert stem == "audio_file"
        assert suffix == ".wav"

    def test_valid_extension_preserved(self) -> None:
        """Valid extensions like .wav, .mp3, .WAV are preserved."""
        _, suffix = sanitize_filename("file", ".WAV")
        assert suffix == ".WAV"
        _, suffix = sanitize_filename("file", ".mp3")
        assert suffix == ".mp3"
        _, suffix = sanitize_filename("file", ".flac")
        assert suffix == ".flac"

    def test_invalid_extension_removed(self) -> None:
        """Invalid extensions are removed for safety."""
        _, suffix = sanitize_filename("file", ".wav.exe")
        assert suffix == ""
        _, suffix = sanitize_filename("file", "")
        assert suffix == ""
        _, suffix = sanitize_filename("file", "noperiod")
        assert suffix == ""

    def test_custom_default(self) -> None:
        """Custom default stem can be provided."""
        stem, _ = sanitize_filename("..", "", default="unknown")
        assert stem == "unknown"


class TestEnsureWithinDir:
    """Tests for ensure_within_dir helper."""

    def test_valid_path_within_dir(self, tmp_path: Path) -> None:
        """Valid paths within the base directory pass."""
        base = tmp_path / "raw"
        base.mkdir()
        target = base / "file.wav"

        result = ensure_within_dir(target, base)
        assert result.is_relative_to(base.resolve())

    def test_prefix_sibling_rejected(self, tmp_path: Path) -> None:
        """Paths that are prefix-siblings (e.g., /raw2 vs /raw) are rejected.

        This catches the startswith() bug where '/tmp/raw2/x' passes
        startswith('/tmp/raw') but isn't actually contained.
        """
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        sibling_dir = tmp_path / "raw2"
        sibling_dir.mkdir()
        target = sibling_dir / "file.wav"

        with pytest.raises(ValueError, match="escapes base directory"):
            ensure_within_dir(target, raw_dir)

    def test_parent_traversal_rejected(self, tmp_path: Path) -> None:
        """Parent traversal via .. is rejected after resolution."""
        base = tmp_path / "raw"
        base.mkdir()
        # This path resolves to tmp_path/file.wav, outside of base
        target = base / ".." / "file.wav"

        with pytest.raises(ValueError, match="escapes base directory"):
            ensure_within_dir(target, base)

    def test_nonexistent_base_rejected(self, tmp_path: Path) -> None:
        """Non-existent base directory raises ValueError."""
        base = tmp_path / "nonexistent"
        target = base / "file.wav"

        with pytest.raises(ValueError, match="does not exist"):
            ensure_within_dir(target, base)

    @pytest.mark.skipif(os.name == "nt", reason="Symlinks require admin on Windows")
    def test_symlink_escape_rejected(self, tmp_path: Path) -> None:
        """Symlinks that escape the base directory are rejected.

        raw/link -> ../outside/
        raw/link/file.wav resolves to outside/file.wav
        """
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        # Create symlink: raw/link -> ../outside
        link = raw_dir / "link"
        link.symlink_to(outside_dir, target_is_directory=True)

        # This path appears to be under raw/ but resolves outside
        target = link / "file.wav"

        with pytest.raises(ValueError, match="escapes base directory"):
            ensure_within_dir(target, raw_dir)


class TestUniquePath:
    """Tests for unique_path helper."""

    def test_nonexistent_path_unchanged(self, tmp_path: Path) -> None:
        """Non-existent paths are returned unchanged."""
        target = tmp_path / "newfile.wav"
        result = unique_path(target)
        assert result == target

    def test_existing_path_gets_suffix(self, tmp_path: Path) -> None:
        """Existing paths get a random hex suffix."""
        target = tmp_path / "existing.wav"
        target.touch()

        result = unique_path(target)
        assert result != target
        assert result.parent == target.parent
        assert result.suffix == ".wav"
        assert result.stem.startswith("existing-")
        # Hex suffix should be 8 chars (4 bytes)
        suffix_part = result.stem.split("-")[-1]
        assert len(suffix_part) == 8
        # Verify it's valid hex
        int(suffix_part, 16)

    def test_unique_path_does_not_exist(self, tmp_path: Path) -> None:
        """The returned unique path should not exist."""
        target = tmp_path / "collision.wav"
        target.touch()

        result = unique_path(target)
        assert not result.exists()


class TestGetSafeAudioExtension:
    """Tests for get_safe_audio_extension helper."""

    def test_valid_extensions(self) -> None:
        """Valid extensions are returned."""
        assert get_safe_audio_extension("file.wav") == ".wav"
        assert get_safe_audio_extension("song.mp3") == ".mp3"
        assert get_safe_audio_extension("audio.m4a") == ".m4a"
        assert get_safe_audio_extension("track.flac") == ".flac"
        assert get_safe_audio_extension("voice.ogg") == ".ogg"
        assert get_safe_audio_extension("sound.aac") == ".aac"
        assert get_safe_audio_extension("windows.wma") == ".wma"

    def test_case_insensitive(self) -> None:
        """Extensions are case-insensitive."""
        assert get_safe_audio_extension("FILE.WAV") == ".WAV"
        assert get_safe_audio_extension("Song.Mp3") == ".Mp3"

    def test_invalid_extensions(self) -> None:
        """Invalid extensions return empty string."""
        assert get_safe_audio_extension("script.sh") == ""
        assert get_safe_audio_extension("image.png") == ""
        assert get_safe_audio_extension("data.json") == ""
        assert get_safe_audio_extension("archive.zip") == ""
        assert get_safe_audio_extension("program.exe") == ""

    def test_no_extension(self) -> None:
        """Filenames without extension return empty string."""
        assert get_safe_audio_extension("README") == ""
        assert get_safe_audio_extension("makefile") == ""

    def test_double_extension(self) -> None:
        """Only the last extension matters."""
        assert get_safe_audio_extension("archive.tar.gz") == ""
        assert get_safe_audio_extension("recording.wav.bak") == ""
        assert get_safe_audio_extension("backup.old.wav") == ".wav"


class TestValidatePathSafety:
    """Tests for validate_path_safety helper."""

    def test_safe_path(self) -> None:
        """Safe paths pass validation."""
        validate_path_safety("file.wav")
        validate_path_safety("path/to/file.wav")
        validate_path_safety(Path("file.wav"))

    def test_shell_injection_chars(self) -> None:
        """Paths with shell metacharacters raise ValueError."""
        unsafe_chars = ["&", "|", ";", "`", "$", "(", ")", '"', "'", "<", ">", "\\"]
        for char in unsafe_chars:
            with pytest.raises(ValueError, match="Invalid characters"):
                validate_path_safety(f"file{char}.wav")

    def test_option_injection(self) -> None:
        """Paths starting with dash raise ValueError."""
        with pytest.raises(ValueError, match="cannot start with '-'"):
            validate_path_safety("-option.wav")

        with pytest.raises(ValueError, match="cannot start with '-'"):
            validate_path_safety(Path("-file.wav"))

    def test_safe_dash_in_middle(self) -> None:
        """Dash in middle of filename is safe."""
        validate_path_safety("my-file.wav")
        validate_path_safety("dir/-file.wav")
