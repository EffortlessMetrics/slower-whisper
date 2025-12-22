"""Audio I/O and normalization utilities using ffmpeg.

This module provides functions for audio file normalization and directory
management. All audio is normalized to 16kHz mono WAV format for ASR processing.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from secrets import token_hex

from .config import Paths

logger = logging.getLogger(__name__)


def sanitize_filename(
    stem: str,
    suffix: str,
    *,
    default: str = "audio_file",
) -> tuple[str, str]:
    """
    Sanitize a filename stem and suffix for safe filesystem use.

    - Replaces non-word characters (except .-_) with underscores
    - Strips leading/trailing dots, underscores, and hyphens to prevent
      hidden files and pathological names like ".." or "."
    - Validates suffix matches expected extension pattern

    Args:
        stem: Original filename stem (without extension)
        suffix: Original filename suffix (including leading dot)
        default: Fallback stem if sanitization produces empty string

    Returns:
        Tuple of (safe_stem, safe_suffix)

    Example:
        >>> sanitize_filename("../../../etc/passwd", ".mp3")
        ('etc_passwd', '.mp3')
        >>> sanitize_filename("..", "")
        ('audio_file', '')
        >>> sanitize_filename("my file (1)", ".WAV")
        ('my_file__1_', '.WAV')
    """
    # Replace unsafe characters, then strip pathological leading/trailing chars
    safe_stem = re.sub(r"[^\w\-_.]", "_", stem).strip("._-") or default

    # Only allow simple extensions like .mp3, .wav, .WAV
    safe_suffix = suffix if re.match(r"^\.\w+$", suffix) else ""

    return safe_stem, safe_suffix


def ensure_within_dir(path: Path, base_dir: Path) -> Path:
    """
    Validate that a path resolves to within a base directory.

    Uses strict resolution for the base directory (must exist) and
    non-strict for the target path (may not exist yet).

    Args:
        path: Path to validate (may not exist)
        base_dir: Base directory that must contain the resolved path

    Returns:
        The resolved path if validation passes

    Raises:
        ValueError: If base_dir doesn't exist or path escapes it
        OSError: If path resolution fails

    Example:
        >>> ensure_within_dir(Path("/tmp/raw/file.wav"), Path("/tmp/raw"))
        PosixPath('/tmp/raw/file.wav')
        >>> ensure_within_dir(Path("/tmp/raw/../other/file.wav"), Path("/tmp/raw"))
        ValueError: Path escapes base directory: /tmp/other/file.wav
    """
    try:
        resolved_base = base_dir.resolve(strict=True)
    except (OSError, FileNotFoundError) as e:
        raise ValueError(f"Base directory does not exist: {base_dir}") from e

    resolved_path = path.resolve()

    if not resolved_path.is_relative_to(resolved_base):
        raise ValueError(f"Path escapes base directory: {resolved_path}")

    return resolved_path


def unique_path(path: Path) -> Path:
    """
    Return a unique path by appending a random suffix if the path exists.

    Useful for avoiding overwrites when copying files.

    Args:
        path: Desired path

    Returns:
        Original path if it doesn't exist, otherwise path with random suffix

    Example:
        >>> unique_path(Path("/tmp/file.wav"))  # if exists
        PosixPath('/tmp/file-a3b2c1d4.wav')
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    return parent / f"{stem}-{token_hex(4)}{suffix}"


def ensure_dirs(paths: Paths) -> None:
    """
    Ensure that all working directories exist.
    """
    for d in (paths.raw_dir, paths.norm_dir, paths.transcripts_dir, paths.json_dir):
        d.mkdir(parents=True, exist_ok=True)


def ffmpeg_available() -> bool:
    """
    Return True if ffmpeg is available on PATH.
    """
    return shutil.which("ffmpeg") is not None


def normalize_all(paths: Paths) -> None:
    """
    Convert all files in raw_dir to 16 kHz mono WAV in norm_dir using ffmpeg.

    Existing normalized WAVs are skipped so the operation is idempotent.
    Failures for individual files are logged and do not abort the entire run.
    """
    logger.info("Starting audio normalization with ffmpeg")

    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (for example via Chocolatey) "
            "and make sure 'ffmpeg' works in a new shell."
        )

    any_src = False
    for src in sorted(paths.raw_dir.iterdir()):
        if not src.is_file():
            continue

        any_src = True
        dst = paths.norm_dir / f"{src.stem}.wav"

        # If a normalized file already exists, skip only when it is up-to-date.
        if dst.exists():
            try:
                src_mtime = src.stat().st_mtime
                dst_mtime = dst.stat().st_mtime
                if dst_mtime >= src_mtime:
                    logger.info(
                        "Skipping already normalized file (up to date)",
                        extra={"file": src.name, "output": dst.name},
                    )
                    continue
                else:
                    logger.info(
                        "Re-normalizing file (source is newer)",
                        extra={"file": src.name, "output": dst.name},
                    )
            except OSError as stat_err:
                logger.warning(
                    "Could not compare timestamps for %s: %s; re-normalizing",
                    src.name,
                    stat_err,
                    extra={"file": src.name},
                )

        logger.info("Normalizing audio file", extra={"file": src.name, "output": dst.name})
        # Security fix: Use argument list instead of shell command to prevent command injection
        # Validate file paths to ensure they don't contain malicious characters
        try:
            # Validate source file path
            src_str = str(src)
            dst_str = str(dst)

            # Basic path validation - reject paths with potentially dangerous characters
            # This prevents path traversal and command injection attempts
            if any(
                char in src_str
                for char in ["&", "|", ";", "`", "$", "(", ")", '"', "'", "<", ">", "\\"]
            ):
                raise ValueError(f"Invalid characters in source path: {src_str}")
            if any(
                char in dst_str
                for char in ["&", "|", ";", "`", "$", "(", ")", '"', "'", "<", ">", "\\"]
            ):
                raise ValueError(f"Invalid characters in destination path: {dst_str}")

            # Ensure paths are within expected directories (use is_relative_to for safety)
            ensure_within_dir(src, paths.raw_dir)
            ensure_within_dir(dst, paths.norm_dir)

            # Use argument list to prevent shell injection
            cmd = [
                "ffmpeg",
                "-y",  # overwrite
                "-i",
                src_str,
                "-ac",
                "1",  # mono
                "-ar",
                "16000",  # 16 kHz
                dst_str,
            ]
            # Security fix: Use argument list without shell=True to prevent command injection
            # The default is shell=False, so we don't need to specify it explicitly
            # This ensures the command is executed as a list of arguments, not a shell string
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(
                    "Failed to normalize %s (exit code %d): %s",
                    src.name,
                    result.returncode,
                    result.stderr.strip() if result.stderr else "No error output",
                    extra={"file": src.name},
                )
        except OSError as e:
            # Handle subprocess launch failures (e.g., ffmpeg not found)
            logger.error(
                "Failed to run ffmpeg for %s: %s",
                src.name,
                e,
                exc_info=True,
                extra={"file": src.name},
            )

    if not any_src:
        logger.info("No files found in raw_audio/ directory")
    else:
        logger.info("Audio normalization complete")
