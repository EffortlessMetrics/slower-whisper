"""Audio I/O and normalization utilities using ffmpeg.

This module provides functions for audio file normalization and directory
management. All audio is normalized to 16kHz mono WAV format for ASR processing.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from secrets import token_hex
from typing import NamedTuple

from .config import Paths

logger = logging.getLogger(__name__)


# User-friendly error message for missing ffmpeg dependency
FFMPEG_MISSING_MESSAGE = """
================================================================================
FFmpeg is not installed or not found on your system PATH.

FFmpeg is required for audio normalization (converting audio to 16kHz mono WAV).

INSTALLATION INSTRUCTIONS:
--------------------------

  Linux (Debian/Ubuntu):
    sudo apt update && sudo apt install ffmpeg

  Linux (Fedora):
    sudo dnf install ffmpeg

  Linux (Arch):
    sudo pacman -S ffmpeg

  macOS (Homebrew):
    brew install ffmpeg

  Windows (Chocolatey):
    choco install ffmpeg

  Windows (Scoop):
    scoop install ffmpeg

  Nix (recommended for this project):
    nix develop  # ffmpeg is included in the dev shell

VERIFICATION:
-------------
After installation, open a NEW terminal and run:
    ffmpeg -version

If you see version information, ffmpeg is correctly installed.

DOCUMENTATION:
--------------
  Official site: https://ffmpeg.org/
  Download page: https://ffmpeg.org/download.html

================================================================================
"""


class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg is not available on the system PATH."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or FFMPEG_MISSING_MESSAGE)


class FFmpegError(RuntimeError):
    """Raised when ffmpeg command fails during audio processing."""

    def __init__(
        self,
        message: str,
        returncode: int | None = None,
        stderr: str | None = None,
    ) -> None:
        self.returncode = returncode
        self.stderr = stderr
        full_message = message
        if returncode is not None:
            full_message += f" (exit code: {returncode})"
        if stderr:
            full_message += f"\n\nffmpeg stderr:\n{stderr}"
        super().__init__(full_message)


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


def _check_within_resolved_base(path: Path, resolved_base: Path) -> Path:
    """
    Validate that a path resolves to within a pre-resolved base directory.

    This is an optimized internal helper for batch operations where the base
    directory has already been resolved. Avoids repeated resolve(strict=True)
    syscalls when checking many files against the same base.

    Args:
        path: Path to validate (may not exist)
        resolved_base: Already-resolved base directory path

    Returns:
        The resolved path if validation passes

    Raises:
        ValueError: If path escapes the base directory
    """
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_base):
        raise ValueError(f"Path escapes base directory: {resolved_path}")
    return resolved_path


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
    except OSError as e:
        raise ValueError(f"Base directory does not exist: {base_dir}") from e

    return _check_within_resolved_base(path, resolved_base)


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


def get_ffmpeg_version() -> str | None:
    """
    Get ffmpeg version string if available.

    Returns:
        Version string (e.g., "6.1.1") or None if ffmpeg is not available.
    """
    if not ffmpeg_available():
        return None

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout:
            # Parse version from first line: "ffmpeg version 6.1.1 Copyright..."
            first_line = result.stdout.split("\n")[0]
            parts = first_line.split()
            if len(parts) >= 3 and parts[0] == "ffmpeg" and parts[1] == "version":
                return parts[2]
        return None
    except (subprocess.TimeoutExpired, OSError):
        return None


def check_ffmpeg_installation() -> None:
    """
    Check if ffmpeg is properly installed and raise a helpful error if not.

    Raises:
        FFmpegNotFoundError: If ffmpeg is not found or not working.
    """
    if not ffmpeg_available():
        system = platform.system()
        hint = ""

        if system == "Linux":
            hint = "\n\nHint: On Linux, install with: sudo apt install ffmpeg"
        elif system == "Darwin":
            hint = "\n\nHint: On macOS, install with: brew install ffmpeg"
        elif system == "Windows":
            hint = "\n\nHint: On Windows, install with: choco install ffmpeg"

        logger.error(
            "ffmpeg not found on PATH. System: %s, PATH: %s",
            system,
            os.environ.get("PATH", "<not set>"),
        )
        raise FFmpegNotFoundError(FFMPEG_MISSING_MESSAGE + hint)

    # Verify ffmpeg actually works
    version = get_ffmpeg_version()
    if version:
        logger.debug("ffmpeg version %s detected", version)
    else:
        logger.warning(
            "ffmpeg found on PATH but version could not be determined. "
            "Proceeding anyway, but audio normalization may fail."
        )


def _log_ffmpeg_error(filename: str, returncode: int, stderr: str) -> None:
    """
    Log a detailed error message for ffmpeg failures with debugging hints.

    Analyzes the stderr output to provide more specific guidance.
    """
    # Common ffmpeg error patterns and helpful messages
    error_hints = []

    if "No such file or directory" in stderr:
        error_hints.append("The input file may have been moved or deleted.")
    if "Invalid data found" in stderr or "Invalid argument" in stderr:
        error_hints.append(
            "The audio file may be corrupted or in an unsupported format. "
            "Try re-encoding the source file."
        )
    if "Permission denied" in stderr:
        error_hints.append(
            "Check file permissions on input/output paths. Ensure the output directory is writable."
        )
    if "codec not found" in stderr.lower() or "decoder" in stderr.lower():
        error_hints.append(
            "A required codec may be missing. "
            "Ensure ffmpeg was built with the necessary codec support."
        )
    if "out of memory" in stderr.lower() or "cannot allocate" in stderr.lower():
        error_hints.append("System ran out of memory. Try closing other applications.")
    if "disk" in stderr.lower() and ("full" in stderr.lower() or "space" in stderr.lower()):
        error_hints.append("Disk may be full. Free up space and try again.")

    # Build the error message
    hint_text = ""
    if error_hints:
        hint_text = "\n  Possible causes:\n    - " + "\n    - ".join(error_hints)

    logger.error(
        "ffmpeg failed to normalize '%s' (exit code %d)%s\n  Full ffmpeg stderr output:\n%s",
        filename,
        returncode,
        hint_text,
        _indent_stderr(stderr) if stderr else "    (no stderr output)",
        extra={"file": filename, "returncode": returncode, "stderr": stderr},
    )


def _indent_stderr(stderr: str, indent: str = "    ") -> str:
    """Indent each line of stderr for readable logging."""
    if not stderr:
        return ""
    lines = stderr.strip().split("\n")
    return "\n".join(f"{indent}{line}" for line in lines)


def _validate_path_safety(path: Path | str) -> None:
    """
    Validate that a path is safe for subprocess arguments.

    Prevents:
    - Shell injection (although we use list args, it's good defense in depth)
    - Option injection (paths starting with -)

    Args:
        path: Path to validate

    Raises:
        ValueError: If path contains unsafe characters or patterns.
    """
    path_str = str(path)

    # Check for shell metacharacters
    # This list covers characters that could be dangerous in shell context
    forbidden_chars = ["&", "|", ";", "`", "$", "(", ")", '"', "'", "<", ">", "\\"]
    if any(char in path_str for char in forbidden_chars):
        raise ValueError(f"Invalid characters in path: {path_str}")

    # Check for option injection (leading dash)
    # ffmpeg might interpret files starting with - as options
    if path_str.startswith("-"):
        raise ValueError(f"Path cannot start with '-': {path_str}. Use ./{path_str} instead.")


class _NormalizeResult(NamedTuple):
    """Result of normalizing a single file."""

    src: Path
    success: bool
    skipped: bool = False
    error: Exception | None = None
    abort: bool = False  # True if we should abort the entire run


def _normalize_one_file(
    src: Path, dst: Path, raw_dir_resolved: Path, norm_dir_resolved: Path
) -> _NormalizeResult:
    """
    Process a single file for normalization (thread-safe).

    Returns a result indicating success, skip, or error status.
    Sets abort=True for FFmpegNotFoundError to signal run termination.
    """
    # Check if already up-to-date
    if dst.exists():
        try:
            src_mtime = src.stat().st_mtime
            dst_mtime = dst.stat().st_mtime
            if dst_mtime >= src_mtime:
                logger.info(
                    "Skipping already normalized file (up to date)",
                    extra={"file": src.name, "output": dst.name},
                )
                return _NormalizeResult(src=src, success=True, skipped=True)
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

    try:
        _validate_path_safety(src)
        _validate_path_safety(dst)
        _ = _check_within_resolved_base(src, raw_dir_resolved)
        _ = _check_within_resolved_base(dst, norm_dir_resolved)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr_output = result.stderr.strip() if result.stderr else ""
            _log_ffmpeg_error(src.name, result.returncode, stderr_output)
            if dst.exists():
                try:
                    dst.unlink()
                    logger.debug("Removed partial output: %s", dst.name)
                except OSError:
                    pass
            return _NormalizeResult(src=src, success=False)
        return _NormalizeResult(src=src, success=True)

    except FileNotFoundError:
        # ffmpeg binary not found - signal abort
        logger.error(
            "ffmpeg executable not found while processing '%s'.",
            src.name,
            exc_info=True,
            extra={"file": src.name},
        )
        return _NormalizeResult(src=src, success=False, error=FFmpegNotFoundError(), abort=True)

    except ValueError as e:
        logger.warning("Skipping '%s': %s", src.name, e, extra={"file": src.name})
        return _NormalizeResult(src=src, success=False, error=e)

    except OSError as e:
        logger.error(
            "Failed to run ffmpeg for '%s': %s.",
            src.name,
            e,
            exc_info=True,
            extra={"file": src.name},
        )
        if dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        return _NormalizeResult(src=src, success=False, error=e)


def normalize_single(src: Path, dst: Path) -> None:
    """
    Normalize a single audio file to 16kHz mono WAV using ffmpeg.

    Args:
        src: Source audio file (any format supported by ffmpeg)
        dst: Destination path for the normalized WAV file

    Raises:
        FFmpegNotFoundError: If ffmpeg is not installed or not on PATH.
        FFmpegError: If ffmpeg fails to process the file.
        FileNotFoundError: If the source file does not exist.
        ValueError: If paths contain unsafe characters.
    """
    # Security fix: Validate paths first
    _validate_path_safety(src)
    _validate_path_safety(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source audio file not found: {src}")

    check_ffmpeg_installation()

    src_str = str(src)
    dst_str = str(dst)

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

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr_output = result.stderr.strip() if result.stderr else ""
            # Clean up partial/corrupt output file on failure
            if dst.exists():
                try:
                    dst.unlink()
                except OSError:
                    pass
            raise FFmpegError(
                f"Failed to normalize audio file: {src.name}",
                returncode=result.returncode,
                stderr=stderr_output,
            )
    except FileNotFoundError as e:
        raise FFmpegNotFoundError() from e


def normalize_all(paths: Paths) -> None:
    """
    Convert all files in raw_dir to 16 kHz mono WAV in norm_dir using ffmpeg.

    Processing is performed in parallel using a ThreadPoolExecutor for improved
    throughput when normalizing multiple files.

    Existing normalized WAVs are skipped so the operation is idempotent.
    Failures for individual files are logged and do not abort the entire run.

    Raises:
        FFmpegNotFoundError: If ffmpeg is not installed or not on PATH.
    """
    logger.info("Starting audio normalization with ffmpeg")

    # Check ffmpeg installation before spawning threads
    check_ffmpeg_installation()

    # Resolve base directories once to avoid repeated syscalls in workers
    try:
        raw_dir_resolved = paths.raw_dir.resolve(strict=True)
        norm_dir_resolved = paths.norm_dir.resolve(strict=True)
    except OSError as e:
        logger.error("Required directories missing: %s", e)
        raise

    # Collect files to process
    files_to_process: list[tuple[Path, Path]] = []
    skipped_count = 0

    for src in sorted(paths.raw_dir.iterdir()):
        if src.is_file():
            dst = paths.norm_dir / f"{src.stem}.wav"

            # Check if already up-to-date
            # We do this here to avoid creating thread/future overhead for
            # files that don't need processing (common case in re-runs).
            should_skip = False
            if dst.exists():
                try:
                    src_mtime = src.stat().st_mtime
                    dst_mtime = dst.stat().st_mtime
                    if dst_mtime >= src_mtime:
                        should_skip = True
                except OSError:
                    # If we can't read stats, assume we need to process
                    pass

            if should_skip:
                skipped_count += 1
                continue

            files_to_process.append((src, dst))

    if skipped_count > 0:
        logger.info("Skipped %d already normalized files", skipped_count)

    if not files_to_process:
        if skipped_count == 0:
            logger.info("No files found in raw_audio/ directory")
        else:
            logger.info("All files are up to date")
        return

    # Process files in parallel using ThreadPoolExecutor
    # Default workers = min(32, os.cpu_count() + 4), good for I/O-bound ffmpeg calls
    abort_error: FFmpegNotFoundError | None = None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_normalize_one_file, src, dst, raw_dir_resolved, norm_dir_resolved): src
            for src, dst in files_to_process
        }

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result.abort and result.error is not None:
                    # FFmpegNotFoundError - cancel remaining work and re-raise
                    abort_error = result.error  # type: ignore[assignment]
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
            except Exception as e:
                # Unexpected exception from thread
                src = futures[future]
                logger.error(
                    "Unexpected error processing '%s': %s",
                    src.name,
                    e,
                    exc_info=True,
                    extra={"file": src.name},
                )

    if abort_error is not None:
        raise abort_error

    logger.info("Audio normalization complete")
