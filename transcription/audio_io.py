"""Audio I/O and normalization utilities using ffmpeg.

This module provides functions for audio file normalization and directory
management. All audio is normalized to 16kHz mono WAV format for ASR processing.
"""

import logging
import shutil
import subprocess

from .config import Paths

logger = logging.getLogger(__name__)


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
        cmd = [
            "ffmpeg",
            "-y",  # overwrite
            "-i",
            str(src),
            "-ac",
            "1",  # mono
            "-ar",
            "16000",  # 16 kHz
            str(dst),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            logger.error(
                "Failed to normalize %s",
                src.name,
                exc_info=True,
                extra={"file": src.name},
            )

    if not any_src:
        logger.info("No files found in raw_audio/ directory")
    else:
        logger.info("Audio normalization complete")
