"""Validation helpers for service request payloads."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from .service_settings import HTTP_413_TOO_LARGE, STREAMING_CHUNK_SIZE

logger = logging.getLogger(__name__)


def validate_file_size(
    file_content: bytes,
    max_size_mb: int,
    file_type: str = "file",
) -> None:
    """
    Validate that uploaded file size does not exceed the maximum allowed size.

    Args:
        file_content: Raw file content bytes
        max_size_mb: Maximum allowed size in megabytes
        file_type: Type of file for error messages (e.g., "audio", "transcript")

    Raises:
        HTTPException: 413 if file exceeds maximum size
    """
    file_size_bytes = len(file_content)
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_mb > max_size_mb:
        logger.warning(
            "File too large: %.2f MB (max: %d MB) for %s",
            file_size_mb,
            max_size_mb,
            file_type,
        )
        raise HTTPException(
            status_code=HTTP_413_TOO_LARGE,
            detail=(
                f"{file_type.capitalize()} file too large: {file_size_mb:.2f} MB. "
                f"Maximum allowed size is {max_size_mb} MB."
            ),
        )


async def save_upload_file_streaming(
    upload: UploadFile,
    dest: Path,
    *,
    max_bytes: int,
    file_type: str = "file",
) -> None:
    """
    Stream an uploaded file to disk in chunks to prevent memory exhaustion.

    This function writes the uploaded file to disk in chunks rather than reading
    the entire file into memory first. This prevents DoS attacks via large file uploads.

    Args:
        upload: The FastAPI UploadFile to read from.
        dest: Destination path to write the file to.
        max_bytes: Maximum allowed file size in bytes.
        file_type: Type of file for error messages (e.g., "audio", "transcript").

    Raises:
        HTTPException: 413 if file exceeds max_bytes.
        HTTPException: 500 if file write fails.
    """
    total = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await upload.read(STREAMING_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=HTTP_413_TOO_LARGE,
                        detail=f"{file_type.capitalize()} file too large: >{max_bytes // (1024 * 1024)} MB",
                    )
                f.write(chunk)
    except HTTPException:
        # Clean up partial file on size error
        if dest.exists():
            dest.unlink()
        raise
    except Exception as e:
        # Clean up partial file on other errors
        if dest.exists():
            dest.unlink()
        logger.error("Failed to save uploaded %s file", file_type, exc_info=e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded {file_type} file",
        ) from e


def validate_audio_format(audio_path: Path) -> None:
    """
    Validate that the uploaded file is a valid audio file.

    This performs a lightweight check by attempting to probe the file with ffmpeg.
    If ffmpeg cannot identify it as audio, we reject it early.
    When ffprobe is not available, we perform basic Python-based validation.

    Args:
        audio_path: Path to the audio file to validate

    Raises:
        HTTPException: 400 if file is not a valid audio format
    """
    import subprocess

    try:
        # Use ffprobe to check if file is valid audio
        # -v error: only show errors
        # -show_entries format=format_name: show format info
        # -of default=noprint_wrappers=1: simple output format
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=format_name,duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path.resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0 or not result.stdout.strip():
            logger.warning("Invalid audio file: ffprobe failed to identify format")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid audio file format. Please upload a valid audio file "
                    "(e.g., mp3, wav, m4a, flac, ogg)."
                ),
            )

        # Check that duration is present and reasonable
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            logger.warning("Invalid audio file: missing duration information")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid audio file: unable to determine audio duration.",
            )

        try:
            duration = float(lines[1])
            if duration <= 0:
                logger.warning("Invalid audio file: zero or negative duration")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid audio file: audio has zero or negative duration.",
                )
        except (ValueError, IndexError) as e:
            logger.warning("Invalid audio file: cannot parse duration: %s", e)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid audio file: unable to parse audio duration.",
            ) from e

    except subprocess.TimeoutExpired as e:
        logger.error("Audio validation timeout")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio validation timeout: file may be corrupted or invalid.",
        ) from e
    except FileNotFoundError:
        logger.error("ffprobe not found - using Python-based validation")
        # Security fix: When ffprobe is not available, perform basic Python validation
        # instead of accepting any file
        _validate_audio_format_python(audio_path)


def _validate_audio_format_python(audio_path: Path) -> None:
    """
    Basic Python-based audio file validation when ffprobe is not available.

    This is a fallback validation that checks file headers and basic properties
    to ensure the file is likely a valid audio file.

    Args:
        audio_path: Path to the audio file to validate

    Raises:
        HTTPException: 400 if file appears to be invalid
    """
    try:
        # Check file size (must be larger than 0)
        file_size = audio_path.stat().st_size
        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid audio file: file is empty.",
            )

        # Check file extension against allowed list
        allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
        if audio_path.suffix.lower() not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid audio file extension '{audio_path.suffix}'. "
                    f"Allowed extensions: {', '.join(sorted(allowed_extensions))}"
                ),
            )

        # Basic header validation for common formats
        with open(audio_path, "rb") as f:
            # Read first 16 bytes for header check (enough for WMA GUID)
            header = f.read(16)

            suffix = audio_path.suffix.lower()

            # WAV files start with "RIFF" and have "WAVE" at bytes 8-11
            if suffix == ".wav":
                if len(header) < 12 or not header.startswith(b"RIFF") or header[8:12] != b"WAVE":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid WAV file: incorrect header format.",
                    )

            # MP3 files typically start with ID3 tag or MPEG frame sync
            elif suffix == ".mp3":
                if len(header) < 3 or not (
                    header.startswith(b"ID3") or header.startswith(b"\xff\xfb")
                ):
                    # Not a definitive check, but catches obvious non-MP3 files
                    logger.warning("Possible invalid MP3 file: missing ID3 tag or MPEG sync")

            # M4A/AAC files start with "ftyp" box
            elif suffix in (".m4a", ".aac"):
                if len(header) < 8 or header[4:8] != b"ftyp":
                    logger.warning("Possible invalid M4A/AAC file: missing ftyp box")

            # FLAC files start with "fLaC"
            elif suffix == ".flac":
                if len(header) < 4 or header[:4] != b"fLaC":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid FLAC file: incorrect header format.",
                    )

            # OGG files start with "OggS"
            elif suffix == ".ogg":
                if len(header) < 4 or header[:4] != b"OggS":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid OGG file: incorrect header format.",
                    )

            # WMA files start with specific GUID
            elif suffix == ".wma":
                # ASF GUID: 30 26 B2 75 8E 66 CF 11 A6 D9 00 AA 00 62 CE 6C
                wma_guid = b"\x30\x26\xB2\x75\x8E\x66\xCF\x11\xA6\xD9\x00\xAA\x00\x62\xCE\x6C"
                if len(header) < 16 or header[:16] != wma_guid:
                    logger.warning("Possible invalid WMA file: missing ASF GUID")
                    # Strict check for WMA to be safe
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid WMA file: incorrect header format.",
                    )

        # If we get here, basic checks passed
        logger.info("Basic validation passed for %s (ffprobe unavailable)", audio_path.name)

    except HTTPException:
        # Re-raise our own validation errors
        raise
    except Exception as e:
        logger.error("Error during Python audio validation: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio file: validation failed.",
        ) from e


def validate_transcript_json(transcript_content: bytes) -> None:
    """
    Validate that the uploaded transcript is valid JSON with expected structure.

    Args:
        transcript_content: Raw transcript file content bytes

    Raises:
        HTTPException: 400 if transcript is not valid JSON or missing required fields
    """
    try:
        data = json.loads(transcript_content)
    except json.JSONDecodeError as e:
        logger.warning("Invalid transcript JSON: %s", e)
        # Security fix: Provide safe parse error location without raw exception
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transcript JSON: parse error at line {e.lineno}, column {e.colno}",
        ) from e

    # Validate required top-level fields
    required_fields = ["file_name", "segments"]
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        logger.warning("Transcript missing required fields: %s", missing_fields)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid transcript structure: missing required fields {missing_fields}. "
                "Expected a transcript JSON from /transcribe endpoint."
            ),
        )

    # Validate segments is a list
    if not isinstance(data.get("segments"), list):
        logger.warning("Transcript 'segments' field is not a list")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid transcript structure: 'segments' must be a list.",
        )

    # Validate segments is not empty
    if len(data["segments"]) == 0:
        logger.warning("Transcript has no segments")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid transcript: no segments found. Transcript must contain at least one segment.",
        )
