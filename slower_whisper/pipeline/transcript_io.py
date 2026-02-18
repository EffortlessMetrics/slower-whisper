"""
Public transcript I/O wrappers extracted from transcription.api.
"""

from __future__ import annotations

from pathlib import Path

from .exceptions import TranscriptionError
from .models import Transcript
from .writers import load_transcript_from_json, write_json


def load_transcript(json_path: str | Path, *, strict: bool = False) -> Transcript:
    json_path = Path(json_path)

    if not json_path.exists():
        raise TranscriptionError(
            f"Transcript file not found: {json_path}. "
            f"Ensure the file path is correct and the file exists."
        )

    if strict:
        from .validation import validate_transcript_json

        is_valid, errors = validate_transcript_json(json_path)
        if not is_valid:
            error_summary = "; ".join(errors[:3])
            if len(errors) > 3:
                error_summary += f" (and {len(errors) - 3} more)"
            raise TranscriptionError(
                f"Schema validation failed for {json_path.name}: {error_summary}"
            )

    try:
        return load_transcript_from_json(json_path)
    except Exception as e:
        raise TranscriptionError(
            f"Failed to load transcript from {json_path.name}: {e}. "
            f"The JSON file may be corrupted or have an invalid schema."
        ) from e


def save_transcript(transcript: Transcript, json_path: str | Path) -> None:
    json_path = Path(json_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        write_json(transcript, json_path)
    except Exception as e:
        raise TranscriptionError(
            f"Failed to save transcript to {json_path}: {e}. "
            f"Check that you have write permissions and sufficient disk space."
        ) from e
