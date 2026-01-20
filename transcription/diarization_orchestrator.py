"""
Diarization orchestration extracted from transcription.api.

This is kept separate so diarization logic can evolve without ballooning the public API module.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Literal

from .config import TranscriptionConfig
from .models import DiarizationMeta, Transcript

logger = logging.getLogger(__name__)


def _maybe_run_diarization(
    transcript: Transcript,
    wav_path: Path,
    config: TranscriptionConfig,
) -> Transcript:
    """
    Run diarization and turn building if enabled.

    Workflow:
    1. Instantiate pyannote-based Diarizer with provided device/min/max hints.
    2. Run diarizer on the normalized WAV to get speaker turns.
    3. Map turns onto ASR segments via assign_speakers().
    4. Group contiguous segments into turns with build_turns().
    5. Record status and error details in transcript.meta["diarization"].
    """
    if not config.enable_diarization:
        if transcript.meta is None:
            transcript.meta = {}
        transcript.meta["diarization"] = DiarizationMeta(
            requested=False,
            status="disabled",
            backend=None,
            num_speakers=None,
            error=None,
            error_type=None,
        ).to_dict()
        return transcript

    # Keep a snapshot so failures can't leave partial speaker state behind
    original_segment_speakers = [copy.deepcopy(seg.speaker) for seg in transcript.segments]
    original_speakers = copy.deepcopy(transcript.speakers)
    original_turns = copy.deepcopy(transcript.turns)

    try:
        from .diarization import Diarizer, assign_speakers
        from .turns import build_turns

        diarizer = Diarizer(
            device=config.diarization_device,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
        )
        speaker_turns = diarizer.run(wav_path)
        diarization_mode = os.getenv("SLOWER_WHISPER_PYANNOTE_MODE", "auto").lower()

        if len(speaker_turns) == 0:
            logger.warning("Diarization produced no speaker turns for %s", wav_path.name)

        unique_speakers = len({t.speaker_id for t in speaker_turns})
        if unique_speakers > 10:
            logger.warning(
                "Diarization found %d speakers for %s; this may indicate "
                "noisy audio or misconfiguration.",
                unique_speakers,
                wav_path.name,
            )

        transcript = assign_speakers(
            transcript,
            speaker_turns,
            overlap_threshold=config.overlap_threshold,
        )
        transcript = build_turns(transcript)

        if diarization_mode == "stub":
            from .diarization import _normalize_speaker_id

            speaker_map: dict[str, int] = {}
            normalized_turns: list[tuple[Any, str]] = []
            for turn in speaker_turns:
                norm_id = _normalize_speaker_id(turn.speaker_id, speaker_map)
                normalized_turns.append((turn, norm_id))

            aggregates: dict[str, dict[str, Any]] = {}
            for turn, norm_id in normalized_turns:
                agg = aggregates.setdefault(
                    norm_id,
                    {"id": norm_id, "label": None, "total_speech_time": 0.0, "num_segments": 0},
                )
                agg["total_speech_time"] += max(turn.end - turn.start, 0.0)
                agg["num_segments"] += 1

            transcript.speakers = list(aggregates.values())
            transcript.turns = [
                {
                    "id": f"turn_{idx}",
                    "speaker_id": norm_id,
                    "start": turn.start,
                    "end": turn.end,
                    "segment_ids": [seg.id for seg in transcript.segments],
                    "text": " ".join(seg.text for seg in transcript.segments if seg.text).strip(),
                }
                for idx, (turn, norm_id) in enumerate(normalized_turns)
            ]
            if transcript.segments and any(seg.speaker is None for seg in transcript.segments):
                default_id = transcript.speakers[0]["id"] if transcript.speakers else "spk_0"
                for seg in transcript.segments:
                    seg.speaker = {"id": default_id, "confidence": 1.0}

        if transcript.meta is None:
            transcript.meta = {}
        diar_meta = DiarizationMeta(
            status="ok",
            requested=True,
            backend="pyannote.audio",
            num_speakers=len(transcript.speakers) if transcript.speakers else 0,
            error=None,
            error_type=None,
        )
        transcript.meta["diarization"] = diar_meta.to_dict()
        return transcript

    except Exception as exc:
        # Restore original speaker annotations if anything was partially written
        try:
            segment_pairs = zip(transcript.segments, original_segment_speakers, strict=True)
        except ValueError:
            logger.warning(
                "Diarization failure changed segment count (expected %d, got %d); "
                "resetting speaker labels best-effort.",
                len(original_segment_speakers),
                len(transcript.segments),
            )
            segment_pairs = zip(transcript.segments, original_segment_speakers, strict=False)

        for seg, speaker in segment_pairs:
            seg.speaker = speaker
        if len(transcript.segments) > len(original_segment_speakers):
            for seg in transcript.segments[len(original_segment_speakers) :]:
                seg.speaker = None

        transcript.speakers = original_speakers
        transcript.turns = original_turns

        logger.warning(
            "Diarization failed for %s: %s. Proceeding without speakers/turns.",
            wav_path.name,
            exc,
            exc_info=True,
        )
        if transcript.meta is None:
            transcript.meta = {}

        error_msg = str(exc)
        cause = exc.__cause__
        cause_msg = str(cause) if cause else ""
        full_msg = f"{error_msg} {cause_msg}".lower()
        error_type = "unknown"

        if "hf_token" in full_msg or "use_auth_token" in full_msg:
            error_type = "auth"
        elif (
            isinstance(exc, ImportError | ModuleNotFoundError)
            or isinstance(cause, ImportError | ModuleNotFoundError)
            or "pyannote.audio" in full_msg
            or "no module named" in full_msg
        ):
            error_type = "missing_dependency"
        elif (
            "not found" in full_msg
            or isinstance(exc, FileNotFoundError)
            or isinstance(cause, FileNotFoundError)
        ):
            error_type = "file_not_found"

        status: Literal["skipped", "error"] = (
            "skipped" if error_type in {"missing_dependency", "auth"} else "error"
        )

        diar_meta = DiarizationMeta(
            status=status,
            requested=True,
            backend="pyannote.audio",
            num_speakers=None,
            error=error_msg,
            error_type=error_type,
        )
        transcript.meta["diarization"] = diar_meta.to_dict()
        return transcript
