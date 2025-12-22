"""
Public API for slower-whisper library.

This module provides a stable, high-level interface for transcription and
enrichment. It wraps the internal pipeline and audio_enrichment modules with
a clean API suitable for programmatic use.

Example usage:
    >>> from transcription import transcribe_directory, enrich_directory
    >>> from transcription import TranscriptionConfig, EnrichmentConfig
    >>>
    >>> # Stage 1: Transcribe
    >>> cfg = TranscriptionConfig(model="large-v3", language="en")
    >>> transcripts = transcribe_directory("/data/project", cfg)
    >>>
    >>> # Stage 2: Enrich
    >>> ecfg = EnrichmentConfig(enable_prosody=True, enable_emotion=True)
    >>> enriched = enrich_directory("/data/project", ecfg)
"""

from __future__ import annotations

import copy
import logging
import os
import wave
from pathlib import Path
from typing import Any, Literal

from .config import (
    AsrConfig,
    EnrichmentConfig,
    Paths,
    TranscriptionConfig,
    validate_diarization_settings,
)
from .exceptions import EnrichmentError, TranscriptionError
from .meta_utils import build_generation_metadata
from .models import DiarizationMeta, Transcript
from .writers import load_transcript_from_json, write_json

logger = logging.getLogger(__name__)


def _get_wav_duration_seconds(path: Path) -> float:
    """Return WAV duration in seconds, tolerating unreadable files."""
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
        return frames / float(rate) if rate else 0.0
    except Exception as exc:  # noqa: BLE001 - want the raw error for debugging
        logger.warning("Could not read duration for %s: %s", path.name, exc)
        return 0.0


def _neutral_audio_state(error: str | None = None) -> dict[str, Any]:
    """Create a minimal audio_state placeholder used when enrichment fails."""
    extraction_status = {
        "prosody": "skipped",
        "emotion_dimensional": "skipped",
        "emotion_categorical": "skipped",
        "errors": [error] if error else [],
    }
    return {
        "prosody": {
            "pitch": {"level": "unknown", "mean_hz": None, "std_hz": None, "contour": "unknown"},
            "energy": {"level": "unknown", "db_rms": None, "variation": "unknown"},
            "rate": {"level": "unknown", "syllables_per_sec": None, "words_per_sec": None},
            "pauses": {"count": 0, "longest_ms": 0, "density": "unknown"},
        },
        "emotion": {
            "valence": {"level": "neutral", "score": 0.5},
            "arousal": {"level": "medium", "score": 0.5},
            "dominance": {"level": "neutral", "score": 0.5},
        },
        "rendering": "[audio: neutral]",
        "extraction_status": extraction_status,
    }


def _turns_have_metadata(turns: Any) -> bool:
    """Return True when turns are present and each has metadata/meta."""
    if not turns:
        return False

    for turn in turns:
        meta = None
        if isinstance(turn, dict):
            meta = turn.get("metadata") or turn.get("meta")
        elif hasattr(turn, "metadata"):
            meta = getattr(turn, "metadata", None)
        elif hasattr(turn, "meta"):
            meta = getattr(turn, "meta", None)

        if not meta:
            return False
    return True


def _run_speaker_analytics(transcript: Transcript, config: EnrichmentConfig) -> Transcript:
    """Populate turn metadata and speaker_stats when enabled in config."""
    needs_turn_meta = config.enable_turn_metadata or config.enable_speaker_stats

    if needs_turn_meta and not transcript.turns:
        from .turns import build_turns

        transcript = build_turns(transcript, pause_threshold=config.pause_threshold)

    if needs_turn_meta:
        from .turns_enrich import enrich_turns_metadata

        enrich_turns_metadata(transcript)

    if config.enable_speaker_stats:
        from .speaker_stats import compute_speaker_stats

        compute_speaker_stats(transcript)

    return transcript


def _run_semantic_annotator(transcript: Transcript, config: EnrichmentConfig) -> Transcript:
    """Invoke semantic annotator hook when enabled."""
    if not getattr(config, "enable_semantic_annotator", False):
        return transcript

    annotator = getattr(config, "semantic_annotator", None)
    if annotator is None:
        from .semantic import KeywordSemanticAnnotator

        annotator = KeywordSemanticAnnotator()

    try:
        updated = annotator.annotate(transcript)
        return updated or transcript
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic annotator failed: %s", exc, exc_info=True)
        return transcript


def _maybe_build_chunks(transcript: Transcript, config: TranscriptionConfig) -> Transcript:
    """Attach RAG-friendly chunks when enabled."""
    if not getattr(config, "enable_chunking", False):
        return transcript

    from .chunking import ChunkingConfig, build_chunks

    chunk_cfg = ChunkingConfig(
        target_duration_s=config.chunk_target_duration_s,
        max_duration_s=config.chunk_max_duration_s,
        target_tokens=config.chunk_target_tokens,
        pause_split_threshold_s=config.chunk_pause_split_threshold_s,
    )
    build_chunks(transcript, chunk_cfg)
    return transcript


# ============================================================================
# Internal helper: Diarization orchestration (v1.1)
# ============================================================================


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

    Args:
        transcript: Transcript from ASR (with segments populated).
        wav_path: Path to normalized WAV file.
        config: TranscriptionConfig with diarization settings.

    Returns:
        Updated transcript (with speakers/turns if diarization succeeds,
        or original transcript with meta.diarization.status set).
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

    # v1.1: Real diarization implementation with pyannote.audio
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

        # Check for suspiciously high speaker counts
        unique_speakers = len({t.speaker_id for t in speaker_turns})
        if unique_speakers > 10:
            logger.warning(
                "Diarization found %d speakers for %s; this may indicate "
                "noisy audio or misconfiguration.",
                unique_speakers,
                wav_path.name,
            )

        # Assign speakers to segments based on overlap
        transcript = assign_speakers(
            transcript,
            speaker_turns,
            overlap_threshold=config.overlap_threshold,
        )

        # Build turn structure from speaker-labeled segments
        transcript = build_turns(transcript)

        if diarization_mode == "stub":
            # Ensure deterministic speaker/turn structures even with dummy ASR output.
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
                # Default segment speakers to the first normalized id when ASR output
                # lacks speaker annotations (common with dummy models in tests).
                default_id = transcript.speakers[0]["id"] if transcript.speakers else "spk_0"
                for seg in transcript.segments:
                    seg.speaker = {"id": default_id, "confidence": 1.0}

        # Record success in metadata
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
            segment_pairs = zip(
                transcript.segments,
                original_segment_speakers,
                strict=True,
            )
        except ValueError:
            logger.warning(
                "Diarization failure changed segment count (expected %d, got %d); "
                "resetting speaker labels best-effort.",
                len(original_segment_speakers),
                len(transcript.segments),
            )
            segment_pairs = zip(
                transcript.segments,
                original_segment_speakers,
                strict=False,
            )

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

        # Categorize error for better debugging
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


def transcribe_directory(
    root: str | Path,
    config: TranscriptionConfig,
) -> list[Transcript]:
    """
    Transcribe all audio files under a project root.

    Expected layout:
        raw_audio/      input recordings
        input_audio/    normalized WAVs (auto-created)
        whisper_json/   JSON transcripts (auto-created)
        transcripts/    TXT/SRT outputs (auto-created)

    Args:
        root: Root directory containing raw_audio/ subdirectory
        config: Transcription configuration

    Returns:
        List of Transcript objects (one per audio file)

    Example:
        >>> config = TranscriptionConfig(model="large-v3", language="en")
        >>> transcripts = transcribe_directory("/data/project", config)
        >>> print(f"Transcribed {len(transcripts)} files")
    """
    from .config import AppConfig
    from .pipeline import run_pipeline

    root = Path(root)
    validate_diarization_settings(
        config.min_speakers,
        config.max_speakers,
        config.overlap_threshold,
    )

    # Convert public config to internal AppConfig
    paths = Paths(root=root)
    asr_cfg = AsrConfig(
        model_name=config.model,
        device=config.device,
        compute_type=config.compute_type,
        vad_min_silence_ms=config.vad_min_silence_ms,
        beam_size=config.beam_size,
        language=config.language,
        task=config.task,
        word_timestamps=config.word_timestamps,
    )
    app_cfg = AppConfig(
        paths=paths,
        asr=asr_cfg,
        skip_existing_json=config.skip_existing_json,
    )

    # Run the pipeline (modifies files on disk)
    # Pass the config for diarization if enabled
    result = run_pipeline(app_cfg, diarization_config=config)
    logger.info(
        "Pipeline completed: %d processed, %d skipped, %d failed",
        result.processed,
        result.skipped,
        result.failed,
    )

    # Load and return all transcripts
    json_dir = paths.json_dir
    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        raise TranscriptionError(
            f"No transcripts found in {json_dir}. "
            f"Ensure audio files exist in {paths.raw_dir} and transcription completed successfully."
        )

    transcripts = []
    for json_path in json_files:
        try:
            transcript = load_transcript_from_json(json_path)
            transcripts.append(transcript)
        except Exception as e:
            raise TranscriptionError(
                f"Failed to load transcript from {json_path.name}: {e}. "
                f"The JSON file may be corrupted or have an invalid schema."
            ) from e

    return transcripts


def transcribe_file(
    audio_path: str | Path,
    root: str | Path,
    config: TranscriptionConfig,
) -> Transcript:
    """
    Transcribe a single audio file into the project structure.

    The file will be normalized into root/input_audio/ and the transcript
    written to root/whisper_json/ and root/transcripts/.

    Args:
        audio_path: Path to input audio file (any format supported by ffmpeg)
        root: Root directory for outputs
        config: Transcription configuration

    Returns:
        Transcript object

    Example:
        >>> config = TranscriptionConfig(model="large-v3")
        >>> transcript = transcribe_file("interview.mp3", "/data/project", config)
        >>> print(transcript.segments[0].text)
    """
    from .asr_engine import TranscriptionEngine

    audio_path = Path(audio_path)
    root = Path(root)
    validate_diarization_settings(
        config.min_speakers,
        config.max_speakers,
        config.overlap_threshold,
    )

    # Validate input
    if not audio_path.exists():
        raise TranscriptionError(
            f"Audio file not found: {audio_path}. "
            f"Please verify the file path and ensure the file exists."
        )
    if not audio_path.is_file():
        raise TranscriptionError(
            f"Audio path is not a file: {audio_path}. "
            "Provide a path to an audio file instead of a directory."
        )

    # For single-file mode, we use the ASR engine directly
    # (pipeline.run_pipeline is designed for batch processing)
    asr_cfg = AsrConfig(
        model_name=config.model,
        device=config.device,
        compute_type=config.compute_type,
        vad_min_silence_ms=config.vad_min_silence_ms,
        beam_size=config.beam_size,
        language=config.language,
        task=config.task,
        word_timestamps=config.word_timestamps,
    )

    # Normalize the audio file
    from . import audio_io

    paths = Paths(root=root)
    audio_io.ensure_dirs(paths)

    # Copy to raw_audio if not already there
    # Security fix: Validate and sanitize file paths to prevent directory traversal
    import shutil

    # Validate the source path is safe
    try:
        source_str = str(audio_path.resolve())
        # Check for path traversal attempts in the source path
        if ".." in source_str or source_str.startswith("/"):
            # Allow absolute paths but validate they point to actual files
            if not audio_path.is_file():
                raise TranscriptionError(f"Invalid source path: {audio_path}")
    except (OSError, ValueError) as e:
        raise TranscriptionError(f"Invalid source path: {audio_path}") from e

    # Create a safe destination filename
    # Use only the stem (filename without extension) to avoid extension-based attacks
    safe_name = audio_path.stem
    # Sanitize the filename to remove dangerous characters
    import re

    safe_name = re.sub(r"[^\w\-_.]", "_", safe_name)
    # Ensure the name is not empty after sanitization
    if not safe_name:
        safe_name = "audio_file"

    # Reconstruct the filename with the original extension
    safe_filename = f"{safe_name}{audio_path.suffix}"
    raw_dest = paths.raw_dir / safe_filename

    # Validate the destination path is within the expected directory
    try:
        dest_str = str(raw_dest.resolve())
        if not dest_str.startswith(str(paths.raw_dir.resolve())):
            raise TranscriptionError(f"Destination path outside allowed directory: {raw_dest}")
    except (OSError, ValueError) as e:
        raise TranscriptionError(f"Invalid destination path: {raw_dest}") from e

    raw_dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        same_file = raw_dest.resolve() == audio_path.resolve()
    except OSError:
        same_file = False

    if not same_file:
        # Always refresh the raw copy so repeated calls with the same filename
        # pick up updated audio instead of reusing a stale file.
        shutil.copy2(audio_path, raw_dest)

    # Normalize to input_audio
    from .audio_io import normalize_all

    normalize_all(paths)

    # Find the normalized file
    stem = audio_path.stem
    norm_wav = paths.norm_dir / f"{stem}.wav"

    if not norm_wav.exists():
        raise TranscriptionError(
            f"Audio normalization failed: {norm_wav} not found. "
            f"Ensure ffmpeg is installed and the input audio format is supported."
        )

    duration_sec = _get_wav_duration_seconds(norm_wav)

    # Transcribe
    engine = TranscriptionEngine(asr_cfg)
    transcript = engine.transcribe_file(norm_wav)

    # v1.1: Run diarization if enabled (skeleton for now)
    transcript = _maybe_run_diarization(
        transcript=transcript,
        wav_path=norm_wav,
        config=config,
    )
    transcript = _maybe_build_chunks(transcript, config)

    # Write outputs
    from . import writers

    json_path = paths.json_dir / f"{stem}.json"
    txt_path = paths.transcripts_dir / f"{stem}.txt"
    srt_path = paths.transcripts_dir / f"{stem}.srt"

    # Build metadata
    from . import __version__

    engine_cfg = getattr(engine, "cfg", None)
    engine_device = getattr(engine_cfg, "device", None) if engine_cfg else None
    engine_compute_type = getattr(engine_cfg, "compute_type", None) if engine_cfg else None

    transcript.meta = build_generation_metadata(
        transcript,
        duration_sec=duration_sec,
        model_name=config.model,
        config_device=config.device,
        config_compute_type=config.compute_type,
        beam_size=config.beam_size,
        vad_min_silence_ms=config.vad_min_silence_ms,
        language_hint=config.language,
        task=config.task,
        pipeline_version=__version__,
        root=root,
        runtime_device_candidates=(engine_device,),
        runtime_compute_candidates=(engine_compute_type,),
    )

    writers.write_json(transcript, json_path)
    writers.write_txt(transcript, txt_path)
    writers.write_srt(transcript, srt_path)

    return transcript


def enrich_directory(
    root: str | Path,
    config: EnrichmentConfig,
) -> list[Transcript]:
    """
    Enrich all transcripts under whisper_json/ with audio-derived features.

    Reads transcripts from root/whisper_json/ and corresponding audio from
    root/input_audio/, enriches them with prosody/emotion features, and
    writes updated transcripts back to whisper_json/.

    Args:
        root: Root directory
        config: Enrichment configuration

    Returns:
        List of enriched Transcript objects

    Example:
        >>> config = EnrichmentConfig(enable_prosody=True, enable_emotion=True)
        >>> enriched = enrich_directory("/data/project", config)
        >>> print(enriched[0].segments[0].audio_state["rendering"])
    """
    root = Path(root)
    paths = Paths(root=root)

    json_dir = paths.json_dir
    audio_dir = paths.norm_dir

    if not json_dir.exists():
        raise EnrichmentError(
            f"JSON directory does not exist: {json_dir}. "
            f"Run transcription first using transcribe_directory() or ensure the project structure is correct."
        )
    if not audio_dir.exists():
        raise EnrichmentError(
            f"Audio directory does not exist: {audio_dir}. "
            f"Run transcription first to generate normalized audio files in {audio_dir}."
        )

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise EnrichmentError(
            f"No JSON transcript files found in {json_dir}. "
            f"Run transcription first using transcribe_directory() to generate transcript files."
        )

    enriched_transcripts = []
    errors = []
    total = len(json_files)

    for idx, json_path in enumerate(json_files, start=1):
        logger.info("[%d/%d] %s", idx, total, json_path.name)
        stem = json_path.stem
        wav_path = audio_dir / f"{stem}.wav"

        if not wav_path.exists():
            errors.append(f"{json_path.name}: Audio file not found at {wav_path}")
            # Skip gracefully without failing the entire directory
            continue

        # Load transcript
        try:
            transcript = load_transcript_from_json(json_path)
        except Exception as e:
            errors.append(f"{json_path.name}: Failed to load transcript - {e}")
            continue

        audio_ready = bool(transcript.segments) and all(
            seg.audio_state is not None for seg in transcript.segments
        )
        turns_ready = _turns_have_metadata(transcript.turns)
        stats_ready = bool(transcript.speaker_stats)
        semantic_ready = not getattr(config, "enable_semantic_annotator", False) or bool(
            (getattr(transcript, "annotations", None) or {}).get("semantic")
        )
        already_enriched = (
            audio_ready
            and (not config.enable_turn_metadata or turns_ready)
            and (not config.enable_speaker_stats or stats_ready)
            and semantic_ready
        )

        # Skip if already enriched and skip_existing=True
        if config.skip_existing:
            if already_enriched:
                enriched_transcripts.append(transcript)
                continue

            if audio_ready:
                # Use existing audio_state but upgrade analytics
                transcript = _run_speaker_analytics(transcript, config)
                transcript = _run_semantic_annotator(transcript, config)
                write_json(transcript, json_path)
                enriched_transcripts.append(transcript)
                continue

        # Enrich (lazy import to avoid requiring enrichment dependencies)
        try:
            from .audio_enrichment import enrich_transcript_audio as _enrich_transcript_internal

            enriched = _enrich_transcript_internal(
                transcript=transcript,
                wav_path=wav_path,
                enable_prosody=config.enable_prosody,
                enable_emotion=config.enable_emotion,
                enable_categorical_emotion=config.enable_categorical_emotion,
                compute_baseline=True,
            )
            enriched = _run_speaker_analytics(enriched, config)
            enriched = _run_semantic_annotator(enriched, config)

            # Write back
            write_json(enriched, json_path)
            enriched_transcripts.append(enriched)

        except ImportError as e:
            raise EnrichmentError(
                f"Missing required dependencies for audio enrichment. "
                f'Install with: uv sync --extra full (repo) or pip install "slower-whisper[full]". '
                f"Error: {e}"
            ) from e
        except Exception as e:
            errors.append(f"{json_path.name}: Enrichment failed - {e}")
            logger.warning(f"Enrichment failed for {json_path.name}: {e}", exc_info=True)
            # Skip this transcript but continue processing others

    if errors and not enriched_transcripts:
        raise EnrichmentError(
            "Failed to enrich any transcripts. Errors encountered:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )

    return enriched_transcripts


def enrich_transcript(
    transcript: Transcript,
    audio_path: str | Path,
    config: EnrichmentConfig,
) -> Transcript:
    """
    Enrich a single Transcript with audio-derived features.

    This is a pure function that does not write to disk. Use with
    load_transcript() and save_transcript() for file-based workflows.

    Args:
        transcript: Transcript object to enrich
        audio_path: Path to corresponding audio file (WAV)
        config: Enrichment configuration

    Returns:
        Enriched Transcript object (new instance)

    Raises:
        EnrichmentError: If audio file not found or enrichment dependencies missing
        EnrichmentError: If enrichment processing fails

    Example:
        >>> from transcription import load_transcript, save_transcript
        >>>
        >>> transcript = load_transcript("transcript.json")
        >>> config = EnrichmentConfig(enable_prosody=True)
        >>> enriched = enrich_transcript(transcript, "audio.wav", config)
        >>> save_transcript(enriched, "enriched.json")
    """
    audio_path = Path(audio_path)

    audio_ready = bool(transcript.segments) and all(
        seg.audio_state is not None for seg in transcript.segments
    )
    turns_ready = _turns_have_metadata(transcript.turns)
    stats_ready = bool(transcript.speaker_stats)
    semantic_ready = not getattr(config, "enable_semantic_annotator", False) or bool(
        (getattr(transcript, "annotations", None) or {}).get("semantic")
    )

    already_enriched = (
        audio_ready
        and (not config.enable_turn_metadata or turns_ready)
        and (not config.enable_speaker_stats or stats_ready)
        and semantic_ready
    )

    if config.skip_existing and already_enriched:
        return transcript

    if not audio_path.exists():
        raise EnrichmentError(
            f"Audio file not found: {audio_path}. "
            f"Ensure the audio file exists and the path is correct."
        )

    try:
        from .audio_enrichment import enrich_transcript_audio as _enrich_transcript_internal
    except ImportError as e:
        raise EnrichmentError(
            f"Missing required dependencies for audio enrichment. "
            f'Install with: uv sync --extra full (repo) or pip install "slower-whisper[full]". '
            f"Error: {e}"
        ) from e

    try:
        enriched = _enrich_transcript_internal(
            transcript=transcript,
            wav_path=audio_path,
            enable_prosody=config.enable_prosody,
            enable_emotion=config.enable_emotion,
            enable_categorical_emotion=config.enable_categorical_emotion,
            compute_baseline=True,
        )
        enriched = _run_speaker_analytics(enriched, config)
        enriched = _run_semantic_annotator(enriched, config)
        return enriched
    except Exception as e:
        logger.warning(
            f"Enrichment failed for {audio_path.name}: {e}; returning neutral audio_state",
            exc_info=True,
        )
        for segment in transcript.segments:
            segment.audio_state = _neutral_audio_state(str(e))
        return _run_semantic_annotator(_run_speaker_analytics(transcript, config), config)


# Convenience I/O wrappers for public API


def load_transcript(json_path: str | Path, *, strict: bool = False) -> Transcript:
    """
    Load a Transcript from a JSON file.

    Args:
        json_path: Path to JSON transcript file
        strict: If True, validate against JSON schema before loading.
                Raises TranscriptionError if validation fails.
                Default is False for backward compatibility.

    Returns:
        Transcript object

    Raises:
        TranscriptionError: If file not found, JSON is invalid, or
                           validation fails (when strict=True)

    Example:
        >>> transcript = load_transcript("output.json")
        >>> print(transcript.file_name, transcript.language)

        >>> # Strict mode validates against schema
        >>> transcript = load_transcript("output.json", strict=True)
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise TranscriptionError(
            f"Transcript file not found: {json_path}. "
            f"Ensure the file path is correct and the file exists."
        )

    # Validate against schema if strict mode is enabled
    if strict:
        from .validation import validate_transcript_json

        is_valid, errors = validate_transcript_json(json_path)
        if not is_valid:
            error_summary = "; ".join(errors[:3])  # Show first 3 errors
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
    """
    Save a Transcript to a JSON file.

    Args:
        transcript: Transcript object to save
        json_path: Output path

    Raises:
        TranscriptionError: If file cannot be written

    Example:
        >>> save_transcript(transcript, "output.json")
    """
    json_path = Path(json_path)

    # Ensure parent directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        write_json(transcript, json_path)
    except Exception as e:
        raise TranscriptionError(
            f"Failed to save transcript to {json_path}: {e}. "
            f"Check that you have write permissions and sufficient disk space."
        ) from e
