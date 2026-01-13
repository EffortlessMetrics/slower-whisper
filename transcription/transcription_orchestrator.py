"""
Transcription orchestration extracted from transcription.api.

These are the "real" implementations. transcription.api stays as a thin façade so
tests (and downstream users) can continue patching transcription.api.* helpers.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from .config import AsrConfig, Paths, TranscriptionConfig, validate_diarization_settings
from .exceptions import TranscriptionError
from .meta_utils import build_generation_metadata
from .models import Transcript
from .writers import load_transcript_from_json

logger = logging.getLogger(__name__)

GetWavDurationFn = Callable[[Path], float]
MaybeRunDiarizationFn = Callable[[Transcript, Path, TranscriptionConfig], Transcript]
MaybeBuildChunksFn = Callable[[Transcript, TranscriptionConfig], Transcript]


def _transcribe_directory_impl(
    root: str | Path,
    config: TranscriptionConfig,
) -> list[Transcript]:
    from .config import AppConfig
    from .pipeline import run_pipeline

    root = Path(root)
    validate_diarization_settings(
        config.min_speakers,
        config.max_speakers,
        config.overlap_threshold,
    )

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

    result = run_pipeline(app_cfg, diarization_config=config)
    logger.info(
        "Pipeline completed: %d processed, %d skipped, %d failed",
        result.processed,
        result.skipped,
        result.failed,
    )

    json_dir = paths.json_dir
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise TranscriptionError(
            f"No transcripts found in {json_dir}. "
            f"Ensure audio files exist in {paths.raw_dir} and transcription completed successfully."
        )

    transcripts: list[Transcript] = []
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


def _transcribe_file_impl(
    audio_path: str | Path,
    root: str | Path,
    config: TranscriptionConfig,
    *,
    get_wav_duration_seconds: GetWavDurationFn,
    maybe_run_diarization: MaybeRunDiarizationFn,
    maybe_build_chunks: MaybeBuildChunksFn,
) -> Transcript:
    from .asr_engine import TranscriptionEngine
    from .audio_io import ensure_dirs, ensure_within_dir, normalize_all, sanitize_filename

    audio_path = Path(audio_path)
    root = Path(root)

    validate_diarization_settings(
        config.min_speakers,
        config.max_speakers,
        config.overlap_threshold,
    )

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

    paths = Paths(root=root)
    ensure_dirs(paths)

    safe_stem, safe_suffix = sanitize_filename(audio_path.stem, audio_path.suffix)
    raw_dest = paths.raw_dir / f"{safe_stem}{safe_suffix}"

    try:
        ensure_within_dir(raw_dest, paths.raw_dir)
    except ValueError as e:
        raise TranscriptionError(str(e)) from e

    raw_dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        same_file = raw_dest.resolve() == audio_path.resolve()
    except OSError:
        same_file = False

    if not same_file:
        shutil.copy2(audio_path, raw_dest)

    normalize_all(paths)

    stem = raw_dest.stem
    norm_wav = paths.norm_dir / f"{stem}.wav"

    if not norm_wav.exists():
        raise TranscriptionError(
            f"Audio normalization failed: {norm_wav} not found. "
            f"Ensure ffmpeg is installed and the input audio format is supported."
        )

    duration_sec = get_wav_duration_seconds(norm_wav)

    engine = TranscriptionEngine(asr_cfg)
    transcript = engine.transcribe_file(norm_wav)

    transcript = maybe_run_diarization(
        transcript,
        norm_wav,
        config,
    )
    transcript = maybe_build_chunks(transcript, config)

    from . import __version__, writers

    json_path = paths.json_dir / f"{stem}.json"
    txt_path = paths.transcripts_dir / f"{stem}.txt"
    srt_path = paths.transcripts_dir / f"{stem}.srt"

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


def _transcribe_bytes_impl(
    audio_data: bytes,
    config: TranscriptionConfig,
    format: str = "wav",
    *,
    get_wav_duration_seconds: GetWavDurationFn,
    maybe_run_diarization: MaybeRunDiarizationFn,
    maybe_build_chunks: MaybeBuildChunksFn,
) -> Transcript:
    import re
    import tempfile

    from .asr_engine import TranscriptionEngine

    if not audio_data:
        raise TranscriptionError(
            "Audio data is empty. Provide non-empty audio bytes for transcription."
        )

    format_lower = format.lower().lstrip(".")
    if not re.fullmatch(r"[a-z0-9][a-z0-9+.-]{0,15}", format_lower):
        raise TranscriptionError(
            f"Invalid audio format hint: '{format}'. "
            f"Format must be a simple extension like 'wav', 'mp3', or 'flac'."
        )

    validate_diarization_settings(
        config.min_speakers,
        config.max_speakers,
        config.overlap_threshold,
    )

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

    temp_file = None
    norm_temp = None
    try:
        temp_file = tempfile.NamedTemporaryFile(
            suffix=f".{format_lower}",
            delete=False,
        )
        temp_path = Path(temp_file.name)
        temp_file.write(audio_data)
        temp_file.close()

        norm_temp = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        )
        norm_path = Path(norm_temp.name)
        norm_temp.close()

        from .audio_io import normalize_single

        try:
            normalize_single(temp_path, norm_path)
        except Exception as e:
            raise TranscriptionError(
                f"Failed to normalize audio: {e}. "
                f"Ensure ffmpeg is installed and the audio data is valid {format_lower.upper()} format."
            ) from e

        if not norm_path.exists():
            raise TranscriptionError(
                "Audio normalization failed: normalized file not created. "
                "Ensure ffmpeg is installed and the audio data is valid."
            )

        duration_sec = get_wav_duration_seconds(norm_path)

        engine = TranscriptionEngine(asr_cfg)
        transcript = engine.transcribe_file(norm_path)
        transcript.file_name = f"<bytes:{format_lower}>"

        transcript = maybe_run_diarization(
            transcript,
            norm_path,
            config,
        )
        transcript = maybe_build_chunks(transcript, config)

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
            root="<in-memory>",
            runtime_device_candidates=(engine_device,),
            runtime_compute_candidates=(engine_compute_type,),
        )

        return transcript

    finally:
        if temp_file is not None:
            try:
                Path(temp_file.name).unlink(missing_ok=True)
            except Exception as cleanup_exc:
                logger.debug("Failed to clean up temp file %s: %s", temp_file.name, cleanup_exc)
        if norm_temp is not None:
            try:
                Path(norm_temp.name).unlink(missing_ok=True)
            except Exception as cleanup_exc:
                logger.debug(
                    "Failed to clean up normalized temp file %s: %s",
                    norm_temp.name,
                    cleanup_exc,
                )
