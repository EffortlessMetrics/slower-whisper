"""Stage 1: ASR transcription pipeline.

This module implements the core transcription pipeline (run_pipeline) that:
1. Normalizes audio files to 16kHz mono WAV using ffmpeg
2. Transcribes audio using faster-whisper
3. Writes output to JSON, TXT, and SRT formats

The pipeline is the entry point for all transcription operations and is called
by both the CLI and public API. It handles batch processing, progress tracking,
and error recovery.
"""

import logging
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__ as PIPELINE_VERSION
from . import audio_io, writers
from .asr_engine import TranscriptionEngine
from .config import AppConfig, TranscriptionConfig
from .meta_utils import build_generation_metadata
from .models import Transcript

logger = logging.getLogger(__name__)


@dataclass
class PipelineFileResult:
    """Result of processing a single audio file in the pipeline.

    Tracks the outcome of normalizing, transcribing, and optionally
    diarizing a single audio file through the run_pipeline() function.
    """

    file_name: str
    status: str  # "success", "skipped", "diarized_only", "error"
    error_message: str | None = None


@dataclass
class PipelineBatchResult:
    """Result of batch pipeline processing in run_pipeline().

    Tracks success/failure statistics, timing metrics, and per-file results
    for structured error reporting and performance monitoring.

    This is distinct from BatchProcessingResult (used by the public API layer)
    as it includes pipeline-specific metrics like diarized_only count and
    real-time factor calculations.
    """

    total_files: int
    processed: int
    skipped: int
    diarized_only: int
    failed: int
    total_audio_seconds: float
    total_time_seconds: float
    file_results: list[PipelineFileResult] = field(default_factory=list)

    @property
    def overall_rtf(self) -> float:
        """Real-time factor (processing time / audio duration)."""
        if self.total_audio_seconds > 0:
            return self.total_time_seconds / self.total_audio_seconds
        return 0.0


def _get_duration_seconds(path: Path) -> float:
    """
    Return the duration of a WAV file in seconds.

    Assumes path points to a valid WAV file.
    """
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
        return frames / float(rate) if rate else 0.0
    except Exception as e:
        logger.warning("Could not read duration for %s: %s", path.name, e, exc_info=True)
        return 0.0


def _build_meta(
    cfg: AppConfig, transcript: Transcript, audio_path: Path, duration_sec: float
) -> dict[str, Any]:
    """
    Build a metadata dictionary describing this transcript generation run.
    """
    return build_generation_metadata(
        transcript,
        duration_sec=duration_sec,
        model_name=cfg.asr.model_name,
        config_device=cfg.asr.device,
        config_compute_type=cfg.asr.compute_type,
        beam_size=cfg.asr.beam_size,
        vad_min_silence_ms=cfg.asr.vad_min_silence_ms,
        language_hint=cfg.asr.language,
        task=cfg.asr.task,
        pipeline_version=PIPELINE_VERSION,
        root=cfg.paths.root,
    )


def run_pipeline(
    cfg: AppConfig, diarization_config: TranscriptionConfig | None = None
) -> PipelineBatchResult:
    """
    Orchestrate the full pipeline:
    1) Ensure directories.
    2) Normalize raw audio to 16 kHz mono WAV.
    3) Transcribe normalized audio with Whisper.
    4) (v1.1) Optionally run diarization if diarization_config.enable_diarization=True.
    5) Write JSON, TXT, and SRT outputs per file.

    If cfg.skip_existing_json is True, files that already have a JSON
    output will be skipped at the transcription step.

    Args:
        cfg: AppConfig (internal pipeline config).
        diarization_config: Optional TranscriptionConfig with diarization settings.
                           If provided and enable_diarization=True, diarization
                           will run on each transcript before writing outputs.

    Returns:
        PipelineBatchResult with success/failure statistics and per-file results.
    """
    paths = cfg.paths

    audio_io.ensure_dirs(paths)
    audio_io.normalize_all(paths)

    norm_files = sorted(paths.norm_dir.glob("*.wav"))
    if not norm_files:
        logger.warning("No .wav files found in input_audio/. Nothing to transcribe.")
        return PipelineBatchResult(
            total_files=0,
            processed=0,
            skipped=0,
            diarized_only=0,
            failed=0,
            total_audio_seconds=0.0,
            total_time_seconds=0.0,
        )

    engine = TranscriptionEngine(cfg.asr)

    logger.info("=== Step 3: Transcribing normalized audio ===")
    total = len(norm_files)
    processed = skipped = failed = 0
    diarized_only = 0
    total_audio = 0.0
    total_time = 0.0
    file_results: list[PipelineFileResult] = []

    for idx, wav in enumerate(norm_files, start=1):
        logger.info("[%d/%d] %s", idx, total, wav.name)
        stem = Path(wav.name).stem
        json_path = paths.json_dir / f"{stem}.json"
        txt_path = paths.transcripts_dir / f"{stem}.txt"
        srt_path = paths.transcripts_dir / f"{stem}.srt"

        if cfg.skip_existing_json and json_path.exists():
            # Check what operations we need to perform on existing files
            enable_chunking = diarization_config and getattr(
                diarization_config, "enable_chunking", False
            )
            enable_diarization = diarization_config and diarization_config.enable_diarization

            # If we need to process, we must load the transcript first.
            # Optimization: Load ONCE for both chunking and diarization checks.
            transcript = None
            load_error = None
            if enable_chunking or enable_diarization:
                try:
                    transcript = writers.load_transcript_from_json(json_path)
                except Exception as exc:
                    load_error = exc

            # 1. Update chunks if enabled
            if enable_chunking:
                if transcript:
                    try:
                        from .transcription_helpers import _maybe_build_chunks

                        transcript = _maybe_build_chunks(transcript, diarization_config)
                        writers.write_json(transcript, json_path)
                    except Exception as exc:
                        logger.error(
                            "Failed to update chunks for %s: %s",
                            json_path.name,
                            exc,
                            exc_info=True,
                        )
                elif load_error:
                    # Log error but continue (matches original behavior where chunking error isn't fatal)
                    logger.error(
                        "Failed to update chunks for %s (load failed): %s",
                        json_path.name,
                        load_error,
                        exc_info=True,
                    )

            # 2. Update diarization if enabled
            if enable_diarization:
                # Upgrade existing transcript with diarization without re-transcribing
                if not transcript:
                    # If we failed to load, fail the file (matches original behavior)
                    err = load_error or "Transcript not loaded"
                    logger.error("Failed to load %s: %s", json_path.name, err, exc_info=True)
                    failed += 1
                    file_results.append(
                        PipelineFileResult(
                            file_name=wav.name,
                            status="error",
                            error_message=f"Load failed: {err}",
                        )
                    )
                    continue

                diar_meta = (transcript.meta or {}).get("diarization", {})
                if diar_meta.get("status") in {"success", "ok"}:
                    logger.debug(
                        "[skip-transcribe] %s because %s already exists (diarization present)",
                        wav.name,
                        json_path.name,
                    )
                    skipped += 1
                    file_results.append(PipelineFileResult(file_name=wav.name, status="skipped"))
                    continue

                logger.info("[diarize-existing] %s (reusing existing transcript)", wav.name)
                try:
                    from .diarization_orchestrator import _maybe_run_diarization

                    transcript = _maybe_run_diarization(
                        transcript=transcript,
                        wav_path=wav,
                        config=diarization_config,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to run diarization for %s: %s",
                        wav.name,
                        exc,
                        exc_info=True,
                    )
                    failed += 1
                    file_results.append(
                        PipelineFileResult(
                            file_name=wav.name,
                            status="error",
                            error_message=f"Diarization failed: {exc}",
                        )
                    )
                    continue

                writers.write_json(transcript, json_path)
                writers.write_txt(transcript, txt_path)
                writers.write_srt(transcript, srt_path)

                diarized_only += 1
                logger.info("  → [diarization-only] %s", json_path)
                logger.info("  → [diarization-only] %s", txt_path)
                logger.info("  → [diarization-only] %s", srt_path)
                file_results.append(PipelineFileResult(file_name=wav.name, status="diarized_only"))
            else:
                logger.debug(
                    "[skip-transcribe] %s because %s already exists",
                    wav.name,
                    json_path.name,
                )
                skipped += 1
                file_results.append(PipelineFileResult(file_name=wav.name, status="skipped"))
            continue

        duration = _get_duration_seconds(wav)
        total_audio += duration

        start = time.time()
        try:
            transcript = engine.transcribe_file(wav)
        except Exception as e:
            logger.error("Failed to transcribe %s: %s", wav.name, e, exc_info=True)
            failed += 1
            file_results.append(
                PipelineFileResult(
                    file_name=wav.name,
                    status="error",
                    error_message=f"Transcription failed: {e}",
                )
            )
            continue
        elapsed = time.time() - start
        total_time += elapsed

        rtf = elapsed / duration if duration > 0 else 0.0
        logger.info(
            "  [stats] audio=%.1f min, wall=%.1fs, RTF=%.2fx",
            duration / 60,
            elapsed,
            rtf,
        )

        # Attach metadata to transcript before writing JSON.
        transcript.meta = _build_meta(cfg, transcript, wav, duration)

        # v1.1+: Run diarization (or record disabled state) if config provided
        if diarization_config:
            from .diarization_orchestrator import _maybe_run_diarization

            transcript = _maybe_run_diarization(
                transcript=transcript,
                wav_path=wav,
                config=diarization_config,
            )
            if getattr(diarization_config, "enable_chunking", False):
                from .transcription_helpers import _maybe_build_chunks

                transcript = _maybe_build_chunks(transcript, diarization_config)

        writers.write_json(transcript, json_path)
        writers.write_txt(transcript, txt_path)
        writers.write_srt(transcript, srt_path)

        logger.info("  → JSON: %s", json_path)
        logger.info("  → TXT:  %s", txt_path)
        logger.info("  → SRT:  %s", srt_path)

        processed += 1
        file_results.append(PipelineFileResult(file_name=wav.name, status="success"))

    logger.info("=== Summary ===")
    logger.info(
        "  transcribed=%d, diarized_only=%d, skipped=%d, failed=%d, total=%d",
        processed,
        diarized_only,
        skipped,
        failed,
        total,
    )
    if total_audio > 0 and total_time > 0:
        overall_rtf = total_time / total_audio
        logger.info(
            "  audio=%.1f min, wall=%.1f min, RTF=%.2fx",
            total_audio / 60,
            total_time / 60,
            overall_rtf,
        )
    logger.info("All done.")

    return PipelineBatchResult(
        total_files=total,
        processed=processed,
        skipped=skipped,
        diarized_only=diarized_only,
        failed=failed,
        total_audio_seconds=total_audio,
        total_time_seconds=total_time,
        file_results=file_results,
    )
