"""
Unified CLI for slower-whisper.

Provides a single entry point with two subcommands:
- slower-whisper transcribe: Stage 1 transcription
- slower-whisper enrich: Stage 2 audio enrichment

This replaces the separate slower-whisper and slower-whisper-enrich commands
with a more coherent interface.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from . import api as api_module
from .config import EnrichmentConfig, TranscriptionConfig, validate_diarization_settings
from .exceptions import ConfigurationError, SlowerWhisperError
from .exporters import SUPPORTED_EXPORT_FORMATS, export_transcript
from .models import Transcript
from .validation import DEFAULT_SCHEMA_PATH, validate_many


# Expose API functions for compatibility with tests/patching while delegating to api module
def transcribe_directory(*args, **kwargs) -> list[Transcript]:
    return api_module.transcribe_directory(*args, **kwargs)


def enrich_directory(*args, **kwargs) -> list[Transcript]:
    return api_module.enrich_directory(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="slower-whisper",
        description="Local transcription and audio enrichment pipeline.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ============================================================================
    # transcribe subcommand
    # ============================================================================
    p_trans = subparsers.add_parser(
        "transcribe",
        help="Transcribe audio under a project root (Stage 1).",
    )

    p_trans.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root (contains raw_audio/, input_audio/, whisper_json/, transcripts/).",
    )
    p_trans.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to TranscriptionConfig JSON file. Precedence: CLI flags > config file > env vars > defaults.",
    )
    p_trans.add_argument(
        "--model",
        default=None,
        help="Whisper model name (default: large-v3).",
    )
    p_trans.add_argument(
        "--device",
        default=None,
        help="Device: cuda or cpu (default: cuda).",
    )
    p_trans.add_argument(
        "--compute-type",
        default=None,
        help="faster-whisper compute type (e.g. float16, int8).",
    )
    p_trans.add_argument(
        "--language",
        default=None,
        help="Force language (e.g. en). Leave empty for auto-detect.",
    )
    p_trans.add_argument(
        "--task",
        default=None,
        choices=["transcribe", "translate"],
        help="Whisper task (default: transcribe).",
    )
    p_trans.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=None,
        help="Minimum silence duration in ms to split segments (default: 500).",
    )
    p_trans.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size for decoding (default: 5).",
    )
    p_trans.add_argument(
        "--skip-existing-json",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip files with existing JSON in whisper_json/ (default: True).",
    )
    p_trans.add_argument(
        "--enable-chunking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Emit turn-aware chunks for RAG/export (default: False).",
    )
    p_trans.add_argument(
        "--chunk-target-duration-s",
        type=float,
        default=None,
        help="Soft target chunk duration in seconds (default: 30).",
    )
    p_trans.add_argument(
        "--chunk-max-duration-s",
        type=float,
        default=None,
        help="Hard max chunk duration in seconds (default: 45).",
    )
    p_trans.add_argument(
        "--chunk-target-tokens",
        type=int,
        default=None,
        help="Approximate max tokens per chunk before splitting (default: 400).",
    )
    p_trans.add_argument(
        "--chunk-pause-split-threshold-s",
        type=float,
        default=None,
        help="Split on pauses >= this length when near target size (default: 1.5).",
    )
    p_trans.add_argument(
        "--enable-diarization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable speaker diarization (experimental, default: False). See docs/SPEAKER_DIARIZATION.md for setup.",
    )
    p_trans.add_argument(
        "--diarization-device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Device for diarization ('auto', 'cuda', or 'cpu'). Defaults to 'auto'.",
    )
    p_trans.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers expected (diarization hint, optional).",
    )
    p_trans.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers expected (diarization hint, optional).",
    )
    p_trans.add_argument(
        "--overlap-threshold",
        type=float,
        default=None,
        help="Minimum overlap ratio (0.0-1.0) required to assign a speaker to a segment (default: 0.3).",
    )

    # ============================================================================
    # enrich subcommand
    # ============================================================================
    p_enrich = subparsers.add_parser(
        "enrich",
        help="Enrich existing transcripts with audio-derived features (Stage 2).",
    )

    p_enrich.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root.",
    )
    p_enrich.add_argument(
        "--enrich-config",
        type=Path,
        default=None,
        help="Path to EnrichmentConfig JSON file. Precedence: CLI flags > config file > env vars > defaults.",
    )
    p_enrich.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip segments that already have audio_state (default: True).",
    )
    p_enrich.add_argument(
        "--enable-prosody",
        dest="enable_prosody",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable prosody extraction (default: True).",
    )
    p_enrich.add_argument(
        "--enable-emotion",
        dest="enable_emotion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable dimensional emotion extraction (default: True).",
    )
    p_enrich.add_argument(
        "--enable-categorical-emotion",
        dest="enable_categorical_emotion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable categorical emotion (slower, default: False).",
    )
    p_enrich.add_argument(
        "--enable-turn-metadata",
        dest="enable_turn_metadata",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Populate turn metadata (questions, pauses, disfluency) (default: True).",
    )
    p_enrich.add_argument(
        "--enable-speaker-stats",
        dest="enable_speaker_stats",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute per-speaker aggregates (default: True).",
    )
    p_enrich.add_argument(
        "--enable-semantics",
        dest="enable_semantic_annotator",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run keyword-based semantic annotation, writing to annotations.semantic (default: False).",
    )
    p_enrich.add_argument(
        "--enable-speaker-analytics",
        dest="enable_speaker_analytics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Convenience flag to enable/disable both turn metadata and speaker stats together.",
    )
    p_enrich.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device to run emotion models on (default: cpu).",
    )

    # ============================================================================
    # cache subcommand
    # ============================================================================
    p_cache = subparsers.add_parser(
        "cache",
        help="Inspect or clear model caches.",
    )

    cache_group = p_cache.add_mutually_exclusive_group(required=True)
    cache_group.add_argument(
        "--show",
        action="store_true",
        help="Show cache locations and sizes.",
    )
    cache_group.add_argument(
        "--clear",
        choices=["all", "whisper", "emotion", "diarization", "hf", "torch", "samples"],
        help="Clear selected cache.",
    )

    # ============================================================================
    # samples subcommand
    # ============================================================================
    p_samples = subparsers.add_parser(
        "samples",
        help="Manage sample datasets for testing and evaluation.",
    )

    samples_subparsers = p_samples.add_subparsers(dest="samples_action", required=True)

    # samples list
    samples_subparsers.add_parser(
        "list",
        help="List available sample datasets.",
    )

    # samples download
    p_samples_download = samples_subparsers.add_parser(
        "download",
        help="Download a sample dataset to cache.",
    )
    p_samples_download.add_argument(
        "dataset",
        help="Dataset name to download (e.g., 'mini_diarization').",
    )
    p_samples_download.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if already cached.",
    )

    # samples copy
    p_samples_copy = samples_subparsers.add_parser(
        "copy",
        help="Copy sample dataset to project's raw_audio directory.",
    )
    p_samples_copy.add_argument(
        "dataset",
        help="Dataset name to copy (e.g., 'mini_diarization').",
    )
    p_samples_copy.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory).",
    )

    # samples generate
    p_samples_generate = samples_subparsers.add_parser(
        "generate",
        help="Generate synthetic sample audio for testing.",
    )
    p_samples_generate.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: ./raw_audio/).",
    )
    p_samples_generate.add_argument(
        "--speakers",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of speakers to generate (default: 2).",
    )

    # ============================================================================
    # export subcommand
    # ============================================================================
    p_export = subparsers.add_parser(
        "export",
        help="Export a transcript JSON into CSV/HTML/VTT/TextGrid.",
    )
    p_export.add_argument(
        "transcript",
        type=Path,
        help="Path to transcript JSON.",
    )
    p_export.add_argument(
        "--format",
        choices=sorted(SUPPORTED_EXPORT_FORMATS),
        required=True,
        help="Target export format.",
    )
    p_export.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: derive from transcript name).",
    )
    p_export.add_argument(
        "--unit",
        choices=["segments", "turns"],
        default="segments",
        help="Row unit to export (default: segments).",
    )

    # ============================================================================
    # validate subcommand
    # ============================================================================
    p_validate = subparsers.add_parser(
        "validate",
        help="Validate transcript JSON against the v2 schema.",
    )
    p_validate.add_argument(
        "transcripts",
        nargs="+",
        type=Path,
        help="Transcript JSON files to validate.",
    )
    p_validate.add_argument(
        "--schema",
        type=Path,
        default=None,
        help=f"Override schema path (default: {DEFAULT_SCHEMA_PATH})",
    )

    return parser


def _merge_configs(base: TranscriptionConfig, override: TranscriptionConfig) -> TranscriptionConfig:
    """
    Merge two TranscriptionConfig instances, with override taking precedence.

    This function implements proper config layering by checking which fields
    were explicitly set in the override config (via _source_fields attribute).
    Only explicitly-set fields override the base config, preventing the bug where
    a file value that equals the default would be ignored.

    The merge strategy:
    - If field was explicitly set in override (in _source_fields): use override value
    - Otherwise: keep base value

    This correctly handles the case where:
    1. Env sets device="cpu"
    2. File explicitly sets device="cuda" (which happens to equal the default)
    3. Result is "cuda" (file overrides env) ✓

    And also handles:
    1. Env sets device="cpu"
    2. File doesn't mention device (so device="cuda" is just a default)
    3. Result is "cpu" (env value preserved) ✓

    Args:
        base: Base configuration (lower precedence).
        override: Override configuration (higher precedence).

    Returns:
        New TranscriptionConfig with merged values.
    """
    # Check if override has _source_fields (set by from_file/from_env)
    # If not, or if it's empty, fall back to comparing values (for backward compatibility)
    source_fields = getattr(override, "_source_fields", None)

    if source_fields is None or not source_fields:
        # No source tracking - fall back to value comparison
        # This shouldn't happen in normal CLI usage, but provides safety
        return TranscriptionConfig(
            model=override.model if override.model != base.model else base.model,
            device=override.device if override.device != base.device else base.device,
            compute_type=override.compute_type
            if override.compute_type != base.compute_type
            else base.compute_type,
            language=override.language if override.language != base.language else base.language,
            task=override.task if override.task != base.task else base.task,
            vad_min_silence_ms=override.vad_min_silence_ms
            if override.vad_min_silence_ms != base.vad_min_silence_ms
            else base.vad_min_silence_ms,
            beam_size=override.beam_size
            if override.beam_size != base.beam_size
            else base.beam_size,
            skip_existing_json=override.skip_existing_json
            if override.skip_existing_json != base.skip_existing_json
            else base.skip_existing_json,
            enable_chunking=override.enable_chunking
            if override.enable_chunking != base.enable_chunking
            else base.enable_chunking,
            chunk_target_duration_s=override.chunk_target_duration_s
            if override.chunk_target_duration_s != base.chunk_target_duration_s
            else base.chunk_target_duration_s,
            chunk_max_duration_s=override.chunk_max_duration_s
            if override.chunk_max_duration_s != base.chunk_max_duration_s
            else base.chunk_max_duration_s,
            chunk_target_tokens=override.chunk_target_tokens
            if override.chunk_target_tokens != base.chunk_target_tokens
            else base.chunk_target_tokens,
            chunk_pause_split_threshold_s=override.chunk_pause_split_threshold_s
            if override.chunk_pause_split_threshold_s != base.chunk_pause_split_threshold_s
            else base.chunk_pause_split_threshold_s,
            enable_diarization=override.enable_diarization
            if override.enable_diarization != base.enable_diarization
            else base.enable_diarization,
            diarization_device=override.diarization_device
            if override.diarization_device != base.diarization_device
            else base.diarization_device,
            min_speakers=override.min_speakers
            if override.min_speakers != base.min_speakers
            else base.min_speakers,
            max_speakers=override.max_speakers
            if override.max_speakers != base.max_speakers
            else base.max_speakers,
            overlap_threshold=override.overlap_threshold
            if override.overlap_threshold != base.overlap_threshold
            else base.overlap_threshold,
        )

    # Use source_fields to determine which values to override
    return TranscriptionConfig(
        model=override.model if "model" in source_fields else base.model,
        device=override.device if "device" in source_fields else base.device,
        compute_type=override.compute_type
        if "compute_type" in source_fields
        else base.compute_type,
        language=override.language if "language" in source_fields else base.language,
        task=override.task if "task" in source_fields else base.task,
        vad_min_silence_ms=override.vad_min_silence_ms
        if "vad_min_silence_ms" in source_fields
        else base.vad_min_silence_ms,
        beam_size=override.beam_size if "beam_size" in source_fields else base.beam_size,
        skip_existing_json=override.skip_existing_json
        if "skip_existing_json" in source_fields
        else base.skip_existing_json,
        enable_chunking=override.enable_chunking
        if "enable_chunking" in source_fields
        else base.enable_chunking,
        chunk_target_duration_s=override.chunk_target_duration_s
        if "chunk_target_duration_s" in source_fields
        else base.chunk_target_duration_s,
        chunk_max_duration_s=override.chunk_max_duration_s
        if "chunk_max_duration_s" in source_fields
        else base.chunk_max_duration_s,
        chunk_target_tokens=override.chunk_target_tokens
        if "chunk_target_tokens" in source_fields
        else base.chunk_target_tokens,
        chunk_pause_split_threshold_s=override.chunk_pause_split_threshold_s
        if "chunk_pause_split_threshold_s" in source_fields
        else base.chunk_pause_split_threshold_s,
        enable_diarization=override.enable_diarization
        if "enable_diarization" in source_fields
        else base.enable_diarization,
        diarization_device=override.diarization_device
        if "diarization_device" in source_fields
        else base.diarization_device,
        min_speakers=override.min_speakers
        if "min_speakers" in source_fields
        else base.min_speakers,
        max_speakers=override.max_speakers
        if "max_speakers" in source_fields
        else base.max_speakers,
        overlap_threshold=override.overlap_threshold
        if "overlap_threshold" in source_fields
        else base.overlap_threshold,
    )


def _merge_enrich_configs(base: EnrichmentConfig, override: EnrichmentConfig) -> EnrichmentConfig:
    """
    Merge two EnrichmentConfig instances, with override taking precedence.

    This function implements proper config layering by checking which fields
    were explicitly set in the override config (via _source_fields attribute).
    Only explicitly-set fields override the base config.

    See _merge_configs() for detailed explanation of the merge strategy.

    Args:
        base: Base configuration (lower precedence).
        override: Override configuration (higher precedence).

    Returns:
        New EnrichmentConfig with merged values.
    """
    # Check if override has _source_fields (set by from_file/from_env)
    # If not, or if it's empty, fall back to comparing values (for backward compatibility)
    source_fields = getattr(override, "_source_fields", None)

    if source_fields is None or not source_fields:
        # No source tracking - fall back to value comparison
        return EnrichmentConfig(
            skip_existing=override.skip_existing
            if override.skip_existing != base.skip_existing
            else base.skip_existing,
            enable_prosody=override.enable_prosody
            if override.enable_prosody != base.enable_prosody
            else base.enable_prosody,
            enable_emotion=override.enable_emotion
            if override.enable_emotion != base.enable_emotion
            else base.enable_emotion,
            enable_categorical_emotion=override.enable_categorical_emotion
            if override.enable_categorical_emotion != base.enable_categorical_emotion
            else base.enable_categorical_emotion,
            enable_semantic_annotator=override.enable_semantic_annotator
            if override.enable_semantic_annotator != base.enable_semantic_annotator
            else base.enable_semantic_annotator,
            enable_turn_metadata=override.enable_turn_metadata
            if override.enable_turn_metadata != base.enable_turn_metadata
            else base.enable_turn_metadata,
            enable_speaker_stats=override.enable_speaker_stats
            if override.enable_speaker_stats != base.enable_speaker_stats
            else base.enable_speaker_stats,
            device=override.device if override.device != base.device else base.device,
            dimensional_model_name=override.dimensional_model_name
            if override.dimensional_model_name != base.dimensional_model_name
            else base.dimensional_model_name,
            categorical_model_name=override.categorical_model_name
            if override.categorical_model_name != base.categorical_model_name
            else base.categorical_model_name,
            semantic_annotator=override.semantic_annotator or base.semantic_annotator,
        )

    # Use source_fields to determine which values to override
    return EnrichmentConfig(
        skip_existing=override.skip_existing
        if "skip_existing" in source_fields
        else base.skip_existing,
        enable_prosody=override.enable_prosody
        if "enable_prosody" in source_fields
        else base.enable_prosody,
        enable_emotion=override.enable_emotion
        if "enable_emotion" in source_fields
        else base.enable_emotion,
        enable_categorical_emotion=override.enable_categorical_emotion
        if "enable_categorical_emotion" in source_fields
        else base.enable_categorical_emotion,
        enable_semantic_annotator=override.enable_semantic_annotator
        if "enable_semantic_annotator" in source_fields
        else base.enable_semantic_annotator,
        enable_turn_metadata=override.enable_turn_metadata
        if "enable_turn_metadata" in source_fields
        else base.enable_turn_metadata,
        enable_speaker_stats=override.enable_speaker_stats
        if "enable_speaker_stats" in source_fields
        else base.enable_speaker_stats,
        device=override.device if "device" in source_fields else base.device,
        dimensional_model_name=override.dimensional_model_name
        if "dimensional_model_name" in source_fields
        else base.dimensional_model_name,
        categorical_model_name=override.categorical_model_name
        if "categorical_model_name" in source_fields
        else base.categorical_model_name,
        semantic_annotator=override.semantic_annotator or base.semantic_annotator,
    )


def _config_from_transcribe_args(args: argparse.Namespace) -> TranscriptionConfig:
    """
    Build TranscriptionConfig from CLI arguments with proper precedence.

    Precedence order (highest to lowest):
    1. Explicit CLI flags
    2. Config file (if --config provided)
    3. Environment variables
    4. Defaults

    Args:
        args: Parsed command-line arguments from argparse.

    Returns:
        TranscriptionConfig with merged settings from all sources.
    """
    # Step 1: Start with defaults
    config = TranscriptionConfig()

    # Step 2: Override with environment variables
    env_config = TranscriptionConfig.from_env()
    env_compute_type_explicit = "compute_type" in getattr(env_config, "_source_fields", set())
    config = _merge_configs(config, env_config)

    # Step 3: Override with config file if provided
    file_compute_type_explicit = False
    if args.config is not None:
        file_config = TranscriptionConfig.from_file(args.config)
        file_compute_type_explicit = "compute_type" in getattr(
            file_config,
            "_source_fields",
            set(),
        )
        config = _merge_configs(config, file_config)

    # Step 4: Override with explicit CLI flags (only if not None)
    # Older tests/builders may not populate the newer chunking fields, so use
    # getattr with a default to keep backwards compatibility.
    enable_chunking = getattr(args, "enable_chunking", None)
    chunk_target_duration_s = getattr(args, "chunk_target_duration_s", None)
    chunk_max_duration_s = getattr(args, "chunk_max_duration_s", None)
    chunk_target_tokens = getattr(args, "chunk_target_tokens", None)
    chunk_pause_split_threshold_s = getattr(args, "chunk_pause_split_threshold_s", None)
    enable_diarization = getattr(args, "enable_diarization", None)
    diarization_device = getattr(args, "diarization_device", None)
    min_speakers = getattr(args, "min_speakers", None)
    max_speakers = getattr(args, "max_speakers", None)
    overlap_threshold = getattr(args, "overlap_threshold", None)

    config = TranscriptionConfig(
        model=args.model if args.model is not None else config.model,
        device=args.device if args.device is not None else config.device,
        compute_type=args.compute_type if args.compute_type is not None else config.compute_type,
        language=args.language if args.language is not None else config.language,
        task=args.task if args.task is not None else config.task,
        vad_min_silence_ms=args.vad_min_silence_ms
        if args.vad_min_silence_ms is not None
        else config.vad_min_silence_ms,
        beam_size=args.beam_size if args.beam_size is not None else config.beam_size,
        skip_existing_json=args.skip_existing_json
        if args.skip_existing_json is not None
        else config.skip_existing_json,
        enable_chunking=enable_chunking if enable_chunking is not None else config.enable_chunking,
        chunk_target_duration_s=chunk_target_duration_s
        if chunk_target_duration_s is not None
        else config.chunk_target_duration_s,
        chunk_max_duration_s=chunk_max_duration_s
        if chunk_max_duration_s is not None
        else config.chunk_max_duration_s,
        chunk_target_tokens=chunk_target_tokens
        if chunk_target_tokens is not None
        else config.chunk_target_tokens,
        chunk_pause_split_threshold_s=chunk_pause_split_threshold_s
        if chunk_pause_split_threshold_s is not None
        else config.chunk_pause_split_threshold_s,
        enable_diarization=enable_diarization
        if enable_diarization is not None
        else config.enable_diarization,
        diarization_device=diarization_device
        if diarization_device is not None
        else config.diarization_device,
        min_speakers=min_speakers if min_speakers is not None else config.min_speakers,
        max_speakers=max_speakers if max_speakers is not None else config.max_speakers,
        overlap_threshold=overlap_threshold
        if overlap_threshold is not None
        else config.overlap_threshold,
    )

    # If compute_type wasn't explicitly provided by CLI, file, or env,
    # re-derive the default based on the final device selection.
    compute_type_explicit = (
        args.compute_type is not None or env_compute_type_explicit or file_compute_type_explicit
    )
    if not compute_type_explicit:
        config.compute_type = "int8" if config.device == "cpu" else "float16"

    validate_diarization_settings(
        config.min_speakers,
        config.max_speakers,
        config.overlap_threshold,
    )

    return config


def _config_from_enrich_args(args: argparse.Namespace) -> EnrichmentConfig:
    """
    Build EnrichmentConfig from CLI arguments with proper precedence.

    Precedence order (highest to lowest):
    1. Explicit CLI flags
    2. Config file (if --enrich-config provided)
    3. Environment variables
    4. Defaults

    Args:
        args: Parsed command-line arguments from argparse.

    Returns:
        EnrichmentConfig with merged settings from all sources.
    """
    # Step 1: Start with defaults
    config = EnrichmentConfig()

    # Step 2: Override with environment variables
    env_config = EnrichmentConfig.from_env()
    config = _merge_enrich_configs(config, env_config)

    # Step 3: Override with config file if provided
    if args.enrich_config is not None:
        file_config = EnrichmentConfig.from_file(args.enrich_config)
        config = _merge_enrich_configs(config, file_config)

    # Step 4: Override with explicit CLI flags (only if not None)
    config = EnrichmentConfig(
        skip_existing=args.skip_existing
        if args.skip_existing is not None
        else config.skip_existing,
        enable_prosody=args.enable_prosody
        if args.enable_prosody is not None
        else config.enable_prosody,
        enable_emotion=args.enable_emotion
        if args.enable_emotion is not None
        else config.enable_emotion,
        enable_categorical_emotion=args.enable_categorical_emotion
        if args.enable_categorical_emotion is not None
        else config.enable_categorical_emotion,
        enable_semantic_annotator=args.enable_semantic_annotator
        if args.enable_semantic_annotator is not None
        else config.enable_semantic_annotator,
        enable_turn_metadata=args.enable_turn_metadata
        if args.enable_turn_metadata is not None
        else config.enable_turn_metadata,
        enable_speaker_stats=args.enable_speaker_stats
        if args.enable_speaker_stats is not None
        else config.enable_speaker_stats,
        device=args.device if args.device is not None else config.device,
        dimensional_model_name=config.dimensional_model_name,
        categorical_model_name=config.categorical_model_name,
        semantic_annotator=config.semantic_annotator,
    )

    # Convenience flag overrides granular analytics flags if provided
    if args.enable_speaker_analytics is not None:
        config.enable_turn_metadata = args.enable_speaker_analytics
        config.enable_speaker_stats = args.enable_speaker_analytics

    return config


def _get_cache_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except (OSError, PermissionError):
                pass
    return total


def _format_size(bytes_size: int) -> str:
    """Format bytes as human-readable string."""
    size = float(bytes_size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _handle_cache_command(args: argparse.Namespace) -> int:
    """Handle cache subcommand: --show or --clear."""
    import shutil

    from .cache import CachePaths
    from .samples import get_samples_cache_dir

    paths = CachePaths.from_env().ensure_dirs()
    samples_dir = get_samples_cache_dir()

    if args.show:
        print("slower-whisper cache locations:")
        print(f"  Root:         {paths.root}")
        print(f"  HF_HOME:      {paths.hf_home} ({_format_size(_get_cache_size(paths.hf_home))})")
        print(
            f"  TORCH_HOME:   {paths.torch_home} ({_format_size(_get_cache_size(paths.torch_home))})"
        )
        print(
            f"  Whisper:      {paths.whisper_root} ({_format_size(_get_cache_size(paths.whisper_root))})"
        )
        print(
            f"  Emotion:      {paths.emotion_root} ({_format_size(_get_cache_size(paths.emotion_root))})"
        )
        print(
            f"  Diarization:  {paths.diarization_root} ({_format_size(_get_cache_size(paths.diarization_root))})"
        )
        print(f"  Samples:      {samples_dir} ({_format_size(_get_cache_size(samples_dir))})")
        total = sum(
            [
                _get_cache_size(paths.hf_home),
                _get_cache_size(paths.torch_home),
                _get_cache_size(paths.whisper_root),
                _get_cache_size(paths.emotion_root),
                _get_cache_size(paths.diarization_root),
                _get_cache_size(samples_dir),
            ]
        )
        print(f"\n  Total:        {_format_size(total)}")
        return 0

    if args.clear:
        targets = []
        if args.clear == "all":
            targets = [
                ("Whisper", paths.whisper_root),
                ("Emotion", paths.emotion_root),
                ("Diarization", paths.diarization_root),
                ("HF", paths.hf_home),
                ("Torch", paths.torch_home),
                ("Samples", samples_dir),
            ]
        elif args.clear == "whisper":
            targets = [("Whisper", paths.whisper_root)]
        elif args.clear == "emotion":
            targets = [("Emotion", paths.emotion_root)]
        elif args.clear == "diarization":
            targets = [("Diarization", paths.diarization_root)]
        elif args.clear == "hf":
            targets = [("HF", paths.hf_home)]
        elif args.clear == "torch":
            targets = [("Torch", paths.torch_home)]
        elif args.clear == "samples":
            targets = [("Samples", samples_dir)]

        for name, target in targets:
            if target.exists():
                shutil.rmtree(target)
                print(f"Cleared {name} cache: {target}")
            target.mkdir(parents=True, exist_ok=True)

        return 0

    # Should not reach here due to required=True
    raise RuntimeError("Invalid cache command: expected either --show or --clear")


def _handle_samples_command(args: argparse.Namespace) -> int:
    """Handle samples subcommand: list/download/copy/generate."""
    from .samples import (
        copy_sample_to_project,
        download_sample_dataset,
        get_sample_test_files,
        list_sample_datasets,
    )

    if args.samples_action == "list":
        datasets = list_sample_datasets()
        print("Available sample datasets:\n")
        for name, metadata in datasets.items():
            print(f"  {name}")
            print(f"    Description: {metadata['description']}")
            print(f"    Source:      {metadata['source_url']}")
            print(f"    License:     {metadata['license']}")
            print(f"    Test files:  {metadata['test_files']}")
            print()
        return 0

    elif args.samples_action == "download":
        try:
            download_sample_dataset(args.dataset, force_download=args.force)
            test_files = get_sample_test_files(args.dataset)
            print("\nTest files ready:")
            for f in test_files:
                print(f"  {f}")
            return 0
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.samples_action == "copy":
        try:
            project_dir = args.root / "raw_audio"
            copied_files = copy_sample_to_project(args.dataset, project_dir)
            print(f"\nCopied {len(copied_files)} files to {project_dir}:")
            for f in copied_files:
                print(f"  {f.name}")
            print("\nReady to transcribe with:")
            print(f"  cd {args.root}")
            print("  uv run slower-whisper transcribe --enable-diarization")
            return 0
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.samples_action == "generate":
        # Generate synthetic samples
        from .samples import generate_synthetic_2speaker

        output_dir = args.output or (Path.cwd() / "raw_audio")
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.speakers == 2:
            output_file = output_dir / "synthetic_2speaker.wav"
            try:
                generate_synthetic_2speaker(output_file)
                print("\nReady to transcribe with:")
                print(
                    "  uv run slower-whisper transcribe --enable-diarization --min-speakers 2 --max-speakers 2"
                )
                return 0
            except ImportError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
        else:
            print(
                f"Error: Generating {args.speakers}-speaker samples not yet implemented",
                file=sys.stderr,
            )
            return 1

    else:
        raise RuntimeError(f"Unknown samples action: {args.samples_action}")


def _default_export_path(input_path: Path, fmt: str) -> Path:
    suffix_map = {"csv": ".csv", "html": ".html", "vtt": ".vtt", "textgrid": ".TextGrid"}
    suffix = suffix_map.get(fmt.lower(), f".{fmt}")
    return input_path.with_suffix(suffix)


def _handle_export_command(args: argparse.Namespace) -> int:
    transcript = api_module.load_transcript(args.transcript)
    output_path = args.output or _default_export_path(args.transcript, args.format)
    export_transcript(transcript, args.format, output_path, unit=args.unit)
    print(f"[done] Wrote {args.format} to {output_path}")
    return 0


def _handle_validate_command(args: argparse.Namespace) -> int:
    schema_path = args.schema or DEFAULT_SCHEMA_PATH
    failures = validate_many(args.transcripts, schema_path=schema_path)
    if failures:
        print("Validation failed:")
        for err in failures:
            print(f"- {err}")
        return 1

    print(f"[ok] {len(args.transcripts)} transcript(s) valid against {schema_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    Main CLI entry point.

    Returns:
        0 on success, 1 on SlowerWhisperError, 2 on unexpected error.
    """
    try:
        parser = build_parser()
        args = parser.parse_args(argv)

        if args.command == "transcribe":
            try:
                cfg = _config_from_transcribe_args(args)
            except (OSError, ValueError) as e:
                raise ConfigurationError(f"Invalid transcription configuration: {e}") from e

            # Check for experimental diarization flag (v1.1 experimental)
            if cfg.enable_diarization:
                print(
                    "\n[INFO] Speaker diarization is EXPERIMENTAL in v1.1.\n"
                    "Requires: uv sync --extra diarization\n"
                    "Requires: HF_TOKEN environment variable (huggingface.co/settings/tokens)\n"
                    "See docs/SPEAKER_DIARIZATION.md for details.\n",
                    file=sys.stderr,
                )

            transcripts = transcribe_directory(args.root, config=cfg)
            print(f"\n[done] Transcribed {len(transcripts)} files")

        elif args.command == "enrich":
            try:
                enrich_cfg = _config_from_enrich_args(args)
            except (OSError, ValueError) as e:
                raise ConfigurationError(f"Invalid enrichment configuration: {e}") from e
            enriched = enrich_directory(args.root, config=enrich_cfg)
            print(f"\n[done] Enriched {len(enriched)} transcripts")

        elif args.command == "cache":
            return _handle_cache_command(args)

        elif args.command == "samples":
            return _handle_samples_command(args)

        elif args.command == "export":
            return _handle_export_command(args)

        elif args.command == "validate":
            return _handle_validate_command(args)

        else:
            parser.error(f"Unknown command: {args.command}")

        return 0

    except SlowerWhisperError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
