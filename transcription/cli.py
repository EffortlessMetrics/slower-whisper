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
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from . import api as api_module
from .benchmark_cli import build_benchmark_parser, handle_benchmark_command
from .config import (
    EnrichmentConfig,
    Paths,
    TranscriptionConfig,
)
from .device import DeviceChoice, format_preflight_banner, resolve_device
from .exceptions import ConfigurationError, SlowerWhisperError
from .exporters import SUPPORTED_EXPORT_FORMATS, export_transcript
from .models import Transcript
from .validation import DEFAULT_SCHEMA_PATH, validate_many

logger = logging.getLogger(__name__)


# Expose API functions for compatibility with tests/patching while delegating to api module
def transcribe_directory(*args, **kwargs) -> list[Transcript]:
    return api_module.transcribe_directory(*args, **kwargs)


def enrich_directory(*args, **kwargs) -> list[Transcript]:
    return api_module.enrich_directory(*args, **kwargs)


def _setup_progress_logging(show_progress: bool) -> None:
    """
    Configure logging to show/hide progress messages based on --progress flag.

    When show_progress=True, sets root logger to INFO to show file counters.
    When show_progress=False, sets to WARNING to hide progress messages.

    Args:
        show_progress: Whether to show progress indicators (file counters).
    """
    level = logging.INFO if show_progress else logging.WARNING

    # Configure basic format first (only works on first call)
    logging.basicConfig(format="%(message)s")

    # Always set level directly (works on subsequent calls)
    logging.getLogger().setLevel(level)


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
        choices=["auto", "cuda", "cpu"],
        help="Device for ASR (Whisper) inference. 'auto' detects CUDA availability (default: auto).",
    )
    p_trans.add_argument(
        "--compute-type",
        default=None,
        help="faster-whisper compute type: float16, float32, int8, int8_float16, etc. "
        "(default: auto-selected based on device).",
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
        "--word-timestamps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Extract word-level timestamps (v1.8+, default: False).",
    )
    # Skip existing transcripts flag - aliased for consistency with enrich command
    skip_existing_group = p_trans.add_mutually_exclusive_group()
    skip_existing_group.add_argument(
        "--skip-existing-json",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="skip_existing_json",
        help="Skip files with existing JSON in whisper_json/ (default: True).",
    )
    skip_existing_group.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="skip_existing_json",
        help="Alias for --skip-existing-json (for consistency with enrich command).",
    )
    p_trans.add_argument(
        "--progress",
        action="store_true",
        default=False,
        help="Show progress indicator during transcription (file counter).",
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
        help="Device for diarization. 'auto' selects cuda if available, else cpu (default: auto).",
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
        "--config",
        "--enrich-config",  # deprecated alias, kept for backward compatibility
        type=Path,
        default=None,
        dest="enrich_config",
        help="Path to EnrichmentConfig JSON file. Precedence: CLI flags > config file > env vars > defaults.",
    )
    p_enrich.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip segments that already have audio_state (default: True).",
    )
    p_enrich.add_argument(
        "--progress",
        action="store_true",
        default=False,
        help="Show progress indicator during enrichment (file counter).",
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
        help="Convenience flag: overrides both --enable-turn-metadata and --enable-speaker-stats. "
        "Omit to control each feature individually.",
    )
    p_enrich.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Device for emotion model inference. 'auto' detects CUDA availability (default: cpu).",
    )
    p_enrich.add_argument(
        "--pause-threshold",
        type=float,
        default=None,
        help="Minimum pause duration (seconds) to split turns for same speaker. "
        "If not set, only speaker changes trigger turn splits. "
        "Example: --pause-threshold 2.0 splits turns on pauses >= 2 seconds.",
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

    # ============================================================================
    # benchmark subcommand
    # ============================================================================
    build_benchmark_parser(subparsers)

    return parser


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
    # Extract CLI overrides (only non-None values)
    # from_sources() handles filtering and validation
    config = TranscriptionConfig.from_sources(
        config_file=args.config,
        # Core ASR settings
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        task=args.task,
        vad_min_silence_ms=args.vad_min_silence_ms,
        beam_size=args.beam_size,
        skip_existing_json=args.skip_existing_json,
        # Word-level alignment (v1.8+)
        word_timestamps=getattr(args, "word_timestamps", None),
        # Chunking settings (with getattr for backward compatibility)
        enable_chunking=getattr(args, "enable_chunking", None),
        chunk_target_duration_s=getattr(args, "chunk_target_duration_s", None),
        chunk_max_duration_s=getattr(args, "chunk_max_duration_s", None),
        chunk_target_tokens=getattr(args, "chunk_target_tokens", None),
        chunk_pause_split_threshold_s=getattr(args, "chunk_pause_split_threshold_s", None),
        # Diarization settings (with getattr for backward compatibility)
        enable_diarization=getattr(args, "enable_diarization", None),
        diarization_device=getattr(args, "diarization_device", None),
        min_speakers=getattr(args, "min_speakers", None),
        max_speakers=getattr(args, "max_speakers", None),
        overlap_threshold=getattr(args, "overlap_threshold", None),
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
    # Extract CLI overrides into a dict (only non-None values)
    cli_overrides = {}

    if args.skip_existing is not None:
        cli_overrides["skip_existing"] = args.skip_existing
    if args.enable_prosody is not None:
        cli_overrides["enable_prosody"] = args.enable_prosody
    if args.enable_emotion is not None:
        cli_overrides["enable_emotion"] = args.enable_emotion
    if args.enable_categorical_emotion is not None:
        cli_overrides["enable_categorical_emotion"] = args.enable_categorical_emotion
    if args.enable_semantic_annotator is not None:
        cli_overrides["enable_semantic_annotator"] = args.enable_semantic_annotator
    if args.enable_turn_metadata is not None:
        cli_overrides["enable_turn_metadata"] = args.enable_turn_metadata
    if args.enable_speaker_stats is not None:
        cli_overrides["enable_speaker_stats"] = args.enable_speaker_stats
    if args.device is not None:
        cli_overrides["device"] = args.device
    if getattr(args, "pause_threshold", None) is not None:
        cli_overrides["pause_threshold"] = args.pause_threshold

    # Use from_sources to handle the full config chain
    config = EnrichmentConfig.from_sources(
        config_file=args.enrich_config,
        **cli_overrides,
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


def _handle_transcribe_command(args: argparse.Namespace) -> int:
    """Handle transcribe command and return exit code (0=success, 1=partial failure)."""
    try:
        cfg = _config_from_transcribe_args(args)
    except (OSError, ValueError) as e:
        raise ConfigurationError(f"Invalid transcription configuration: {e}") from e

    # Configure progress logging based on --progress flag
    _setup_progress_logging(args.progress)

    # Resolve device with auto-detection and print preflight banner
    # CLI flag is the only "explicit" signal we can trust for device
    requested_device: DeviceChoice = (
        args.device if args.device in ("auto", "cuda", "cpu") else "auto"
    )
    resolved = resolve_device(requested_device, allow_fallback=True)

    # Override config with resolved values
    cfg.device = resolved.device
    # CRITICAL: Only keep user's explicit --compute-type; otherwise sync with resolved device
    # This prevents CPU fallback from retaining incompatible GPU compute types (e.g., float16)
    if args.compute_type is None:
        cfg.compute_type = resolved.compute_type

    # Print preflight banner to stderr (keeps stdout clean for structured output)
    banner = format_preflight_banner(resolved, cfg.model)
    print(f"\n{banner}\n", file=sys.stderr)

    # Check for experimental diarization flag (v1.1 experimental)
    if cfg.enable_diarization:
        print(
            "[INFO] Speaker diarization is EXPERIMENTAL in v1.1.\n"
            "Requires: uv sync --extra diarization\n"
            "Requires: HF_TOKEN environment variable (huggingface.co/settings/tokens)\n"
            "See docs/SPEAKER_DIARIZATION.md for details.\n",
            file=sys.stderr,
        )

    # Call internal pipeline to get result with failure count
    from .config import AppConfig, AsrConfig
    from .pipeline import run_pipeline

    root = Path(args.root)
    paths = Paths(root=root)
    asr_cfg = AsrConfig(
        model_name=cfg.model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        vad_min_silence_ms=cfg.vad_min_silence_ms,
        beam_size=cfg.beam_size,
        language=cfg.language,
        task=cfg.task,
    )
    app_cfg = AppConfig(
        paths=paths,
        asr=asr_cfg,
        skip_existing_json=cfg.skip_existing_json,
    )

    result = run_pipeline(app_cfg, diarization_config=cfg)

    # Display structured results
    print("\n=== Transcription Summary ===")
    print(f"Total files:      {result.total_files}")
    print(f"Processed:        {result.processed}")
    print(f"Skipped:          {result.skipped}")
    if result.diarized_only > 0:
        print(f"Diarized only:    {result.diarized_only}")
    print(f"Failed:           {result.failed}")

    # Show RTF if available
    if result.total_audio_seconds > 0 and result.total_time_seconds > 0:
        rtf = result.overall_rtf
        print("\nPerformance:")
        print(f"  Audio duration: {result.total_audio_seconds / 60:.1f} min")
        print(f"  Wall time:      {result.total_time_seconds / 60:.1f} min")
        print(f"  RTF:            {rtf:.2f}x")

    # Show first 5 failures with error messages
    failures = [r for r in result.file_results if r.status == "error"]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for fail in failures[:5]:
            error_msg = fail.error_message or "Unknown error"
            print(f"  - {fail.file_name}: {error_msg}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")

    if result.failed > 0:
        return 1
    return 0


def _handle_enrich_command(args: argparse.Namespace) -> int:
    """Handle enrich command and return exit code (0=success, 1=partial failure)."""
    try:
        enrich_cfg = _config_from_enrich_args(args)
    except (OSError, ValueError) as e:
        raise ConfigurationError(f"Invalid enrichment configuration: {e}") from e

    # Resolve "auto" device for emotion inference (uses torch, not ctranslate2)
    if enrich_cfg.device in (None, "auto"):
        resolved = resolve_device("auto", backend="torch")
        enrich_cfg.device = resolved.device
        if resolved.is_fallback:
            print(
                f"[Device] {resolved.device.upper()} for enrichment "
                f"(reason: {resolved.fallback_reason})",
                file=sys.stderr,
            )

    # Configure progress logging based on --progress flag
    _setup_progress_logging(args.progress)

    # Track success/failure counts by calling enrich_directory and checking for errors
    root = Path(args.root)
    paths = Paths(root=root)
    json_dir = paths.json_dir
    audio_dir = paths.norm_dir

    if not json_dir.exists():
        raise api_module.EnrichmentError(
            f"JSON directory does not exist: {json_dir}. "
            f"Run transcription first using transcribe command."
        )

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise api_module.EnrichmentError(
            f"No JSON transcript files found in {json_dir}. "
            f"Run transcription first to generate transcript files."
        )

    total_files = len(json_files)
    enriched_count = 0
    skipped_count = 0
    failed_count = 0
    failures = []  # Track failures with details

    for idx, json_path in enumerate(json_files, start=1):
        logger.info("[%d/%d] %s", idx, total_files, json_path.name)
        try:
            stem = json_path.stem
            wav_path = audio_dir / f"{stem}.wav"

            if not wav_path.exists():
                error_msg = f"Audio file not found: {wav_path}"
                logger.warning(f"Audio file not found for {json_path.name}: {wav_path}")
                failed_count += 1
                failures.append((json_path.name, error_msg))
                continue

            # Use the single-file enrichment API
            from .writers import load_transcript_from_json, write_json

            transcript = load_transcript_from_json(json_path)

            # Check if already enriched when skip_existing is enabled
            if enrich_cfg.skip_existing:
                audio_ready = bool(transcript.segments) and all(
                    seg.audio_state is not None for seg in transcript.segments
                )
                if audio_ready:
                    skipped_count += 1
                    continue

            enriched = api_module.enrich_transcript(transcript, wav_path, enrich_cfg)
            write_json(enriched, json_path)
            enriched_count += 1

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Enrichment failed for {json_path.name}: {e}", exc_info=True)
            failed_count += 1
            failures.append((json_path.name, error_msg))

    # Display structured results
    print("\n=== Enrichment Summary ===")
    print(f"Total files:      {total_files}")
    print(f"Enriched:         {enriched_count}")
    if skipped_count > 0:
        print(f"Skipped:          {skipped_count} (already enriched)")
    print(f"Failed:           {failed_count}")

    # Show first 5 failures with error messages
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for file_name, error_msg in failures[:5]:
            print(f"  - {file_name}: {error_msg}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")

    if failed_count > 0:
        return 1
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
        0 on success, 1 on SlowerWhisperError or partial failures, 2 on unexpected error.
    """
    try:
        parser = build_parser()
        args = parser.parse_args(argv)

        if args.command == "transcribe":
            return _handle_transcribe_command(args)

        elif args.command == "enrich":
            return _handle_enrich_command(args)

        elif args.command == "cache":
            return _handle_cache_command(args)

        elif args.command == "samples":
            return _handle_samples_command(args)

        elif args.command == "export":
            return _handle_export_command(args)

        elif args.command == "validate":
            return _handle_validate_command(args)

        elif args.command == "benchmark":
            return handle_benchmark_command(args)

        else:
            parser.error(f"Unknown command: {args.command}")

    except SlowerWhisperError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
