"""
Audio enrichment CLI module.

This module provides command-line interface for enriching existing transcripts
with audio features including prosody and emotion analysis.
"""

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__ as PIPELINE_VERSION
from .audio_enrichment import enrich_transcript_audio as enrich_transcript_comprehensive
from .config import Paths
from .exceptions import SlowerWhisperError
from .models import AUDIO_STATE_VERSION, Transcript
from .writers import load_transcript_from_json, write_json

logger = logging.getLogger(__name__)


class LegacyEnrichmentConfig:
    """Configuration for audio enrichment process.

    .. deprecated:: 1.6.0
        This class is deprecated. Use the modern EnrichmentConfig from
        transcription.config module instead. This legacy implementation will
        be removed in v2.0.0.
    """

    def __init__(
        self,
        root: Path,
        skip_existing: bool = False,
        enable_prosody: bool = True,
        enable_emotion: bool = True,
        enable_categorical_emotion: bool = False,
        device: str = "cuda",
        single_file: str | None = None,
    ) -> None:
        self.paths = Paths(root=root)
        self.skip_existing = skip_existing
        self.enable_prosody = enable_prosody
        self.enable_emotion = enable_emotion
        self.enable_categorical_emotion = enable_categorical_emotion
        self.device = device
        self.single_file = single_file


def enrich_transcript_audio(
    transcript: Transcript,
    audio_path: Path,
    config: LegacyEnrichmentConfig,
) -> Transcript:
    """
    Enrich all segments in a transcript with audio features.

    This function delegates to the comprehensive audio_enrichment module
    which includes full feature extraction, rendering, and error handling.

    Args:
        transcript: Transcript object to enrich
        audio_path: Path to the audio file
        config: Enrichment configuration

    Returns:
        Enriched transcript with audio_state populated for each segment
    """
    # Use the comprehensive enrichment module
    enriched_transcript = enrich_transcript_comprehensive(
        transcript=transcript,
        wav_path=audio_path,
        enable_prosody=config.enable_prosody,
        enable_emotion=config.enable_emotion,
        enable_categorical_emotion=config.enable_categorical_emotion,
        compute_baseline=True,  # Enable speaker normalization
    )

    # Count enriched segments for logging
    enriched_count = sum(1 for seg in enriched_transcript.segments if seg.audio_state is not None)
    skipped_count = len(enriched_transcript.segments) - enriched_count

    logger.info(
        "Enrichment stats",
        extra={"enriched": enriched_count, "skipped": skipped_count},
    )

    return enriched_transcript


def _build_enrichment_meta(config: LegacyEnrichmentConfig) -> dict:
    """Build metadata for the enrichment process."""
    models_used = []
    if config.enable_prosody:
        models_used.append("prosody_analysis")
    if config.enable_emotion:
        models_used.append("emotion_dimensional")
    if config.enable_categorical_emotion:
        models_used.append("emotion_categorical")

    return {
        "enrichment_version": AUDIO_STATE_VERSION,
        "enriched_at": datetime.now(timezone.utc).isoformat(),
        "models_used": models_used,
        "device": config.device,
        "pipeline_version": PIPELINE_VERSION,
    }


def run_enrichment_pipeline(config: LegacyEnrichmentConfig) -> None:
    """
    Orchestrate the audio enrichment pipeline.

    Process flow:
    1. Find all JSON files in whisper_json/
    2. For each JSON, locate matching WAV in input_audio/
    3. Load transcript from JSON
    4. Enrich with audio features
    5. Write back to JSON (preserving original or creating _enriched.json)
    6. Report progress and statistics
    """
    paths = config.paths

    # Handle single file mode
    if config.single_file:
        json_files = [Path(config.single_file)]
        logger.info("Enriching single file", extra={"file": config.single_file})
    else:
        json_files = sorted(paths.json_dir.glob("*.json"))
        logger.info(
            "Starting audio enrichment pipeline",
            extra={
                "root": str(paths.root),
                "json_dir": str(paths.json_dir),
                "audio_dir": str(paths.norm_dir),
                "file_count": len(json_files),
            },
        )

    if not json_files:
        logger.warning("No JSON files found. Nothing to enrich.")
        return

    # Validation
    if not paths.norm_dir.exists():
        logger.error("Audio directory does not exist", extra={"audio_dir": str(paths.norm_dir)})
        return

    total = len(json_files)
    processed = 0
    skipped = 0
    failed = 0
    total_time = 0.0

    for idx, json_path in enumerate(json_files, start=1):
        logger.info(
            "Processing file",
            extra={"index": idx, "total": total, "file": json_path.name},
        )

        # Find matching audio file
        stem = json_path.stem
        wav_path = paths.norm_dir / f"{stem}.wav"

        if not wav_path.exists():
            logger.warning(
                "Audio file not found, skipping",
                extra={"expected_file": wav_path.name, "json_file": json_path.name},
            )
            skipped += 1
            continue

        start = time.time()

        try:
            # Load transcript
            transcript = load_transcript_from_json(json_path)

            # Check if already enriched
            if config.skip_existing:
                has_audio_state = any(seg.audio_state is not None for seg in transcript.segments)
                if has_audio_state:
                    logger.info(
                        "Transcript already enriched, skipping (use --no-skip-existing to re-enrich)",
                        extra={"file": json_path.name},
                    )
                    skipped += 1
                    continue

            # Enrich transcript
            logger.info(
                "Enriching transcript",
                extra={
                    "segment_count": len(transcript.segments),
                    "audio_file": wav_path.name,
                },
            )
            transcript = enrich_transcript_audio(transcript, wav_path, config)

            # Update metadata
            if transcript.meta is None:
                transcript.meta = {}
            if "audio_enrichment" not in transcript.meta:
                transcript.meta["audio_enrichment"] = {}

            transcript.meta["audio_enrichment"].update(_build_enrichment_meta(config))

            # Write enriched JSON back
            write_json(transcript, json_path)

            elapsed = time.time() - start
            total_time += elapsed

            logger.info(
                "Enrichment completed",
                extra={
                    "file": json_path.name,
                    "elapsed_seconds": round(elapsed, 1),
                    "output_path": str(json_path),
                },
            )

            processed += 1

        except Exception as e:
            logger.error(
                "Failed to enrich file",
                extra={"file": json_path.name, "error": str(e)},
                exc_info=True,
            )
            failed += 1
            continue

    # Summary
    print("\n=== Summary ===")
    print(f"  processed={processed}, skipped={skipped}, failed={failed}, total={total}")
    if total_time > 0:
        print(f"  total_time={total_time:.1f}s, avg={total_time / max(processed, 1):.1f}s per file")
    print("All done.")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for audio enrichment CLI."""
    parser = argparse.ArgumentParser(
        description="Audio enrichment pipeline - add prosody and emotion features to transcripts"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(),
        help="Root directory containing whisper_json/ and input_audio/ (default: current directory)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip JSON files that already have audio_state",
    )
    parser.add_argument(
        "--enable-prosody",
        action="store_true",
        default=True,
        help="Enable prosody feature extraction (default: True)",
    )
    parser.add_argument(
        "--no-enable-prosody",
        dest="enable_prosody",
        action="store_false",
        help="Disable prosody feature extraction",
    )
    parser.add_argument(
        "--enable-emotion",
        action="store_true",
        default=True,
        help="Enable dimensional emotion analysis (default: True)",
    )
    parser.add_argument(
        "--no-enable-emotion",
        dest="enable_emotion",
        action="store_false",
        help="Disable dimensional emotion analysis",
    )
    parser.add_argument(
        "--enable-categorical-emotion",
        action="store_true",
        default=False,
        help="Enable categorical emotion classification (slower, default: False)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for emotion models (default: cuda)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Enrich a single JSON file instead of batch processing",
    )

    return parser


def main() -> int:
    """
    Main entry point for audio enrichment CLI.

    Returns:
        0 on success, 1 on SlowerWhisperError, 2 on unexpected error.
    """
    try:
        parser = build_parser()
        args = parser.parse_args()

        # Deprecation warning
        logger.warning(
            "audio_enrich_cli is deprecated as of v1.6.0. "
            "Use 'slower-whisper enrich' from the modern CLI instead. "
            "This legacy interface will be removed in v2.0.0."
        )

        config = LegacyEnrichmentConfig(
            root=args.root,
            skip_existing=args.skip_existing,
            enable_prosody=args.enable_prosody,
            enable_emotion=args.enable_emotion,
            enable_categorical_emotion=args.enable_categorical_emotion,
            device=args.device,
            single_file=args.file,
        )

        run_enrichment_pipeline(config)
        return 0

    except SlowerWhisperError as e:
        logger.error("SlowerWhisper error occurred", extra={"error": str(e)})
        return 1

    except Exception as e:
        logger.error("Unexpected error occurred", extra={"error": str(e)}, exc_info=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
