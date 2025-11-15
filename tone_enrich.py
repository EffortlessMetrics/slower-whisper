#!/usr/bin/env python3
"""
Tone enrichment CLI.

Enriches existing transcripts with tone labels using LLM analysis.

Usage:
    python tone_enrich.py                           # Enrich all transcripts
    python tone_enrich.py --file meeting1.json      # Enrich specific file
    python tone_enrich.py --skip-existing           # Skip already enriched files
    python tone_enrich.py --model claude-sonnet-4.5 --batch-size 5
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transcription.models import Transcript, Segment
from transcription.enrichment.tone import ToneAnalyzer, ToneConfig
from transcription.writers import write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_transcript_from_json(json_path: Path) -> Transcript:
    """Load a Transcript object from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = [Segment(**seg) for seg in data["segments"]]
    return Transcript(
        file_name=data["file_name"],
        language=data["language"],
        segments=segments,
        meta=data.get("meta")
    )


def should_skip_file(json_path: Path, skip_existing: bool) -> bool:
    """Check if file should be skipped."""
    if not skip_existing:
        return False

    # Check if already enriched with tone
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            enrichments = data.get("meta", {}).get("enrichments", {})
            if "tone_version" in enrichments:
                logger.info(f"[skip] {json_path.name} - already has tone enrichment")
                return True
    except Exception as e:
        logger.warning(f"Could not check enrichment status for {json_path}: {e}")

    return False


def enrich_file(json_path: Path, analyzer: ToneAnalyzer, backup: bool) -> bool:
    """
    Enrich a single transcript file with tone.

    Args:
        json_path: Path to JSON file
        analyzer: ToneAnalyzer instance
        backup: Whether to create backup before overwriting

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load transcript
        transcript = load_transcript_from_json(json_path)

        # Create backup if requested
        if backup:
            backup_path = json_path.with_suffix(".json.bak")
            import shutil
            shutil.copy2(json_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")

        # Analyze tone
        start_time = time.time()
        transcript = analyzer.annotate(transcript)
        elapsed = time.time() - start_time

        # Write updated JSON
        write_json(transcript, json_path)

        logger.info(
            f"✓ {json_path.name} - {len(transcript.segments)} segments, "
            f"{elapsed:.1f}s"
        )
        return True

    except Exception as e:
        logger.error(f"✗ {json_path.name} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Enrich transcripts with tone analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tone_enrich.py
  python tone_enrich.py --file meeting1.json
  python tone_enrich.py --skip-existing --backup
  python tone_enrich.py --provider openai --model gpt-4
  python tone_enrich.py --provider mock  # For testing without API
        """
    )

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory containing whisper_json/ folder (default: current directory)"
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Specific JSON file to enrich (basename only, e.g., 'meeting1.json')"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have tone enrichment"
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak backup before overwriting JSON files"
    )

    # Tone analyzer config
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai", "mock"],
        help="API provider for tone analysis (default: anthropic)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Model name (default: claude-sonnet-4-5-20250929)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (if not set via environment variable)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of segments per API call (default: 10)"
    )

    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Disable including surrounding segments as context"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum confidence to assign tone (default: 0.6)"
    )

    args = parser.parse_args()

    # Setup paths
    root = Path(args.root).resolve()
    json_dir = root / "whisper_json"

    if not json_dir.exists():
        logger.error(f"whisper_json directory not found: {json_dir}")
        sys.exit(1)

    # Find files to process
    if args.file:
        json_files = [json_dir / args.file]
        if not json_files[0].exists():
            logger.error(f"File not found: {json_files[0]}")
            sys.exit(1)
    else:
        json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found to process")
        sys.exit(0)

    logger.info(f"Found {len(json_files)} transcript(s) to process")

    # Initialize tone analyzer
    tone_config = ToneConfig(
        api_provider=args.provider,
        model_name=args.model,
        api_key=args.api_key,
        batch_size=args.batch_size,
        include_context=not args.no_context,
        confidence_threshold=args.confidence_threshold,
    )

    try:
        analyzer = ToneAnalyzer(tone_config)
    except Exception as e:
        logger.error(f"Failed to initialize tone analyzer: {e}")
        sys.exit(1)

    # Process files
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    start_time = time.time()

    for i, json_path in enumerate(json_files, 1):
        logger.info(f"[{i}/{len(json_files)}] {json_path.name}")

        if should_skip_file(json_path, args.skip_existing):
            stats["skipped"] += 1
            continue

        success = enrich_file(json_path, analyzer, args.backup)
        if success:
            stats["processed"] += 1
        else:
            stats["failed"] += 1

    # Summary
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("TONE ENRICHMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Processed:  {stats['processed']}")
    logger.info(f"Skipped:    {stats['skipped']}")
    logger.info(f"Failed:     {stats['failed']}")
    logger.info(f"Total:      {len(json_files)}")
    logger.info(f"Wall time:  {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
