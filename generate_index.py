#!/usr/bin/env python3
"""
Generate transcript index and analytics reports.

Creates indices and reports from existing transcript JSON files.

Usage:
    python generate_index.py                      # Generate all outputs
    python generate_index.py --index-only         # Just generate index
    python generate_index.py --reports-only       # Just generate reports
    python generate_index.py --output-dir reports/
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transcription.enrichment.analytics import indexer, reports

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate transcript indices and reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_index.py
  python generate_index.py --index-only
  python generate_index.py --tone-report --speaker-report
  python generate_index.py --output-dir reports/

Outputs:
  - transcripts_index.json: JSON index of all transcripts
  - transcripts_index.csv: CSV index (for Excel/spreadsheet analysis)
  - tone_analysis.md: Tone distribution report (if --tone-report)
  - speaker_analysis.md: Speaker diarization report (if --speaker-report)
        """
    )

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory containing whisper_json/ folder (default: current directory)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for reports (default: root directory)"
    )

    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only generate indices (skip reports)"
    )

    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Only generate reports (skip indices)"
    )

    parser.add_argument(
        "--tone-report",
        action="store_true",
        help="Generate tone analysis report"
    )

    parser.add_argument(
        "--speaker-report",
        action="store_true",
        help="Generate speaker diarization report"
    )

    parser.add_argument(
        "--all-reports",
        action="store_true",
        help="Generate all available reports"
    )

    args = parser.parse_args()

    # Setup paths
    root = Path(args.root).resolve()
    json_dir = root / "whisper_json"

    if not json_dir.exists():
        logger.error(f"whisper_json directory not found: {json_dir}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = root

    logger.info(f"Processing transcripts from: {json_dir}")
    logger.info(f"Writing outputs to: {output_dir}")

    # Determine what to generate
    generate_indices = not args.reports_only
    generate_tone = args.tone_report or args.all_reports or (not args.index_only and not args.reports_only)
    generate_speaker = args.speaker_report or args.all_reports or (not args.index_only and not args.reports_only)

    # Generate indices
    if generate_indices:
        logger.info("\n=== Generating Indices ===")

        try:
            json_index_path = output_dir / "transcripts_index.json"
            indexer.generate_index(json_dir, json_index_path, format="json")
            logger.info(f"✓ JSON index: {json_index_path}")
        except Exception as e:
            logger.error(f"✗ Failed to generate JSON index: {e}")

        try:
            csv_index_path = output_dir / "transcripts_index.csv"
            indexer.generate_index(json_dir, csv_index_path, format="csv")
            logger.info(f"✓ CSV index: {csv_index_path}")
        except Exception as e:
            logger.error(f"✗ Failed to generate CSV index: {e}")

    # Generate reports
    if not args.index_only:
        logger.info("\n=== Generating Reports ===")

        if generate_tone:
            try:
                tone_report_path = output_dir / "tone_analysis.md"
                reports.generate_tone_report(json_dir, tone_report_path)
                logger.info(f"✓ Tone report: {tone_report_path}")
            except Exception as e:
                logger.error(f"✗ Failed to generate tone report: {e}")

        if generate_speaker:
            try:
                speaker_report_path = output_dir / "speaker_analysis.md"
                reports.generate_speaker_report(json_dir, speaker_report_path)
                logger.info(f"✓ Speaker report: {speaker_report_path}")
            except Exception as e:
                logger.error(f"✗ Failed to generate speaker report: {e}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
