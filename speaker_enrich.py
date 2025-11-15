#!/usr/bin/env python3
"""
Speaker diarization CLI.

Enriches existing transcripts with speaker labels using pyannote.audio.

Usage:
    python speaker_enrich.py                              # Diarize all transcripts
    python speaker_enrich.py --file meeting1.json         # Diarize specific file
    python speaker_enrich.py --num-speakers 3             # Hint: 3 speakers
    python speaker_enrich.py --speaker-map speakers.json  # Use custom names
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transcription.models import Transcript, Segment
from transcription.enrichment.speaker import SpeakerDiarizer, DiarizationConfig
from transcription.enrichment.speaker.diarizer import load_speaker_mapping
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

    # Check if already enriched with speakers
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            enrichments = data.get("meta", {}).get("enrichments", {})
            if "speaker_version" in enrichments:
                logger.info(f"[skip] {json_path.name} - already has speaker enrichment")
                return True
    except Exception as e:
        logger.warning(f"Could not check enrichment status for {json_path}: {e}")

    return False


def enrich_file(json_path: Path, input_audio_dir: Path, diarizer: SpeakerDiarizer,
                speaker_mapping: dict, backup: bool) -> bool:
    """
    Enrich a single transcript file with speaker labels.

    Args:
        json_path: Path to JSON file
        input_audio_dir: Directory containing normalized WAV files
        diarizer: SpeakerDiarizer instance
        speaker_mapping: Optional speaker name mapping
        backup: Whether to create backup before overwriting

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load transcript
        transcript = load_transcript_from_json(json_path)

        # Find corresponding audio file
        audio_name = Path(transcript.file_name).stem + ".wav"
        audio_path = input_audio_dir / audio_name

        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return False

        # Create backup if requested
        if backup:
            backup_path = json_path.with_suffix(".json.bak")
            import shutil
            shutil.copy2(json_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")

        # Diarize
        start_time = time.time()
        transcript = diarizer.annotate(transcript, audio_path, speaker_mapping)
        elapsed = time.time() - start_time

        # Write updated JSON
        write_json(transcript, json_path)

        # Count unique speakers
        speakers = set(seg.speaker for seg in transcript.segments if seg.speaker)
        logger.info(
            f"✓ {json_path.name} - {len(speakers)} speakers, "
            f"{len(transcript.segments)} segments, {elapsed:.1f}s"
        )
        return True

    except Exception as e:
        logger.error(f"✗ {json_path.name} - {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Enrich transcripts with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python speaker_enrich.py
  python speaker_enrich.py --file meeting1.json
  python speaker_enrich.py --num-speakers 3
  python speaker_enrich.py --speaker-map speakers.json
  python speaker_enrich.py --device cpu  # Use CPU instead of GPU

Speaker mapping file format (JSON):
  {
    "SPEAKER_00": "Alice",
    "SPEAKER_01": "Bob",
    "SPEAKER_02": "Charlie"
  }
        """
    )

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory containing whisper_json/ and input_audio/ folders (default: current directory)"
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Specific JSON file to enrich (basename only, e.g., 'meeting1.json')"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have speaker enrichment"
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak backup before overwriting JSON files"
    )

    # Diarization config
    parser.add_argument(
        "--model",
        type=str,
        default="pyannote/speaker-diarization-3.1",
        help="Pyannote model name (default: pyannote/speaker-diarization-3.1)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace token (if not set via HF_TOKEN environment variable)"
    )

    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Fixed number of speakers (if known)"
    )

    parser.add_argument(
        "--min-speakers",
        type=int,
        default=1,
        help="Minimum number of speakers for automatic detection (default: 1)"
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=10,
        help="Maximum number of speakers for automatic detection (default: 10)"
    )

    parser.add_argument(
        "--speaker-map",
        type=str,
        help="JSON file mapping SPEAKER_XX to custom names"
    )

    args = parser.parse_args()

    # Setup paths
    root = Path(args.root).resolve()
    json_dir = root / "whisper_json"
    audio_dir = root / "input_audio"

    if not json_dir.exists():
        logger.error(f"whisper_json directory not found: {json_dir}")
        sys.exit(1)

    if not audio_dir.exists():
        logger.error(f"input_audio directory not found: {audio_dir}")
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

    # Load speaker mapping if provided
    speaker_mapping = None
    if args.speaker_map:
        mapping_path = Path(args.speaker_map)
        if not mapping_path.exists():
            logger.error(f"Speaker mapping file not found: {mapping_path}")
            sys.exit(1)
        speaker_mapping = load_speaker_mapping(mapping_path)
        logger.info(f"Loaded speaker mapping with {len(speaker_mapping)} entries")

    # Initialize diarizer
    diarization_config = DiarizationConfig(
        model_name=args.model,
        device=args.device,
        hf_token=args.hf_token,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    try:
        diarizer = SpeakerDiarizer(diarization_config)
    except Exception as e:
        logger.error(f"Failed to initialize diarizer: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

    # Process files
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    start_time = time.time()

    for i, json_path in enumerate(json_files, 1):
        logger.info(f"[{i}/{len(json_files)}] {json_path.name}")

        if should_skip_file(json_path, args.skip_existing):
            stats["skipped"] += 1
            continue

        success = enrich_file(json_path, audio_dir, diarizer, speaker_mapping, args.backup)
        if success:
            stats["processed"] += 1
        else:
            stats["failed"] += 1

    # Summary
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPEAKER DIARIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Processed:  {stats['processed']}")
    logger.info(f"Skipped:    {stats['skipped']}")
    logger.info(f"Failed:     {stats['failed']}")
    logger.info(f"Total:      {len(json_files)}")
    logger.info(f"Wall time:  {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
