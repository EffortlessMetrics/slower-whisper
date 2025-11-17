#!/usr/bin/env python3
"""
Batch Research Data Processing Workflow

Process large collections of audio recordings for research studies:
- Batch transcription and enrichment
- Metadata extraction from filenames
- Aggregated statistics and exports
- Quality control reporting
- CSV exports for statistical analysis

Usage:
    python batch_research_processing.py --input-dir raw_audio/study_2025 --output-dir outputs/processed
    python batch_research_processing.py --input-dir raw_audio --pattern "{condition}_{subject}_{trial}.wav"
    python batch_research_processing.py --output-dir outputs/processed --qc-only
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transcription.writers import load_transcript_from_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    extract_metadata_from_filename: bool = True
    filename_pattern: str | None = None
    checkpoint_interval: int = 10
    generate_aggregated_csv: bool = True
    generate_statistics_report: bool = True
    generate_qc_report: bool = True


@dataclass
class FileMetadata:
    """Metadata extracted from filename and processing."""

    filename: str
    file_path: str
    duration: float = 0.0
    segment_count: int = 0
    enriched: bool = False
    language: str = ""

    # Extracted metadata fields (customizable)
    metadata_fields: dict[str, str] = None

    # Quality metrics
    avg_confidence: float = 0.0
    segments_with_audio_state: int = 0

    # Statistics
    avg_pitch: float = 0.0
    avg_energy: float = 0.0
    avg_speech_rate: float = 0.0
    avg_valence: float = 0.0
    avg_arousal: float = 0.0

    def __post_init__(self):
        if self.metadata_fields is None:
            self.metadata_fields = {}


class BatchProcessor:
    """Process batches of audio transcripts."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.processed_files: list[FileMetadata] = []
        self.failed_files: list[tuple[str, str]] = []  # (filename, error)

    def extract_metadata_from_filename(
        self, filename: str, pattern: str | None = None
    ) -> dict[str, str]:
        """
        Extract metadata from filename using pattern.

        Example patterns:
            "{condition}_{subject_id}_{trial}.wav"
            "{study}_{participant}_{session}_{date}.mp3"
        """
        metadata = {}

        if not pattern:
            # Default: try to parse common patterns
            parts = Path(filename).stem.split("_")
            for i, part in enumerate(parts):
                metadata[f"field_{i + 1}"] = part
            return metadata

        # Convert pattern to regex
        # Replace {field_name} with named capture groups
        pattern_regex = re.escape(pattern)
        pattern_regex = re.sub(r"\\{(\w+)\\}", r"(?P<\1>[^_]+)", pattern_regex)
        pattern_regex = pattern_regex.replace(r"\*", ".*")

        match = re.match(pattern_regex, filename)
        if match:
            metadata = match.groupdict()

        return metadata

    def process_file(self, json_path: Path, audio_path: Path | None = None) -> FileMetadata | None:
        """Process a single transcript file and extract statistics."""
        try:
            transcript = load_transcript_from_json(json_path)

            metadata = FileMetadata(
                filename=json_path.name,
                file_path=str(json_path),
                duration=transcript.segments[-1].end if transcript.segments else 0.0,
                segment_count=len(transcript.segments),
                language=transcript.language,
            )

            # Extract filename metadata
            if self.config.extract_metadata_from_filename:
                metadata.metadata_fields = self.extract_metadata_from_filename(
                    json_path.name, self.config.filename_pattern
                )

            # Calculate statistics
            pitch_values = []
            energy_values = []
            rate_values = []
            valence_values = []
            arousal_values = []
            segments_with_state = 0

            for segment in transcript.segments:
                if not segment.audio_state:
                    continue

                segments_with_state += 1

                # Prosody
                pitch = segment.audio_state.get("pitch", {})
                if "mean_hz" in pitch:
                    pitch_values.append(pitch["mean_hz"])

                energy = segment.audio_state.get("energy", {})
                if "db_rms" in energy:
                    energy_values.append(energy["db_rms"])

                rate = segment.audio_state.get("rate", {})
                if "syllables_per_sec" in rate:
                    rate_values.append(rate["syllables_per_sec"])

                # Emotion
                emotion = segment.audio_state.get("emotion", {})
                if "dimensional" in emotion:
                    dim = emotion["dimensional"]

                    valence = dim.get("valence", {}).get("score")
                    if valence is not None:
                        valence_values.append(valence)

                    arousal = dim.get("arousal", {}).get("score")
                    if arousal is not None:
                        arousal_values.append(arousal)

            metadata.segments_with_audio_state = segments_with_state
            metadata.enriched = segments_with_state > 0

            # Calculate averages
            if pitch_values:
                metadata.avg_pitch = sum(pitch_values) / len(pitch_values)
            if energy_values:
                metadata.avg_energy = sum(energy_values) / len(energy_values)
            if rate_values:
                metadata.avg_speech_rate = sum(rate_values) / len(rate_values)
            if valence_values:
                metadata.avg_valence = sum(valence_values) / len(valence_values)
            if arousal_values:
                metadata.avg_arousal = sum(arousal_values) / len(arousal_values)

            return metadata

        except Exception as e:
            logger.error(f"Failed to process {json_path.name}: {e}")
            self.failed_files.append((json_path.name, str(e)))
            return None

    def process_directory(self, input_dir: Path, output_dir: Path) -> None:
        """Process all JSON transcripts in a directory."""
        json_files = sorted(input_dir.glob("*.json"))

        logger.info(f"Found {len(json_files)} JSON files to process")

        start_time = time.time()

        for i, json_path in enumerate(json_files, 1):
            logger.info(f"Processing {i}/{len(json_files)}: {json_path.name}")

            metadata = self.process_file(json_path)
            if metadata:
                self.processed_files.append(metadata)

            # Checkpoint
            if i % self.config.checkpoint_interval == 0:
                self.save_checkpoint(output_dir)
                logger.info(f"Checkpoint saved at {i} files")

        elapsed_time = time.time() - start_time
        logger.info(f"Batch processing complete in {elapsed_time:.1f}s")

    def save_checkpoint(self, output_dir: Path):
        """Save processing state for resume capability."""
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / "batch_state.json"

        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "processed_count": len(self.processed_files),
            "failed_count": len(self.failed_files),
            "processed_files": [f.filename for f in self.processed_files],
            "failed_files": self.failed_files,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def generate_aggregated_csv(self, output_dir: Path):
        """Generate aggregated CSV with all segments from all files."""
        logger.info("Generating aggregated segments CSV...")

        output_dir = output_dir / "aggregated"
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "all_segments.csv"

        fieldnames = [
            "filename",
            "segment_id",
            "start_time",
            "end_time",
            "duration",
            "text",
            "word_count",
            "pitch_level",
            "pitch_mean_hz",
            "pitch_contour",
            "energy_level",
            "energy_db_rms",
            "rate_level",
            "syllables_per_sec",
            "pause_count",
            "pause_density",
            "emotion_valence",
            "emotion_arousal",
            "emotion_category",
        ]

        # Add metadata fields from first file
        if self.processed_files and self.processed_files[0].metadata_fields:
            for field in self.processed_files[0].metadata_fields.keys():
                if field not in fieldnames:
                    fieldnames.insert(1, field)

        total_segments = 0

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for file_meta in self.processed_files:
                # Load transcript
                json_path = Path(file_meta.file_path)
                try:
                    transcript = load_transcript_from_json(json_path)
                except Exception as e:
                    logger.error(f"Failed to load {json_path.name}: {e}")
                    continue

                for segment in transcript.segments:
                    row = {
                        "filename": file_meta.filename,
                        "segment_id": segment.id,
                        "start_time": f"{segment.start:.3f}",
                        "end_time": f"{segment.end:.3f}",
                        "duration": f"{segment.end - segment.start:.3f}",
                        "text": segment.text,
                        "word_count": len(segment.text.split()),
                    }

                    # Add metadata fields
                    for field, value in file_meta.metadata_fields.items():
                        row[field] = value

                    # Extract audio state features
                    if segment.audio_state:
                        # Pitch
                        pitch = segment.audio_state.get("pitch", {})
                        row["pitch_level"] = pitch.get("level", "")
                        row["pitch_mean_hz"] = pitch.get("mean_hz", "")
                        row["pitch_contour"] = pitch.get("contour", "")

                        # Energy
                        energy = segment.audio_state.get("energy", {})
                        row["energy_level"] = energy.get("level", "")
                        row["energy_db_rms"] = energy.get("db_rms", "")

                        # Rate
                        rate = segment.audio_state.get("rate", {})
                        row["rate_level"] = rate.get("level", "")
                        row["syllables_per_sec"] = rate.get("syllables_per_sec", "")

                        # Pauses
                        pauses = segment.audio_state.get("pauses", {})
                        row["pause_count"] = pauses.get("count", "")
                        row["pause_density"] = pauses.get("density", "")

                        # Emotion
                        emotion = segment.audio_state.get("emotion", {})
                        if "dimensional" in emotion:
                            dim = emotion["dimensional"]
                            row["emotion_valence"] = dim.get("valence", {}).get("score", "")
                            row["emotion_arousal"] = dim.get("arousal", {}).get("score", "")

                        if "categorical" in emotion:
                            row["emotion_category"] = emotion["categorical"].get("primary", "")

                    writer.writerow(row)
                    total_segments += 1

        logger.info(f"Saved {total_segments} segments to {csv_path}")

        # Also generate file-level metadata CSV
        self._generate_file_metadata_csv(output_dir)

    def _generate_file_metadata_csv(self, output_dir: Path):
        """Generate CSV with file-level metadata and statistics."""
        csv_path = output_dir / "file_metadata.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            # Collect all possible metadata fields
            all_metadata_fields = set()
            for file_meta in self.processed_files:
                all_metadata_fields.update(file_meta.metadata_fields.keys())

            fieldnames = [
                "filename",
                "duration",
                "segment_count",
                "enriched",
                "language",
                "avg_pitch",
                "avg_energy",
                "avg_speech_rate",
                "avg_valence",
                "avg_arousal",
            ] + sorted(all_metadata_fields)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for file_meta in self.processed_files:
                row = {
                    "filename": file_meta.filename,
                    "duration": f"{file_meta.duration:.2f}",
                    "segment_count": file_meta.segment_count,
                    "enriched": file_meta.enriched,
                    "language": file_meta.language,
                    "avg_pitch": f"{file_meta.avg_pitch:.1f}" if file_meta.avg_pitch else "",
                    "avg_energy": f"{file_meta.avg_energy:.1f}" if file_meta.avg_energy else "",
                    "avg_speech_rate": f"{file_meta.avg_speech_rate:.2f}"
                    if file_meta.avg_speech_rate
                    else "",
                    "avg_valence": f"{file_meta.avg_valence:.3f}" if file_meta.avg_valence else "",
                    "avg_arousal": f"{file_meta.avg_arousal:.3f}" if file_meta.avg_arousal else "",
                }

                # Add metadata fields
                for field in all_metadata_fields:
                    row[field] = file_meta.metadata_fields.get(field, "")

                writer.writerow(row)

        logger.info(f"Saved file metadata to {csv_path}")

    def generate_statistics_report(self, output_dir: Path) -> str:
        """Generate comprehensive statistics report."""
        lines = []
        lines.append("RESEARCH CORPUS PROCESSING REPORT")
        lines.append("=" * 80)
        lines.append("")

        lines.append(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Total Files: {len(self.processed_files) + len(self.failed_files)}")
        lines.append("")

        # Processing summary
        lines.append("--- Processing Summary ---")
        lines.append("")
        success_count = len(self.processed_files)
        fail_count = len(self.failed_files)
        total = success_count + fail_count

        success_pct = (success_count / total * 100) if total > 0 else 0

        lines.append(f"Successfully Processed: {success_count} ({success_pct:.1f}%)")

        if fail_count > 0:
            lines.append(f"Failed: {fail_count} ({100 - success_pct:.1f}%)")
            for filename, error in self.failed_files[:5]:
                lines.append(f"  - {filename}: {error}")
            if len(self.failed_files) > 5:
                lines.append(f"  ... and {len(self.failed_files) - 5} more")

        lines.append("")

        # Corpus statistics
        if self.processed_files:
            total_duration = sum(f.duration for f in self.processed_files)
            total_segments = sum(f.segment_count for f in self.processed_files)
            avg_duration = total_duration / len(self.processed_files)

            lines.append("--- Corpus Statistics ---")
            lines.append("")
            lines.append(f"Total Audio Duration: {total_duration / 3600:.1f} hours")
            lines.append(f"Average File Duration: {avg_duration / 60:.1f} minutes")
            lines.append(f"Total Segments: {total_segments:,}")
            lines.append(
                f"Average Segments per File: {total_segments / len(self.processed_files):.0f}"
            )
            lines.append("")

            # Language distribution
            languages = Counter(f.language for f in self.processed_files)
            lines.append("Language Distribution:")
            for lang, count in languages.most_common():
                pct = (count / len(self.processed_files)) * 100
                lines.append(f"  {lang}: {count} ({pct:.1f}%)")
            lines.append("")

            # Enrichment statistics
            enriched_count = sum(1 for f in self.processed_files if f.enriched)
            enriched_pct = (enriched_count / len(self.processed_files)) * 100

            lines.append("--- Enrichment Statistics ---")
            lines.append("")
            lines.append(f"Files with Audio Features: {enriched_count} ({enriched_pct:.1f}%)")

            if enriched_count > 0:
                enriched_files = [f for f in self.processed_files if f.enriched]

                avg_pitch = sum(f.avg_pitch for f in enriched_files if f.avg_pitch) / len(
                    [f for f in enriched_files if f.avg_pitch]
                )
                avg_rate = sum(
                    f.avg_speech_rate for f in enriched_files if f.avg_speech_rate
                ) / len([f for f in enriched_files if f.avg_speech_rate])
                avg_valence = sum(f.avg_valence for f in enriched_files if f.avg_valence) / len(
                    [f for f in enriched_files if f.avg_valence]
                )
                avg_arousal = sum(f.avg_arousal for f in enriched_files if f.avg_arousal) / len(
                    [f for f in enriched_files if f.avg_arousal]
                )

                lines.append("")
                lines.append("Average Features Across Corpus:")
                lines.append(f"  Mean Pitch: {avg_pitch:.1f} Hz")
                lines.append(f"  Mean Speech Rate: {avg_rate:.2f} syllables/sec")
                lines.append(
                    f"  Mean Valence: {avg_valence:.3f} ({'Positive' if avg_valence > 0.5 else 'Negative'})"
                )
                lines.append(
                    f"  Mean Arousal: {avg_arousal:.3f} ({'High' if avg_arousal > 0.5 else 'Low'})"
                )

            lines.append("")

            # Metadata summary
            if self.processed_files[0].metadata_fields:
                lines.append("--- Metadata Fields Extracted ---")
                lines.append("")
                for field in sorted(self.processed_files[0].metadata_fields.keys()):
                    unique_values = len(
                        {f.metadata_fields.get(field, "") for f in self.processed_files}
                    )
                    lines.append(f"  ✓ {field} ({unique_values} unique values)")
                lines.append("")

        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save report
        report_path = output_dir / "reports" / "statistics_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Saved statistics report: {report_path}")

        return report

    def generate_qc_report(self, output_dir: Path):
        """Generate quality control report."""
        lines = []
        lines.append("QUALITY CONTROL REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Files without enrichment
        unenriched = [f for f in self.processed_files if not f.enriched]
        if unenriched:
            lines.append(f"⚠️  Files Without Audio Features: {len(unenriched)}")
            for file_meta in unenriched[:10]:
                lines.append(f"  - {file_meta.filename}")
            if len(unenriched) > 10:
                lines.append(f"  ... and {len(unenriched) - 10} more")
            lines.append("")

        # Very short files
        short_files = [f for f in self.processed_files if f.duration < 10.0]
        if short_files:
            lines.append(f"⚠️  Very Short Files (<10s): {len(short_files)}")
            for file_meta in short_files[:10]:
                lines.append(f"  - {file_meta.filename}: {file_meta.duration:.1f}s")
            lines.append("")

        # Files with few segments
        few_segments = [f for f in self.processed_files if f.segment_count < 5]
        if few_segments:
            lines.append(f"⚠️  Files with Few Segments (<5): {len(few_segments)}")
            for file_meta in few_segments[:10]:
                lines.append(f"  - {file_meta.filename}: {file_meta.segment_count} segments")
            lines.append("")

        if not unenriched and not short_files and not few_segments:
            lines.append("✅ No quality issues detected!")
            lines.append("")

        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save report
        report_path = output_dir / "reports" / "quality_control.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Saved QC report: {report_path}")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Batch Research Data Processing Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all transcripts in directory
  python batch_research_processing.py --input-dir whisper_json --output-dir outputs/batch

  # Extract metadata from filenames with pattern
  python batch_research_processing.py --input-dir whisper_json --output-dir outputs/batch \\
      --pattern "{condition}_{subject_id}_{trial}.json"

  # Quality control only (no reprocessing)
  python batch_research_processing.py --output-dir outputs/batch --qc-only
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("whisper_json"),
        help="Directory containing JSON transcripts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/research_batch"),
        help="Output directory for aggregated data",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help='Filename pattern for metadata extraction (e.g., "{study}_{id}.json")',
    )
    parser.add_argument(
        "--qc-only", action="store_true", help="Only generate QC report, skip processing"
    )

    args = parser.parse_args()

    # Create configuration
    config = BatchConfig()
    if args.pattern:
        config.filename_pattern = args.pattern

    # Create processor
    processor = BatchProcessor(config)

    # Process files
    if not args.qc_only:
        if not args.input_dir.exists():
            logger.error(f"Input directory not found: {args.input_dir}")
            sys.exit(1)

        processor.process_directory(args.input_dir, args.output_dir)

        # Generate outputs
        if config.generate_aggregated_csv and processor.processed_files:
            processor.generate_aggregated_csv(args.output_dir)

    # Generate reports
    if config.generate_statistics_report and processor.processed_files:
        report = processor.generate_statistics_report(args.output_dir)
        print("\n" + report + "\n")

    if config.generate_qc_report and processor.processed_files:
        qc_report = processor.generate_qc_report(args.output_dir)
        print("\n" + qc_report + "\n")

    logger.info("Batch processing complete!")


if __name__ == "__main__":
    main()
