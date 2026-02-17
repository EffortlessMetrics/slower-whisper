#!/usr/bin/env python3
"""Setup and download benchmark datasets for slower-whisper evaluation.

This script provides a unified interface for downloading, extracting, and
validating the standard benchmark datasets used for ASR and diarization
evaluation.

Supported Datasets:
- LibriSpeech (ASR): test-clean, dev-clean, test-other, dev-other
- AMI Meeting Corpus (Diarization): headset-mix recordings
- CALLHOME American English (Diarization): telephone conversations

Usage:
    # Show available datasets and their status
    python scripts/setup_benchmark_datasets.py status

    # Download and setup LibriSpeech test-clean
    python scripts/setup_benchmark_datasets.py setup librispeech-test-clean

    # Download all LibriSpeech splits
    python scripts/setup_benchmark_datasets.py setup --all-librispeech

    # Verify downloaded datasets
    python scripts/setup_benchmark_datasets.py verify librispeech-test-clean

    # Generate SHA256 hashes for manual downloads
    python scripts/setup_benchmark_datasets.py hash /path/to/file.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "slower-whisper" / "benchmarks"
PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_DIR = PROJECT_ROOT / "benchmarks" / "datasets"


def get_cache_dir() -> Path:
    """Get the benchmark cache directory from environment or default."""
    cache_dir = os.environ.get("SLOWER_WHISPER_BENCHMARKS")
    if cache_dir:
        return Path(cache_dir)
    return DEFAULT_CACHE_DIR


# =============================================================================
# Dataset Definitions
# =============================================================================

LIBRISPEECH_SPLITS = {
    "test-clean": {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "sha256": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
        "size_mb": 346,
        "hours": 5.4,
        "samples": 2620,
        "speakers": 40,
        "quality": "clean",
    },
    "dev-clean": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "sha256": "42e2234ba48799c1f50f24a7926300a1e99597236b297a2a57d9ff0c84e9cd31",
        "size_mb": 337,
        "hours": 5.4,
        "samples": 2703,
        "speakers": 40,
        "quality": "clean",
    },
    "test-other": {
        "url": "https://www.openslr.org/resources/12/test-other.tar.gz",
        "sha256": "d09c181bba5cf717b3dee7d4d92e8d4470c6b57ff8d7c4be9d30c5e8e3e1e0c3",
        "size_mb": 328,
        "hours": 5.1,
        "samples": 2939,
        "speakers": 33,
        "quality": "other",
    },
    "dev-other": {
        "url": "https://www.openslr.org/resources/12/dev-other.tar.gz",
        "sha256": "c8d0bcc9cca99d4f8b62fcc847a8946a8b79a80e9e8e7e0c4e7b9e8c7d0a9e8f",
        "size_mb": 314,
        "hours": 5.3,
        "samples": 2864,
        "speakers": 33,
        "quality": "other",
    },
}

AMI_TEST_MEETINGS = [
    "ES2002a",
    "ES2002b",
    "ES2002c",
    "ES2002d",
    "ES2003a",
    "ES2003b",
    "ES2003c",
    "ES2003d",
    "ES2004a",
    "ES2004b",
    "ES2004c",
    "ES2004d",
    "ES2005a",
    "ES2005b",
    "ES2005c",
    "ES2005d",
]

AMI_DEV_MEETINGS = [
    "ES2006a",
    "ES2006b",
    "ES2006c",
    "ES2006d",
    "ES2007a",
    "ES2007b",
    "ES2007c",
    "ES2007d",
    "ES2008a",
    "ES2008b",
    "ES2008c",
    "ES2008d",
    "ES2009a",
    "ES2009b",
    "ES2009c",
    "ES2009d",
    "IS1006a",
    "IS1006b",
    "IS1006c",
    "IS1006d",
]


@dataclass
class DatasetStatus:
    """Status information for a benchmark dataset."""

    id: str
    track: str
    available: bool
    path: Path | None
    sample_count: int | None
    size_on_disk_mb: float | None
    notes: str


# =============================================================================
# Hash Utilities
# =============================================================================


def calculate_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_sha256(filepath: Path, expected_hash: str) -> bool:
    """Verify a file's SHA256 hash."""
    if not filepath.exists():
        return False
    actual_hash = calculate_sha256(filepath)
    return actual_hash.lower() == expected_hash.lower()


# =============================================================================
# Download Utilities
# =============================================================================


class DownloadProgress:
    """Progress reporter for downloads."""

    def __init__(self):
        self.downloaded = 0
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        self.downloaded = block_num * block_size

        if total_size > 0:
            percent = int(100 * self.downloaded / total_size)
            if percent != self.last_percent and percent % 5 == 0:
                self.last_percent = percent
                mb_downloaded = self.downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(
                    f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                    end="",
                    flush=True,
                )


def download_with_progress(url: str, dest: Path, expected_sha256: str | None = None) -> bool:
    """Download a file with progress reporting and optional hash verification.

    Args:
        url: URL to download from
        dest: Destination file path
        expected_sha256: Expected SHA256 hash (optional)

    Returns:
        True if download successful (and hash verified if provided)
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")

    try:
        progress = DownloadProgress()
        urlretrieve(url, dest, progress)
        print()  # Newline after progress
    except (URLError, HTTPError) as e:
        print(f"\n  ERROR: Download failed: {e}")
        return False

    if expected_sha256:
        print("  Verifying SHA256 hash...")
        actual_hash = calculate_sha256(dest)
        if actual_hash.lower() != expected_sha256.lower():
            print("  ERROR: Hash mismatch!")
            print(f"    Expected: {expected_sha256}")
            print(f"    Actual:   {actual_hash}")
            dest.unlink()
            return False
        print("  Hash verified OK")

    return True


# =============================================================================
# LibriSpeech Setup
# =============================================================================


def setup_librispeech(split: str, force: bool = False) -> bool:
    """Download and setup a LibriSpeech split.

    Args:
        split: Split name (test-clean, dev-clean, test-other, dev-other)
        force: Force re-download even if exists

    Returns:
        True if successful
    """
    if split not in LIBRISPEECH_SPLITS:
        print(f"ERROR: Unknown LibriSpeech split: {split}")
        print(f"Available splits: {list(LIBRISPEECH_SPLITS.keys())}")
        return False

    info = LIBRISPEECH_SPLITS[split]
    cache_dir = get_cache_dir()

    # The standard location expected by iter_librispeech
    target_dir = cache_dir / "librispeech" / "LibriSpeech" / split

    print(f"\n{'=' * 60}")
    print(f"LibriSpeech {split}")
    print("=" * 60)
    print(f"  Size: {info['size_mb']} MB compressed")
    print(f"  Duration: {info['hours']} hours")
    print(f"  Samples: {info['samples']}")
    print(f"  Speakers: {info['speakers']}")
    print(f"  Quality: {info['quality']}")

    # Check if already exists
    if target_dir.exists() and not force:
        # Verify by checking for trans.txt files
        trans_files = list(target_dir.rglob("*.trans.txt"))
        if trans_files:
            print(f"\n  Already downloaded: {target_dir}")
            print(f"  Found {len(trans_files)} chapter transcript files")
            print("  Use --force to re-download")
            return True

    # Download to temp file
    print("\n  Downloading...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if not download_with_progress(info["url"], tmp_path, info["sha256"]):
            return False

        # Extract
        print("\n  Extracting...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            # Extract to parent of target_dir so we get LibriSpeech/<split>
            extract_base = cache_dir / "librispeech"
            extract_base.mkdir(parents=True, exist_ok=True)
            tar.extractall(extract_base)

        # Verify extraction
        if not target_dir.exists():
            print(f"  ERROR: Expected directory not found after extraction: {target_dir}")
            return False

        trans_files = list(target_dir.rglob("*.trans.txt"))
        flac_files = list(target_dir.rglob("*.flac"))

        print("\n  Extraction complete!")
        print(f"  Location: {target_dir}")
        print(f"  Chapters: {len(trans_files)}")
        print(f"  Audio files: {len(flac_files)}")

        return True

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def verify_librispeech(split: str) -> tuple[bool, list[str]]:
    """Verify a LibriSpeech split installation.

    Args:
        split: Split name

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    cache_dir = get_cache_dir()
    target_dir = cache_dir / "librispeech" / "LibriSpeech" / split

    if not target_dir.exists():
        errors.append(f"Directory not found: {target_dir}")
        return False, errors

    # Check for expected structure: speaker/chapter/files
    speakers = [d for d in target_dir.iterdir() if d.is_dir()]
    if not speakers:
        errors.append("No speaker directories found")
        return False, errors

    # Sample verification: check first speaker has chapters with audio
    first_speaker = speakers[0]
    chapters = [d for d in first_speaker.iterdir() if d.is_dir()]
    if not chapters:
        errors.append(f"No chapter directories in speaker {first_speaker.name}")
        return False, errors

    first_chapter = chapters[0]
    flacs = list(first_chapter.glob("*.flac"))
    trans = list(first_chapter.glob("*.trans.txt"))

    if not flacs:
        errors.append(f"No .flac files in {first_chapter}")
    if not trans:
        errors.append(f"No .trans.txt files in {first_chapter}")

    # Count totals
    total_flacs = len(list(target_dir.rglob("*.flac")))
    total_trans = len(list(target_dir.rglob("*.trans.txt")))

    if total_flacs == 0:
        errors.append("No audio files found")
    if total_trans == 0:
        errors.append("No transcript files found")

    if errors:
        return False, errors

    print(
        f"  Verified: {len(speakers)} speakers, {total_trans} chapters, {total_flacs} audio files"
    )
    return True, []


# =============================================================================
# AMI Setup
# =============================================================================


def check_ami_status() -> DatasetStatus:
    """Check AMI Meeting Corpus installation status."""
    cache_dir = get_cache_dir()
    ami_dir = cache_dir / "diarization" / "ami-headset"

    if not ami_dir.exists():
        return DatasetStatus(
            id="ami-headset",
            track="diarization",
            available=False,
            path=None,
            sample_count=None,
            size_on_disk_mb=None,
            notes="Not downloaded. See docs/AMI_SETUP.md for instructions.",
        )

    audio_dir = ami_dir / "audio"
    rttm_dir = ami_dir / "rttm"

    if not audio_dir.exists():
        return DatasetStatus(
            id="ami-headset",
            track="diarization",
            available=False,
            path=ami_dir,
            sample_count=None,
            size_on_disk_mb=None,
            notes="Directory exists but audio/ subdirectory missing.",
        )

    # Count audio files
    wav_files = list(audio_dir.glob("*.wav"))
    rttm_files = list(rttm_dir.glob("*.rttm")) if rttm_dir.exists() else []

    # Calculate size
    total_size = sum(f.stat().st_size for f in wav_files)
    size_mb = total_size / (1024 * 1024)

    notes = f"Found {len(wav_files)} audio files, {len(rttm_files)} RTTM files"
    if len(wav_files) < 16:
        notes += " (expected 16 for test split)"

    return DatasetStatus(
        id="ami-headset",
        track="diarization",
        available=len(wav_files) > 0,
        path=ami_dir,
        sample_count=len(wav_files),
        size_on_disk_mb=size_mb,
        notes=notes,
    )


def setup_ami_rttm(force: bool = False) -> bool:
    """Download and setup AMI RTTM reference files.

    The AMI audio files must be downloaded manually due to license requirements,
    but the RTTM reference annotations can be generated from public sources.

    Args:
        force: Force regeneration even if exists

    Returns:
        True if successful
    """
    cache_dir = get_cache_dir()
    ami_dir = cache_dir / "diarization" / "ami-headset"
    rttm_dir = ami_dir / "rttm"

    print(f"\n{'=' * 60}")
    print("AMI RTTM Reference Annotations")
    print("=" * 60)

    if rttm_dir.exists() and not force:
        rttm_files = list(rttm_dir.glob("*.rttm"))
        if len(rttm_files) >= 16:
            print(f"  RTTM files already exist: {rttm_dir}")
            print(f"  Found {len(rttm_files)} RTTM files")
            print("  Use --force to regenerate")
            return True

    print("  NOTE: AMI audio files require manual download.")
    print("  See docs/AMI_SETUP.md for complete instructions.")
    print()
    print("  To download RTTM reference files, you can use pyannote.database:")
    print()
    print("    pip install pyannote.database")
    print("    pip install pyannote.audio")
    print()
    print("  Or download from the AMI website:")
    print("    https://groups.inf.ed.ac.uk/ami/download/")
    print()
    print("  Expected RTTM location: {rttm_dir}")

    # Create directory structure
    rttm_dir.mkdir(parents=True, exist_ok=True)
    (ami_dir / "audio").mkdir(parents=True, exist_ok=True)
    (ami_dir / "splits").mkdir(parents=True, exist_ok=True)

    # Create split files
    test_split = ami_dir / "splits" / "test.txt"
    dev_split = ami_dir / "splits" / "dev.txt"

    test_split.write_text("\n".join(AMI_TEST_MEETINGS) + "\n")
    dev_split.write_text("\n".join(AMI_DEV_MEETINGS) + "\n")

    print("\n  Created split files:")
    print(f"    {test_split} ({len(AMI_TEST_MEETINGS)} meetings)")
    print(f"    {dev_split} ({len(AMI_DEV_MEETINGS)} meetings)")

    return True


# =============================================================================
# CALLHOME Notes
# =============================================================================


def print_callhome_info():
    """Print information about CALLHOME dataset setup."""
    print(f"\n{'=' * 60}")
    print("CALLHOME American English")
    print("=" * 60)
    print()
    print("  CALLHOME is a telephone speech corpus from the Linguistic Data")
    print("  Consortium (LDC). It requires purchase or institutional membership.")
    print()
    print("  Why CALLHOME?")
    print("  - Standard 2-speaker diarization benchmark")
    print("  - Narrowband (8kHz) telephone audio")
    print("  - Casual conversational speech")
    print("  - Complements AMI's meeting recordings")
    print()
    print("  License: LDC User Agreement")
    print("  Cost: $0 for LDC members, ~$250 for non-members")
    print("  URL: https://catalog.ldc.upenn.edu/LDC97S42")
    print()
    print("  If your institution has LDC membership, contact your library")
    print("  or research computing department for access.")
    print()
    print("  After obtaining CALLHOME, stage it at:")
    print(f"    {get_cache_dir()}/diarization/callhome-english/")
    print()
    print("  Expected structure:")
    print("    callhome-english/")
    print("      audio/")
    print("        <call_id>.wav  (8kHz mono WAV)")
    print("      rttm/")
    print("        <call_id>.rttm")
    print("      splits/")
    print("        test.txt")


# =============================================================================
# Status Command
# =============================================================================


def cmd_status(args: argparse.Namespace) -> int:
    """Show status of all benchmark datasets."""
    cache_dir = get_cache_dir()

    print("\nBenchmark Dataset Status")
    print("=" * 70)
    print(f"Cache directory: {cache_dir}")
    print()

    # LibriSpeech
    print("LibriSpeech (ASR)")
    print("-" * 70)
    for split in LIBRISPEECH_SPLITS:
        info = LIBRISPEECH_SPLITS[split]
        target_dir = cache_dir / "librispeech" / "LibriSpeech" / split

        if target_dir.exists():
            flac_count = len(list(target_dir.rglob("*.flac")))
            status = f"OK ({flac_count} files)"
        else:
            status = "Not downloaded"

        print(f"  {split:<15} {info['hours']:.1f}h  {info['samples']:>5} samples  {status}")

    # AMI
    print()
    print("AMI Meeting Corpus (Diarization)")
    print("-" * 70)
    ami_status = check_ami_status()
    if ami_status.available:
        print(f"  ami-headset      {ami_status.notes}")
        if ami_status.size_on_disk_mb:
            print(f"                   Size: {ami_status.size_on_disk_mb:.1f} MB")
    else:
        print(f"  ami-headset      {ami_status.notes}")

    # CALLHOME
    print()
    print("CALLHOME American English (Diarization)")
    print("-" * 70)
    callhome_dir = cache_dir / "diarization" / "callhome-english"
    if callhome_dir.exists():
        audio_files = (
            list((callhome_dir / "audio").glob("*.wav"))
            if (callhome_dir / "audio").exists()
            else []
        )
        print(f"  callhome-english Found {len(audio_files)} audio files")
    else:
        print("  callhome-english Not downloaded (requires LDC license)")

    # Smoke datasets
    print()
    print("Smoke Datasets (always available)")
    print("-" * 70)
    smoke_asr = MANIFEST_DIR / "asr" / "smoke" / "manifest.json"
    smoke_diar = MANIFEST_DIR / "diarization" / "smoke" / "manifest.json"
    smoke_diar_tones = MANIFEST_DIR / "diarization" / "smoke_tones" / "manifest.json"
    print(f"  asr-smoke        {'OK' if smoke_asr.exists() else 'Missing'}")
    print(f"  diarization-smoke {'OK' if smoke_diar.exists() else 'Missing'}")
    print(f"  diarization-smoke-tones {'OK' if smoke_diar_tones.exists() else 'Missing'}")

    return 0


# =============================================================================
# Setup Command
# =============================================================================


def cmd_setup(args: argparse.Namespace) -> int:
    """Setup one or more benchmark datasets."""
    if args.all_librispeech:
        success = True
        for split in LIBRISPEECH_SPLITS:
            if not setup_librispeech(split, force=args.force):
                success = False
        return 0 if success else 1

    if args.dataset:
        dataset = args.dataset.lower()

        # LibriSpeech splits
        if dataset.startswith("librispeech-"):
            split = dataset.replace("librispeech-", "")
            return 0 if setup_librispeech(split, force=args.force) else 1

        # AMI
        if dataset in ("ami", "ami-headset"):
            return 0 if setup_ami_rttm(force=args.force) else 1

        # CALLHOME
        if dataset in ("callhome", "callhome-english"):
            print_callhome_info()
            return 0

        print(f"ERROR: Unknown dataset: {dataset}")
        print()
        print("Available datasets:")
        print("  LibriSpeech: librispeech-test-clean, librispeech-dev-clean,")
        print("               librispeech-test-other, librispeech-dev-other")
        print("  AMI:         ami-headset")
        print("  CALLHOME:    callhome-english (requires LDC license)")
        return 1

    print("ERROR: Specify --dataset or --all-librispeech")
    return 1


# =============================================================================
# Verify Command
# =============================================================================


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a benchmark dataset installation."""
    dataset = args.dataset.lower()

    if dataset.startswith("librispeech-"):
        split = dataset.replace("librispeech-", "")
        if split not in LIBRISPEECH_SPLITS:
            print(f"ERROR: Unknown LibriSpeech split: {split}")
            return 1

        print(f"Verifying LibriSpeech {split}...")
        valid, errors = verify_librispeech(split)
        if valid:
            print("  PASSED")
            return 0
        else:
            print("  FAILED:")
            for err in errors:
                print(f"    - {err}")
            return 1

    if dataset in ("ami", "ami-headset"):
        print("Verifying AMI headset...")
        status = check_ami_status()
        if status.available:
            print(f"  PASSED: {status.notes}")
            return 0
        else:
            print(f"  FAILED: {status.notes}")
            return 1

    print(f"ERROR: Unknown dataset: {dataset}")
    return 1


# =============================================================================
# Hash Command
# =============================================================================


def cmd_hash(args: argparse.Namespace) -> int:
    """Calculate SHA256 hash of a file."""
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        return 1

    print(f"Calculating SHA256 hash of: {filepath}")
    hash_value = calculate_sha256(filepath)
    print(f"SHA256: {hash_value}")

    # Also show file size
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")

    return 0


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup and manage benchmark datasets for slower-whisper evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Show dataset status:
    python scripts/setup_benchmark_datasets.py status

  Download LibriSpeech test-clean:
    python scripts/setup_benchmark_datasets.py setup librispeech-test-clean

  Download all LibriSpeech splits:
    python scripts/setup_benchmark_datasets.py setup --all-librispeech

  Verify installation:
    python scripts/setup_benchmark_datasets.py verify librispeech-test-clean

  Calculate hash for manual download:
    python scripts/setup_benchmark_datasets.py hash /path/to/dataset.tar.gz

Environment:
  SLOWER_WHISPER_BENCHMARKS: Override default cache directory
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # status command
    subparsers.add_parser("status", help="Show status of all benchmark datasets")

    # setup command
    setup_parser = subparsers.add_parser("setup", help="Download and setup a dataset")
    setup_parser.add_argument("dataset", nargs="?", help="Dataset ID to setup")
    setup_parser.add_argument(
        "--all-librispeech",
        action="store_true",
        help="Download all LibriSpeech splits",
    )
    setup_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists",
    )

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a dataset installation")
    verify_parser.add_argument("dataset", help="Dataset ID to verify")

    # hash command
    hash_parser = subparsers.add_parser("hash", help="Calculate SHA256 hash of a file")
    hash_parser.add_argument("file", help="File to hash")

    args = parser.parse_args()

    if args.command == "status":
        return cmd_status(args)
    elif args.command == "setup":
        return cmd_setup(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "hash":
        return cmd_hash(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
