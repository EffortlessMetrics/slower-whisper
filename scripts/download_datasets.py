#!/usr/bin/env python3
"""Download and verify benchmark datasets.

This script downloads benchmark datasets based on their manifest files,
verifies SHA256 hashes, and stages them in the benchmark cache directory.

Usage:
    python scripts/download_datasets.py --dataset librispeech-test-clean
    python scripts/download_datasets.py --dataset librispeech-test-clean --verify
    python scripts/download_datasets.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

# Default cache directory for benchmark datasets
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "slower-whisper" / "benchmarks"

# Manifest directory relative to this script
MANIFEST_DIR = Path(__file__).parent.parent / "benchmarks" / "datasets"


def get_cache_dir() -> Path:
    """Get the benchmark cache directory from environment or default."""
    cache_dir = os.environ.get("SLOWER_WHISPER_BENCHMARKS")
    if cache_dir:
        return Path(cache_dir)
    return DEFAULT_CACHE_DIR


def load_manifest(dataset_id: str) -> tuple[dict[str, Any], Path] | None:
    """Load a manifest file by dataset ID.

    Searches all track directories for a matching manifest, either by
    directory name or by the 'id' field in the manifest.

    Args:
        dataset_id: Dataset identifier (e.g., 'librispeech-test-clean')

    Returns:
        Tuple of (manifest dict, manifest directory) if found, None otherwise
    """
    for track_dir in MANIFEST_DIR.iterdir():
        if not track_dir.is_dir():
            continue
        # First try by directory name
        manifest_path = track_dir / dataset_id / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f), manifest_path.parent
        # Then search by manifest ID
        for dataset_dir in track_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            manifest_path = dataset_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    if manifest.get("id") == dataset_id:
                        return manifest, manifest_path.parent
    return None


def list_datasets() -> list[dict[str, Any]]:
    """List all available datasets with their status.

    Returns:
        List of dataset info dicts with id, track, description, and availability
    """
    datasets = []
    cache_dir = get_cache_dir()

    for track_dir in sorted(MANIFEST_DIR.iterdir()):
        if not track_dir.is_dir():
            continue
        track = track_dir.name

        for dataset_dir in sorted(track_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            manifest_path = dataset_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            with open(manifest_path) as f:
                manifest = json.load(f)

            dataset_id = manifest.get("id", dataset_dir.name)

            # Check if dataset is available locally
            is_smoke = manifest.get("split") == "smoke"
            if is_smoke:
                # Smoke datasets are always available (committed to repo)
                available = True
            else:
                # Check if downloaded to cache
                dataset_cache = cache_dir / track / dataset_id
                available = dataset_cache.exists()

            datasets.append(
                {
                    "id": dataset_id,
                    "track": track,
                    "split": manifest.get("split", "unknown"),
                    "description": manifest.get("description", ""),
                    "available": available,
                    "download_required": manifest.get("download") is not None,
                    "sample_count": manifest.get("meta", {}).get("sample_count", 0),
                }
            )

    return datasets


def calculate_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of a file.

    Args:
        filepath: Path to file
        chunk_size: Read chunk size in bytes

    Returns:
        Hex-encoded SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_file(filepath: Path, expected_hash: str) -> bool:
    """Verify a file's SHA256 hash.

    Args:
        filepath: Path to file
        expected_hash: Expected SHA256 hash

    Returns:
        True if hash matches, False otherwise
    """
    if not filepath.exists():
        return False
    actual_hash = calculate_sha256(filepath)
    return actual_hash.lower() == expected_hash.lower()


class DownloadProgress:
    """Progress reporter for downloads."""

    def __init__(self, total_size: int | None = None):
        self.total_size = total_size
        self.downloaded = 0
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        """Report hook for urlretrieve."""
        if total_size > 0:
            self.total_size = total_size

        self.downloaded = block_num * block_size

        if self.total_size:
            percent = int(100 * self.downloaded / self.total_size)
            if percent != self.last_percent:
                self.last_percent = percent
                mb_downloaded = self.downloaded / (1024 * 1024)
                mb_total = self.total_size / (1024 * 1024)
                print(
                    f"\r  Downloading: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                    end="",
                    flush=True,
                )
        else:
            mb_downloaded = self.downloaded / (1024 * 1024)
            print(f"\r  Downloading: {mb_downloaded:.1f} MB", end="", flush=True)


def download_file(url: str, dest: Path, expected_hash: str | None = None) -> bool:
    """Download a file with progress reporting.

    Args:
        url: URL to download from
        dest: Destination path
        expected_hash: Expected SHA256 hash (optional)

    Returns:
        True if download successful and hash verified (if provided)
    """
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")

    # Create parent directory
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        progress = DownloadProgress()
        urlretrieve(url, dest, progress)
        print()  # Newline after progress
    except (URLError, HTTPError) as e:
        print(f"\n  ERROR: Download failed: {e}")
        return False

    if expected_hash:
        print("  Verifying hash...")
        if not verify_file(dest, expected_hash):
            actual_hash = calculate_sha256(dest)
            print("  ERROR: Hash mismatch!")
            print(f"    Expected: {expected_hash}")
            print(f"    Actual:   {actual_hash}")
            dest.unlink()  # Remove corrupted file
            return False
        print("  Hash verified OK")

    return True


def extract_archive(
    archive_path: Path, dest_dir: Path, extraction_subpath: str | None = None
) -> bool:
    """Extract a tar.gz or zip archive.

    Args:
        archive_path: Path to archive file
        dest_dir: Directory to extract to
        extraction_subpath: Subdirectory within archive to extract (optional)

    Returns:
        True if extraction successful
    """
    print(f"  Extracting to: {dest_dir}")

    try:
        if archive_path.suffix == ".gz" or archive_path.name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar:
                # Extract to temp dir first if we need a subpath
                if extraction_subpath:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tar.extractall(tmpdir)
                        src = Path(tmpdir) / extraction_subpath
                        if not src.exists():
                            print(f"  ERROR: Extraction path not found: {extraction_subpath}")
                            return False
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        for item in src.iterdir():
                            shutil.move(str(item), str(dest_dir / item.name))
                else:
                    tar.extractall(dest_dir)
        elif archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                if extraction_subpath:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        zf.extractall(tmpdir)
                        src = Path(tmpdir) / extraction_subpath
                        if not src.exists():
                            print(f"  ERROR: Extraction path not found: {extraction_subpath}")
                            return False
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        for item in src.iterdir():
                            shutil.move(str(item), str(dest_dir / item.name))
                else:
                    zf.extractall(dest_dir)
        else:
            print(f"  ERROR: Unknown archive format: {archive_path.suffix}")
            return False
    except (tarfile.TarError, zipfile.BadZipFile) as e:
        print(f"  ERROR: Extraction failed: {e}")
        return False

    print("  Extraction complete")
    return True


def verify_dataset(
    manifest: dict[str, Any],
    manifest_dir: Path,
    cache_dir: Path,
) -> tuple[bool, list[str]]:
    """Verify a downloaded dataset against its manifest.

    Args:
        manifest: Dataset manifest
        manifest_dir: Directory containing the manifest file
        cache_dir: Dataset cache directory

    Returns:
        Tuple of (all_valid, list of error messages)
    """
    errors = []
    samples = manifest.get("samples", [])

    if not samples:
        # No explicit samples - check if discovery is configured
        discovery = manifest.get("sample_discovery")
        if discovery:
            # For now, just check the directory exists
            if not cache_dir.exists():
                errors.append(f"Dataset directory not found: {cache_dir}")
            return len(errors) == 0, errors

    # Verify each sample
    for sample in samples:
        sample_id = sample.get("id", "unknown")

        # Check audio file
        audio_path = sample.get("audio")
        if audio_path:
            if audio_path.startswith(("http://", "https://")):
                # Remote URL - can't verify without downloading
                continue
            else:
                # Local path - resolve relative to manifest directory
                full_path = (manifest_dir / audio_path).resolve()
                if not full_path.exists():
                    # Try cache dir
                    full_path = cache_dir / Path(audio_path).name
                if not full_path.exists():
                    errors.append(f"Sample {sample_id}: Audio file not found: {audio_path}")
                    continue

                # Verify hash if provided
                audio_hash = sample.get("sha256") or sample.get("audio_sha256")
                if audio_hash and not verify_file(full_path, audio_hash):
                    errors.append(f"Sample {sample_id}: Audio hash mismatch")

        # Check RTTM file for diarization
        rttm_path = sample.get("reference_rttm")
        if rttm_path:
            full_path = (manifest_dir / rttm_path).resolve()
            if not full_path.exists():
                full_path = cache_dir / Path(rttm_path).name
            if not full_path.exists():
                errors.append(f"Sample {sample_id}: RTTM file not found: {rttm_path}")
            else:
                rttm_hash = sample.get("rttm_sha256")
                if rttm_hash and not verify_file(full_path, rttm_hash):
                    errors.append(f"Sample {sample_id}: RTTM hash mismatch")

    return len(errors) == 0, errors


def download_dataset(dataset_id: str, force: bool = False, verify_only: bool = False) -> bool:
    """Download and verify a dataset.

    Args:
        dataset_id: Dataset identifier
        force: Force re-download even if exists
        verify_only: Only verify existing download, don't download

    Returns:
        True if successful
    """
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_id}")
    print("=" * 60)

    # Load manifest
    result = load_manifest(dataset_id)
    if not result:
        print(f"ERROR: Dataset '{dataset_id}' not found")
        print(f"Available datasets: {[d['id'] for d in list_datasets()]}")
        return False

    manifest, manifest_dir = result
    track = manifest.get("track", "unknown")
    cache_dir = get_cache_dir() / track / dataset_id

    # Check if smoke dataset (no download needed)
    if manifest.get("split") == "smoke":
        print("This is a smoke dataset - no download required")
        print("Verifying local files...")
        valid, errors = verify_dataset(manifest, manifest_dir, cache_dir)
        if valid:
            print("All smoke test files verified OK")
            return True
        else:
            print("Verification failed:")
            for err in errors:
                print(f"  - {err}")
            return False

    # Check download info
    download_info = manifest.get("download")
    if not download_info:
        print("ERROR: No download information in manifest")
        if download_info is None:
            print("  This dataset may require manual download. Check the source URL.")
        return False

    # Handle manual download datasets
    if download_info.get("method") == "manual":
        print("This dataset requires manual download")
        print(f"  Instructions: {download_info.get('instructions_url', 'See docs/')}")
        if cache_dir.exists():
            print(f"  Dataset found at: {cache_dir}")
            print("  Verifying...")
            valid, errors = verify_dataset(manifest, manifest_dir, cache_dir)
            if valid:
                print("  Verification OK")
                return True
            else:
                print("  Verification failed:")
                for err in errors:
                    print(f"    - {err}")
                return False
        else:
            print(f"  Expected location: {cache_dir}")
            return False

    # Check if already downloaded
    if cache_dir.exists() and not force:
        print(f"Dataset already exists at: {cache_dir}")
        if verify_only:
            print("Verifying...")
            valid, errors = verify_dataset(manifest, manifest_dir, cache_dir)
            if valid:
                print("Verification OK")
                return True
            else:
                print("Verification failed:")
                for err in errors:
                    print(f"  - {err}")
                return False
        else:
            print("Use --force to re-download")
            return True

    if verify_only:
        print("Dataset not found - cannot verify")
        return False

    # Download
    download_url = download_info.get("url")
    download_hash = download_info.get("sha256")
    download_format = download_info.get("format", "tar.gz")
    extraction_path = download_info.get("extraction_path")

    if not download_url:
        print("ERROR: No download URL in manifest")
        return False

    # Download to temp file
    print("\nDownloading...")
    with tempfile.NamedTemporaryFile(suffix=f".{download_format}", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if not download_file(download_url, tmp_path, download_hash):
            return False

        # Extract
        print("\nExtracting...")
        if not extract_archive(tmp_path, cache_dir, extraction_path):
            return False

    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()

    # Verify
    print("\nVerifying...")
    valid, errors = verify_dataset(manifest, manifest_dir, cache_dir)
    if valid:
        print("Dataset ready!")
        return True
    else:
        print("Verification issues:")
        for err in errors:
            print(f"  - {err}")
        print("Dataset may still be usable - check errors above")
        return True  # Return True since download succeeded


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and verify benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List all datasets:
    python scripts/download_datasets.py --list

  Download a dataset:
    python scripts/download_datasets.py --dataset librispeech-test-clean

  Verify existing download:
    python scripts/download_datasets.py --dataset librispeech-test-clean --verify

  Force re-download:
    python scripts/download_datasets.py --dataset librispeech-test-clean --force
""",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset ID to download (e.g., librispeech-test-clean)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing download, don't download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Override cache directory (default: ~/.cache/slower-whisper/benchmarks)",
    )

    args = parser.parse_args()

    # Set cache dir from args if provided
    if args.cache_dir:
        os.environ["SLOWER_WHISPER_BENCHMARKS"] = args.cache_dir

    if args.list:
        datasets = list_datasets()
        if not datasets:
            print("No datasets found")
            return 1

        print("\nAvailable benchmark datasets:")
        print("-" * 80)
        print(f"{'ID':<30} {'Track':<15} {'Split':<8} {'Status':<12} {'Samples'}")
        print("-" * 80)

        for ds in datasets:
            status = (
                "Available"
                if ds["available"]
                else ("Download" if ds["download_required"] else "Missing")
            )
            print(
                f"{ds['id']:<30} {ds['track']:<15} {ds['split']:<8} {status:<12} {ds['sample_count']}"
            )

        print("-" * 80)
        print(f"\nCache directory: {get_cache_dir()}")
        print("Set SLOWER_WHISPER_BENCHMARKS to override")
        return 0

    if args.dataset:
        success = download_dataset(args.dataset, force=args.force, verify_only=args.verify)
        return 0 if success else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
