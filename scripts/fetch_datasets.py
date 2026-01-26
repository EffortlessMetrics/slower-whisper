#!/usr/bin/env python3
"""Fetch and manage benchmark datasets using manifest infrastructure.

This script provides a unified interface for downloading, verifying, and
managing benchmark datasets based on their manifest files.

Features:
- Download datasets by name from manifest
- Verify checksums for integrity
- Show licensing information
- Support partial downloads (smoke vs full)
- Validate manifests against schema

Usage:
    # List all available datasets
    python scripts/fetch_datasets.py list

    # Fetch smoke datasets only (for CI)
    python scripts/fetch_datasets.py fetch --smoke

    # Fetch a specific dataset
    python scripts/fetch_datasets.py fetch --dataset librispeech-test-clean

    # Verify dataset integrity
    python scripts/fetch_datasets.py verify --dataset asr-smoke

    # Show dataset license info
    python scripts/fetch_datasets.py license --dataset commonvoice_en_smoke

    # Validate manifest schema
    python scripts/fetch_datasets.py validate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "slower-whisper" / "benchmarks"
PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_DIR = PROJECT_ROOT / "benchmarks" / "datasets"
SCHEMA_PATH = PROJECT_ROOT / "benchmarks" / "manifest_schema.json"


def get_cache_dir() -> Path:
    """Get the benchmark cache directory from environment or default."""
    cache_dir = os.environ.get("SLOWER_WHISPER_BENCHMARKS")
    if cache_dir:
        return Path(cache_dir)
    return DEFAULT_CACHE_DIR


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class DatasetInfo:
    """Information about a benchmark dataset."""

    id: str
    track: str
    split: str
    description: str
    manifest_path: Path
    manifest: dict[str, Any]
    is_smoke: bool
    is_available: bool
    requires_download: bool
    license_id: str
    license_name: str
    sample_count: int


# ============================================================================
# Manifest Loading
# ============================================================================


def load_all_manifests() -> list[DatasetInfo]:
    """Load all dataset manifests from the manifest directory.

    Returns:
        List of DatasetInfo objects for all discovered manifests
    """
    datasets = []
    cache_dir = get_cache_dir()

    for track_dir in sorted(MANIFEST_DIR.iterdir()):
        if not track_dir.is_dir() or track_dir.name.startswith("."):
            continue

        track = track_dir.name

        for dataset_dir in sorted(track_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            manifest_path = dataset_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in {manifest_path}: {e}", file=sys.stderr)
                continue

            dataset_id = manifest.get("id", dataset_dir.name)
            split = manifest.get("split", "unknown")
            is_smoke = split == "smoke"

            # Determine availability
            if is_smoke:
                # Smoke datasets are committed to repo - check if samples exist
                samples = manifest.get("samples", [])
                if samples:
                    # Check first sample exists
                    first_audio = samples[0].get("audio")
                    if first_audio:
                        audio_path = (manifest_path.parent / first_audio).resolve()
                        is_available = audio_path.exists()
                    else:
                        is_available = True  # No audio to check
                else:
                    is_available = True  # Empty samples list
            else:
                # Full datasets - check cache
                dataset_cache = cache_dir / track / dataset_id
                is_available = dataset_cache.exists()

            download_info = manifest.get("download")
            requires_download = download_info is not None and download_info != {}

            license_info = manifest.get("license", {})
            meta = manifest.get("meta", {})

            datasets.append(
                DatasetInfo(
                    id=dataset_id,
                    track=track,
                    split=split,
                    description=manifest.get("description", ""),
                    manifest_path=manifest_path,
                    manifest=manifest,
                    is_smoke=is_smoke,
                    is_available=is_available,
                    requires_download=requires_download,
                    license_id=license_info.get("id", "Unknown"),
                    license_name=license_info.get("name", "Unknown License"),
                    sample_count=meta.get("sample_count", 0),
                )
            )

    return datasets


def find_dataset(dataset_id: str) -> DatasetInfo | None:
    """Find a dataset by ID.

    Args:
        dataset_id: Dataset identifier

    Returns:
        DatasetInfo if found, None otherwise
    """
    for dataset in load_all_manifests():
        if dataset.id == dataset_id:
            return dataset
    return None


# ============================================================================
# Verification
# ============================================================================


def calculate_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_file(filepath: Path, expected_hash: str) -> bool:
    """Verify a file's SHA256 hash."""
    if not filepath.exists():
        return False
    actual_hash = calculate_sha256(filepath)
    return actual_hash.lower() == expected_hash.lower()


def verify_dataset(dataset: DatasetInfo) -> tuple[bool, list[str]]:
    """Verify a dataset's integrity.

    Args:
        dataset: Dataset to verify

    Returns:
        Tuple of (all_valid, list of error messages)
    """
    errors = []
    samples = dataset.manifest.get("samples", [])

    if not samples:
        # Check sample_source for alternative sample discovery
        sample_source = dataset.manifest.get("sample_source")
        sample_discovery = dataset.manifest.get("sample_discovery")

        if sample_source or sample_discovery:
            # Samples discovered dynamically - check base directory exists
            if dataset.is_smoke:
                # Smoke datasets are in manifest directory
                return True, []
            else:
                cache_dir = get_cache_dir() / dataset.track / dataset.id
                if not cache_dir.exists():
                    errors.append(f"Dataset directory not found: {cache_dir}")
                return len(errors) == 0, errors

        return True, []  # No samples to verify

    # Verify each sample
    verified_count = 0
    for sample in samples:
        sample_id = sample.get("id", "unknown")

        # Verify audio file
        audio_path = sample.get("audio")
        if audio_path:
            full_path = (dataset.manifest_path.parent / audio_path).resolve()
            if not full_path.exists():
                # Try cache dir for downloaded datasets
                cache_path = get_cache_dir() / dataset.track / dataset.id / Path(audio_path).name
                if cache_path.exists():
                    full_path = cache_path
                else:
                    errors.append(f"Sample {sample_id}: Audio not found: {audio_path}")
                    continue

            # Verify hash if provided
            audio_hash = sample.get("sha256") or sample.get("audio_sha256")
            if audio_hash:
                if not verify_file(full_path, audio_hash):
                    actual = calculate_sha256(full_path)
                    errors.append(
                        f"Sample {sample_id}: Hash mismatch. "
                        f"Expected: {audio_hash[:16]}..., Got: {actual[:16]}..."
                    )
                    continue

        # Verify RTTM file (diarization)
        rttm_path = sample.get("reference_rttm")
        if rttm_path:
            full_path = (dataset.manifest_path.parent / rttm_path).resolve()
            if not full_path.exists():
                cache_path = get_cache_dir() / dataset.track / dataset.id / Path(rttm_path).name
                if cache_path.exists():
                    full_path = cache_path
                else:
                    errors.append(f"Sample {sample_id}: RTTM not found: {rttm_path}")
                    continue

            rttm_hash = sample.get("rttm_sha256")
            if rttm_hash and not verify_file(full_path, rttm_hash):
                errors.append(f"Sample {sample_id}: RTTM hash mismatch")
                continue

        verified_count += 1

    if errors:
        print(f"  Verified {verified_count}/{len(samples)} samples")

    return len(errors) == 0, errors


# ============================================================================
# Download
# ============================================================================


class DownloadProgress:
    """Progress reporter for downloads."""

    def __init__(self):
        self.downloaded = 0
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        self.downloaded = block_num * block_size

        if total_size > 0:
            percent = int(100 * self.downloaded / total_size)
            if percent != self.last_percent:
                self.last_percent = percent
                mb_downloaded = self.downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(
                    f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                    end="",
                    flush=True,
                )
        else:
            mb_downloaded = self.downloaded / (1024 * 1024)
            print(f"\r  Downloaded: {mb_downloaded:.1f} MB", end="", flush=True)


def download_file(url: str, dest: Path, expected_hash: str | None = None) -> bool:
    """Download a file with progress reporting.

    Args:
        url: URL to download from
        dest: Destination path
        expected_hash: Expected SHA256 hash (optional)

    Returns:
        True if download successful and hash verified
    """
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
            dest.unlink()
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
        extraction_subpath: Subdirectory within archive to extract

    Returns:
        True if extraction successful
    """
    import shutil

    print(f"  Extracting to: {dest_dir}")

    try:
        if archive_path.suffix == ".gz" or archive_path.name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar:
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
                    dest_dir.mkdir(parents=True, exist_ok=True)
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
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    zf.extractall(dest_dir)
        else:
            print(f"  ERROR: Unknown archive format: {archive_path.suffix}")
            return False
    except (tarfile.TarError, zipfile.BadZipFile) as e:
        print(f"  ERROR: Extraction failed: {e}")
        return False

    print("  Extraction complete")
    return True


def fetch_dataset(dataset: DatasetInfo, force: bool = False) -> bool:
    """Fetch a dataset.

    Args:
        dataset: Dataset to fetch
        force: Force re-download even if exists

    Returns:
        True if successful
    """
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset.id} ({dataset.track})")
    print("=" * 60)

    # Smoke datasets don't need download
    if dataset.is_smoke:
        print("Smoke dataset - no download required")
        print("Verifying local files...")
        valid, errors = verify_dataset(dataset)
        if valid:
            print("All files verified OK")
            return True
        else:
            print("Verification failed:")
            for err in errors:
                print(f"  - {err}")
            return False

    # Check download info
    download_info = dataset.manifest.get("download")
    if not download_info:
        print("ERROR: No download information in manifest")
        print("This dataset may require manual download.")
        return False

    cache_dir = get_cache_dir() / dataset.track / dataset.id

    # Handle different download methods
    method = download_info.get("method", "http")

    if method == "manual":
        print("This dataset requires manual download")
        instructions_url = download_info.get("instructions_url")
        if instructions_url:
            print(f"  Instructions: {instructions_url}")
        print(f"  Expected location: {cache_dir}")
        if cache_dir.exists():
            print("  Dataset found - verifying...")
            valid, errors = verify_dataset(dataset)
            if valid:
                print("  Verification OK")
                return True
            else:
                print("  Verification failed:")
                for err in errors:
                    print(f"    - {err}")
                return False
        return False

    if method == "huggingface":
        print("Hugging Face dataset requires special handling")
        hf_dataset = download_info.get("dataset")
        hf_config = download_info.get("config")
        print(f"  Dataset: {hf_dataset}")
        print(f"  Config: {hf_config}")
        if download_info.get("requires_auth"):
            print("  Note: Requires authentication. Accept terms on Hugging Face first.")
        print("  Use: python benchmarks/scripts/stage_commonvoice.py")
        return False

    # HTTP download
    if cache_dir.exists() and not force:
        print(f"Dataset already exists at: {cache_dir}")
        print("Use --force to re-download")
        print("Verifying...")
        valid, errors = verify_dataset(dataset)
        if valid:
            print("Verification OK")
            return True
        else:
            print("Verification issues:")
            for err in errors:
                print(f"  - {err}")
            return True  # Return True since it exists

    download_url = download_info.get("url")
    download_hash = download_info.get("sha256")
    download_format = download_info.get("format", "tar.gz")
    extraction_path = download_info.get("extraction_path")

    if not download_url:
        print("ERROR: No download URL in manifest")
        return False

    # Show license info before download
    print(f"\nLicense: {dataset.license_name} ({dataset.license_id})")
    license_url = dataset.manifest.get("license", {}).get("url")
    if license_url:
        print(f"  {license_url}")
    license_notes = dataset.manifest.get("license", {}).get("notes")
    if license_notes:
        print(f"  Note: {license_notes}")

    # Download
    print(f"\nDownloading from: {download_url}")
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
        if tmp_path.exists():
            tmp_path.unlink()

    # Verify
    print("\nVerifying...")
    valid, errors = verify_dataset(dataset)
    if valid:
        print("Dataset ready!")
        return True
    else:
        print("Verification issues (dataset may still be usable):")
        for err in errors:
            print(f"  - {err}")
        return True


# ============================================================================
# Schema Validation
# ============================================================================


def validate_manifests() -> tuple[int, int]:
    """Validate all manifests against the JSON schema.

    Returns:
        Tuple of (valid_count, error_count)
    """
    try:
        import jsonschema
    except ImportError:
        print("ERROR: jsonschema package required for validation")
        print("  Install with: pip install jsonschema")
        return 0, 0

    if not SCHEMA_PATH.exists():
        print(f"ERROR: Schema not found at {SCHEMA_PATH}")
        return 0, 0

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    valid_count = 0
    error_count = 0

    for dataset in load_all_manifests():
        try:
            jsonschema.validate(dataset.manifest, schema)
            print(f"  [OK] {dataset.id}")
            valid_count += 1
        except jsonschema.ValidationError as e:
            print(f"  [FAIL] {dataset.id}: {e.message}")
            error_count += 1

    return valid_count, error_count


# ============================================================================
# CLI Commands
# ============================================================================


def cmd_list(args: argparse.Namespace) -> int:
    """List all available datasets."""
    datasets = load_all_manifests()

    if not datasets:
        print("No datasets found")
        return 1

    print("\nBenchmark Datasets")
    print("=" * 90)
    print(f"{'ID':<30} {'Track':<15} {'Split':<8} {'Status':<12} {'License':<10} Samples")
    print("-" * 90)

    for ds in datasets:
        if args.smoke_only and not ds.is_smoke:
            continue
        if args.track and ds.track != args.track:
            continue

        if ds.is_available:
            status = "Ready"
        elif ds.is_smoke:
            status = "Missing"
        elif ds.requires_download:
            status = "Download"
        else:
            status = "Manual"

        print(
            f"{ds.id:<30} {ds.track:<15} {ds.split:<8} {status:<12} {ds.license_id:<10} {ds.sample_count}"
        )

    print("-" * 90)
    print(f"\nCache directory: {get_cache_dir()}")
    print("Set SLOWER_WHISPER_BENCHMARKS to override")

    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Fetch datasets."""
    datasets = load_all_manifests()

    if args.dataset:
        # Fetch specific dataset
        dataset = find_dataset(args.dataset)
        if not dataset:
            print(f"ERROR: Dataset '{args.dataset}' not found")
            print("Use 'list' command to see available datasets")
            return 1
        success = fetch_dataset(dataset, force=args.force)
        return 0 if success else 1

    if args.smoke:
        # Fetch all smoke datasets
        smoke_datasets = [ds for ds in datasets if ds.is_smoke]
        if not smoke_datasets:
            print("No smoke datasets found")
            return 1

        print(f"Fetching {len(smoke_datasets)} smoke dataset(s)...")
        all_success = True
        for dataset in smoke_datasets:
            if not fetch_dataset(dataset, force=args.force):
                all_success = False

        return 0 if all_success else 1

    if args.track:
        # Fetch all datasets for a track
        track_datasets = [ds for ds in datasets if ds.track == args.track]
        if not track_datasets:
            print(f"No datasets found for track '{args.track}'")
            return 1

        print(f"Fetching {len(track_datasets)} dataset(s) for track '{args.track}'...")
        all_success = True
        for dataset in track_datasets:
            if not fetch_dataset(dataset, force=args.force):
                all_success = False

        return 0 if all_success else 1

    print("Specify --dataset, --smoke, or --track")
    return 1


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify dataset integrity."""
    if args.dataset:
        dataset = find_dataset(args.dataset)
        if not dataset:
            print(f"ERROR: Dataset '{args.dataset}' not found")
            return 1

        print(f"Verifying: {dataset.id}")
        valid, errors = verify_dataset(dataset)
        if valid:
            print("Verification: PASSED")
            return 0
        else:
            print("Verification: FAILED")
            for err in errors:
                print(f"  - {err}")
            return 1

    # Verify all available datasets
    datasets = load_all_manifests()
    available = [ds for ds in datasets if ds.is_available]

    if not available:
        print("No datasets available to verify")
        return 1

    print(f"Verifying {len(available)} dataset(s)...")
    all_valid = True

    for dataset in available:
        print(f"\n{dataset.id}:")
        valid, errors = verify_dataset(dataset)
        if valid:
            print("  PASSED")
        else:
            print("  FAILED")
            for err in errors:
                print(f"    - {err}")
            all_valid = False

    return 0 if all_valid else 1


def cmd_license(args: argparse.Namespace) -> int:
    """Show license information."""
    if args.dataset:
        dataset = find_dataset(args.dataset)
        if not dataset:
            print(f"ERROR: Dataset '{args.dataset}' not found")
            return 1
        datasets = [dataset]
    else:
        datasets = load_all_manifests()

    print("\nDataset Licenses")
    print("=" * 80)

    for ds in datasets:
        license_info = ds.manifest.get("license", {})
        print(f"\n{ds.id}")
        print(f"  License: {license_info.get('name', 'Unknown')} ({license_info.get('id', '?')})")
        if url := license_info.get("url"):
            print(f"  URL: {url}")
        if notes := license_info.get("notes"):
            print(f"  Notes: {notes}")

        source = ds.manifest.get("source", {})
        if citation := source.get("citation"):
            print(f"  Citation: {citation}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate manifest schemas."""
    print("Validating manifests against schema...")
    print(f"Schema: {SCHEMA_PATH}")
    print()

    valid, errors = validate_manifests()

    print()
    print(f"Results: {valid} valid, {errors} errors")

    return 0 if errors == 0 else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show overall status of benchmark infrastructure."""
    datasets = load_all_manifests()

    smoke_datasets = [ds for ds in datasets if ds.is_smoke]
    full_datasets = [ds for ds in datasets if not ds.is_smoke]
    available = [ds for ds in datasets if ds.is_available]

    print("\nBenchmark Infrastructure Status")
    print("=" * 60)
    print(f"Cache directory: {get_cache_dir()}")
    print(f"Manifest directory: {MANIFEST_DIR}")
    print()
    print(f"Total datasets: {len(datasets)}")
    print(f"  Smoke datasets: {len(smoke_datasets)}")
    print(f"  Full datasets: {len(full_datasets)}")
    print(f"  Available: {len(available)}")
    print()

    # By track
    tracks = {}
    for ds in datasets:
        if ds.track not in tracks:
            tracks[ds.track] = {"total": 0, "available": 0, "smoke": 0}
        tracks[ds.track]["total"] += 1
        if ds.is_available:
            tracks[ds.track]["available"] += 1
        if ds.is_smoke:
            tracks[ds.track]["smoke"] += 1

    print("By Track:")
    for track, counts in sorted(tracks.items()):
        print(
            f"  {track}: {counts['available']}/{counts['total']} available ({counts['smoke']} smoke)"
        )

    return 0


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch and manage benchmark datasets using manifest infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list command
    list_parser = subparsers.add_parser("list", help="List available datasets")
    list_parser.add_argument("--smoke-only", action="store_true", help="Show only smoke datasets")
    list_parser.add_argument("--track", type=str, help="Filter by track")

    # fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch datasets")
    fetch_parser.add_argument("--dataset", type=str, help="Specific dataset ID to fetch")
    fetch_parser.add_argument("--smoke", action="store_true", help="Fetch all smoke datasets")
    fetch_parser.add_argument("--track", type=str, help="Fetch all datasets for a track")
    fetch_parser.add_argument("--force", action="store_true", help="Force re-download")

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify dataset integrity")
    verify_parser.add_argument("--dataset", type=str, help="Specific dataset to verify")

    # license command
    license_parser = subparsers.add_parser("license", help="Show license information")
    license_parser.add_argument("--dataset", type=str, help="Specific dataset")

    # validate command
    subparsers.add_parser("validate", help="Validate manifest schemas")

    # status command
    subparsers.add_parser("status", help="Show infrastructure status")

    args = parser.parse_args()

    if args.command == "list":
        return cmd_list(args)
    elif args.command == "fetch":
        return cmd_fetch(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "license":
        return cmd_license(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "status":
        return cmd_status(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
