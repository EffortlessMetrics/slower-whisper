"""Sample dataset management for dogfooding and evaluation.

This module provides automated download and caching of public audio datasets
for testing diarization and LLM integration without requiring manual setup.

Sample datasets are cached under:
    $SLOWER_WHISPER_CACHE_ROOT/samples (default: ~/.cache/slower-whisper/samples)

Environment variables respected:
- SLOWER_WHISPER_CACHE_ROOT: Root cache directory
- SLOWER_WHISPER_SAMPLES: Override samples cache location
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from transcription.cache import CachePaths
from transcription.exceptions import SampleExistsError

logger = logging.getLogger(__name__)

SampleDatasetName = Literal[
    "mini_diarization",  # 2-speaker student/professor conversations (Kaggle)
    "ami_sample",  # AMI Meeting Corpus sample (future)
]


@dataclass
class SampleDataset:
    """Metadata for a sample dataset.

    Attributes:
        name: Unique dataset identifier
        description: Human-readable description
        url: Direct download URL (no auth required)
        sha256: Expected SHA256 hash for integrity check
        archive_format: Archive type ('zip' or 'tar.gz')
        test_files: List of test files to extract (relative to archive root)
        source_url: Original dataset homepage (for attribution)
        license: License type (e.g., "CC BY 4.0", "MIT")
    """

    name: str
    description: str
    url: str
    sha256: str
    archive_format: Literal["zip", "tar.gz"]
    test_files: list[str]
    source_url: str
    license: str


# Registry of available sample datasets
# Note: URLs must be direct downloads (no auth, no redirects requiring cookies)
SAMPLE_DATASETS: dict[SampleDatasetName, SampleDataset] = {
    # Placeholder - actual URLs require Kaggle API or manual download
    # We'll generate synthetic samples instead for now
    "mini_diarization": SampleDataset(
        name="mini_diarization",
        description="2-speaker student/professor conversations for diarization testing",
        url="",  # Requires Kaggle auth - see download_mini_diarization()
        sha256="",
        archive_format="zip",
        test_files=["dataset/test/test.wav"],
        source_url="https://www.kaggle.com/datasets/wiradkp/mini-speech-diarization",
        license="CC BY 4.0 (Kaggle)",
    ),
    # Future: AMI Meeting Corpus sample
}


def get_samples_cache_dir() -> Path:
    """Get the samples cache directory.

    Returns:
        Path to samples cache directory
    """
    # Respect SLOWER_WHISPER_SAMPLES if set, otherwise use cache root
    if samples_override := os.environ.get("SLOWER_WHISPER_SAMPLES"):
        return Path(samples_override).expanduser()

    paths = CachePaths.from_env()
    return paths.root / "samples"


def verify_sha256(file_path: Path, expected_sha256: str) -> bool:
    """Verify file integrity using SHA256.

    Args:
        file_path: Path to file to verify
        expected_sha256: Expected SHA256 hex digest

    Returns:
        True if hash matches, False otherwise
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_sha256


def download_file(url: str, dest: Path, show_progress: bool = True) -> None:
    """Download a file from URL to dest with progress reporting.

    Args:
        url: Download URL
        dest: Destination file path
        show_progress: Show download progress (default: True)
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if not show_progress or total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        logger.info(
            f"Downloading: {percent:5.1f}% ({mb_downloaded:6.1f}/{mb_total:6.1f} MB)",
            extra={"percent": percent, "mb_downloaded": mb_downloaded, "mb_total": mb_total},
        )
        if downloaded >= total_size:
            logger.info("Download completed")

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)


def extract_archive(
    archive_path: Path,
    extract_to: Path,
    archive_format: Literal["zip", "tar.gz"],
    members: list[str] | None = None,
) -> None:
    """Extract archive to destination directory.

    Args:
        archive_path: Path to archive file
        extract_to: Destination directory
        archive_format: Archive type ('zip' or 'tar.gz')
        members: Specific members to extract (None = extract all)
    """
    extract_to.mkdir(parents=True, exist_ok=True)

    if archive_format == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            if members:
                zf.extractall(extract_to, members=members)
            else:
                zf.extractall(extract_to)
    elif archive_format == "tar.gz":
        with tarfile.open(archive_path, "r:gz") as tf:
            if members:
                # Filter members by name
                to_extract = [m for m in tf.getmembers() if m.name in members]
                tf.extractall(extract_to, members=to_extract, filter="data")
            else:
                tf.extractall(extract_to, filter="data")
    else:
        raise ValueError(f"Unsupported archive format: {archive_format}")


def download_sample_dataset(
    dataset_name: SampleDatasetName,
    cache_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download and cache a sample dataset.

    Args:
        dataset_name: Name of dataset to download
        cache_dir: Override cache directory (default: auto-detect)
        force_download: Re-download even if cached (default: False)

    Returns:
        Path to extracted dataset directory

    Raises:
        ValueError: If dataset requires manual download (e.g., Kaggle auth)
        RuntimeError: If download or extraction fails
    """
    dataset = SAMPLE_DATASETS.get(dataset_name)
    if not dataset:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cache_dir = cache_dir or get_samples_cache_dir()
    dataset_dir = cache_dir / dataset.name
    archive_path = cache_dir / f"{dataset.name}.{dataset.archive_format}"

    # Check if already downloaded and valid
    if not force_download:
        if dataset_dir.exists():
            # Verify test files exist
            all_exist = all((dataset_dir / f).exists() for f in dataset.test_files)
            if all_exist:
                logger.info(
                    f"Dataset '{dataset.name}' already cached",
                    extra={"dataset": dataset.name, "path": str(dataset_dir)},
                )
                return dataset_dir

    # Special handling for datasets requiring auth
    if dataset.name == "mini_diarization":
        raise ValueError(
            f"Dataset '{dataset.name}' requires manual download.\n"
            f"Please download from: {dataset.source_url}\n"
            f"Or use the Kaggle CLI:\n"
            f"  kaggle datasets download -d wiradkp/mini-speech-diarization \\\n"
            f"    -p {cache_dir} --unzip\n"
            f"See docs/DOGFOOD_SETUP.md for detailed instructions."
        )

    # Download if URL is provided
    if not dataset.url:
        raise ValueError(
            f"Dataset '{dataset.name}' has no direct download URL. "
            f"See {dataset.source_url} for manual download instructions."
        )

    logger.info(
        f"Starting download of '{dataset.name}'",
        extra={"dataset": dataset.name, "url": dataset.url},
    )
    download_file(dataset.url, archive_path)

    # Verify integrity
    if dataset.sha256:
        logger.info(
            "Verifying download integrity",
            extra={"dataset": dataset.name},
        )
        if not verify_sha256(archive_path, dataset.sha256):
            raise RuntimeError(
                f"Download integrity check failed for {dataset.name}. "
                f"Expected SHA256: {dataset.sha256}"
            )

    # Extract
    logger.info(
        "Extracting archive",
        extra={"dataset": dataset.name, "path": str(dataset_dir)},
    )
    extract_archive(archive_path, dataset_dir, dataset.archive_format, dataset.test_files)

    # Clean up archive to save space
    archive_path.unlink()

    logger.info(
        f"Dataset '{dataset.name}' ready",
        extra={"dataset": dataset.name, "path": str(dataset_dir)},
    )
    return dataset_dir


def list_sample_datasets() -> dict[str, dict[str, str]]:
    """List all available sample datasets with metadata.

    Returns:
        Dict mapping dataset name to metadata dict
    """
    return {
        name: {
            "description": ds.description,
            "source_url": ds.source_url,
            "license": ds.license,
            "test_files": ", ".join(ds.test_files),
        }
        for name, ds in SAMPLE_DATASETS.items()
    }


def get_sample_test_files(dataset_name: SampleDatasetName) -> list[Path]:
    """Get paths to test files for a sample dataset.

    Args:
        dataset_name: Name of dataset

    Returns:
        List of absolute paths to test files

    Raises:
        FileNotFoundError: If dataset not cached
    """
    dataset = SAMPLE_DATASETS.get(dataset_name)
    if not dataset:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cache_dir = get_samples_cache_dir()
    dataset_dir = cache_dir / dataset.name

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not cached. "
            f"Run download_sample_dataset('{dataset_name}') first."
        )

    test_files = [dataset_dir / f for f in dataset.test_files]

    # Verify all exist
    missing = [f for f in test_files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Test files missing for '{dataset_name}': {missing}. "
            f"Try re-downloading with force_download=True."
        )

    return test_files


def copy_sample_to_project(
    dataset_name: SampleDatasetName, project_raw_audio_dir: Path, force: bool = False
) -> list[Path]:
    """Copy sample dataset test files to a project's raw_audio directory.

    Args:
        dataset_name: Name of dataset
        project_raw_audio_dir: Path to project's raw_audio/ directory
        force: Overwrite existing files without error (default: False)

    Returns:
        List of paths to copied files in project directory

    Raises:
        SampleExistsError: If files exist and force=False

    Example:
        >>> from pathlib import Path
        >>> copied = copy_sample_to_project("mini_diarization", Path("raw_audio"))
        >>> # Now can run: slower-whisper transcribe --enable-diarization
    """
    test_files = get_sample_test_files(dataset_name)
    project_raw_audio_dir.mkdir(parents=True, exist_ok=True)

    # Calculate destinations first
    destinations: list[tuple[Path, Path]] = []
    for src in test_files:
        # Use dataset name prefix to avoid collisions
        dest_name = f"{dataset_name}_{src.name}"
        dest = project_raw_audio_dir / dest_name
        destinations.append((src, dest))

    if not force:
        existing = [d for _, d in destinations if d.exists()]
        if existing:
            raise SampleExistsError(
                f"Sample files already exist in {project_raw_audio_dir}",
                existing_files=existing,
            )

    copied_files = []
    for src, dest in destinations:
        shutil.copy(src, dest)
        copied_files.append(dest)
        logger.info(
            "Copied sample file",
            extra={"dataset": dataset_name, "source": src.name, "destination": str(dest)},
        )

    return copied_files


def generate_synthetic_2speaker(output_path: Path) -> None:
    """Generate synthetic 2-speaker audio for diarization testing.

    Creates a deterministic audio file with two distinct speakers
    alternating in a known pattern using different frequency tones.

    Pattern:
        Speaker A (120 Hz tone): 0.0-3.0s, 6.2-9.2s
        Speaker B (220 Hz tone): 3.2-6.2s, 9.4-12.4s
        Total duration: ~12.6s

    Args:
        output_path: Where to write the WAV file

    Raises:
        ImportError: If soundfile/numpy not available
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "numpy is required to generate fixtures. Install with: uv sync --extra enrich-basic"
        ) from e

    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required to generate fixtures. Install with: uv sync --extra enrich-basic"
        ) from e

    SR = 16_000  # 16 kHz mono (Whisper standard)

    def tone(f_hz: float, duration: float):  # noqa: UP037
        """Generate a pure sine wave tone."""
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        return 0.1 * np.sin(2 * np.pi * f_hz * t).astype(np.float32)

    def silence(duration: float):  # noqa: UP037
        """Generate silence (zeros)."""
        return np.zeros(int(SR * duration), dtype=np.float32)

    # Speaker A: 120 Hz tone (low pitch)
    a1 = tone(120.0, 3.0)  # 0.0 - 3.0s
    a2 = tone(120.0, 3.0)  # 6.2 - 9.2s

    # Speaker B: 220 Hz tone (higher pitch)
    b1 = tone(220.0, 3.0)  # 3.2 - 6.2s
    b2 = tone(220.0, 3.0)  # 9.4 - 12.4s

    gap = silence(0.2)  # 200ms gaps between speakers

    # Concatenate in A-B-A-B pattern
    audio = np.concatenate([a1, gap, b1, gap, a2, gap, b2], axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, SR)

    duration_sec = len(audio) / SR
    logger.info(
        "Generated synthetic 2-speaker audio",
        extra={
            "path": str(output_path),
            "duration_sec": duration_sec,
            "sample_rate": SR,
            "size_mb": output_path.stat().st_size / (1024 * 1024),
        },
    )
    logger.info(
        "Expected speaker turns: A: 0.0-3.0s, 6.2-9.2s | B: 3.2-6.2s, 9.4-12.4s",
        extra={"speaker_a_turns": "0.0-3.0s, 6.2-9.2s", "speaker_b_turns": "3.2-6.2s, 9.4-12.4s"},
    )
