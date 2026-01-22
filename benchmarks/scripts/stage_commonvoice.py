#!/usr/bin/env python3
"""
Common Voice smoke slice staging script.

Downloads and stages the fixed Common Voice EN smoke slice for ASR benchmarking.
Only downloads the specific clips listed in selection.csv to ensure deterministic
regression testing.

Requirements:
    - Hugging Face account with Common Voice terms accepted
    - HF_TOKEN environment variable or huggingface-cli login

Usage:
    python benchmarks/scripts/stage_commonvoice.py
    python benchmarks/scripts/stage_commonvoice.py --verify
    python benchmarks/scripts/stage_commonvoice.py --output-dir /path/to/cache
    python benchmarks/scripts/stage_commonvoice.py --generate-selection

License compliance:
    - Common Voice audio is CC0 1.0 (public domain)
    - Do NOT attempt to identify speakers
    - Do NOT redistribute the dataset
    - See: https://commonvoice.mozilla.org/terms

Note:
    Make this file executable with: chmod +x benchmarks/scripts/stage_commonvoice.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Pinned Common Voice version for reproducibility
CV_VERSION = "17.0"
CV_DATASET = "mozilla-foundation/common_voice_17_0"
CV_CONFIG = "en"
CV_SPLIT = "test"

# Default cache location
DEFAULT_CACHE_DIR = (
    Path.home() / ".cache" / "slower-whisper" / "benchmarks" / "commonvoice_en_smoke"
)

# Deterministic seed for clip selection
SELECTION_SEED = "slower-whisper-commonvoice-en-smoke-v1"

# Text feature regex patterns
TEXT_FEATURES = {
    "digits": re.compile(r"\d"),
    "apostrophe": re.compile(r"['']"),
    "dash": re.compile(r"[-–—]"),
    "abbrev": re.compile(r"\b[A-Za-z]\.[A-Za-z]\.|(?:\bDr\.|\bMr\.|\bMs\.|\bSt\.)"),
    "quotes": re.compile(r'[""\"()]'),
    "comma": re.compile(r","),
}

# Accent bucket mapping (lowercase patterns)
ACCENT_BUCKETS = {
    "us": ["us", "united states", "american", "canada", "canadian"],
    "uk": ["uk", "united kingdom", "england", "welsh", "british"],
    "india": ["india", "indian"],
    "au_nz": ["australia", "australian", "new zealand", "nz", "kiwi"],
    "ireland": ["ireland", "irish"],
    "scotland": ["scotland", "scottish"],
    "south_africa": ["south africa", "southafrica", "sa english"],
}

# Selection slots: (bucket, required_feature, count)
# Total: 15 clips
SELECTION_SLOTS = [
    ("us", "digits", 1),
    ("us", "apostrophe", 1),
    ("us", "dash", 1),
    ("uk", "quotes", 1),
    ("uk", "comma", 1),
    ("uk", "abbrev", 1),
    ("india", "digits", 1),
    ("india", "comma", 1),
    ("au_nz", "apostrophe", 1),
    ("au_nz", "dash", 1),
    ("ireland", "comma", 1),
    ("scotland", "apostrophe", 1),
    ("south_africa", "digits", 1),
    ("other", "any", 1),  # Unspecified/Other - hard mode candidate 1
    ("other", "any", 1),  # Unspecified/Other - hard mode candidate 2
]


def get_selection_file() -> Path:
    """Get path to selection.csv."""
    return (
        Path(__file__).parent.parent / "datasets" / "asr" / "commonvoice_en_smoke" / "selection.csv"
    )


def get_manifest_file() -> Path:
    """Get path to manifest.json."""
    return (
        Path(__file__).parent.parent / "datasets" / "asr" / "commonvoice_en_smoke" / "manifest.json"
    )


def load_selection() -> list[dict]:
    """Load clip selection from CSV."""
    selection_file = get_selection_file()
    if not selection_file.exists():
        raise FileNotFoundError(f"Selection file not found: {selection_file}")

    clips = []
    with open(selection_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clips.append(row)

    logger.info(f"Loaded {len(clips)} clips from selection")
    return clips


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_selection_hash(seed: str, path: str) -> str:
    """Compute deterministic hash for selection ordering."""
    return hashlib.sha256(f"{seed}{path}".encode()).hexdigest()


def detect_text_features(text: str) -> set[str]:
    """Detect which text features are present in a string."""
    features = set()
    for name, pattern in TEXT_FEATURES.items():
        if pattern.search(text):
            features.add(name)
    return features


def classify_accent(accent_str: str | None) -> str:
    """
    Classify an accent string into a bucket.

    Returns bucket name or "other" if no match.

    Uses word-boundary matching to avoid false positives like
    'Australia' matching 'us'.
    """
    if not accent_str:
        return "other"

    accent_lower = accent_str.lower().strip()
    if not accent_lower:
        return "other"

    # Check buckets in a specific order to avoid false matches
    # (e.g., "australia" contains "us" but should match au_nz)
    # Longer/more specific patterns should be checked first
    bucket_order = ["au_nz", "south_africa", "uk", "us", "india", "ireland", "scotland"]

    for bucket in bucket_order:
        patterns = ACCENT_BUCKETS[bucket]
        for pattern in patterns:
            # Use word boundary matching for short patterns to avoid
            # false matches like "us" in "Australia"
            if len(pattern) <= 3:
                # For short patterns, require word boundaries
                word_pattern = re.compile(r"\b" + re.escape(pattern) + r"\b", re.IGNORECASE)
                if word_pattern.search(accent_lower):
                    return bucket
            else:
                # For longer patterns, substring match is fine
                if pattern in accent_lower:
                    return bucket

    return "other"


def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


def is_valid_candidate(example: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    """
    Check if an example meets the selection criteria.

    Args:
        example: HuggingFace dataset example

    Returns:
        Tuple of (is_valid, clip_info_dict or None)
    """
    sentence = example.get("sentence", "")
    if not sentence:
        return False, None

    # Word count check: 5-25 words
    word_count = count_words(sentence)
    if not (5 <= word_count <= 25):
        return False, None

    # Duration check: 3-12 seconds
    audio = example.get("audio", {})
    audio_array = audio.get("array")
    sample_rate = audio.get("sampling_rate", 48000)

    if audio_array is None:
        return False, None

    duration_s = len(audio_array) / sample_rate
    if not (3.0 <= duration_s <= 12.0):
        return False, None

    # Vote quality check: up_votes >= 2, down_votes == 0
    up_votes = example.get("up_votes", 0)
    down_votes = example.get("down_votes", 0)

    # Some entries may not have vote data, be lenient
    if up_votes is not None and down_votes is not None:
        if down_votes > 0:
            return False, None
        if up_votes < 2:
            return False, None

    path = example.get("path", "")
    client_id = example.get("client_id", "")
    accent = example.get("accent", "")

    clip_info = {
        "path": path,
        "client_id": client_id,
        "accent": accent,
        "sentence": sentence,
        "up_votes": up_votes if up_votes is not None else 0,
        "down_votes": down_votes if down_votes is not None else 0,
        "duration_s": round(duration_s, 2),
        "word_count": word_count,
        "text_features": detect_text_features(sentence),
        "accent_bucket": classify_accent(accent),
    }

    return True, clip_info


def generate_selection() -> list[dict[str, Any]]:
    """
    Generate deterministic clip selection from Common Voice.

    Loads the dataset, filters candidates, and selects clips
    deterministically based on accent buckets and text features.

    Returns:
        List of selected clip info dicts
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        sys.exit(1)

    logger.info(f"Loading Common Voice {CV_VERSION} ({CV_CONFIG}) from Hugging Face...")
    logger.info(
        "Note: This requires accepting terms at "
        "https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0"
    )

    try:
        ds = load_dataset(CV_DATASET, CV_CONFIG, split=CV_SPLIT, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Make sure you have accepted the terms and are logged in:")
        logger.error("  huggingface-cli login")
        sys.exit(1)

    logger.info(f"Dataset loaded with {len(ds)} examples")

    # First pass: collect all valid candidates
    logger.info("Filtering candidates...")
    candidates: list[dict[str, Any]] = []

    for i, example in enumerate(ds):
        if i % 10000 == 0:
            logger.debug(f"Processing example {i}...")

        is_valid, clip_info = is_valid_candidate(example)
        if is_valid and clip_info:
            candidates.append(clip_info)

    logger.info(f"Found {len(candidates)} valid candidates")

    # Group candidates by accent bucket
    by_bucket: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in ACCENT_BUCKETS}
    by_bucket["other"] = []

    for clip in candidates:
        bucket = clip["accent_bucket"]
        by_bucket[bucket].append(clip)

    for bucket, clips in by_bucket.items():
        logger.info(f"  {bucket}: {len(clips)} candidates")

    # Sort each bucket by deterministic hash
    for bucket in by_bucket:
        by_bucket[bucket].sort(key=lambda c: compute_selection_hash(SELECTION_SEED, c["path"]))

    # Select clips for each slot
    selected: list[dict[str, Any]] = []
    used_client_ids: set[str] = set()

    for slot_idx, (bucket, required_feature, count) in enumerate(SELECTION_SLOTS):
        bucket_candidates = by_bucket.get(bucket, [])

        for _ in range(count):
            found = False
            for clip in bucket_candidates:
                # Skip if speaker already used
                if clip["client_id"] in used_client_ids:
                    continue

                # Check feature requirement
                if required_feature != "any":
                    if required_feature not in clip["text_features"]:
                        continue

                # Found a match
                clip_with_slot = {
                    "slot_bucket": bucket,
                    "slot_feature": required_feature,
                    **clip,
                }
                selected.append(clip_with_slot)
                used_client_ids.add(clip["client_id"])
                bucket_candidates.remove(clip)
                found = True
                logger.info(
                    f"Slot {slot_idx + 1} ({bucket}/{required_feature}): selected {clip['path']}"
                )
                break

            if not found:
                logger.warning(
                    f"Slot {slot_idx + 1} ({bucket}/{required_feature}): "
                    f"no matching candidate found"
                )

    logger.info(f"Selected {len(selected)} clips")
    return selected


def write_selection_csv(clips: list[dict[str, Any]], output_path: Path) -> None:
    """
    Write selection to CSV file.

    Args:
        clips: List of selected clip info dicts
        output_path: Path to write CSV
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "slot_bucket",
        "slot_feature",
        "path",
        "client_id",
        "accent",
        "sentence",
        "up_votes",
        "down_votes",
        "duration_s",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for clip in clips:
            writer.writerow(clip)

    logger.info(f"Wrote selection to {output_path}")


def stage_clips(output_dir: Path, verify: bool = False) -> dict:
    """
    Stage Common Voice clips to local cache.

    Args:
        output_dir: Directory to save audio files
        verify: If True, only verify existing files without downloading

    Returns:
        Dict with staging results including any errors
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        sys.exit(1)

    selection = load_selection()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list] = {"staged": [], "verified": [], "errors": [], "skipped": []}

    # Support both old format (clip_id) and new format (path)
    def get_clip_path(clip: dict) -> str:
        """Extract the clip path/filename from selection row."""
        if "path" in clip and clip["path"]:
            return clip["path"]
        # Fallback to old clip_id format
        return clip.get("clip_id", "")

    if verify:
        # Verify mode: check existing files
        for clip in selection:
            clip_path = get_clip_path(clip)
            clip_name = Path(clip_path).name if clip_path else ""

            if not clip_name:
                continue

            # Try both .mp3 and .wav extensions
            audio_path_wav = output_dir / clip_name.replace(".mp3", ".wav")
            audio_path_mp3 = output_dir / clip_name

            if audio_path_wav.exists():
                results["verified"].append(clip_name)
                logger.info(f"Verified: {clip_name}")
            elif audio_path_mp3.exists():
                results["verified"].append(clip_name)
                logger.info(f"Verified: {clip_name}")
            else:
                results["errors"].append({"clip_id": clip_name, "error": "file_not_found"})

        return results

    # Download mode
    logger.info(f"Loading Common Voice {CV_VERSION} ({CV_CONFIG}) from Hugging Face...")
    logger.info(
        "Note: This requires accepting terms at "
        "https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0"
    )

    try:
        # Load dataset (this will prompt for auth if needed)
        ds = load_dataset(CV_DATASET, CV_CONFIG, split=CV_SPLIT, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Make sure you have accepted the terms and are logged in:")
        logger.error("  huggingface-cli login")
        sys.exit(1)

    # Create index by path for quick lookup
    # Support both old clip_id and new path column
    clip_paths = set()
    for clip in selection:
        clip_path = get_clip_path(clip)
        if clip_path:
            # Store both full path and just the filename for matching
            clip_paths.add(clip_path)
            clip_paths.add(Path(clip_path).name)

    logger.info(f"Searching for {len(selection)} selected clips...")

    found_count = 0
    for example in ds:
        path = example.get("path", "")
        clip_name = Path(path).name if path else ""

        # Match by full path or just filename
        if path in clip_paths or clip_name in clip_paths:
            # Save audio
            audio = example.get("audio", {})
            audio_array = audio.get("array")
            sample_rate = audio.get("sampling_rate", 48000)

            if audio_array is not None:
                # Save as WAV for consistency
                output_path = output_dir / clip_name.replace(".mp3", ".wav")

                try:
                    import soundfile as sf

                    sf.write(output_path, audio_array, sample_rate)

                    sha256 = compute_sha256(output_path)
                    results["staged"].append(
                        {
                            "clip_id": clip_name,
                            "path": path,
                            "output_path": str(output_path),
                            "sha256": sha256,
                            "sample_rate": sample_rate,
                        }
                    )
                    found_count += 1
                    logger.info(f"Staged: {clip_name} -> {output_path}")
                except Exception as e:
                    results["errors"].append({"clip_id": clip_name, "error": str(e)})

            if found_count >= len(selection):
                break

    # Report any clips not found
    found_paths = {r.get("path", r.get("clip_id", "")) for r in results["staged"]}
    found_names = {Path(p).name for p in found_paths if p}

    for clip in selection:
        clip_path = get_clip_path(clip)
        clip_name = Path(clip_path).name if clip_path else ""

        if clip_path not in found_paths and clip_name not in found_names:
            if clip_name not in [e["clip_id"] for e in results["errors"]]:
                results["skipped"].append({"clip_id": clip_name, "reason": "not_found_in_dataset"})

    return results


def update_manifest_with_hashes(results: dict) -> None:
    """Update manifest.json with SHA256 hashes from staged files."""
    manifest_file = get_manifest_file()
    if not manifest_file.exists():
        logger.warning("Manifest file not found, skipping hash update")
        return

    with open(manifest_file) as f:
        manifest = json.load(f)

    # Build samples list from staged results
    samples = []
    selection = load_selection()

    # Build index supporting both old (clip_id) and new (path) formats
    selection_by_id: dict[str, dict] = {}
    for c in selection:
        # Index by filename (from path or clip_id)
        if "path" in c and c["path"]:
            filename = Path(c["path"]).name
            selection_by_id[filename] = c
        if "clip_id" in c:
            selection_by_id[c["clip_id"]] = c

    for staged in results.get("staged", []):
        clip_id = staged["clip_id"]
        sel = selection_by_id.get(clip_id, {})

        # Support both old format (expected_transcript) and new format (sentence)
        transcript = sel.get("sentence", sel.get("expected_transcript", ""))

        samples.append(
            {
                "id": clip_id.replace(".mp3", "").replace(".wav", ""),
                "audio": staged["output_path"],
                "sha256": staged["sha256"],
                "duration_s": float(sel.get("duration_s", 0)),
                "language": "en",
                "reference_transcript": transcript,
                "license": "CC0-1.0",
                "source": "common-voice",
                "accent": sel.get("accent", ""),
                "notes": sel.get("slot_bucket", "") + "/" + sel.get("slot_feature", ""),
            }
        )

    manifest["samples"] = samples
    manifest["meta"]["sample_count"] = len(samples)

    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Updated manifest with {len(samples)} samples")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Stage Common Voice smoke slice for ASR benchmarking"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Output directory for audio files (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Only verify existing files, don't download"
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Update manifest.json with SHA256 hashes after staging",
    )
    parser.add_argument(
        "--generate-selection",
        action="store_true",
        help="Generate deterministic clip selection from Common Voice metadata",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s"
    )

    # Generate selection mode
    if args.generate_selection:
        logger.info("Common Voice Clip Selection Generator")
        logger.info(f"  Version: {CV_VERSION}")
        logger.info(f"  Seed: {SELECTION_SEED}")

        clips = generate_selection()
        output_path = get_selection_file()
        write_selection_csv(clips, output_path)

        # Print summary
        print("\n" + "=" * 60)
        print("SELECTION SUMMARY")
        print("=" * 60)
        print(f"Total clips: {len(clips)}")
        print(f"Output file: {output_path}")
        print("\nBy accent bucket:")
        bucket_counts: dict[str, int] = {}
        for clip in clips:
            bucket = clip["slot_bucket"]
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        for bucket, count in sorted(bucket_counts.items()):
            print(f"  {bucket}: {count}")

        return 0

    # Staging mode
    logger.info("Common Voice Smoke Slice Staging")
    logger.info(f"  Version: {CV_VERSION}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Mode: {'verify' if args.verify else 'download'}")

    results = stage_clips(args.output_dir, verify=args.verify)

    # Print summary
    print("\n" + "=" * 60)
    print("STAGING SUMMARY")
    print("=" * 60)
    print(f"Staged:   {len(results['staged'])}")
    print(f"Verified: {len(results['verified'])}")
    print(f"Errors:   {len(results['errors'])}")
    print(f"Skipped:  {len(results['skipped'])}")

    if results["errors"]:
        print("\nErrors:")
        for err in results["errors"]:
            print(f"  - {err['clip_id']}: {err['error']}")

    if results["skipped"]:
        print("\nSkipped:")
        for skip in results["skipped"]:
            print(f"  - {skip['clip_id']}: {skip['reason']}")

    # Update manifest if requested
    if args.update_manifest and results["staged"]:
        update_manifest_with_hashes(results)

    # Exit with error if any clips failed
    if results["errors"]:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
