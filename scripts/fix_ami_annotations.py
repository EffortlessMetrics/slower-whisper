#!/usr/bin/env python3
"""Fix AMI annotation files to use correct field names.

This script converts annotation files that use non-standard field names
(like 'reference_summary', 'reference_transcript') to the standard names
expected by the evaluation infrastructure ('summary', 'transcript').

Usage:
    # Fix all annotations in the AMI directory
    python scripts/fix_ami_annotations.py

    # Fix specific directory
    python scripts/fix_ami_annotations.py --annotations-dir /path/to/annotations

    # Dry run (show what would be changed without modifying files)
    python scripts/fix_ami_annotations.py --dry-run
"""

import argparse
import json
from pathlib import Path


def fix_annotation_file(path: Path, dry_run: bool = False) -> dict:
    """Fix field names in a single annotation file.

    Args:
        path: Path to annotation JSON file
        dry_run: If True, don't modify the file

    Returns:
        Dict with keys:
        - 'changed': bool, whether any changes were made
        - 'fixes': list of field name changes applied
    """
    with open(path) as f:
        data = json.load(f)

    fixes = []
    changed = False

    # Fix reference_summary -> summary
    if "reference_summary" in data and "summary" not in data:
        data["summary"] = data.pop("reference_summary")
        fixes.append("reference_summary → summary")
        changed = True

    # Fix reference_transcript -> transcript
    if "reference_transcript" in data and "transcript" not in data:
        data["transcript"] = data.pop("reference_transcript")
        fixes.append("reference_transcript → transcript")
        changed = True

    # Fix reference_speakers -> speakers (less common)
    if "reference_speakers" in data and "speakers" not in data:
        data["speakers"] = data.pop("reference_speakers")
        fixes.append("reference_speakers → speakers")
        changed = True

    # Remove redundant fields that are just duplicates
    removed = []
    if "meeting_id" in data:
        # meeting_id is redundant (encoded in filename)
        removed.append("meeting_id")
        data.pop("meeting_id")
        changed = True

    if "participants" in data and "metadata" in data:
        # Move participants to metadata if metadata exists
        if "num_speakers" not in data["metadata"]:
            data["metadata"]["num_speakers"] = data.pop("participants")
            removed.append("participants (moved to metadata.num_speakers)")
            changed = True

    if "scenario" in data and "metadata" in data:
        # Move scenario to metadata if metadata exists
        if "scenario" not in data["metadata"]:
            data["metadata"]["scenario"] = data.pop("scenario")
            removed.append("scenario (moved to metadata.scenario)")
            changed = True

    # Ensure metadata exists if we need to add top-level fields to it
    if "metadata" not in data:
        # Collect any top-level fields that should be in metadata
        metadata_fields = {}
        for key in ["scenario", "participants", "duration", "roles"]:
            if key in data:
                metadata_fields[key] = data.pop(key)
                removed.append(f"{key} (moved to metadata)")
                changed = True

        if metadata_fields:
            data["metadata"] = metadata_fields

    if changed and not dry_run:
        # Write back with clean formatting
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")  # Add trailing newline

    return {
        "changed": changed,
        "fixes": fixes,
        "removed": removed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fix AMI annotation field names to match evaluation expectations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        help="Path to annotations directory (default: auto-detect from benchmarks cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )

    args = parser.parse_args()

    # Determine annotations directory
    if args.annotations_dir:
        annotations_dir = args.annotations_dir
    else:
        # Auto-detect from benchmarks cache
        try:
            from slower_whisper.pipeline.benchmarks import get_benchmarks_root

            benchmarks_root = get_benchmarks_root()
            annotations_dir = benchmarks_root / "ami" / "annotations"
        except ImportError:
            print("Error: Could not auto-detect annotations directory.")
            print("Please specify --annotations-dir explicitly.")
            return 1

    if not annotations_dir.exists():
        print(f"Error: Annotations directory not found: {annotations_dir}")
        return 1

    print(f"Scanning annotations in: {annotations_dir}")
    if args.dry_run:
        print("DRY RUN MODE - no files will be modified\n")

    # Process all JSON files
    json_files = list(annotations_dir.glob("*.json"))
    if not json_files:
        print("No JSON files found.")
        return 0

    total_changed = 0
    for json_file in sorted(json_files):
        result = fix_annotation_file(json_file, dry_run=args.dry_run)

        if result["changed"]:
            total_changed += 1
            status = "WOULD FIX" if args.dry_run else "FIXED"
            print(f"{status}: {json_file.name}")
            for fix in result["fixes"]:
                print(f"  - {fix}")
            for removal in result["removed"]:
                print(f"  - Removed: {removal}")
        else:
            print(f"OK: {json_file.name}")

    print()
    if args.dry_run:
        print(f"Summary: {total_changed}/{len(json_files)} files would be modified")
        print("Run without --dry-run to apply changes.")
    else:
        print(f"Summary: {total_changed}/{len(json_files)} files modified")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
