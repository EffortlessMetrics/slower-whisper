#!/usr/bin/env python3
"""Validate gold label files against the semantic benchmark schema.

Checks all JSON files in benchmarks/gold/semantic/ for:
1. Schema compliance (JSON Schema validation)
2. Sanity checks (non-empty topics/risks/actions arrays)
3. Consistency (segment_id references, etc.)

Usage:
    python scripts/validate_gold_labels.py
    python scripts/validate_gold_labels.py --verbose
    python scripts/validate_gold_labels.py --strict  # exit 1 on warnings too

Exit codes:
    0 - All files valid
    1 - Validation errors found
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Optional: use jsonschema if available, fall back to basic validation
try:
    from jsonschema import Draft202012Validator

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "benchmarks" / "gold" / "semantic" / "schema.json"
GOLD_DIR = ROOT / "benchmarks" / "gold" / "semantic"

# Controlled vocabulary for topics (extensible)
TOPIC_VOCABULARY = {
    # Business domains
    "pricing",
    "contract",
    "renewal",
    "onboarding",
    "support",
    "escalation",
    "feedback",
    "feature_request",
    "bug_report",
    "billing",
    "cancellation",
    "upsell",
    "cross_sell",
    "negotiation",
    "timeline",
    "process",
    "project",
    # Technical domains
    "integration",
    "api",
    "security",
    "performance",
    "migration",
    "training",
    "documentation",
    "technical",
    "design",
    # Meeting types
    "status_update",
    "planning",
    "retrospective",
    "demo",
    "review",
    # General
    "introduction",
    "closing",
    "off_topic",
    "other",
}


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Load the JSON schema from disk."""
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    with schema_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_gold_file(path: Path) -> dict[str, Any]:
    """Load a gold label JSON file."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_schema(data: dict[str, Any], schema: dict[str, Any], path: Path) -> list[str]:
    """Validate data against JSON schema. Returns list of error messages."""
    errors: list[str] = []

    if HAS_JSONSCHEMA:
        validator = Draft202012Validator(schema)
        for error in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
            loc = ".".join(str(p) for p in error.path) if error.path else "(root)"
            errors.append(f"{path.name}: schema error at {loc}: {error.message}")
    else:
        # Basic validation without jsonschema library
        errors.extend(_basic_schema_check(data, path))

    return errors


def _basic_schema_check(data: dict[str, Any], path: Path) -> list[str]:
    """Fallback validation when jsonschema is not available."""
    errors: list[str] = []

    # Check required fields
    required = ["schema_version", "meeting_id", "topics", "risks", "actions"]
    for field in required:
        if field not in data:
            errors.append(f"{path.name}: missing required field '{field}'")

    # Check schema_version
    if data.get("schema_version") != 1:
        errors.append(f"{path.name}: schema_version must be 1, got {data.get('schema_version')}")

    # Check meeting_id is non-empty string
    meeting_id = data.get("meeting_id")
    if not isinstance(meeting_id, str) or not meeting_id:
        errors.append(f"{path.name}: meeting_id must be a non-empty string")

    # Check topics array structure
    topics = data.get("topics", [])
    if not isinstance(topics, list):
        errors.append(f"{path.name}: topics must be an array")
    else:
        for i, topic in enumerate(topics):
            if not isinstance(topic, dict):
                errors.append(f"{path.name}: topics[{i}] must be an object")
                continue
            if "label" not in topic:
                errors.append(f"{path.name}: topics[{i}] missing required field 'label'")

    # Check risks array structure
    risks = data.get("risks", [])
    if not isinstance(risks, list):
        errors.append(f"{path.name}: risks must be an array")
    else:
        valid_types = {"escalation", "churn", "pricing"}
        valid_severities = {"low", "medium", "high"}
        for i, risk in enumerate(risks):
            if not isinstance(risk, dict):
                errors.append(f"{path.name}: risks[{i}] must be an object")
                continue
            for field in ["type", "severity", "segment_id"]:
                if field not in risk:
                    errors.append(f"{path.name}: risks[{i}] missing required field '{field}'")
            if risk.get("type") not in valid_types:
                errors.append(
                    f"{path.name}: risks[{i}].type must be one of {valid_types}, "
                    f"got '{risk.get('type')}'"
                )
            if risk.get("severity") not in valid_severities:
                errors.append(
                    f"{path.name}: risks[{i}].severity must be one of {valid_severities}, "
                    f"got '{risk.get('severity')}'"
                )

    # Check actions array structure
    actions = data.get("actions", [])
    if not isinstance(actions, list):
        errors.append(f"{path.name}: actions must be an array")
    else:
        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                errors.append(f"{path.name}: actions[{i}] must be an object")
                continue
            if "text" not in action:
                errors.append(f"{path.name}: actions[{i}] missing required field 'text'")
            elif not action.get("text"):
                errors.append(f"{path.name}: actions[{i}].text must be non-empty")

    return errors


def sanity_checks(data: dict[str, Any], path: Path, *, strict: bool = False) -> list[str]:
    """Run sanity checks on gold label data. Returns list of warning/error messages."""
    warnings: list[str] = []

    # Check for empty arrays (warn, not error)
    topics = data.get("topics", [])
    risks = data.get("risks", [])
    actions = data.get("actions", [])

    if not topics:
        msg = f"{path.name}: topics array is empty"
        warnings.append(f"{'ERROR' if strict else 'WARNING'}: {msg}")

    if not risks:
        msg = f"{path.name}: risks array is empty"
        warnings.append(f"{'ERROR' if strict else 'WARNING'}: {msg}")

    if not actions:
        msg = f"{path.name}: actions array is empty"
        warnings.append(f"{'ERROR' if strict else 'WARNING'}: {msg}")

    # Check topic labels against vocabulary
    for i, topic in enumerate(topics):
        label = topic.get("label", "")
        if label and label not in TOPIC_VOCABULARY:
            msg = f"{path.name}: topics[{i}].label '{label}' not in controlled vocabulary"
            warnings.append(f"WARNING: {msg}")

    # Check for duplicate segment_ids in same risk type
    risk_segments: dict[str, set[int]] = {}
    for _i, risk in enumerate(risks):
        risk_type = risk.get("type", "")
        segment_id = risk.get("segment_id")
        if risk_type and segment_id is not None:
            if risk_type not in risk_segments:
                risk_segments[risk_type] = set()
            if segment_id in risk_segments[risk_type]:
                msg = (
                    f"{path.name}: duplicate risk entry for type='{risk_type}' "
                    f"segment_id={segment_id}"
                )
                warnings.append(f"WARNING: {msg}")
            risk_segments[risk_type].add(segment_id)

    return warnings


def validate_file(
    path: Path, schema: dict[str, Any], *, verbose: bool = False, strict: bool = False
) -> tuple[list[str], list[str]]:
    """Validate a single gold label file.

    Returns:
        Tuple of (errors, warnings)
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        data = load_gold_file(path)
    except json.JSONDecodeError as e:
        errors.append(f"{path.name}: invalid JSON - {e}")
        return errors, warnings
    except Exception as e:
        errors.append(f"{path.name}: failed to load - {e}")
        return errors, warnings

    # Schema validation
    schema_errors = validate_schema(data, schema, path)
    errors.extend(schema_errors)

    # Skip sanity checks if schema validation failed
    if not schema_errors:
        sanity_warnings = sanity_checks(data, path, strict=strict)
        for msg in sanity_warnings:
            if msg.startswith("ERROR:"):
                errors.append(msg.replace("ERROR: ", ""))
            else:
                warnings.append(msg.replace("WARNING: ", ""))

    return errors, warnings


def validate_all(
    gold_dir: Path,
    schema_path: Path,
    *,
    verbose: bool = False,
    strict: bool = False,
) -> tuple[int, int, int]:
    """Validate all gold label files in directory.

    Returns:
        Tuple of (total_files, error_count, warning_count)
    """
    schema = load_schema(schema_path)

    json_files = sorted(gold_dir.glob("*.json"))
    # Exclude schema.json from validation
    json_files = [f for f in json_files if f.name != "schema.json"]

    if not json_files:
        print(f"No gold label files found in {gold_dir}")
        return 0, 0, 0

    total_errors = 0
    total_warnings = 0
    valid_files = 0

    print(f"Validating {len(json_files)} gold label file(s) in {gold_dir}")
    print(f"Schema: {schema_path}")
    if not HAS_JSONSCHEMA:
        print("Note: jsonschema not installed, using basic validation")
    print()

    for path in json_files:
        errors, warnings = validate_file(path, schema, verbose=verbose, strict=strict)

        if errors:
            print(f"FAIL: {path.name}")
            for err in errors:
                print(f"  - {err}")
            total_errors += len(errors)
        elif warnings:
            print(f"WARN: {path.name}")
            for warn in warnings:
                print(f"  - {warn}")
            valid_files += 1
        else:
            print(f"OK: {path.name}")
            valid_files += 1

        total_warnings += len(warnings)

    print()
    print("=" * 60)
    print(f"Validation Report: {valid_files}/{len(json_files)} files valid")
    print(f"  Errors: {total_errors}")
    print(f"  Warnings: {total_warnings}")
    print("=" * 60)

    return len(json_files), total_errors, total_warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate gold label files against semantic benchmark schema.",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=GOLD_DIR,
        help=f"Directory containing gold label JSON files (default: {GOLD_DIR})",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=SCHEMA_PATH,
        help=f"Path to JSON schema file (default: {SCHEMA_PATH})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (exit 1 on any issue)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        total_files, errors, warnings = validate_all(
            args.gold_dir,
            args.schema,
            verbose=args.verbose,
            strict=args.strict,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if total_files == 0:
        print("No files to validate.", file=sys.stderr)
        return 0

    if errors > 0:
        return 1

    if args.strict and warnings > 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
