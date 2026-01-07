#!/usr/bin/env python3
"""
Check DevLT estimation calibration against human-reported values.

This script loads calibration samples and computes accuracy metrics
to validate and improve the estimation model.

Usage:
    python scripts/check-calibration.py
    python scripts/check-calibration.py --samples-path path/to/samples.json
    python scripts/check-calibration.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Load calibration samples from JSON file."""
    if not path.exists():
        print(f"Error: Samples file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"Error: Expected JSON array in {path}", file=sys.stderr)
        sys.exit(1)

    return data


def validate_sample(sample: dict[str, Any], index: int) -> list[str]:
    """Validate a calibration sample, returning a list of errors."""
    errors = []

    if "pr_number" not in sample:
        errors.append(f"Sample {index}: missing required field 'pr_number'")

    if "estimated_lb" not in sample:
        errors.append(f"Sample {index}: missing required field 'estimated_lb'")

    if "estimated_ub" not in sample:
        errors.append(f"Sample {index}: missing required field 'estimated_ub'")

    if "estimated_lb" in sample and "estimated_ub" in sample:
        if sample["estimated_lb"] > sample["estimated_ub"]:
            errors.append(
                f"Sample {index} (PR #{sample.get('pr_number', '?')}): "
                f"estimated_lb ({sample['estimated_lb']}) > estimated_ub ({sample['estimated_ub']})"
            )

    return errors


def compute_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute calibration metrics from samples.

    Returns dict with:
        - total_samples: total number of samples
        - samples_with_actuals: samples with human_reported_minutes
        - within_bounds: samples where actual is within [lb, ub]
        - coverage_rate: % within bounds
        - mean_absolute_error: average |midpoint - actual|
        - bias: average (midpoint - actual)
        - avg_bound_width: average (ub - lb)
        - details: per-sample results (for verbose mode)
    """
    total = len(samples)
    with_actuals = []
    details = []

    for sample in samples:
        pr = sample.get("pr_number", "?")
        lb = sample.get("estimated_lb", 0)
        ub = sample.get("estimated_ub", 0)
        actual = sample.get("human_reported_minutes")
        midpoint = (lb + ub) / 2
        width = ub - lb

        detail = {
            "pr_number": pr,
            "estimated_lb": lb,
            "estimated_ub": ub,
            "midpoint": midpoint,
            "width": width,
            "actual": actual,
            "within_bounds": None,
            "error": None,
        }

        if actual is not None:
            within = lb <= actual <= ub
            error = midpoint - actual
            detail["within_bounds"] = within
            detail["error"] = error
            with_actuals.append(detail)

        details.append(detail)

    # Compute aggregate metrics
    if not with_actuals:
        return {
            "total_samples": total,
            "samples_with_actuals": 0,
            "within_bounds": 0,
            "coverage_rate": None,
            "mean_absolute_error": None,
            "bias": None,
            "avg_bound_width": sum(d["width"] for d in details) / total if total else 0,
            "details": details,
        }

    within_bounds = sum(1 for d in with_actuals if d["within_bounds"])
    errors = [d["error"] for d in with_actuals]
    mae = sum(abs(e) for e in errors) / len(errors)
    bias = sum(errors) / len(errors)

    return {
        "total_samples": total,
        "samples_with_actuals": len(with_actuals),
        "within_bounds": within_bounds,
        "coverage_rate": within_bounds / len(with_actuals) * 100,
        "mean_absolute_error": mae,
        "bias": bias,
        "avg_bound_width": sum(d["width"] for d in details) / total if total else 0,
        "details": details,
    }


def print_report(metrics: dict[str, Any], verbose: bool = False) -> None:
    """Print calibration report to stdout."""
    print("=" * 60)
    print("DevLT Estimation Calibration Report")
    print("=" * 60)
    print()

    print(f"Total samples:        {metrics['total_samples']}")
    print(f"Samples with actuals: {metrics['samples_with_actuals']}")
    print(f"Average bound width:  {metrics['avg_bound_width']:.1f} minutes")
    print()

    if metrics["samples_with_actuals"] == 0:
        print("No samples with human-reported values to evaluate.")
        print()
        print("To add calibration data:")
        print("  1. Record actual time spent on a PR")
        print("  2. Run: python scripts/generate-pr-ledger.py --pr <N> --dump-bundle")
        print("  3. Add human_reported_minutes to the sample in samples.json")
        return

    print("-" * 40)
    print("Accuracy Metrics")
    print("-" * 40)
    print()

    coverage = metrics["coverage_rate"]
    within = metrics["within_bounds"]
    total = metrics["samples_with_actuals"]

    print(f"Within bounds:        {within}/{total} ({coverage:.1f}%)")
    print(f"Mean absolute error:  {metrics['mean_absolute_error']:.1f} minutes")
    print(f"Bias:                 {metrics['bias']:+.1f} minutes", end="")
    if metrics["bias"] > 0:
        print(" (overestimates)")
    elif metrics["bias"] < 0:
        print(" (underestimates)")
    else:
        print()

    print()

    # Assessment
    print("-" * 40)
    print("Assessment")
    print("-" * 40)
    print()

    if coverage >= 90:
        print("[GOOD] Coverage rate >= 90%: bounds are well-calibrated")
    elif coverage >= 70:
        print("[OK] Coverage rate 70-90%: bounds are reasonable but could be tighter")
    else:
        print("[WARN] Coverage rate < 70%: bounds may be too narrow")

    if abs(metrics["bias"]) <= 5:
        print("[GOOD] Bias within +/- 5 minutes: no systematic error")
    elif abs(metrics["bias"]) <= 15:
        print("[OK] Bias 5-15 minutes: minor systematic error")
    else:
        print(
            f"[WARN] Bias > 15 minutes: systematic {'over' if metrics['bias'] > 0 else 'under'}estimation"
        )

    print()

    if verbose:
        print("-" * 40)
        print("Per-Sample Details")
        print("-" * 40)
        print()
        print(f"{'PR#':<6} {'LB':<6} {'UB':<6} {'Mid':<6} {'Actual':<8} {'Within':<8} {'Error':<8}")
        print("-" * 56)

        for d in metrics["details"]:
            pr = str(d["pr_number"])
            lb = str(d["estimated_lb"])
            ub = str(d["estimated_ub"])
            mid = f"{d['midpoint']:.0f}"
            actual = str(d["actual"]) if d["actual"] is not None else "-"
            within = "Yes" if d["within_bounds"] else ("No" if d["within_bounds"] is False else "-")
            error = f"{d['error']:+.0f}" if d["error"] is not None else "-"

            print(f"{pr:<6} {lb:<6} {ub:<6} {mid:<6} {actual:<8} {within:<8} {error:<8}")

        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check DevLT estimation calibration against human-reported values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --samples-path custom/samples.json
  %(prog)s --verbose
""",
    )

    parser.add_argument(
        "--samples-path",
        type=Path,
        default=Path("docs/audit/calibration/samples.json"),
        help="Path to calibration samples JSON file (default: docs/audit/calibration/samples.json)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-sample details",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output metrics as JSON instead of human-readable report",
    )

    args = parser.parse_args()

    # Load samples
    samples = load_samples(args.samples_path)

    # Validate samples
    all_errors = []
    for i, sample in enumerate(samples):
        errors = validate_sample(sample, i)
        all_errors.extend(errors)

    if all_errors:
        print("Validation errors:", file=sys.stderr)
        for error in all_errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    # Compute metrics
    metrics = compute_metrics(samples)

    # Output
    if args.json:
        # Remove details from JSON output unless verbose
        output = {k: v for k, v in metrics.items() if k != "details" or args.verbose}
        print(json.dumps(output, indent=2))
    else:
        print_report(metrics, verbose=args.verbose)


if __name__ == "__main__":
    main()
