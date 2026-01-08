#!/usr/bin/env python3
"""
Check DevLT estimation calibration against human-reported values.

This script loads calibration samples and computes accuracy metrics
to validate and improve the estimation model.

Usage:
    python scripts/check-calibration.py
    python scripts/check-calibration.py --samples-path path/to/samples.json
    python scripts/check-calibration.py --verbose
    python scripts/check-calibration.py --tuning-advice
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Decision time bands from decision_extractor.py (for tuning reference)
DECISION_TIME_BANDS = {
    "scope": (2, 8),
    "design": (6, 20),
    "quality": (4, 15),
    "debug": (8, 30),
    "publish": (3, 12),
}


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


def normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a sample to the canonical format with both human and estimated bounds.

    Handles both old format (flat) and new format (nested human_reported/estimated).
    """
    normalized: dict[str, Any] = {
        "pr_number": sample.get("pr_number"),
        "title": sample.get("title", ""),
    }

    # Handle human_reported (new format) vs flat fields (old format)
    if "human_reported" in sample:
        hr = sample["human_reported"]
        # If only point estimate, create bounds from it
        if "point_estimate_minutes" in hr and "lb_minutes" not in hr:
            point = hr["point_estimate_minutes"]
            # Use +/- 20% as default range for point estimates
            normalized["human_lb"] = int(point * 0.8)
            normalized["human_ub"] = int(point * 1.2)
            normalized["human_point"] = point
        else:
            normalized["human_lb"] = hr.get("lb_minutes")
            normalized["human_ub"] = hr.get("ub_minutes")
            normalized["human_point"] = hr.get("point_estimate_minutes")
        normalized["human_confidence"] = hr.get("confidence", "unknown")
        normalized["human_source"] = hr.get("source", "unknown")
    else:
        # Old format compatibility
        normalized["human_lb"] = None
        normalized["human_ub"] = None
        normalized["human_point"] = sample.get("human_reported_minutes")
        normalized["human_confidence"] = sample.get("source", "unknown")
        normalized["human_source"] = sample.get("source", "unknown")

    # Handle estimated (new format) vs flat fields (old format)
    if "estimated" in sample:
        est = sample["estimated"]
        normalized["estimated_lb"] = est.get("lb_minutes")
        normalized["estimated_ub"] = est.get("ub_minutes")
        normalized["method"] = est.get("method", "unknown")
        normalized["coverage"] = est.get("coverage", "unknown")
        normalized["decision_count"] = est.get("decision_count")
    else:
        # Old format compatibility
        normalized["estimated_lb"] = sample.get("estimated_lb")
        normalized["estimated_ub"] = sample.get("estimated_ub")
        normalized["method"] = "unknown"
        normalized["coverage"] = "unknown"
        normalized["decision_count"] = None

    # Decision summary
    normalized["decisions_summary"] = sample.get("decisions_summary", [])
    normalized["notes"] = sample.get("notes", "")
    normalized["recorded_at"] = sample.get("recorded_at", "")

    return normalized


def validate_sample(sample: dict[str, Any], index: int) -> list[str]:
    """Validate a normalized calibration sample, returning a list of errors."""
    errors = []

    if sample["pr_number"] is None:
        errors.append(f"Sample {index}: missing required field 'pr_number'")

    if sample["estimated_lb"] is None:
        errors.append(f"Sample {index}: missing required field 'estimated_lb'")

    if sample["estimated_ub"] is None:
        errors.append(f"Sample {index}: missing required field 'estimated_ub'")

    if sample["estimated_lb"] is not None and sample["estimated_ub"] is not None:
        if sample["estimated_lb"] > sample["estimated_ub"]:
            errors.append(
                f"Sample {index} (PR #{sample['pr_number']}): "
                f"estimated_lb ({sample['estimated_lb']}) > estimated_ub ({sample['estimated_ub']})"
            )

    # Check human bounds if both provided
    if sample["human_lb"] is not None and sample["human_ub"] is not None:
        if sample["human_lb"] > sample["human_ub"]:
            errors.append(
                f"Sample {index} (PR #{sample['pr_number']}): "
                f"human_lb ({sample['human_lb']}) > human_ub ({sample['human_ub']})"
            )

    return errors


def compute_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute calibration metrics from normalized samples.

    Returns dict with:
        - total_samples: total number of samples
        - samples_with_actuals: samples with human_reported data
        - within_bounds: samples where human midpoint is within [est_lb, est_ub]
        - coverage_rate: % within bounds
        - mean_absolute_error: average |est_midpoint - human_midpoint|
        - bias: average (est_midpoint - human_midpoint)
        - avg_bound_width: average (est_ub - est_lb)
        - width_ratio: average ratio of estimated width to human width
        - by_method: metrics grouped by estimation method
        - by_coverage: metrics grouped by coverage level
        - by_decision_type: error analysis per decision type
        - details: per-sample results (for verbose mode)
    """
    total = len(samples)
    with_actuals: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_coverage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decision_type_errors: dict[str, list[float]] = defaultdict(list)

    for sample in samples:
        pr = sample.get("pr_number", "?")
        est_lb = sample.get("estimated_lb", 0)
        est_ub = sample.get("estimated_ub", 0)
        est_midpoint = (est_lb + est_ub) / 2
        est_width = est_ub - est_lb

        # Get human values
        human_lb = sample.get("human_lb")
        human_ub = sample.get("human_ub")
        human_point = sample.get("human_point")

        # Compute human midpoint
        if human_lb is not None and human_ub is not None:
            human_midpoint = (human_lb + human_ub) / 2
            human_width = human_ub - human_lb
        elif human_point is not None:
            human_midpoint = human_point
            human_width = human_point * 0.4  # Default +/- 20%
            human_lb = int(human_point * 0.8)
            human_ub = int(human_point * 1.2)
        else:
            human_midpoint = None
            human_width = None

        detail: dict[str, Any] = {
            "pr_number": pr,
            "title": sample.get("title", "")[:40],
            "estimated_lb": est_lb,
            "estimated_ub": est_ub,
            "est_midpoint": est_midpoint,
            "est_width": est_width,
            "human_lb": human_lb,
            "human_ub": human_ub,
            "human_midpoint": human_midpoint,
            "human_width": human_width,
            "method": sample.get("method", "unknown"),
            "coverage": sample.get("coverage", "unknown"),
            "decision_count": sample.get("decision_count"),
            "within_bounds": None,
            "error": None,
            "width_ratio": None,
            "confidence": sample.get("human_confidence", "unknown"),
        }

        if human_midpoint is not None:
            within = est_lb <= human_midpoint <= est_ub
            error = est_midpoint - human_midpoint
            width_ratio = est_width / human_width if human_width and human_width > 0 else None

            detail["within_bounds"] = within
            detail["error"] = error
            detail["width_ratio"] = width_ratio
            with_actuals.append(detail)

            # Group by method and coverage
            by_method[detail["method"]].append(detail)
            by_coverage[detail["coverage"]].append(detail)

            # Attribute error to decision types
            decisions = sample.get("decisions_summary", [])
            if decisions and abs(error) > 0:
                error_per_decision = error / len(decisions)
                for d in decisions:
                    dtype = d.get("type", "unknown")
                    decision_type_errors[dtype].append(error_per_decision)

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
            "avg_bound_width": sum(d["est_width"] for d in details) / total if total else 0,
            "width_ratio": None,
            "by_method": {},
            "by_coverage": {},
            "by_decision_type": {},
            "details": details,
        }

    within_bounds = sum(1 for d in with_actuals if d["within_bounds"])
    errors = [d["error"] for d in with_actuals]
    mae = sum(abs(e) for e in errors) / len(errors)
    bias = sum(errors) / len(errors)
    width_ratios = [d["width_ratio"] for d in with_actuals if d["width_ratio"] is not None]
    avg_width_ratio = sum(width_ratios) / len(width_ratios) if width_ratios else None

    # Method-level metrics
    method_metrics = {}
    for method, method_samples in by_method.items():
        m_within = sum(1 for d in method_samples if d["within_bounds"])
        m_errors = [d["error"] for d in method_samples]
        method_metrics[method] = {
            "count": len(method_samples),
            "within_bounds": m_within,
            "coverage_rate": m_within / len(method_samples) * 100 if method_samples else 0,
            "mae": sum(abs(e) for e in m_errors) / len(m_errors) if m_errors else 0,
            "bias": sum(m_errors) / len(m_errors) if m_errors else 0,
        }

    # Coverage-level metrics
    coverage_metrics = {}
    for cov, cov_samples in by_coverage.items():
        c_within = sum(1 for d in cov_samples if d["within_bounds"])
        c_errors = [d["error"] for d in cov_samples]
        coverage_metrics[cov] = {
            "count": len(cov_samples),
            "within_bounds": c_within,
            "coverage_rate": c_within / len(cov_samples) * 100 if cov_samples else 0,
            "mae": sum(abs(e) for e in c_errors) / len(c_errors) if c_errors else 0,
            "bias": sum(c_errors) / len(c_errors) if c_errors else 0,
        }

    # Decision type error analysis
    decision_type_metrics = {}
    for dtype, dtype_errors in decision_type_errors.items():
        if dtype_errors:
            decision_type_metrics[dtype] = {
                "count": len(dtype_errors),
                "mean_error": sum(dtype_errors) / len(dtype_errors),
                "mae": sum(abs(e) for e in dtype_errors) / len(dtype_errors),
                "current_band": DECISION_TIME_BANDS.get(dtype, (0, 0)),
            }

    return {
        "total_samples": total,
        "samples_with_actuals": len(with_actuals),
        "within_bounds": within_bounds,
        "coverage_rate": within_bounds / len(with_actuals) * 100,
        "mean_absolute_error": mae,
        "bias": bias,
        "avg_bound_width": sum(d["est_width"] for d in details) / total if total else 0,
        "width_ratio": avg_width_ratio,
        "by_method": method_metrics,
        "by_coverage": coverage_metrics,
        "by_decision_type": decision_type_metrics,
        "details": details,
    }


def print_report(metrics: dict[str, Any], verbose: bool = False) -> None:
    """Print calibration report to stdout."""
    print("=" * 70)
    print("DevLT Estimation Calibration Report")
    print("=" * 70)
    print()

    print(f"Total samples:        {metrics['total_samples']}")
    print(f"Samples with actuals: {metrics['samples_with_actuals']}")
    print(f"Average bound width:  {metrics['avg_bound_width']:.1f} minutes")
    if metrics["width_ratio"]:
        print(f"Avg width ratio:      {metrics['width_ratio']:.2f}x (est/human)")
    print()

    if metrics["samples_with_actuals"] == 0:
        print("No samples with human-reported values to evaluate.")
        print()
        print("To add calibration data:")
        print("  1. Record actual time spent on a PR")
        print("  2. Run: python scripts/generate-pr-ledger.py --pr <N> --dump-bundle")
        print("  3. Add sample to docs/audit/calibration/samples.json")
        return

    print("-" * 50)
    print("Accuracy Metrics")
    print("-" * 50)
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

    # By-method breakdown if multiple methods
    if len(metrics["by_method"]) > 1:
        print("-" * 50)
        print("By Estimation Method")
        print("-" * 50)
        for method, m in metrics["by_method"].items():
            print(
                f"  {method}: {m['count']} samples, {m['coverage_rate']:.0f}% coverage, MAE {m['mae']:.1f}m, bias {m['bias']:+.1f}m"
            )
        print()

    # By-coverage breakdown if multiple coverage levels
    if len(metrics["by_coverage"]) > 1:
        print("-" * 50)
        print("By Coverage Level")
        print("-" * 50)
        for cov, c in metrics["by_coverage"].items():
            print(
                f"  {cov}: {c['count']} samples, {c['coverage_rate']:.0f}% coverage, MAE {c['mae']:.1f}m, bias {c['bias']:+.1f}m"
            )
        print()

    # Assessment
    print("-" * 50)
    print("Assessment")
    print("-" * 50)
    print()

    if coverage >= 85:
        print("[GOOD] Coverage rate >= 85%: bounds are well-calibrated")
    elif coverage >= 70:
        print("[OK] Coverage rate 70-85%: bounds are reasonable but could improve")
    else:
        print("[WARN] Coverage rate < 70%: bounds may be too narrow")

    mae = metrics["mean_absolute_error"]
    if mae <= 10:
        print("[GOOD] MAE <= 10 minutes: midpoint estimates are accurate")
    elif mae <= 20:
        print("[OK] MAE 10-20 minutes: midpoint estimates are reasonable")
    else:
        print("[WARN] MAE > 20 minutes: midpoint estimates need improvement")

    bias = metrics["bias"]
    if abs(bias) <= 5:
        print("[GOOD] Bias within +/- 5 minutes: no systematic error")
    elif abs(bias) <= 15:
        print("[OK] Bias 5-15 minutes: minor systematic error")
    else:
        direction = "overestimation" if bias > 0 else "underestimation"
        print(f"[WARN] Bias > 15 minutes: systematic {direction}")

    print()

    if verbose:
        print("-" * 50)
        print("Per-Sample Details")
        print("-" * 50)
        print()
        print(
            f"{'PR#':<6} {'Est LB':<7} {'Est UB':<7} {'Hum Mid':<8} {'Within':<8} {'Error':<8} {'Conf':<6}"
        )
        print("-" * 58)

        for d in metrics["details"]:
            pr = str(d["pr_number"])
            lb = str(d["estimated_lb"])
            ub = str(d["estimated_ub"])
            human_mid = f"{d['human_midpoint']:.0f}" if d["human_midpoint"] is not None else "-"
            within = "Yes" if d["within_bounds"] else ("No" if d["within_bounds"] is False else "-")
            error = f"{d['error']:+.0f}" if d["error"] is not None else "-"
            conf = d.get("confidence", "-")[:4]

            print(f"{pr:<6} {lb:<7} {ub:<7} {human_mid:<8} {within:<8} {error:<8} {conf:<6}")

        print()


def print_tuning_advice(metrics: dict[str, Any]) -> None:
    """Print actionable tuning advice based on error patterns."""
    print()
    print("=" * 70)
    print("Tuning Recommendations")
    print("=" * 70)
    print()

    if not metrics["by_decision_type"]:
        print("No decision-type error data available.")
        print("Add decisions_summary to samples for per-type analysis.")
        return

    print("Decision Type Error Analysis:")
    print("-" * 50)
    print(
        f"{'Type':<10} {'Count':<6} {'Mean Err':<10} {'MAE':<8} {'Current Band':<15} {'Suggested':<15}"
    )
    print("-" * 70)

    for dtype, data in sorted(metrics["by_decision_type"].items()):
        count = data["count"]
        mean_err = data["mean_error"]
        mae = data["mae"]
        current = data["current_band"]

        # Suggest adjustment based on error
        suggested = current
        if abs(mean_err) > 3:  # Significant bias
            if mean_err > 0:
                # Overestimating - reduce bounds
                new_min = max(1, current[0] - int(mean_err * 0.5))
                new_max = max(new_min + 2, current[1] - int(mean_err * 0.5))
                suggested = (new_min, new_max)
            else:
                # Underestimating - increase bounds
                new_min = current[0] + int(abs(mean_err) * 0.5)
                new_max = current[1] + int(abs(mean_err) * 0.5)
                suggested = (new_min, new_max)

        current_str = f"({current[0]}, {current[1]})"
        suggested_str = f"({suggested[0]}, {suggested[1]})" if suggested != current else "-"

        print(
            f"{dtype:<10} {count:<6} {mean_err:+.1f}m{'':>4} {mae:.1f}m{'':>3} {current_str:<15} {suggested_str:<15}"
        )

    print()
    print("Interpretation:")
    print("  - Positive Mean Err = overestimating (reduce band)")
    print("  - Negative Mean Err = underestimating (increase band)")
    print("  - High MAE with low Mean Err = high variance, consider wider bounds")
    print()

    # Overall bias advice
    bias = metrics.get("bias", 0)
    if bias is not None and abs(bias) > 10:
        print("Overall Model Adjustment:")
        if bias > 0:
            print(f"  Model overestimates by ~{bias:.0f} minutes on average.")
            print("  Consider reducing floor values in ESTIMATION_CONSTANTS or")
            print("  reducing decision time bands across the board.")
        else:
            print(f"  Model underestimates by ~{abs(bias):.0f} minutes on average.")
            print("  Consider increasing floor values in ESTIMATION_CONSTANTS or")
            print("  increasing decision time bands across the board.")
        print()

    # Coverage advice
    coverage = metrics.get("coverage_rate", 0)
    if coverage is not None and coverage < 70:
        print("Bounds Calibration:")
        print(f"  Only {coverage:.0f}% of human estimates fall within bounds.")
        print("  Bounds are too narrow. Consider:")
        print("    - Increasing session_slack_minutes in estimation.py")
        print("    - Widening decision time bands (increase max values)")
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
  %(prog)s --tuning-advice
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
    parser.add_argument(
        "--tuning-advice",
        "-t",
        action="store_true",
        help="Show actionable tuning recommendations based on error patterns",
    )

    args = parser.parse_args()

    # Load samples
    raw_samples = load_samples(args.samples_path)

    # Normalize samples to canonical format
    samples = [normalize_sample(s) for s in raw_samples]

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
        print(json.dumps(output, indent=2, default=str))
    else:
        print_report(metrics, verbose=args.verbose)
        if args.tuning_advice:
            print_tuning_advice(metrics)


if __name__ == "__main__":
    main()
