#!/usr/bin/env python3
"""Compare two evaluation result files to show improvements/regressions.

Usage:
    uv run python benchmarks/compare_results.py \\
        --before benchmarks/results/baseline.json \\
        --after benchmarks/results/improved.json

    # With failure analysis diff
    uv run python benchmarks/compare_results.py \\
        --before benchmarks/results/baseline.json \\
        --after benchmarks/results/improved.json \\
        --show-failures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_scores(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Compare aggregate scores between two runs."""
    before_scores = before.get("aggregate_scores", {})
    after_scores = after.get("aggregate_scores", {})

    print("\n" + "=" * 70)
    print("AGGREGATE SCORES COMPARISON")
    print("=" * 70)

    # Header
    print(f"\n{'Metric':<20} {'Before':>10} {'After':>10} {'Delta':>10} {'Change':<10}")
    print("-" * 70)

    # Compare each metric
    metrics = sorted(set(before_scores.keys()) | set(after_scores.keys()))
    for metric in metrics:
        before_val = before_scores.get(metric, 0.0)
        after_val = after_scores.get(metric, 0.0)
        delta = after_val - before_val

        # Determine if this is an improvement
        if abs(delta) < 0.05:
            change = "~"
        elif delta > 0:
            change = "↑ BETTER"
        else:
            change = "↓ WORSE"

        # Format with color indicators (simple text, no ANSI)
        print(f"{metric:<20} {before_val:>10.2f} {after_val:>10.2f} {delta:>+10.2f} {change:<10}")

    # Overall assessment
    avg_delta = sum(after_scores.get(m, 0) - before_scores.get(m, 0) for m in metrics) / len(
        metrics
    )

    print("\n" + "-" * 70)
    print(f"Average delta: {avg_delta:+.2f}")
    if avg_delta > 0.1:
        print("Overall assessment: SIGNIFICANT IMPROVEMENT ✓")
    elif avg_delta > 0:
        print("Overall assessment: Slight improvement")
    elif avg_delta > -0.1:
        print("Overall assessment: No meaningful change")
    else:
        print("Overall assessment: REGRESSION ✗")


def compare_failures(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Compare failure patterns between two runs."""
    before_analysis = before.get("failure_analysis", {})
    after_analysis = after.get("failure_analysis", {})

    if not before_analysis or not after_analysis:
        print("\n(Failure analysis not available in one or both runs)")
        return

    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS COMPARISON")
    print("=" * 70)

    # Bad cases count
    before_bad = len(before_analysis.get("bad_cases", []))
    after_bad = len(after_analysis.get("bad_cases", []))
    delta_bad = after_bad - before_bad

    print(f"\nBad cases: {before_bad} → {after_bad} ({delta_bad:+d})")

    # Category breakdown
    before_cats = before_analysis.get("categories", {})
    after_cats = after_analysis.get("categories", {})

    if before_cats or after_cats:
        print(f"\n{'Category':<20} {'Before':>10} {'After':>10} {'Delta':>10}")
        print("-" * 70)

        categories = sorted(set(before_cats.keys()) | set(after_cats.keys()))
        for cat in categories:
            before_count = before_cats.get(cat, 0)
            after_count = after_cats.get(cat, 0)
            delta = after_count - before_count
            print(f"{cat:<20} {before_count:>10} {after_count:>10} {delta:>+10}")

    # Recommendations comparison
    after_recs = after_analysis.get("recommendations", [])

    if after_recs:
        print("\nNew recommendations:")
        for rec in after_recs:
            print(f"  • {rec}")


def compare_metadata(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Show key metadata differences."""
    before_meta = before.get("metadata", {})
    after_meta = after.get("metadata", {})

    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    # Show side-by-side only if different
    keys = sorted(set(before_meta.keys()) | set(after_meta.keys()))
    differences = []

    for key in keys:
        before_val = before_meta.get(key)
        after_val = after_meta.get(key)
        if before_val != after_val:
            differences.append((key, before_val, after_val))

    if differences:
        print("\nConfiguration changes:")
        for key, before_val, after_val in differences:
            print(f"  {key}: {before_val} → {after_val}")
    else:
        print("\n(Same configuration)")

    # Show sample count
    before_n = before_meta.get("n_meetings") or before_meta.get("n_samples")
    after_n = after_meta.get("n_meetings") or after_meta.get("n_samples")
    print(f"\nSamples: {before_n} (before), {after_n} (after)")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare two evaluation result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python benchmarks/compare_results.py --before baseline.json --after improved.json

  # With failure analysis
  python benchmarks/compare_results.py \\
      --before baseline.json \\
      --after improved.json \\
      --show-failures
""",
    )
    parser.add_argument(
        "--before",
        type=Path,
        required=True,
        help="Baseline results JSON file",
    )
    parser.add_argument(
        "--after",
        type=Path,
        required=True,
        help="New results JSON file to compare",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Show detailed failure analysis comparison",
    )

    args = parser.parse_args()

    # Load files
    try:
        before = load_results(args.before)
        after = load_results(args.after)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        return 1

    # Show comparison
    compare_metadata(before, after)
    compare_scores(before, after)

    if args.show_failures:
        compare_failures(before, after)

    print("\n" + "=" * 70)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
