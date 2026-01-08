#!/usr/bin/env python3
"""
Collect hard probes telemetry for PR analysis.

This script runs deterministic static analysis tools on changed files and
outputs a structured JSON telemetry bundle. This telemetry feeds into the
historian pipeline as measured facts (not estimates).

Usage:
    # Run against staged changes
    python scripts/collect-telemetry.py

    # Run against specific files
    python scripts/collect-telemetry.py --files src/feature.py tests/test_feature.py

    # Run against git diff (base branch)
    python scripts/collect-telemetry.py --diff main

    # Output to file
    python scripts/collect-telemetry.py --output telemetry.json

Hard probes included:
    - ruff: Lint + format checks
    - pytest: Test execution with coverage
    - radon: Complexity metrics (cc/mi)
    - pip-audit: Security vulnerability scan (optional)

Output schema:
    {
        "collected_at": ISO8601,
        "files_analyzed": [...],
        "probes": {
            "ruff": {...},
            "pytest": {...},
            "radon": {...},
            "pip_audit": {...}
        },
        "summary": {
            "total_issues": int,
            "by_severity": {...},
            "quality_band": "high|medium|low"
        }
    }
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def run_command(cmd: list[str], capture: bool = True) -> tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=300,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -2, "", f"Command not found: {cmd[0]}"


def get_changed_files(base: str = "main") -> list[str]:
    """Get list of Python files changed vs base branch."""
    rc, stdout, _ = run_command(["git", "diff", "--name-only", base, "--", "*.py"])
    if rc != 0:
        return []
    return [f.strip() for f in stdout.strip().split("\n") if f.strip()]


def collect_ruff(files: list[str]) -> dict[str, Any]:
    """Run ruff check and return results."""
    if not files:
        return {"skipped": True, "reason": "no files"}

    # Check for lint issues
    cmd = ["ruff", "check", "--output-format=json", *files]
    rc, stdout, stderr = run_command(cmd)

    issues = []
    if stdout.strip():
        try:
            issues = json.loads(stdout)
        except json.JSONDecodeError:
            pass

    # Also check formatting
    format_cmd = ["ruff", "format", "--check", "--diff", *files]
    format_rc, _, _ = run_command(format_cmd)

    return {
        "exit_code": rc,
        "issues": issues,
        "issue_count": len(issues),
        "format_ok": format_rc == 0,
        "by_rule": _group_by_key(issues, "code"),
        "by_file": _group_by_key(issues, "filename"),
    }


def collect_pytest(files: list[str] | None = None) -> dict[str, Any]:
    """Run pytest and return results."""
    cmd = ["pytest", "--json-report", "--json-report-file=-", "-q"]
    if files:
        # Find related test files
        test_files = [f for f in files if "test" in f]
        if test_files:
            cmd.extend(test_files)
        else:
            # Run tests that might cover these files
            cmd.extend(["--co", "-q"])  # Just collect, don't run

    rc, stdout, stderr = run_command(cmd)

    result: dict[str, Any] = {
        "exit_code": rc,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration_seconds": 0,
    }

    # Try to parse JSON report
    try:
        # pytest-json-report outputs JSON to the specified file
        # When using -, it goes to stdout but may be mixed with other output
        for line in stdout.split("\n"):
            if line.strip().startswith("{"):
                report = json.loads(line)
                summary = report.get("summary", {})
                result["passed"] = summary.get("passed", 0)
                result["failed"] = summary.get("failed", 0)
                result["skipped"] = summary.get("skipped", 0)
                result["errors"] = summary.get("error", 0)
                result["duration_seconds"] = report.get("duration", 0)
                result["tests"] = [
                    {
                        "name": t.get("nodeid"),
                        "outcome": t.get("outcome"),
                        "duration": t.get("call", {}).get("duration"),
                    }
                    for t in report.get("tests", [])
                ]
                break
    except (json.JSONDecodeError, KeyError):
        # Fallback: parse text output
        if "passed" in stderr or "passed" in stdout:
            result["raw_output"] = stderr or stdout

    return result


def collect_radon(files: list[str]) -> dict[str, Any]:
    """Run radon complexity analysis and return results."""
    if not files:
        return {"skipped": True, "reason": "no files"}

    # Cyclomatic complexity
    cc_cmd = ["radon", "cc", "-j", *files]
    cc_rc, cc_stdout, _ = run_command(cc_cmd)

    cc_results: dict[str, Any] = {}
    if cc_stdout.strip():
        try:
            cc_results = json.loads(cc_stdout)
        except json.JSONDecodeError:
            pass

    # Maintainability index
    mi_cmd = ["radon", "mi", "-j", *files]
    mi_rc, mi_stdout, _ = run_command(mi_cmd)

    mi_results: dict[str, Any] = {}
    if mi_stdout.strip():
        try:
            mi_results = json.loads(mi_stdout)
        except json.JSONDecodeError:
            pass

    # Summarize
    high_complexity = []
    low_maintainability = []

    for filename, functions in cc_results.items():
        for func in functions:
            if func.get("complexity", 0) >= 10:
                high_complexity.append(
                    {
                        "file": filename,
                        "name": func.get("name"),
                        "complexity": func.get("complexity"),
                        "rank": func.get("rank"),
                    }
                )

    for filename, mi_data in mi_results.items():
        # radon mi returns {"filename": {"mi": score, "rank": "A/B/C"}}
        if isinstance(mi_data, dict):
            mi_score = mi_data.get("mi", 100)
            if mi_score < 65:  # Low maintainability threshold
                low_maintainability.append(
                    {
                        "file": filename,
                        "mi": mi_score,
                        "rank": mi_data.get("rank"),
                    }
                )

    return {
        "cyclomatic_complexity": cc_results,
        "maintainability_index": mi_results,
        "high_complexity_functions": high_complexity,
        "low_maintainability_files": low_maintainability,
        "cc_available": cc_rc == 0,
        "mi_available": mi_rc == 0,
    }


def collect_pip_audit() -> dict[str, Any]:
    """Run pip-audit for security vulnerabilities."""
    cmd = ["pip-audit", "-f", "json"]
    rc, stdout, stderr = run_command(cmd)

    if rc == -2:
        return {"skipped": True, "reason": "pip-audit not installed"}

    vulnerabilities = []
    if stdout.strip():
        try:
            vulnerabilities = json.loads(stdout)
        except json.JSONDecodeError:
            pass

    return {
        "exit_code": rc,
        "vulnerabilities": vulnerabilities,
        "vuln_count": len(vulnerabilities),
        "by_severity": _group_vulnerabilities(vulnerabilities),
    }


def _group_by_key(items: list[dict], key: str) -> dict[str, int]:
    """Group items by a key and count occurrences."""
    counts: dict[str, int] = {}
    for item in items:
        value = item.get(key, "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _group_vulnerabilities(vulns: list[dict]) -> dict[str, int]:
    """Group vulnerabilities by severity."""
    counts: dict[str, int] = {}
    for vuln in vulns:
        # pip-audit doesn't have severity, so we count by package
        pkg = vuln.get("name", "unknown")
        counts[pkg] = counts.get(pkg, 0) + 1
    return counts


def compute_quality_band(telemetry: dict[str, Any]) -> str:
    """Compute overall quality band from telemetry."""
    score = 100

    # Deduct for ruff issues
    ruff = telemetry.get("probes", {}).get("ruff", {})
    issue_count = ruff.get("issue_count", 0)
    if issue_count > 20:
        score -= 30
    elif issue_count > 10:
        score -= 20
    elif issue_count > 0:
        score -= 10

    if not ruff.get("format_ok", True):
        score -= 10

    # Deduct for test failures
    pytest_data = telemetry.get("probes", {}).get("pytest", {})
    if pytest_data.get("failed", 0) > 0:
        score -= 25
    if pytest_data.get("errors", 0) > 0:
        score -= 15

    # Deduct for complexity issues
    radon = telemetry.get("probes", {}).get("radon", {})
    if len(radon.get("high_complexity_functions", [])) > 3:
        score -= 15
    elif len(radon.get("high_complexity_functions", [])) > 0:
        score -= 5

    if len(radon.get("low_maintainability_files", [])) > 0:
        score -= 10

    # Deduct for security issues
    pip_audit = telemetry.get("probes", {}).get("pip_audit", {})
    if pip_audit.get("vuln_count", 0) > 0:
        score -= 20

    if score >= 80:
        return "high"
    elif score >= 50:
        return "medium"
    else:
        return "low"


def collect_telemetry(
    files: list[str] | None = None,
    base: str | None = None,
    skip_pytest: bool = False,
    skip_pip_audit: bool = False,
) -> dict[str, Any]:
    """Collect all telemetry probes."""
    # Determine files to analyze
    if files:
        py_files = [f for f in files if f.endswith(".py")]
    elif base:
        py_files = get_changed_files(base)
    else:
        # Default to staged changes
        rc, stdout, _ = run_command(["git", "diff", "--cached", "--name-only", "--", "*.py"])
        py_files = [f.strip() for f in stdout.strip().split("\n") if f.strip()]

    telemetry: dict[str, Any] = {
        "collected_at": datetime.now(UTC).isoformat(),
        "files_analyzed": py_files,
        "file_count": len(py_files),
        "probes": {},
    }

    # Run probes
    print("Collecting ruff telemetry...", file=sys.stderr)
    telemetry["probes"]["ruff"] = collect_ruff(py_files)

    if not skip_pytest:
        print("Collecting pytest telemetry...", file=sys.stderr)
        telemetry["probes"]["pytest"] = collect_pytest(py_files if py_files else None)

    print("Collecting radon telemetry...", file=sys.stderr)
    telemetry["probes"]["radon"] = collect_radon(py_files)

    if not skip_pip_audit:
        print("Collecting pip-audit telemetry...", file=sys.stderr)
        telemetry["probes"]["pip_audit"] = collect_pip_audit()

    # Compute summary
    ruff = telemetry["probes"].get("ruff", {})
    pytest_data = telemetry["probes"].get("pytest", {})
    radon = telemetry["probes"].get("radon", {})
    pip_audit = telemetry["probes"].get("pip_audit", {})

    total_issues = (
        ruff.get("issue_count", 0)
        + pytest_data.get("failed", 0)
        + pytest_data.get("errors", 0)
        + len(radon.get("high_complexity_functions", []))
        + pip_audit.get("vuln_count", 0)
    )

    telemetry["summary"] = {
        "total_issues": total_issues,
        "by_category": {
            "lint": ruff.get("issue_count", 0),
            "format": 0 if ruff.get("format_ok", True) else 1,
            "test_failures": pytest_data.get("failed", 0),
            "test_errors": pytest_data.get("errors", 0),
            "complexity": len(radon.get("high_complexity_functions", [])),
            "maintainability": len(radon.get("low_maintainability_files", [])),
            "security": pip_audit.get("vuln_count", 0),
        },
        "quality_band": compute_quality_band(telemetry),
    }

    return telemetry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect hard probes telemetry for PR analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Specific files to analyze (default: git staged files)",
    )
    parser.add_argument(
        "--diff",
        metavar="BASE",
        help="Analyze files changed vs BASE branch (e.g., main)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip pytest execution",
    )
    parser.add_argument(
        "--skip-pip-audit",
        action="store_true",
        help="Skip pip-audit security scan",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    # Collect telemetry
    telemetry = collect_telemetry(
        files=args.files,
        base=args.diff,
        skip_pytest=args.skip_pytest,
        skip_pip_audit=args.skip_pip_audit,
    )

    # Output
    indent = 2 if args.pretty else None
    output = json.dumps(telemetry, indent=indent)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Telemetry written to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Exit with non-zero if quality is low
    if telemetry["summary"]["quality_band"] == "low":
        sys.exit(1)


if __name__ == "__main__":
    main()
