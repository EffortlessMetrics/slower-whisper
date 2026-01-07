#!/usr/bin/env python3
"""
Generate PR ledger and dossier with LLM analysis.

This script provides a multi-stage pipeline for PR analysis:

Stage 1 - Fact Bundle (deterministic, no LLM):
    python scripts/generate-pr-ledger.py --pr 123 --dump-bundle
    # Output: docs/audit/pr/_bundles/123.json

Stage 2 - LLM Analysis:
    python scripts/generate-pr-ledger.py --pr 123 --analyze --llm claude-code
    # Output: dossier JSON to stdout

Stage 3 - Full Publish:
    python scripts/generate-pr-ledger.py --pr 123 --publish --llm claude-code
    # Output: docs/audit/pr/123.json + optionally updates PR body

Legacy modes (still supported):
    python scripts/generate-pr-ledger.py --pr 123 --raw
    python scripts/generate-pr-ledger.py --pr 123  # Analysis prompt for manual LLM

Requires:
    - gh CLI installed and authenticated
    - claude-agent-sdk (for --llm mode)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PR ledger and dossier with LLM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1: Dump fact bundle (no LLM)
  %(prog)s --pr 123 --dump-bundle

  # Stage 2: Analyze with LLM
  %(prog)s --pr 123 --analyze --llm claude-code

  # Stage 3: Full publish
  %(prog)s --pr 123 --publish --llm claude-code --update-pr

  # Legacy: Generate analysis prompt for manual LLM
  %(prog)s --pr 123

  # Legacy: Output raw data as JSON
  %(prog)s --pr 123 --raw
""",
    )

    # Required
    parser.add_argument("--pr", "-p", type=int, required=True, help="PR number to analyze")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dump-bundle",
        action="store_true",
        help="Output fact bundle JSON (deterministic, no LLM)",
    )
    mode_group.add_argument(
        "--analyze",
        "-a",
        action="store_true",
        help="Run LLM analysis and output dossier (requires --llm)",
    )
    mode_group.add_argument(
        "--publish",
        action="store_true",
        help="Run full pipeline and publish dossier (requires --llm)",
    )
    mode_group.add_argument(
        "--raw",
        "-r",
        action="store_true",
        help="[Legacy] Output raw gathered data as JSON",
    )

    # LLM options
    parser.add_argument(
        "--llm",
        type=str,
        choices=["claude-code", "mock"],
        help="LLM provider for analysis (required for --analyze/--publish)",
    )

    # Output options
    parser.add_argument("--output", "-o", type=str, help="Output file path (default: stdout)")
    parser.add_argument(
        "--include-diff",
        action="store_true",
        help="Include full diff in bundle (can be large)",
    )

    # Publish options
    parser.add_argument(
        "--update-pr",
        action="store_true",
        help="Update PR description via gh (with --publish)",
    )
    parser.add_argument(
        "--skip-exhibits",
        action="store_true",
        help="Skip auto-updating EXHIBITS.md (with --publish)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )

    args = parser.parse_args()

    # Validate LLM requirement
    if (args.analyze or args.publish) and not args.llm:
        parser.error("--llm is required for --analyze and --publish modes")

    # Route to appropriate handler
    if args.dump_bundle:
        _handle_dump_bundle(args)
    elif args.analyze:
        _handle_analyze(args)
    elif args.publish:
        _handle_publish(args)
    elif args.raw:
        _handle_raw(args)
    else:
        # Default: generate analysis prompt (legacy mode)
        _handle_legacy_prompt(args)


def _handle_dump_bundle(args: argparse.Namespace) -> None:
    """Handle --dump-bundle mode."""
    from transcription.historian import gather_pr_data
    from transcription.historian.estimation import compute_bounded_estimate

    # Gather data
    bundle = gather_pr_data(args.pr, include_diff=args.include_diff)

    # Compute estimation
    compute_bounded_estimate(bundle)

    # Output
    output = json.dumps(bundle.to_dict(), indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Bundle written to: {args.output}", file=sys.stderr)
    else:
        # Default: write to bundles directory
        bundles_dir = Path("docs/audit/pr/_bundles")
        bundles_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundles_dir / f"{args.pr}.json"
        with open(bundle_path, "w") as f:
            f.write(output)
        print(f"Bundle written to: {bundle_path}", file=sys.stderr)


def _handle_analyze(args: argparse.Namespace) -> None:
    """Handle --analyze mode."""
    from transcription.historian import gather_pr_data
    from transcription.historian.estimation import compute_bounded_estimate
    from transcription.historian.llm_client import LLMConfig, create_llm_provider
    from transcription.historian.synthesis import run_all_analyzers, synthesize_dossier

    # Gather and estimate
    print(f"Gathering data for PR #{args.pr}...", file=sys.stderr)
    bundle = gather_pr_data(args.pr, include_diff=True)
    compute_bounded_estimate(bundle)

    # Create LLM provider
    config = LLMConfig(provider=args.llm)
    llm = create_llm_provider(config)

    # Run analyzers
    print("Running LLM analyzers...", file=sys.stderr)
    results = asyncio.run(run_all_analyzers(bundle, llm))

    # Report analyzer status
    for result in results:
        status = "ok" if result.success else "FAILED"
        print(f"  {result.name}: {status} ({result.duration_ms}ms)", file=sys.stderr)
        if result.errors:
            for err in result.errors[:3]:
                print(f"    - {err}", file=sys.stderr)

    # Synthesize dossier
    print("Synthesizing dossier...", file=sys.stderr)
    synthesis = synthesize_dossier(bundle, results)

    if synthesis.validation_errors:
        print("Validation warnings:", file=sys.stderr)
        for err in synthesis.validation_errors[:5]:
            print(f"  - {err}", file=sys.stderr)

    # Output
    output = json.dumps(synthesis.dossier, indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Dossier written to: {args.output}", file=sys.stderr)
    else:
        print(output)


def _handle_publish(args: argparse.Namespace) -> None:
    """Handle --publish mode."""
    from transcription.historian import gather_pr_data
    from transcription.historian.estimation import compute_bounded_estimate
    from transcription.historian.llm_client import LLMConfig, create_llm_provider
    from transcription.historian.publisher import publish_dossier
    from transcription.historian.synthesis import run_all_analyzers, synthesize_dossier

    # Gather and estimate
    print(f"Gathering data for PR #{args.pr}...", file=sys.stderr)
    bundle = gather_pr_data(args.pr, include_diff=True)
    compute_bounded_estimate(bundle)

    # Create LLM provider
    config = LLMConfig(provider=args.llm)
    llm = create_llm_provider(config)

    # Run analyzers
    print("Running LLM analyzers...", file=sys.stderr)
    results = asyncio.run(run_all_analyzers(bundle, llm))

    for result in results:
        status = "ok" if result.success else "FAILED"
        print(f"  {result.name}: {status}", file=sys.stderr)

    # Synthesize
    print("Synthesizing dossier...", file=sys.stderr)
    synthesis = synthesize_dossier(bundle, results)

    # Publish
    print("Publishing...", file=sys.stderr)
    result = publish_dossier(
        dossier=synthesis.dossier,
        pr_number=args.pr,
        update_pr=args.update_pr,
        update_exhibits=not args.skip_exhibits,
        dry_run=args.dry_run,
    )

    # Report
    if result.success:
        print("Published successfully:", file=sys.stderr)
        for note in result.notes:
            print(f"  - {note}", file=sys.stderr)
        if result.errors:
            print("Warnings:", file=sys.stderr)
            for err in result.errors:
                print(f"  - {err}", file=sys.stderr)
    else:
        print("Publish failed:", file=sys.stderr)
        for err in result.errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)


def _handle_raw(args: argparse.Namespace) -> None:
    """Handle --raw mode (legacy)."""
    data = _legacy_gather_pr_data(args.pr)

    if not args.include_diff:
        diff_len = len(data.get("diff", ""))
        data["diff"] = f"[diff omitted, {diff_len} chars, use --include-diff to include]"

    output = json.dumps(data, indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)


def _handle_legacy_prompt(args: argparse.Namespace) -> None:
    """Handle legacy analysis prompt mode."""
    data = _legacy_gather_pr_data(args.pr)

    if not args.include_diff:
        diff_len = len(data.get("diff", ""))
        data["diff"] = f"[diff omitted, {diff_len} chars, use --include-diff to include]"

    output = _generate_analysis_prompt(data)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)


# --- Legacy functions (preserved for backward compatibility) ---

import re
import subprocess
from datetime import datetime


def _run_gh(args: list[str], check: bool = True) -> dict[str, Any] | list[Any] | str | None:
    """Run gh CLI command and return parsed JSON or raw output."""
    cmd = ["gh"] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running gh command: {' '.join(cmd)}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("Error: gh CLI not found. Install from https://cli.github.com/", file=sys.stderr)
        sys.exit(1)


def _legacy_gather_pr_data(pr_number: int) -> dict[str, Any]:
    """Gather all PR data for analysis (legacy format)."""
    print(f"Gathering data for PR #{pr_number}...", file=sys.stderr)

    # Fetch metadata
    fields = [
        "number",
        "title",
        "url",
        "state",
        "createdAt",
        "mergedAt",
        "closedAt",
        "additions",
        "deletions",
        "changedFiles",
        "body",
        "labels",
        "author",
        "mergedBy",
        "baseRefName",
        "headRefName",
    ]
    metadata = _run_gh(["pr", "view", str(pr_number), "--json", ",".join(fields)])
    if not isinstance(metadata, dict):
        print(f"Error: Could not fetch PR #{pr_number}", file=sys.stderr)
        sys.exit(1)

    # Files
    files_data = _run_gh(["pr", "view", str(pr_number), "--json", "files"])
    files = files_data.get("files", []) if isinstance(files_data, dict) else []

    # Commits
    commits_data = _run_gh(["pr", "view", str(pr_number), "--json", "commits"])
    commits = commits_data.get("commits", []) if isinstance(commits_data, dict) else []

    # Comments and reviews
    comments_data = _run_gh(["pr", "view", str(pr_number), "--json", "comments,reviews"])
    comments = []
    if isinstance(comments_data, dict):
        comments.extend(comments_data.get("comments", []))
        for review in comments_data.get("reviews", []):
            if review.get("body"):
                comments.append(
                    {
                        "author": review.get("author", {}).get("login", "unknown"),
                        "body": review.get("body"),
                        "createdAt": review.get("submittedAt"),
                        "type": "review",
                        "state": review.get("state"),
                    }
                )

    # Check runs
    checks_data = _run_gh(["pr", "view", str(pr_number), "--json", "statusCheckRollup"])
    checks = checks_data.get("statusCheckRollup", []) if isinstance(checks_data, dict) else []

    # Diff
    diff = _run_gh(["pr", "diff", str(pr_number)], check=False)
    diff = diff if isinstance(diff, str) else ""
    if len(diff) > 50000:
        diff = diff[:50000] + "\n\n[... diff truncated at 50k chars ...]"

    # Derived analysis
    active_work = _estimate_active_work(commits)
    wall_clock = _calculate_wall_clock(metadata.get("createdAt"), metadata.get("mergedAt"))
    commit_patterns = _analyze_commit_patterns(commits)
    issue_refs = _extract_issue_refs(metadata.get("body", ""))
    file_categories = _categorize_files(files)

    changed = metadata.get("changedFiles", 0)
    blast_radius = "low" if changed <= 5 else ("medium" if changed <= 20 else "high")

    return {
        "pr_number": pr_number,
        "metadata": {
            "title": metadata.get("title"),
            "url": metadata.get("url"),
            "state": metadata.get("state"),
            "created_at": metadata.get("createdAt"),
            "merged_at": metadata.get("mergedAt"),
            "author": metadata.get("author", {}).get("login"),
            "merged_by": metadata.get("mergedBy", {}).get("login")
            if metadata.get("mergedBy")
            else None,
            "base_branch": metadata.get("baseRefName"),
            "head_branch": metadata.get("headRefName"),
            "labels": [label.get("name") for label in metadata.get("labels", [])],
        },
        "body": metadata.get("body", ""),
        "scope": {
            "files_changed": changed,
            "insertions": metadata.get("additions", 0),
            "deletions": metadata.get("deletions", 0),
            "blast_radius": blast_radius,
            "top_directories": _get_top_directories(files),
            "key_files": _get_key_files(files),
            "file_categories": file_categories,
        },
        "commits": {
            "count": len(commits),
            "messages": [
                {
                    "sha": c.get("oid", "")[:7] or c.get("commit", {}).get("oid", "")[:7],
                    "message": c.get("messageHeadline", "")
                    or c.get("commit", {}).get("messageHeadline", ""),
                    "date": c.get("committedDate") or c.get("commit", {}).get("committedDate"),
                }
                for c in commits
            ],
            "patterns": commit_patterns,
        },
        "comments": {
            "count": len(comments),
            "items": [
                {
                    "author": c.get("author", {}).get("login")
                    if isinstance(c.get("author"), dict)
                    else c.get("author"),
                    "body": c.get("body", "")[:1000],
                    "type": c.get("type", "comment"),
                    "state": c.get("state"),
                }
                for c in comments
            ],
        },
        "checks": {
            "items": [
                {
                    "name": ch.get("name") or ch.get("context"),
                    "status": ch.get("status") or ch.get("state"),
                    "conclusion": ch.get("conclusion"),
                }
                for ch in checks
            ]
            if checks
            else [],
        },
        "analysis": {
            "issue_refs": issue_refs,
            "active_work": active_work,
            "wall_clock": wall_clock,
        },
        "diff": diff,
    }


def _estimate_active_work(commits: list[dict], gap_minutes: int = 45) -> dict[str, Any]:
    if not commits:
        return {"hours": 0.5, "sessions": 1, "method": "minimum estimate (no commits)"}

    timestamps = []
    for commit in commits:
        date = commit.get("committedDate") or commit.get("commit", {}).get("committedDate")
        if date:
            try:
                timestamps.append(datetime.fromisoformat(date.replace("Z", "+00:00")))
            except (ValueError, TypeError):
                pass

    if len(timestamps) < 2:
        return {"hours": 0.5, "sessions": 1, "method": "minimum estimate (<2 commits)"}

    timestamps.sort()
    sessions = []
    start = end = timestamps[0]

    for ts in timestamps[1:]:
        if (ts - end).total_seconds() / 60 <= gap_minutes:
            end = ts
        else:
            sessions.append((start, end))
            start = end = ts
    sessions.append((start, end))

    total = sum((e - s).total_seconds() / 3600 for s, e in sessions) + len(sessions) * 0.25
    return {
        "hours": round(total, 1),
        "sessions": len(sessions),
        "method": f"commit bursts (gaps ≤{gap_minutes}min)",
    }


def _calculate_wall_clock(created: str | None, merged: str | None) -> dict[str, Any]:
    if not created:
        return {"days": None, "hours": None}
    try:
        c = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if merged:
            m = datetime.fromisoformat(merged.replace("Z", "+00:00"))
            delta = m - c
            return {
                "days": round(delta.total_seconds() / 86400, 1),
                "hours": round(delta.total_seconds() / 3600, 1),
            }
        return {"days": None, "hours": None, "note": "PR not merged"}
    except (ValueError, TypeError):
        return {"days": None, "hours": None}


def _extract_issue_refs(text: str) -> list[str]:
    if not text:
        return []
    matches = re.findall(
        r"#(\d+)|(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#?(\d+)", text, re.IGNORECASE
    )
    return sorted({f"#{g}" for m in matches for g in m if g})


def _analyze_commit_patterns(commits: list[dict]) -> dict[str, list]:
    patterns = {
        "fix_commits": [],
        "refactor_commits": [],
        "test_commits": [],
        "doc_commits": [],
        "revert_commits": [],
        "wip_commits": [],
    }
    p = {
        "fix": [r"\bfix\b", r"\bbug\b", r"\bissue\b", r"\bcorrect\b", r"\bresolve\b"],
        "refactor": [r"\brefactor\b", r"\bclean\b", r"\brestructure\b"],
        "test": [r"\btest\b", r"\bspec\b", r"\bcoverage\b"],
        "doc": [r"\bdoc\b", r"\breadme\b", r"\bcomment\b"],
        "revert": [r"\brevert\b", r"\bundo\b", r"\brollback\b"],
        "wip": [r"\bwip\b", r"\bwork in progress\b"],
    }
    for c in commits:
        msg = (
            c.get("messageHeadline", "") or c.get("commit", {}).get("messageHeadline", "")
        ).lower()
        sha = c.get("oid", "")[:7] or c.get("commit", {}).get("oid", "")[:7]
        entry = {"sha": sha, "message": msg}
        for key, regexes in p.items():
            if any(re.search(r, msg) for r in regexes):
                patterns[f"{key}_commits"].append(entry)
    return patterns


def _get_top_directories(files: list[dict], limit: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for f in files:
        parts = f.get("path", "").split("/")
        if len(parts) > 1:
            d = "/".join(parts[:-1]) + "/"
            counts[d] = counts.get(d, 0) + 1
    return [d for d, _ in sorted(counts.items(), key=lambda x: -x[1])[:limit]]


def _get_key_files(files: list[dict], limit: int = 10) -> list[dict]:
    return sorted(
        [
            {
                "path": f.get("path", ""),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
                "total_changes": f.get("additions", 0) + f.get("deletions", 0),
            }
            for f in files
        ],
        key=lambda x: -x["total_changes"],
    )[:limit]


def _categorize_files(files: list[dict]) -> dict[str, list[str]]:
    cats = {"source": [], "tests": [], "docs": [], "config": [], "other": []}
    for f in files:
        p = f.get("path", "")
        if any(x in p for x in ["/test", "test_", "_test."]) or p.startswith("test"):
            cats["tests"].append(p)
        elif p.endswith((".md", ".rst", ".txt")) or "/docs/" in p:
            cats["docs"].append(p)
        elif p.endswith((".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".lock")):
            cats["config"].append(p)
        elif p.endswith((".py", ".rs", ".ts", ".tsx", ".js", ".jsx")):
            cats["source"].append(p)
        else:
            cats["other"].append(p)
    return cats


def _generate_analysis_prompt(data: dict[str, Any]) -> str:
    """Generate a prompt for LLM analysis of the PR."""
    pr_num = data["pr_number"]
    meta = data["metadata"]
    scope = data["scope"]
    commits = data["commits"]
    comments = data["comments"]
    analysis = data["analysis"]

    commit_list = "\n".join(f"  - {c['sha']}: {c['message']}" for c in commits["messages"][:20])
    key_files_list = "\n".join(
        f"  - {f['path']} (+{f['additions']}/-{f['deletions']})" for f in scope["key_files"][:10]
    )
    comment_list = (
        "\n".join(
            f"  - [{c.get('type', 'comment')}] {c.get('author', 'unknown')}: {c.get('body', '')[:200]}"
            for c in comments["items"][:10]
        )
        or "  (no comments)"
    )

    patterns = commits["patterns"]
    friction = []
    if patterns["fix_commits"]:
        friction.append(f"  - {len(patterns['fix_commits'])} fix commits")
    if patterns["revert_commits"]:
        friction.append(f"  - {len(patterns['revert_commits'])} revert commits")
    if patterns["wip_commits"]:
        friction.append(f"  - {len(patterns['wip_commits'])} WIP commits")
    friction_str = "\n".join(friction) or "  (no obvious friction signals)"

    checks = data["checks"]["items"]
    failed = [c for c in checks if c.get("conclusion") in ("failure", "FAILURE")]
    check_summary = f"{len(checks)} checks, {len(failed)} failed" if checks else "no checks"

    return f"""# PR Analysis Request

Analyze PR #{pr_num} and generate a complete PR ledger dossier.

## PR Metadata
- **Title:** {meta["title"]}
- **URL:** {meta["url"]}
- **State:** {meta["state"]}
- **Author:** {meta["author"]}
- **Created:** {meta["created_at"]}
- **Merged:** {meta["merged_at"] or "not merged"}
- **Labels:** {", ".join(meta["labels"]) if meta["labels"] else "none"}

## PR Description
{data["body"][:3000] if data["body"] else "(no description)"}

## Scope
- Files changed: {scope["files_changed"]}
- Insertions: {scope["insertions"]}
- Deletions: {scope["deletions"]}
- Blast radius: {scope["blast_radius"]}
- Top directories: {", ".join(scope["top_directories"][:5])}

### Key Files
{key_files_list}

### File Categories
- Source: {len(scope["file_categories"]["source"])} files
- Tests: {len(scope["file_categories"]["tests"])} files
- Docs: {len(scope["file_categories"]["docs"])} files
- Config: {len(scope["file_categories"]["config"])} files

## Commits ({commits["count"]} total)
{commit_list}

## Friction Signals
{friction_str}

## Comments/Reviews ({comments["count"]} total)
{comment_list}

## CI Checks
{check_summary}

## Computed Metrics
- Wall-clock: {analysis["wall_clock"].get("days", "N/A")} days
- Active work: {analysis["active_work"]["hours"]}h ({analysis["active_work"]["sessions"]} sessions)
- Linked issues: {", ".join(analysis["issue_refs"]) if analysis["issue_refs"] else "none"}

---

## Analysis Tasks

Generate a complete PR dossier following docs/audit/PR_DOSSIER_SCHEMA.md with:

1. **Intent**: goal, type (feature/hardening/mechanization/perf-bench/refactor), phase
2. **Findings**: issues discovered, categorized by failure mode
3. **Process/Friction**: what went wrong events
4. **Evidence**: tests, docs, benchmarks assessment
5. **DevLT**: author and reviewer attention bands
6. **Reflection**: what went well / could be better
7. **Factory Delta**: new gates, contracts, prevention

Output as JSON.
"""


if __name__ == "__main__":
    main()
