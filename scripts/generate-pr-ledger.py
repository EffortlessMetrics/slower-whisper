#!/usr/bin/env python3
"""
Generate PR ledger and dossier with LLM analysis.

This script gathers comprehensive PR data from GitHub and outputs it in a format
suitable for LLM analysis. The LLM then analyzes the data to produce:
- Goal and intent
- Findings and friction events
- DevLT estimates
- Reflection (what went well / could be better)

Usage:
    # Gather data for LLM analysis (outputs analysis prompt)
    python scripts/generate-pr-ledger.py --pr 123 --analyze

    # Output raw data as JSON (for inspection)
    python scripts/generate-pr-ledger.py --pr 123 --raw

    # Generate dossier template (with placeholders)
    python scripts/generate-pr-ledger.py --pr 123 --format json

Requires:
    - gh CLI installed and authenticated
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def run_gh(args: list[str], check: bool = True) -> dict[str, Any] | list[Any] | str | None:
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


def get_pr_metadata(pr_number: int) -> dict[str, Any]:
    """Fetch comprehensive PR metadata."""
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
    data = run_gh(["pr", "view", str(pr_number), "--json", ",".join(fields)])
    return data if isinstance(data, dict) else {}


def get_pr_files(pr_number: int) -> list[dict[str, Any]]:
    """Fetch changed files with diff stats."""
    data = run_gh(["pr", "view", str(pr_number), "--json", "files"])
    if isinstance(data, dict) and "files" in data:
        return data["files"]
    return []


def get_pr_commits(pr_number: int) -> list[dict[str, Any]]:
    """Fetch commits with messages."""
    data = run_gh(["pr", "view", str(pr_number), "--json", "commits"])
    if isinstance(data, dict) and "commits" in data:
        return data["commits"]
    return []


def get_pr_comments(pr_number: int) -> list[dict[str, Any]]:
    """Fetch PR comments (review comments and issue comments)."""
    data = run_gh(["pr", "view", str(pr_number), "--json", "comments,reviews"])
    comments = []
    if isinstance(data, dict):
        if "comments" in data:
            comments.extend(data["comments"])
        if "reviews" in data:
            for review in data["reviews"]:
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
    return comments


def get_pr_diff(pr_number: int) -> str:
    """Fetch the PR diff."""
    result = run_gh(["pr", "diff", str(pr_number)], check=False)
    return result if isinstance(result, str) else ""


def get_check_runs(pr_number: int) -> list[dict[str, Any]]:
    """Fetch CI check runs and their status."""
    data = run_gh(["pr", "view", str(pr_number), "--json", "statusCheckRollup"])
    if isinstance(data, dict) and "statusCheckRollup" in data:
        return data["statusCheckRollup"] or []
    return []


def estimate_active_work(commits: list[dict[str, Any]], gap_minutes: int = 45) -> dict[str, Any]:
    """
    Estimate active work using commit burst heuristic.
    Returns hours and session details.
    """
    if not commits:
        return {"hours": 0.5, "sessions": 1, "method": "minimum estimate (no commits)"}

    timestamps = []
    for commit in commits:
        committed_date = commit.get("committedDate") or commit.get("commit", {}).get(
            "committedDate"
        )
        if committed_date:
            try:
                dt = datetime.fromisoformat(committed_date.replace("Z", "+00:00"))
                timestamps.append(dt)
            except (ValueError, TypeError):
                pass

    if len(timestamps) < 2:
        return {"hours": 0.5, "sessions": 1, "method": "minimum estimate (<2 commits)"}

    timestamps.sort()

    sessions = []
    session_start = timestamps[0]
    session_end = timestamps[0]

    for ts in timestamps[1:]:
        gap = (ts - session_end).total_seconds() / 60
        if gap <= gap_minutes:
            session_end = ts
        else:
            sessions.append((session_start, session_end))
            session_start = ts
            session_end = ts

    sessions.append((session_start, session_end))

    total_hours = sum((end - start).total_seconds() / 3600 for start, end in sessions)
    total_hours += len(sessions) * 0.25  # Add minimum time per session

    return {
        "hours": round(total_hours, 1),
        "sessions": len(sessions),
        "method": f"commit bursts (gaps ≤{gap_minutes}min = same session)",
    }


def calculate_wall_clock(created_at: str | None, merged_at: str | None) -> dict[str, Any]:
    """Calculate wall clock metrics."""
    if not created_at:
        return {"days": None, "hours": None}

    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if merged_at:
            merged = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
            delta = merged - created
            return {
                "days": round(delta.total_seconds() / 86400, 1),
                "hours": round(delta.total_seconds() / 3600, 1),
            }
        else:
            return {"days": None, "hours": None, "note": "PR not merged"}
    except (ValueError, TypeError):
        return {"days": None, "hours": None}


def extract_issue_refs(text: str) -> list[str]:
    """Extract issue references from text."""
    if not text:
        return []
    pattern = r"#(\d+)|(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#?(\d+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    refs = set()
    for m in matches:
        for g in m:
            if g:
                refs.add(f"#{g}")
    return sorted(refs)


def analyze_commit_patterns(commits: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze commit messages for patterns indicating friction/fixes."""
    patterns = {
        "fix_commits": [],
        "refactor_commits": [],
        "test_commits": [],
        "doc_commits": [],
        "revert_commits": [],
        "wip_commits": [],
    }

    fix_patterns = [r"\bfix\b", r"\bbug\b", r"\bissue\b", r"\bcorrect\b", r"\bresolve\b"]
    refactor_patterns = [r"\brefactor\b", r"\bclean\b", r"\brestructure\b"]
    test_patterns = [r"\btest\b", r"\bspec\b", r"\bcoverage\b"]
    doc_patterns = [r"\bdoc\b", r"\breadme\b", r"\bcomment\b"]
    revert_patterns = [r"\brevert\b", r"\bundo\b", r"\brollback\b"]
    wip_patterns = [r"\bwip\b", r"\bwork in progress\b", r"\bwip:\b"]

    for commit in commits:
        msg = commit.get("messageHeadline", "") or commit.get("commit", {}).get(
            "messageHeadline", ""
        )
        msg_lower = msg.lower()
        sha = commit.get("oid", "")[:7] or commit.get("commit", {}).get("oid", "")[:7]

        entry = {"sha": sha, "message": msg}

        if any(re.search(p, msg_lower) for p in fix_patterns):
            patterns["fix_commits"].append(entry)
        if any(re.search(p, msg_lower) for p in refactor_patterns):
            patterns["refactor_commits"].append(entry)
        if any(re.search(p, msg_lower) for p in test_patterns):
            patterns["test_commits"].append(entry)
        if any(re.search(p, msg_lower) for p in doc_patterns):
            patterns["doc_commits"].append(entry)
        if any(re.search(p, msg_lower) for p in revert_patterns):
            patterns["revert_commits"].append(entry)
        if any(re.search(p, msg_lower) for p in wip_patterns):
            patterns["wip_commits"].append(entry)

    return patterns


def get_top_directories(files: list[dict[str, Any]], limit: int = 5) -> list[str]:
    """Extract top directories from changed files."""
    dir_counts: dict[str, int] = {}
    for f in files:
        path = f.get("path", "")
        parts = path.split("/")
        if len(parts) > 1:
            dir_path = "/".join(parts[:-1]) + "/"
            dir_counts[dir_path] = dir_counts.get(dir_path, 0) + 1

    sorted_dirs = sorted(dir_counts.items(), key=lambda x: -x[1])
    return [d for d, _ in sorted_dirs[:limit]]


def get_key_files(files: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    """Extract key files with change stats."""
    scored = []
    for f in files:
        path = f.get("path", "")
        additions = f.get("additions", 0)
        deletions = f.get("deletions", 0)
        scored.append(
            {
                "path": path,
                "additions": additions,
                "deletions": deletions,
                "total_changes": additions + deletions,
            }
        )

    return sorted(scored, key=lambda x: -x["total_changes"])[:limit]


def categorize_file_changes(files: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Categorize files by type."""
    categories = {
        "source": [],
        "tests": [],
        "docs": [],
        "config": [],
        "other": [],
    }

    for f in files:
        path = f.get("path", "")
        if "/test" in path or path.startswith("test") or "_test." in path or "test_" in path:
            categories["tests"].append(path)
        elif path.endswith((".md", ".rst", ".txt")) or "/docs/" in path or "/doc/" in path:
            categories["docs"].append(path)
        elif path.endswith((".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".lock")):
            categories["config"].append(path)
        elif path.endswith((".py", ".rs", ".ts", ".tsx", ".js", ".jsx")):
            categories["source"].append(path)
        else:
            categories["other"].append(path)

    return categories


def gather_pr_data(pr_number: int) -> dict[str, Any]:
    """Gather all PR data for analysis."""
    print(f"Gathering data for PR #{pr_number}...", file=sys.stderr)

    metadata = get_pr_metadata(pr_number)
    if not metadata:
        print(f"Error: Could not fetch PR #{pr_number}", file=sys.stderr)
        sys.exit(1)

    files = get_pr_files(pr_number)
    commits = get_pr_commits(pr_number)
    comments = get_pr_comments(pr_number)
    checks = get_check_runs(pr_number)

    # Get diff (can be large, truncate if needed)
    diff = get_pr_diff(pr_number)
    if len(diff) > 50000:
        diff = diff[:50000] + "\n\n[... diff truncated at 50k chars ...]"

    # Derived analysis
    active_work = estimate_active_work(commits)
    wall_clock = calculate_wall_clock(metadata.get("createdAt"), metadata.get("mergedAt"))
    commit_patterns = analyze_commit_patterns(commits)
    issue_refs = extract_issue_refs(metadata.get("body", ""))
    file_categories = categorize_file_changes(files)

    # Determine blast radius
    changed = metadata.get("changedFiles", 0)
    if changed <= 5:
        blast_radius = "low"
    elif changed <= 20:
        blast_radius = "medium"
    else:
        blast_radius = "high"

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
            "files_changed": metadata.get("changedFiles", 0),
            "insertions": metadata.get("additions", 0),
            "deletions": metadata.get("deletions", 0),
            "blast_radius": blast_radius,
            "top_directories": get_top_directories(files),
            "key_files": get_key_files(files),
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
                    "body": c.get("body", "")[:1000],  # Truncate long comments
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


def generate_analysis_prompt(data: dict[str, Any]) -> str:
    """Generate a prompt for LLM analysis of the PR."""
    pr_num = data["pr_number"]
    meta = data["metadata"]
    scope = data["scope"]
    commits = data["commits"]
    comments = data["comments"]
    analysis = data["analysis"]

    # Format commits
    commit_list = "\n".join(f"  - {c['sha']}: {c['message']}" for c in commits["messages"][:20])

    # Format key files
    key_files_list = "\n".join(
        f"  - {f['path']} (+{f['additions']}/-{f['deletions']})" for f in scope["key_files"][:10]
    )

    # Format comments
    comment_list = (
        "\n".join(
            f"  - [{c.get('type', 'comment')}] {c.get('author', 'unknown')}: {c['body'][:200]}..."
            if len(c.get("body", "")) > 200
            else f"  - [{c.get('type', 'comment')}] {c.get('author', 'unknown')}: {c.get('body', '')}"
            for c in comments["items"][:10]
        )
        if comments["items"]
        else "  (no comments)"
    )

    # Format commit patterns (friction signals)
    patterns = commits["patterns"]
    friction_signals = []
    if patterns["fix_commits"]:
        friction_signals.append(
            f"  - {len(patterns['fix_commits'])} fix commits: {[c['message'] for c in patterns['fix_commits'][:3]]}"
        )
    if patterns["revert_commits"]:
        friction_signals.append(
            f"  - {len(patterns['revert_commits'])} revert commits: {[c['message'] for c in patterns['revert_commits']]}"
        )
    if patterns["wip_commits"]:
        friction_signals.append(f"  - {len(patterns['wip_commits'])} WIP commits")
    friction_str = (
        "\n".join(friction_signals) if friction_signals else "  (no obvious friction signals)"
    )

    # Format checks
    checks = data["checks"]["items"]
    failed_checks = [c for c in checks if c.get("conclusion") in ("failure", "FAILURE")]
    check_summary = (
        f"{len(checks)} checks, {len(failed_checks)} failed" if checks else "no checks found"
    )

    prompt = f"""# PR Analysis Request

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

## Friction Signals (from commit patterns)
{friction_str}

## Comments/Reviews ({comments["count"]} total)
{comment_list}

## CI Checks
{check_summary}

## Computed Metrics
- Wall-clock: {analysis["wall_clock"].get("days", "N/A")} days
- Active work estimate: {analysis["active_work"]["hours"]}h ({analysis["active_work"]["sessions"]} sessions)
- Linked issues: {", ".join(analysis["issue_refs"]) if analysis["issue_refs"] else "none detected"}

---

## Analysis Tasks

Based on the above data, generate a complete PR dossier with:

### 1. Intent
- **Goal:** 1-2 sentence description of what this PR accomplishes
- **Type:** feature | hardening | mechanization | perf/bench | refactor
- **Phase:** infer from context (v1.x, v2.0-track1, etc.) or "unknown"

### 2. Findings
Identify any issues discovered during the PR:
- Look for fix commits, reverts, review comments requesting changes
- Categorize by failure mode type (measurement drift, doc drift, packaging drift, test flake, dependency hazard, process mismatch)
- Note how each was detected (gate, review, post-merge) and disposition (fixed here, deferred)

### 3. Process/Friction Events
Extract "what went wrong" events:
- Multiple fix commits on the same issue
- Review comments pointing out problems
- Reverts or significant rework
- Any iteration patterns visible in commits

### 4. Evidence Assessment
Based on file changes, assess:
- Were tests added/modified?
- Were docs updated?
- Any benchmark/perf files changed?

### 5. DevLT Estimate
Based on:
- Active work time ({analysis["active_work"]["hours"]}h)
- Number of commits ({commits["count"]})
- Comment/review back-and-forth ({comments["count"]} comments)
Estimate author and reviewer attention bands: 10-20m | 20-45m | 45-90m | >90m

### 6. Reflection
- What went well (2-5 bullets)
- What could be better (2-5 actionable items)

### 7. Factory Delta
- Any new gates, contracts, or prevention mechanisms added?
- Any prevention issues that should be filed?

---

Output the analysis as a complete JSON dossier following the schema in docs/audit/PR_DOSSIER_SCHEMA.md
"""

    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PR ledger with LLM analysis")
    parser.add_argument("--pr", "-p", type=int, required=True, help="PR number to analyze")
    parser.add_argument(
        "--analyze", "-a", action="store_true", help="Output analysis prompt for LLM (default mode)"
    )
    parser.add_argument("--raw", "-r", action="store_true", help="Output raw gathered data as JSON")
    parser.add_argument("--output", "-o", type=str, help="Output file path (default: stdout)")
    parser.add_argument(
        "--include-diff", action="store_true", help="Include full diff in output (can be large)"
    )

    args = parser.parse_args()

    # Gather data
    data = gather_pr_data(args.pr)

    # Remove diff if not requested (to reduce output size)
    if not args.include_diff:
        data["diff"] = f"[diff omitted, {len(data['diff'])} chars, use --include-diff to include]"

    if args.raw:
        output = json.dumps(data, indent=2)
    else:
        # Default: generate analysis prompt
        output = generate_analysis_prompt(data)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
