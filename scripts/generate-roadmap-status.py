#!/usr/bin/env python3
"""Generate ROADMAP.md Quick Status table from GitHub issue state.

Usage:
    ./scripts/generate-roadmap-status.py          # Print to stdout
    ./scripts/generate-roadmap-status.py --write  # Update ROADMAP.md in place
    ./scripts/generate-roadmap-status.py --check  # Exit 1 if ROADMAP.md is stale

Requires: gh CLI authenticated
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IssueStatus:
    number: int
    title: str
    state: str  # OPEN or CLOSED
    labels: list[str]


def get_issues(numbers: list[int]) -> dict[int, IssueStatus]:
    """Fetch issue states from GitHub."""
    if not numbers:
        return {}

    # Batch fetch using gh api
    results = {}
    for num in numbers:
        try:
            result = subprocess.run(
                ["gh", "issue", "view", str(num), "--json", "number,title,state,labels"],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)
            results[num] = IssueStatus(
                number=data["number"],
                title=data["title"],
                state=data["state"],
                labels=[lbl["name"] for lbl in data.get("labels", [])],
            )
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            # Issue might not exist or gh not authenticated
            pass
    return results


def compute_track_status(
    issues: dict[int, IssueStatus],
    track_issues: list[int],
    track_name: str,
) -> tuple[str, str]:
    """Compute status emoji and next action for a track."""
    if not track_issues:
        return "⬜ Not Started", "Define issues"

    states = [issues.get(n) for n in track_issues]
    open_issues = [s for s in states if s and s.state == "OPEN"]
    closed_issues = [s for s in states if s and s.state == "CLOSED"]

    # Check for blocked label
    blocked = any("blocked" in (s.labels if s else []) for s in states if s)
    in_progress = any("in-progress" in (s.labels if s else []) for s in states if s)

    if len(closed_issues) == len(track_issues) and all(s is not None for s in states):
        return "✅ Complete", "—"

    if blocked:
        # Find what's blocking
        blocker = next((s for s in states if s and "blocked" in s.labels), None)
        reason = f"Blocked (#{blocker.number})" if blocker else "Blocked"
        return f"⏳ {reason}", "Resolve blocker"

    if in_progress or (open_issues and closed_issues):
        # Some progress made
        next_issue = open_issues[0] if open_issues else None
        next_action = f"#{next_issue.number}" if next_issue else "Continue"
        return "🔄 In Progress", next_action

    # Not started
    first = open_issues[0] if open_issues else None
    return "📋 Ready to Start", f"Begin #{first.number}" if first else "Begin"


# Track definitions: map track name to ordered list of issue numbers
TRACKS = {
    "v1.9.x Closeout": [],  # Already complete
    "API Polish Bundle": [70, 71, 72, 78],
    "Track 1: Benchmarks": [95, 137, 97, 96, 99],
    "Track 2: Streaming": [133, 134, 84, 85, 55, 86],
    "Track 3: Semantics": [88, 90, 91, 92, 89, 98],
}


def generate_status_table(issues: dict[int, IssueStatus]) -> str:
    """Generate the Quick Status markdown table."""
    lines = [
        "## Quick Status",
        "",
        "| Track | Status | Next Action |",
        "|-------|--------|-------------|",
    ]

    for track_name, track_issues in TRACKS.items():
        if track_name == "v1.9.x Closeout":
            # Hardcoded as complete
            status, action = "✅ Complete", "—"
        else:
            status, action = compute_track_status(issues, track_issues, track_name)
        lines.append(f"| {track_name} | {status} | {action} |")

    lines.append("")
    return "\n".join(lines)


def update_roadmap(new_table: str, roadmap_path: Path) -> bool:
    """Update ROADMAP.md with new status table. Returns True if changed."""
    content = roadmap_path.read_text()

    # Find and replace the Quick Status section
    pattern = r"## Quick Status\n\n\|.*?\n\|.*?\n(?:\|.*?\n)+"
    match = re.search(pattern, content)

    if not match:
        print("ERROR: Could not find Quick Status table in ROADMAP.md", file=sys.stderr)
        return False

    old_table = match.group(0)
    new_table_with_newline = new_table.rstrip("\n") + "\n"

    if old_table.strip() == new_table_with_newline.strip():
        return False  # No change

    new_content = content[: match.start()] + new_table_with_newline + content[match.end() :]
    roadmap_path.write_text(new_content)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Update ROADMAP.md in place")
    parser.add_argument("--check", action="store_true", help="Exit 1 if ROADMAP.md is stale")
    args = parser.parse_args()

    # Collect all issue numbers we need
    all_issues = set()
    for track_issues in TRACKS.values():
        all_issues.update(track_issues)

    # Fetch from GitHub
    issues = get_issues(list(all_issues))

    # Generate table
    table = generate_status_table(issues)

    roadmap_path = Path(__file__).parent.parent / "ROADMAP.md"

    if args.check:
        # Check if current ROADMAP matches generated
        content = roadmap_path.read_text()
        pattern = r"## Quick Status\n\n\|.*?\n\|.*?\n(?:\|.*?\n)+"
        match = re.search(pattern, content)
        if match:
            current = match.group(0).strip()
            generated = table.strip()
            if current != generated:
                print("ROADMAP.md Quick Status is stale. Run:", file=sys.stderr)
                print("  ./scripts/generate-roadmap-status.py --write", file=sys.stderr)
                return 1
        print("ROADMAP.md Quick Status is up to date.")
        return 0

    if args.write:
        if update_roadmap(table, roadmap_path):
            print("Updated ROADMAP.md Quick Status table.")
        else:
            print("ROADMAP.md Quick Status already up to date.")
        return 0

    # Default: print to stdout
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
