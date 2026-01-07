"""
Publisher - write dossiers and update PR descriptions.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PublishResult:
    """Result of publishing a dossier."""

    success: bool
    dossier_path: str | None = None
    ledger_path: str | None = None
    pr_updated: bool = False
    exhibit_score: int = 0
    exhibit_added: bool = False
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _render_ledger_markdown(dossier: dict[str, Any]) -> str:
    """Render a dossier to PR ledger markdown format."""
    pr_num = dossier.get("pr_number", "?")
    today = datetime.now().strftime("%Y-%m-%d")

    # Intent section
    intent = dossier.get("intent", {})
    goal = intent.get("goal", "(no goal specified)")
    pr_type = intent.get("type", "unknown")
    out_of_scope = intent.get("out_of_scope", [])

    # Scope section
    scope = dossier.get("scope", {})
    key_files = scope.get("key_files", [])

    # Evidence section
    evidence = dossier.get("evidence", {})
    local_gate = evidence.get("local_gate", {})
    tests = evidence.get("tests", {})
    typing = evidence.get("typing", {})
    benchmarks = evidence.get("benchmarks", {})

    # Process section
    process = dossier.get("process", {})
    friction_events = process.get("friction_events", [])
    design_alignment = process.get("design_alignment", {})
    measurement_integrity = process.get("measurement_integrity", {})

    # Cost section
    cost = dossier.get("cost", {})
    wall_clock = cost.get("wall_clock", {})
    active_work = cost.get("active_work", {})
    devlt = cost.get("devlt", {})

    # Reflection
    reflection = dossier.get("reflection", {})
    went_well = reflection.get("went_well", [])
    could_be_better = reflection.get("could_be_better", [])

    # Follow-ups
    followups = dossier.get("followups", [])

    # Build markdown
    lines = [
        f"## PR Ledger (updated {today})",
        "",
        "### What this PR was",
        f"- **Goal:** {goal}",
        f"- **Type:** {pr_type}",
    ]

    if out_of_scope:
        lines.append(f"- **Out of scope:** {', '.join(out_of_scope)}")

    if key_files:
        lines.append(f"- **Primary surfaces:** `{'`, `'.join(key_files[:5])}`")

    lines.extend(
        [
            "",
            "### What shipped",
            f"- Files changed: {scope.get('files_changed', 0)}",
            f"- Insertions: {scope.get('insertions', 0)}, Deletions: {scope.get('deletions', 0)}",
        ]
    )

    if dossier.get("findings"):
        for finding in dossier["findings"][:5]:
            desc = finding.get("description", "")[:80]
            lines.append(f"- {desc}")

    lines.extend(
        [
            "",
            "### Evidence (receipts)",
        ]
    )

    gate_status = "passed" if local_gate.get("passed") else "unknown"
    lines.append(f"- **Local gate:** {gate_status}")

    if tests.get("added") or tests.get("modified"):
        lines.append(
            f"- **Tests:** +{tests.get('added', 0)} added, {tests.get('modified', 0)} modified"
        )

    mypy_status = "passed" if typing.get("mypy_passed") else "unknown"
    ruff_status = "passed" if typing.get("ruff_passed") else "unknown"
    lines.append(f"- **Type/lint:** mypy {mypy_status}, ruff {ruff_status}")

    if benchmarks.get("metrics"):
        metrics_str = ", ".join(f"{k}={v}" for k, v in benchmarks["metrics"].items())
        baseline = benchmarks.get("baseline_commit", "unknown")
        lines.append(f"- **Bench/perf:** {metrics_str} (baseline: {baseline})")

    lines.extend(
        [
            "",
            "### Process + friction",
            "",
            "| Event | Detected by | Disposition | Prevention |",
            "|-------|-------------|-------------|------------|",
        ]
    )

    if friction_events:
        for event in friction_events[:5]:
            e = event.get("event", "-")[:40]
            d = event.get("detected_by", "-")
            disp = event.get("disposition", "-")[:30]
            prev = event.get("prevention", "-") or "-"
            prev = prev[:30] if prev != "-" else "-"
            lines.append(f"| {e} | {d} | {disp} | {prev} |")
    else:
        lines.append("| (no friction events) | - | - | - |")

    if design_alignment.get("drifted"):
        lines.append(f"\n- **Design alignment:** drifted - {design_alignment.get('notes', '')}")
    else:
        lines.append("\n- **Design alignment:** aligned")

    mi_valid = measurement_integrity.get("valid", "unknown")
    lines.append(f"- **Measurement integrity:** {mi_valid}")

    lines.extend(
        [
            "",
            "### Cost & time (best-effort estimates)",
            "",
            "| Metric | Value | Method |",
            "|--------|-------|--------|",
        ]
    )

    if wall_clock.get("days"):
        lines.append(f"| Wall-clock | {wall_clock['days']} days | `created_at -> merged_at` |")

    if active_work.get("lb_hours") is not None:
        lb = active_work.get("lb_hours", 0)
        ub = active_work.get("ub_hours", lb)
        method = active_work.get("method", "commit bursts")[:30]
        lines.append(f"| Active work | {lb}-{ub}h | {method} |")

    # Handle nested devlt structure
    if isinstance(devlt, dict):
        author = devlt.get("author", {})
        review = devlt.get("review", {})
        if isinstance(author, dict) and author.get("band"):
            lines.append(f"| DevLT (author) | {author.get('band')} | bounded estimation |")
        if isinstance(review, dict) and review.get("band"):
            lines.append(f"| DevLT (review) | {review.get('band')} | bounded estimation |")

    lines.extend(
        [
            "",
            "### What went well",
        ]
    )
    if went_well:
        for item in went_well[:5]:
            lines.append(f"- {item}")
    else:
        lines.append("- (not recorded)")

    lines.extend(
        [
            "",
            "### What could be better",
        ]
    )
    if could_be_better:
        for item in could_be_better[:5]:
            lines.append(f"- {item}")
    else:
        lines.append("- (not recorded)")

    lines.extend(
        [
            "",
            "### Follow-ups",
        ]
    )
    if followups:
        for fu in followups[:5]:
            issue = fu.get("issue", "")
            desc = fu.get("description", "")
            lines.append(f"- {issue}: {desc}")
    else:
        lines.append("- (none)")

    return "\n".join(lines)


def _compute_exhibit_score(dossier: dict[str, Any]) -> int:
    """
    Compute exhibit candidacy score (0-10).

    Higher scores indicate better exhibit candidates.
    """
    score = 0

    # Has documented friction events (+2)
    if dossier.get("process", {}).get("friction_events"):
        score += 2

    # Has findings with prevention (+2)
    findings = dossier.get("findings", [])
    if any(f.get("prevention_added") for f in findings):
        score += 2

    # Has reflection (+1)
    reflection = dossier.get("reflection", {})
    if reflection.get("went_well") or reflection.get("could_be_better"):
        score += 1

    # Has evidence (+1)
    evidence = dossier.get("evidence", {})
    if evidence.get("local_gate", {}).get("passed"):
        score += 1

    # Has DevLT recorded (+1)
    devlt = dossier.get("cost", {}).get("devlt", {})
    if devlt:
        score += 1

    # Has follow-ups (+1)
    if dossier.get("followups"):
        score += 1

    # Non-trivial scope (+1)
    if dossier.get("scope", {}).get("files_changed", 0) >= 5:
        score += 1

    return min(score, 10)


def _update_exhibits_index(
    pr_number: int,
    dossier: dict[str, Any],
    score: int,
    exhibits_path: Path,
) -> bool:
    """Update EXHIBITS.md with a new entry."""
    if not exhibits_path.exists():
        return False

    content = exhibits_path.read_text()

    # Check if already listed
    if f"PR #{pr_number}" in content or f"#{pr_number}" in content:
        return False  # Already there

    # Find the index section to insert
    intent = dossier.get("intent", {})
    goal = intent.get("goal", "")[:50]
    pr_type = intent.get("type", "unknown")

    entry = f"- **PR #{pr_number}** ({pr_type}) - {goal} (score: {score})\n"

    # Insert after "## Index" or at end
    if "## Index" in content:
        parts = content.split("## Index", 1)
        if len(parts) == 2:
            # Find end of list
            lines = parts[1].split("\n")
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("- ") or line.strip() == "":
                    insert_idx = i + 1
                elif line.startswith("#"):
                    break

            lines.insert(insert_idx, entry.rstrip())
            parts[1] = "\n".join(lines)
            content = "## Index".join(parts)
            exhibits_path.write_text(content)
            return True

    return False


def publish_dossier(
    dossier: dict[str, Any],
    pr_number: int,
    repo_root: Path | None = None,
    update_pr: bool = False,
    update_exhibits: bool = True,
    dry_run: bool = False,
) -> PublishResult:
    """
    Publish a dossier to the repository.

    Args:
        dossier: The dossier dict to publish
        pr_number: PR number
        repo_root: Repository root path (defaults to current directory)
        update_pr: Whether to update the PR description via gh
        update_exhibits: Whether to update EXHIBITS.md
        dry_run: If True, don't write files

    Returns:
        PublishResult with status and paths
    """
    from transcription.historian.validation import validate_dossier

    repo_root = repo_root or Path.cwd()
    errors = []
    notes = []

    # Validate first
    valid, validation_errors = validate_dossier(dossier)
    if not valid:
        # Continue with warnings, don't block
        errors.extend(validation_errors)
        notes.append("Dossier has validation warnings")

    # Paths
    dossier_dir = repo_root / "docs" / "audit" / "pr"
    dossier_path = dossier_dir / f"{pr_number}.json"
    ledger_path = dossier_dir / f"{pr_number}.md"
    exhibits_path = repo_root / "docs" / "audit" / "EXHIBITS.md"

    # Render ledger
    ledger_md = _render_ledger_markdown(dossier)

    # Compute exhibit score
    score = _compute_exhibit_score(dossier)
    notes.append(f"Exhibit score: {score}/10")

    if dry_run:
        notes.append("Dry run - no files written")
        return PublishResult(
            success=True,
            dossier_path=str(dossier_path),
            ledger_path=str(ledger_path),
            exhibit_score=score,
            notes=notes,
            errors=errors,
        )

    # Write dossier JSON
    try:
        dossier_dir.mkdir(parents=True, exist_ok=True)
        with open(dossier_path, "w") as f:
            json.dump(dossier, f, indent=2)
        notes.append(f"Wrote dossier to {dossier_path}")
    except Exception as e:
        errors.append(f"Failed to write dossier: {e}")
        return PublishResult(success=False, errors=errors, notes=notes)

    # Write ledger markdown
    try:
        with open(ledger_path, "w") as f:
            f.write(ledger_md)
        notes.append(f"Wrote ledger to {ledger_path}")
    except Exception as e:
        errors.append(f"Failed to write ledger: {e}")

    # Update PR via gh (if requested)
    pr_updated = False
    if update_pr:
        try:
            result = subprocess.run(
                ["gh", "pr", "edit", str(pr_number), "--body-file", str(ledger_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                pr_updated = True
                notes.append("Updated PR description via gh")
            else:
                errors.append(f"gh pr edit failed: {result.stderr}")
        except Exception as e:
            errors.append(f"Failed to update PR: {e}")

    # Update exhibits (if score >= 8)
    exhibit_added = False
    if update_exhibits and score >= 8:
        exhibit_added = _update_exhibits_index(pr_number, dossier, score, exhibits_path)
        if exhibit_added:
            notes.append(f"Added to EXHIBITS.md (score: {score})")

    return PublishResult(
        success=True,
        dossier_path=str(dossier_path),
        ledger_path=str(ledger_path),
        pr_updated=pr_updated,
        exhibit_score=score,
        exhibit_added=exhibit_added,
        errors=errors,
        notes=notes,
    )
