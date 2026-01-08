"""
FrictionMiner analyzer - extracts friction events using FAILURE_MODES taxonomy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from transcription.historian.analyzers.base import (
    BaseAnalyzer,
    SubagentSpec,
    format_comments,
    format_commits,
    format_reviews,
)

if TYPE_CHECKING:
    from transcription.historian.bundle import FactBundle


class FrictionMinerAnalyzer(BaseAnalyzer):
    """Extract friction events and categorize them."""

    @property
    def spec(self) -> SubagentSpec:
        return SubagentSpec(
            name="FrictionMiner",
            description="Extract 2-8 friction events using FAILURE_MODES taxonomy",
            system_prompt="""You are a process analyst. Extract "friction events" - things that went wrong or required rework during this PR.

Friction event categories (from FAILURE_MODES.md):
- measurement_drift: Performance claims don't match reality
- doc_drift: Docs claim features that don't exist or are stale
- packaging_drift: Package installs but runtime fails
- test_flake: Non-deterministic tests
- dependency_hazard: Dependency update breaks functionality
- process_mismatch: PR follows wrong workflow
- implementation_error: Bug or logic error in code
- design_drift: Implementation diverged from plan
- missing_test: Feature added without adequate test coverage
- other: Doesn't fit other categories

Look for signals:
- Fix commits (especially multiple fixes for same issue)
- Revert commits
- WIP commits followed by fixes
- Review comments requesting changes
- Failed check runs that were later fixed

Output a JSON object with these exact fields:
{
  "events": [
    {
      "event": "What went wrong (specific)",
      "type": "category from above",
      "detected_by": "gate|review|post-merge|self",
      "disposition": "fixed here|deferred to #X|known issue",
      "prevention": "What could prevent this in future, or null",
      "evidence": "commit SHA, comment, or other anchor"
    }
  ],
  "iteration_count": N,
  "friction_score": "low|medium|high",
  "notes": "summary observations"
}

Extract 2-8 events. If fewer than 2 events, the PR was smooth - note that.
Be specific about what went wrong, not vague.""",
            output_schema={
                "type": "object",
                "required": ["events", "iteration_count", "friction_score"],
                "properties": {
                    "events": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "required": ["event", "type", "detected_by", "disposition"],
                            "properties": {
                                "event": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "measurement_drift",
                                        "doc_drift",
                                        "packaging_drift",
                                        "test_flake",
                                        "dependency_hazard",
                                        "process_mismatch",
                                        "implementation_error",
                                        "design_drift",
                                        "missing_test",
                                        "other",
                                    ],
                                },
                                "detected_by": {
                                    "type": "string",
                                    "enum": ["gate", "review", "post-merge", "self"],
                                },
                                "disposition": {"type": "string"},
                                "prevention": {"type": ["string", "null"]},
                                "evidence": {"type": "string"},
                            },
                        },
                    },
                    "iteration_count": {"type": "integer", "minimum": 0},
                    "friction_score": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "notes": {"type": "string"},
                },
            },
            required_bundle_fields=["commits", "comments", "reviews", "check_runs"],
        )

    def build_prompt(self, bundle: FactBundle) -> str:
        # Count friction signals
        fix_count = sum(1 for c in bundle.commits if c.is_fix)
        revert_count = sum(1 for c in bundle.commits if c.is_revert)
        wip_count = sum(1 for c in bundle.commits if c.is_wip)

        # Failed checks
        failed_checks = [cr for cr in bundle.check_runs if cr.conclusion == "failure"]

        return f"""Extract friction events from this PR:

## PR: {bundle.metadata.title}
Author: {bundle.metadata.author}

## Friction Signals Summary
- Fix commits: {fix_count}
- Revert commits: {revert_count}
- WIP commits: {wip_count}
- Failed check runs: {len(failed_checks)}

## Commits ({len(bundle.commits)} total)
{format_commits(bundle.commits)}

## Comments ({len(bundle.comments)} total)
{format_comments(bundle.comments)}

## Reviews ({len(bundle.reviews)} total)
{format_reviews(bundle.reviews)}

## Failed Check Runs
{chr(10).join(f"  - {cr.name}: {cr.conclusion}" for cr in failed_checks) if failed_checks else "  (none)"}

Extract friction events. Output your JSON."""
