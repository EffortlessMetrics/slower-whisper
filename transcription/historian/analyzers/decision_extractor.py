"""
DecisionExtractor analyzer - extracts material decision events from PR data.

Taxonomy v1:
- scope: defer to issue, cut from PR, split into follow-up
- design: schema keys, API surface, streaming envelope choices
- quality: choosing to gate, mark invalid, accept limitation
- debug: resolving failing gate, fixing broken assumption
- publish: writing cover sheet, deciding shipped vs deferred
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from transcription.historian.analyzers.base import (
    BaseAnalyzer,
    SubagentSpec,
    format_check_runs,
    format_comments,
    format_commits,
    format_reviews,
)

if TYPE_CHECKING:
    from transcription.historian.bundle import FactBundle


# Decision type time bands (minutes)
DECISION_TIME_BANDS = {
    "scope": (2, 8),
    "design": (6, 20),
    "quality": (4, 15),
    "debug": (8, 30),
    "publish": (3, 12),
}


class DecisionExtractorAnalyzer(BaseAnalyzer):
    """Extract material decision events from PR activity."""

    @property
    def spec(self) -> SubagentSpec:
        return SubagentSpec(
            name="DecisionExtractor",
            description="Extract 5-12 material decisions with anchored evidence",
            system_prompt="""You are a decision archaeologist. Your job is to identify material decisions made during this PR based on observable evidence.

## Decision Taxonomy (v1)

1. **scope** (2-8 min): Decisions about what's in/out of this PR
   - "defer to issue", "cut from PR", "split into follow-up"
   - Visible in: PR body edits, issue comments, labels, "out of scope" notes

2. **design** (6-20 min): Technical choices about contracts, schemas, APIs
   - Schema keys, API surface changes, streaming envelope choices
   - Visible in: design doc diffs, reviews with "this contract must hold" notes

3. **quality** (4-15 min): Accepting or rejecting quality tradeoffs
   - Choosing to gate, mark invalid, accept limitation, add validation
   - Visible in: "wrongness" sections, benchmark notes, doc corrections

4. **debug** (8-30 min): Diagnosing and fixing issues
   - Resolving failing gate, fixing broken assumption, root cause analysis
   - Visible in: fix-commit streaks, review comments, failing→passing checks

5. **publish** (3-12 min): Release and documentation decisions
   - Writing cover sheet, deciding shipped vs deferred, changelog edits
   - Visible in: PR description edits, changelog/roadmap adjustments

## Rules

- ONLY infer decisions from observable evidence (commits, comments, reviews, checks)
- Use "unknown" for type if evidence is ambiguous
- Skip decisions where evidence is insufficient
- Do NOT fabricate drama or invent decisions
- Every decision MUST have a concrete anchor (commit SHA, comment, or check name)
- Aim for 5-12 material decisions; fewer is fine if evidence is sparse
- Confidence should reflect evidence quality:
  - high: Direct statement or clear commit sequence
  - med: Reasonable inference from multiple signals
  - low: Single indirect signal

## Output Format

Return a JSON object with:
{
  "decisions": [
    {
      "type": "scope|design|quality|debug|publish|unknown",
      "description": "What the decision was (one sentence)",
      "anchor": "Specific evidence: commit SHA, comment snippet, or check name",
      "confidence": "high|med|low",
      "minutes_lb": N,  // Lower bound from time band
      "minutes_ub": N   // Upper bound from time band
    }
  ],
  "decision_count": N,
  "dominant_type": "Most common decision type or 'mixed'",
  "evidence_quality": "strong|moderate|weak",
  "notes": "Summary of decision patterns observed"
}""",
            output_schema={
                "type": "object",
                "required": ["decisions", "decision_count", "dominant_type", "evidence_quality"],
                "properties": {
                    "decisions": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 15,
                        "items": {
                            "type": "object",
                            "required": [
                                "type",
                                "description",
                                "anchor",
                                "confidence",
                                "minutes_lb",
                                "minutes_ub",
                            ],
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "scope",
                                        "design",
                                        "quality",
                                        "debug",
                                        "publish",
                                        "unknown",
                                    ],
                                },
                                "description": {"type": "string"},
                                "anchor": {"type": "string"},
                                "confidence": {
                                    "type": "string",
                                    "enum": ["high", "med", "low"],
                                },
                                "minutes_lb": {"type": "integer", "minimum": 0},
                                "minutes_ub": {"type": "integer", "minimum": 0},
                            },
                        },
                    },
                    "decision_count": {"type": "integer", "minimum": 0},
                    "dominant_type": {
                        "type": "string",
                        "enum": [
                            "scope",
                            "design",
                            "quality",
                            "debug",
                            "publish",
                            "mixed",
                        ],
                    },
                    "evidence_quality": {
                        "type": "string",
                        "enum": ["strong", "moderate", "weak"],
                    },
                    "notes": {"type": "string"},
                },
            },
            required_bundle_fields=[
                "metadata",
                "commits",
                "comments",
                "reviews",
                "check_runs",
            ],
        )

    def build_prompt(self, bundle: FactBundle) -> str:
        # Analyze commit patterns for decision signals
        fix_commits = [c for c in bundle.commits if c.is_fix]
        refactor_commits = [c for c in bundle.commits if c.is_refactor]
        test_commits = [c for c in bundle.commits if c.is_test]
        doc_commits = [c for c in bundle.commits if c.is_doc]

        # Check run transitions (failures that became successes)
        failed_checks = [cr for cr in bundle.check_runs if cr.conclusion == "failure"]
        passed_checks = [cr for cr in bundle.check_runs if cr.conclusion == "success"]

        # Look for scope/design keywords in comments/reviews
        scope_keywords = ["out of scope", "defer", "follow-up", "split", "cut from"]
        design_keywords = ["contract", "schema", "api", "interface", "must hold", "invariant"]
        quality_keywords = ["gate", "invalid", "limitation", "accept", "tradeoff"]

        def count_keywords(texts: list[str], keywords: list[str]) -> int:
            count = 0
            for text in texts:
                text_lower = text.lower()
                for kw in keywords:
                    if kw in text_lower:
                        count += 1
            return count

        all_text = [c.body for c in bundle.comments] + [r.body for r in bundle.reviews]
        scope_signals = count_keywords(all_text, scope_keywords)
        design_signals = count_keywords(all_text, design_keywords)
        quality_signals = count_keywords(all_text, quality_keywords)

        # Format time bands for reference
        time_bands_str = "\n".join(
            f"  - {k}: {v[0]}-{v[1]} minutes" for k, v in DECISION_TIME_BANDS.items()
        )

        return f"""Extract material decisions from this PR:

## PR: {bundle.metadata.title}
Author: {bundle.metadata.author}
URL: {bundle.metadata.url}

## PR Description
{bundle.metadata.body if bundle.metadata.body else "(no description)"}

## Labels
{", ".join(bundle.metadata.labels) if bundle.metadata.labels else "(no labels)"}

## Decision Signal Summary
- Fix commits: {len(fix_commits)}
- Refactor commits: {len(refactor_commits)}
- Test commits: {len(test_commits)}
- Doc commits: {len(doc_commits)}
- Failed checks: {len(failed_checks)}
- Passed checks: {len(passed_checks)}
- Scope-related keywords in discussion: {scope_signals}
- Design-related keywords in discussion: {design_signals}
- Quality-related keywords in discussion: {quality_signals}

## Time Bands (for minutes_lb/minutes_ub)
{time_bands_str}

## Commits ({len(bundle.commits)} total)
{format_commits(bundle.commits, limit=30)}

## Comments ({len(bundle.comments)} total)
{format_comments(bundle.comments, limit=15)}

## Reviews ({len(bundle.reviews)} total)
{format_reviews(bundle.reviews, limit=10)}

## Check Runs ({len(bundle.check_runs)} total)
{format_check_runs(bundle.check_runs, limit=15)}

Extract 5-12 material decisions with anchored evidence. Use the time bands above for minutes_lb and minutes_ub based on decision type. Output your JSON."""
