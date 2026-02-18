"""
EvidenceAuditor analyzer - maps claims to artifacts and identifies missing receipts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    BaseAnalyzer,
    SubagentSpec,
    format_check_runs,
)

if TYPE_CHECKING:
    from ..bundle import FactBundle


class EvidenceAuditorAnalyzer(BaseAnalyzer):
    """Audit evidence and receipts, identify what's missing."""

    @property
    def spec(self) -> SubagentSpec:
        return SubagentSpec(
            name="EvidenceAuditor",
            description="Map claims to artifacts, identify missing receipts",
            system_prompt="""You are an evidence auditor. Analyze what claims are made in this PR and what evidence supports them.

Your job is to:
1. Identify claims made (explicitly or implicitly) about testing, quality, performance
2. Map each claim to evidence (or note it's missing)
3. Assess overall evidence completeness

Output a JSON object with these exact fields:
{
  "claims": [
    {
      "claim": "what is being claimed",
      "evidence": "what supports it or null",
      "status": "supported|partial|missing|unknown"
    }
  ],
  "evidence": {
    "local_gate": {"observed": true/false, "path": "path or null"},
    "tests": {"observed": true/false, "added": N, "modified": N},
    "typing": {"mypy_observed": true/false, "ruff_observed": true/false},
    "benchmarks": {"observed": true/false, "path": "path or null"},
    "docs_updated": true/false,
    "schema_validated": true/false/"unknown"
  },
  "missing_receipts": ["list of what's missing that should be present"],
  "evidence_completeness": "high|medium|low",
  "notes": "any additional observations"
}

Be conservative - mark as "unknown" when you can't determine, not "supported".""",
            output_schema={
                "type": "object",
                "required": ["claims", "evidence", "missing_receipts", "evidence_completeness"],
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["claim", "evidence", "status"],
                            "properties": {
                                "claim": {"type": "string"},
                                "evidence": {"type": ["string", "null"]},
                                "status": {
                                    "type": "string",
                                    "enum": ["supported", "partial", "missing", "unknown"],
                                },
                            },
                        },
                    },
                    "evidence": {
                        "type": "object",
                        "properties": {
                            "local_gate": {"type": "object"},
                            "tests": {"type": "object"},
                            "typing": {"type": "object"},
                            "benchmarks": {"type": "object"},
                            "docs_updated": {"type": "boolean"},
                            "schema_validated": {"type": ["boolean", "string"]},
                        },
                    },
                    "missing_receipts": {"type": "array", "items": {"type": "string"}},
                    "evidence_completeness": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "notes": {"type": "string"},
                },
            },
            required_bundle_fields=["metadata", "scope", "check_runs", "receipt_paths"],
        )

    def build_prompt(self, bundle: FactBundle) -> str:
        scope = bundle.scope
        cats = scope.file_categories

        # Check run summary
        passed = sum(1 for cr in bundle.check_runs if cr.conclusion == "success")
        failed = sum(1 for cr in bundle.check_runs if cr.conclusion == "failure")
        total = len(bundle.check_runs)

        return f"""Audit the evidence for this PR:

## PR: {bundle.metadata.title}
URL: {bundle.metadata.url}

## PR Description
{bundle.metadata.body[:2000] if bundle.metadata.body else "(no description)"}

## File Changes
- Source files: {len(cats.get("source", []))}
- Test files: {len(cats.get("tests", []))}
- Doc files: {len(cats.get("docs", []))}
- Config files: {len(cats.get("config", []))}

## Check Runs ({total} total, {passed} passed, {failed} failed)
{format_check_runs(bundle.check_runs)}

## Receipt Paths Found in PR
{chr(10).join(f"  - {p}" for p in bundle.receipt_paths) if bundle.receipt_paths else "  (none found)"}

## Labels
{", ".join(bundle.metadata.labels) if bundle.metadata.labels else "(no labels)"}

Analyze what claims are made and what evidence exists. Output your JSON."""
