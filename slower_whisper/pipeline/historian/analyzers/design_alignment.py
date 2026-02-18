"""
DesignAlignment analyzer - detects drift from original plan/intent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    BaseAnalyzer,
    SubagentSpec,
    format_commits,
    format_key_files,
)

if TYPE_CHECKING:
    from ..bundle import FactBundle


class DesignAlignmentAnalyzer(BaseAnalyzer):
    """Detect design drift between PR intent and actual changes."""

    @property
    def spec(self) -> SubagentSpec:
        return SubagentSpec(
            name="DesignAlignment",
            description="Detect drift between PR intent and implementation",
            system_prompt="""You are a design reviewer. Compare what the PR description says it does vs what the commits/changes actually do.

Look for:
1. Scope creep: Changes beyond stated intent
2. Missing pieces: Stated goals not addressed
3. Undocumented changes: Significant changes not mentioned in description
4. Intent drift: Initial commits vs later commits suggest changed direction

Output a JSON object with these exact fields:
{
  "intent": {
    "stated_goal": "What the PR description says (or 'unclear' if vague)",
    "inferred_type": "feature|hardening|mechanization|perf/bench|refactor"
  },
  "alignment": {
    "drifted": true/false,
    "drift_type": "scope_creep|missing_pieces|undocumented|direction_change|none",
    "details": "explanation if drifted, or why well-aligned"
  },
  "scope_check": {
    "stated_in_scope": ["list of stated scope items"],
    "stated_out_of_scope": ["list of explicit exclusions"],
    "actual_scope": ["what was actually touched"],
    "discrepancies": ["any mismatches"]
  },
  "evidence": ["specific commits, files, or comments that support your assessment"]
}

Be specific. Reference actual commit messages and file paths.""",
            output_schema={
                "type": "object",
                "required": ["intent", "alignment", "scope_check", "evidence"],
                "properties": {
                    "intent": {
                        "type": "object",
                        "required": ["stated_goal", "inferred_type"],
                        "properties": {
                            "stated_goal": {"type": "string"},
                            "inferred_type": {
                                "type": "string",
                                "enum": [
                                    "feature",
                                    "hardening",
                                    "mechanization",
                                    "perf/bench",
                                    "refactor",
                                    "unknown",
                                ],
                            },
                        },
                    },
                    "alignment": {
                        "type": "object",
                        "required": ["drifted", "drift_type", "details"],
                        "properties": {
                            "drifted": {"type": "boolean"},
                            "drift_type": {
                                "type": "string",
                                "enum": [
                                    "scope_creep",
                                    "missing_pieces",
                                    "undocumented",
                                    "direction_change",
                                    "none",
                                ],
                            },
                            "details": {"type": "string"},
                        },
                    },
                    "scope_check": {
                        "type": "object",
                        "properties": {
                            "stated_in_scope": {"type": "array", "items": {"type": "string"}},
                            "stated_out_of_scope": {"type": "array", "items": {"type": "string"}},
                            "actual_scope": {"type": "array", "items": {"type": "string"}},
                            "discrepancies": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
            },
            required_bundle_fields=["metadata", "scope", "commits"],
        )

    def build_prompt(self, bundle: FactBundle) -> str:
        scope = bundle.scope

        return f"""Analyze design alignment for this PR:

## PR: {bundle.metadata.title}
URL: {bundle.metadata.url}

## PR Description
{bundle.metadata.body if bundle.metadata.body else "(no description)"}

## Labels
{", ".join(bundle.metadata.labels) if bundle.metadata.labels else "(no labels)"}

## Scope Summary
- Files changed: {scope.files_changed}
- Top directories: {", ".join(scope.top_directories[:5])}

## Key Files
{format_key_files(scope.key_files)}

## Commits (in order)
{format_commits(bundle.commits, limit=30)}

## File Categories
- Source: {len(scope.file_categories.get("source", []))} files
- Tests: {len(scope.file_categories.get("tests", []))} files
- Docs: {len(scope.file_categories.get("docs", []))} files
- Config: {len(scope.file_categories.get("config", []))} files

Analyze whether the implementation aligns with the stated intent. Output your JSON."""
