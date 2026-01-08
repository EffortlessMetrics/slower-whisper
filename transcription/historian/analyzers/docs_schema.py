"""
DocsSchemaAuditor analyzer - checks documentation and schema integrity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from transcription.historian.analyzers.base import BaseAnalyzer, SubagentSpec

if TYPE_CHECKING:
    from transcription.historian.bundle import FactBundle


class DocsSchemaAuditorAnalyzer(BaseAnalyzer):
    """Audit documentation and schema integrity."""

    @property
    def spec(self) -> SubagentSpec:
        return SubagentSpec(
            name="DocsSchemaAuditor",
            description="Check documentation accuracy and schema integrity",
            system_prompt="""You are a documentation auditor. Check for issues in docs and schema files touched by this PR.

Look for:
1. Stale examples: Code examples that don't match actual API
2. Wrong keys: Schema examples with incorrect field names
3. Broken links: References to files/sections that don't exist
4. Missing docs: New features without documentation
5. Version drift: Version numbers that weren't updated

Output a JSON object with these exact fields:
{
  "docs_touched": true/false,
  "schema_touched": true/false,
  "issues": [
    {
      "type": "stale_example|wrong_key|broken_link|missing_docs|version_drift|other",
      "file": "path to file",
      "description": "what's wrong",
      "severity": "high|medium|low"
    }
  ],
  "doc_files_checked": ["list of doc files in the diff"],
  "schema_files_checked": ["list of schema files in the diff"],
  "recommendations": [
    "list of suggested improvements"
  ],
  "overall_quality": "good|acceptable|needs_work"
}

Be specific about file paths and line numbers when possible.""",
            output_schema={
                "type": "object",
                "required": ["docs_touched", "schema_touched", "issues", "overall_quality"],
                "properties": {
                    "docs_touched": {"type": "boolean"},
                    "schema_touched": {"type": "boolean"},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["type", "file", "description", "severity"],
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "stale_example",
                                        "wrong_key",
                                        "broken_link",
                                        "missing_docs",
                                        "version_drift",
                                        "other",
                                    ],
                                },
                                "file": {"type": "string"},
                                "description": {"type": "string"},
                                "severity": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                },
                            },
                        },
                    },
                    "doc_files_checked": {"type": "array", "items": {"type": "string"}},
                    "schema_files_checked": {"type": "array", "items": {"type": "string"}},
                    "recommendations": {"type": "array", "items": {"type": "string"}},
                    "overall_quality": {
                        "type": "string",
                        "enum": ["good", "acceptable", "needs_work"],
                    },
                },
            },
            required_bundle_fields=["scope", "diff"],
        )

    def should_run(self, bundle: FactBundle) -> bool:
        """Only run if docs or schema files were touched."""
        cats = bundle.scope.file_categories

        # Has doc files
        if cats.get("docs"):
            return True

        # Has schema files
        for f in bundle.scope.key_files:
            path = f["path"].lower()
            if "schema" in path or path.endswith(".schema.json"):
                return True

        return False

    def build_prompt(self, bundle: FactBundle) -> str:
        scope = bundle.scope
        cats = scope.file_categories

        # Collect doc and schema files
        doc_files = cats.get("docs", [])
        schema_files = [
            f["path"]
            for f in scope.key_files
            if "schema" in f["path"].lower() or f["path"].endswith(".schema.json")
        ]

        return f"""Audit documentation and schema integrity for this PR:

## PR: {bundle.metadata.title}
URL: {bundle.metadata.url}

## PR Description
{bundle.metadata.body[:2000] if bundle.metadata.body else "(no description)"}

## Documentation Files Changed
{chr(10).join(f"  - {f}" for f in doc_files) if doc_files else "  (none)"}

## Schema Files Changed
{chr(10).join(f"  - {f}" for f in schema_files) if schema_files else "  (none)"}

## Source Files Changed (for reference)
{chr(10).join(f"  - {f}" for f in cats.get("source", [])[:10])}

## Diff
{bundle.diff[:25000] if bundle.diff else "(diff not available)"}

Check for documentation and schema issues. Output your JSON."""
