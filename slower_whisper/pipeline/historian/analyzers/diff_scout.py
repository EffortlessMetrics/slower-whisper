"""
DiffScout analyzer - maps change surface and identifies key files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    BaseAnalyzer,
    SubagentSpec,
    format_key_files,
)

if TYPE_CHECKING:
    from ..bundle import FactBundle


class DiffScoutAnalyzer(BaseAnalyzer):
    """Analyze the PR diff to map change surface and identify key files."""

    @property
    def spec(self) -> SubagentSpec:
        return SubagentSpec(
            name="DiffScout",
            description="Map change surface, identify key files and blast radius",
            system_prompt="""You are a code review analyst. Analyze the PR diff and produce a structured change map.

Your job is to help reviewers understand:
1. What parts of the codebase were touched
2. Which files are most critical to review
3. How risky/impactful this change is

Output a JSON object with these exact fields:
{
  "review_map": [
    {"directory": "path/", "files_count": N, "change_type": "source|tests|docs|config"}
  ],
  "key_files": [
    {"path": "path/to/file", "reason": "why this file matters"}
  ],
  "blast_radius": {
    "level": "low|medium|high",
    "rationale": "why this blast radius"
  },
  "generated_estimate": 0.0,
  "semantic_summary": "2-3 sentences describing what actually changed"
}

Be precise. Only include what you can see in the data. Limit key_files to 5 most critical.""",
            output_schema={
                "type": "object",
                "required": ["review_map", "key_files", "blast_radius", "semantic_summary"],
                "properties": {
                    "review_map": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["directory", "files_count", "change_type"],
                            "properties": {
                                "directory": {"type": "string"},
                                "files_count": {"type": "integer"},
                                "change_type": {
                                    "type": "string",
                                    "enum": ["source", "tests", "docs", "config", "other"],
                                },
                            },
                        },
                    },
                    "key_files": {
                        "type": "array",
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "required": ["path", "reason"],
                            "properties": {
                                "path": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                        },
                    },
                    "blast_radius": {
                        "type": "object",
                        "required": ["level", "rationale"],
                        "properties": {
                            "level": {"type": "string", "enum": ["low", "medium", "high"]},
                            "rationale": {"type": "string"},
                        },
                    },
                    "generated_estimate": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "semantic_summary": {"type": "string"},
                },
            },
            required_bundle_fields=["scope", "diff"],
        )

    def build_prompt(self, bundle: FactBundle) -> str:
        scope = bundle.scope

        # Build directory summary
        dir_summary = []
        for dir_path in scope.top_directories:
            # Count files in this directory
            count = sum(1 for f in scope.key_files if f["path"].startswith(dir_path.rstrip("/")))
            dir_summary.append(f"  - {dir_path}: ~{count} files")

        dir_str = "\n".join(dir_summary) if dir_summary else "  (no directories)"

        # File category counts
        cats = scope.file_categories
        cat_str = (
            f"Source: {len(cats.get('source', []))}, "
            f"Tests: {len(cats.get('tests', []))}, "
            f"Docs: {len(cats.get('docs', []))}, "
            f"Config: {len(cats.get('config', []))}"
        )

        # Diff excerpt
        diff_str = bundle.diff[:30000] if bundle.diff else "(diff not available)"

        return f"""Analyze this PR:

## PR: {bundle.metadata.title}
{bundle.metadata.body[:1000] if bundle.metadata.body else "(no description)"}

## Scope
- Files changed: {scope.files_changed}
- Insertions: {scope.insertions}
- Deletions: {scope.deletions}
- Existing blast radius estimate: {scope.blast_radius}

## Top Directories
{dir_str}

## File Categories
{cat_str}

## Key Files (by change size)
{format_key_files(scope.key_files)}

## Diff
{diff_str}

Analyze the change surface and produce your JSON output."""
