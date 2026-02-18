"""
Base classes for subagent analyzers.

Each analyzer is a specialized LLM-powered module that analyzes a specific
aspect of a PR and returns structured JSON output.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..bundle import FactBundle
    from ..llm_client import LLMProvider


@dataclass
class SubagentSpec:
    """Specification for a subagent analyzer."""

    name: str
    description: str
    system_prompt: str
    output_schema: dict[str, Any]  # JSON Schema for response validation
    required_bundle_fields: list[str] = field(default_factory=list)


@dataclass
class SubagentResult:
    """Result from a subagent analyzer."""

    name: str
    success: bool
    output: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0
    raw_response: str | None = None


class BaseAnalyzer(ABC):
    """
    Base class for subagent analyzers.

    Subclasses must implement:
    - spec: Return the analyzer specification
    - build_prompt: Build the user prompt from the fact bundle
    """

    @property
    @abstractmethod
    def spec(self) -> SubagentSpec:
        """Return the analyzer specification."""
        ...

    @abstractmethod
    def build_prompt(self, bundle: FactBundle) -> str:
        """
        Build the user prompt from the fact bundle.

        Args:
            bundle: The fact bundle to analyze

        Returns:
            User prompt string
        """
        ...

    def should_run(self, bundle: FactBundle) -> bool:
        """
        Check if this analyzer should run for the given bundle.

        Override in subclasses for conditional execution.

        Args:
            bundle: The fact bundle

        Returns:
            True if analyzer should run
        """
        return True

    def validate_output(self, output: dict[str, Any]) -> list[str]:
        """
        Validate the LLM output against the schema.

        Args:
            output: The parsed JSON output

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            import jsonschema

            validator = jsonschema.Draft7Validator(self.spec.output_schema)
            errors.extend([e.message for e in validator.iter_errors(output)])
        except ImportError:
            # jsonschema not available, skip validation
            pass
        except Exception as e:
            errors.append(f"Schema validation error: {e}")

        return errors

    async def run(
        self,
        bundle: FactBundle,
        llm_provider: LLMProvider,
    ) -> SubagentResult:
        """
        Run the analyzer on a fact bundle.

        Args:
            bundle: The fact bundle to analyze
            llm_provider: The LLM provider to use

        Returns:
            SubagentResult with the analysis output
        """
        name = self.spec.name

        # Check if should run
        if not self.should_run(bundle):
            return SubagentResult(
                name=name,
                success=True,
                output={"skipped": True, "reason": "Analyzer conditions not met"},
                duration_ms=0,
            )

        # Build prompts
        system_prompt = self.spec.system_prompt
        user_prompt = self.build_prompt(bundle)

        # Call LLM
        start_time = time.time()
        try:
            response = await llm_provider.complete(system_prompt, user_prompt)
            duration_ms = int((time.time() - start_time) * 1000)
        except Exception as e:
            return SubagentResult(
                name=name,
                success=False,
                errors=[f"LLM call failed: {e}"],
                duration_ms=int((time.time() - start_time) * 1000),
            )

        # Parse response
        try:
            # Try to extract JSON from response
            text = response.text.strip()

            # Handle markdown code blocks
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()

            output = json.loads(text)
        except json.JSONDecodeError as e:
            return SubagentResult(
                name=name,
                success=False,
                errors=[f"Failed to parse JSON: {e}"],
                duration_ms=duration_ms,
                raw_response=response.text,
            )

        # Validate output
        validation_errors = self.validate_output(output)
        if validation_errors:
            return SubagentResult(
                name=name,
                success=False,
                output=output,
                errors=validation_errors,
                duration_ms=duration_ms,
                raw_response=response.text,
            )

        return SubagentResult(
            name=name,
            success=True,
            output=output,
            duration_ms=duration_ms,
            raw_response=response.text,
        )


# --- Utility functions for building prompts ---


def format_key_files(key_files: list[dict[str, Any]], limit: int = 10) -> str:
    """Format key files for prompt."""
    lines = []
    for f in key_files[:limit]:
        lines.append(f"  - {f['path']} (+{f['additions']}/-{f['deletions']})")
    return "\n".join(lines) if lines else "  (no key files)"


def format_commits(commits: list[Any], limit: int = 20) -> str:
    """Format commits for prompt."""
    lines = []
    for c in commits[:limit]:
        patterns = []
        if c.is_fix:
            patterns.append("fix")
        if c.is_refactor:
            patterns.append("refactor")
        if c.is_test:
            patterns.append("test")
        if c.is_doc:
            patterns.append("doc")
        if c.is_revert:
            patterns.append("revert")
        if c.is_wip:
            patterns.append("wip")

        pattern_str = f" [{','.join(patterns)}]" if patterns else ""
        lines.append(f"  - {c.sha}: {c.message}{pattern_str}")

    if len(commits) > limit:
        lines.append(f"  ... and {len(commits) - limit} more commits")

    return "\n".join(lines) if lines else "  (no commits)"


def format_comments(comments: list[Any], limit: int = 10) -> str:
    """Format comments for prompt."""
    lines = []
    for c in comments[:limit]:
        body = c.body[:200] + "..." if len(c.body) > 200 else c.body
        lines.append(f"  - [{c.comment_type}] {c.author}: {body}")

    if len(comments) > limit:
        lines.append(f"  ... and {len(comments) - limit} more comments")

    return "\n".join(lines) if lines else "  (no comments)"


def format_reviews(reviews: list[Any], limit: int = 5) -> str:
    """Format reviews for prompt."""
    lines = []
    for r in reviews[:limit]:
        body = r.body[:200] + "..." if len(r.body) > 200 else r.body
        lines.append(f"  - [{r.state}] {r.author}: {body}")

    if len(reviews) > limit:
        lines.append(f"  ... and {len(reviews) - limit} more reviews")

    return "\n".join(lines) if lines else "  (no reviews)"


def format_check_runs(check_runs: list[Any], limit: int = 10) -> str:
    """Format check runs for prompt."""
    lines = []
    for cr in check_runs[:limit]:
        status = cr.conclusion or cr.status
        timing = f" ({cr.duration_seconds:.0f}s)" if cr.duration_seconds else ""
        lines.append(f"  - {cr.name}: {status}{timing}")

    if len(check_runs) > limit:
        lines.append(f"  ... and {len(check_runs) - limit} more checks")

    return "\n".join(lines) if lines else "  (no check runs)"


def format_sessions(sessions: list[Any]) -> str:
    """Format sessions for prompt."""
    lines = []
    for s in sessions:
        session_type = "author" if s.is_author_session else "review"
        lines.append(
            f"  - Session {s.session_id} ({session_type}): "
            f"{len(s.commits)} commits, "
            f"{len(s.comments_in_window)} comments, "
            f"{len(s.reviews_in_window)} reviews, "
            f"LB={s.lb_minutes}m UB={s.ub_minutes}m"
        )
    return "\n".join(lines) if lines else "  (no sessions)"
