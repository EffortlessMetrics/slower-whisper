"""
PerfIntegrity analyzer - validates benchmark measurements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseAnalyzer, SubagentSpec

if TYPE_CHECKING:
    from ..bundle import FactBundle


class PerfIntegrityAnalyzer(BaseAnalyzer):
    """Validate benchmark and performance measurement integrity."""

    @property
    def spec(self) -> SubagentSpec:
        return SubagentSpec(
            name="PerfIntegrity",
            description="Validate benchmark measurements and baseline semantics",
            system_prompt="""You are a performance measurement auditor. Analyze any benchmark or performance claims in this PR.

Check for:
1. Baseline pinning: Is there a clear baseline commit/tag?
2. Semantic validity: Are comparisons apples-to-apples?
3. Metric consistency: Are p50/p95/p99 reported? Are units clear?
4. Measurement conditions: Hardware, config, sample size documented?
5. Result drift: Do claimed results match any artifacts?

Output a JSON object with these exact fields:
{
  "has_perf_claims": true/false,
  "measurement_integrity": {
    "valid": true/false/"unknown",
    "invalidation_reason": "reason if invalid, null otherwise",
    "confidence": "high|medium|low"
  },
  "baseline": {
    "specified": true/false,
    "commit_or_tag": "value or null",
    "semantics_unchanged": true/false/"unknown"
  },
  "metrics_found": [
    {
      "name": "p50_ms",
      "value": 42.3,
      "unit": "ms",
      "context": "where this was found"
    }
  ],
  "issues": [
    "list of specific measurement issues found"
  ],
  "recommendations": [
    "list of improvements if any"
  ]
}

If no performance claims exist, set has_perf_claims to false and fill minimal data.""",
            output_schema={
                "type": "object",
                "required": ["has_perf_claims", "measurement_integrity"],
                "properties": {
                    "has_perf_claims": {"type": "boolean"},
                    "measurement_integrity": {
                        "type": "object",
                        "required": ["valid", "confidence"],
                        "properties": {
                            "valid": {"type": ["boolean", "string"]},
                            "invalidation_reason": {"type": ["string", "null"]},
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                    },
                    "baseline": {
                        "type": "object",
                        "properties": {
                            "specified": {"type": "boolean"},
                            "commit_or_tag": {"type": ["string", "null"]},
                            "semantics_unchanged": {"type": ["boolean", "string"]},
                        },
                    },
                    "metrics_found": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": "number"},
                                "unit": {"type": "string"},
                                "context": {"type": "string"},
                            },
                        },
                    },
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "recommendations": {"type": "array", "items": {"type": "string"}},
                },
            },
            required_bundle_fields=["metadata", "diff", "receipt_paths"],
        )

    def should_run(self, bundle: FactBundle) -> bool:
        """Only run if there are benchmark-related files or claims."""
        # Check for benchmark files in scope
        scope = bundle.scope
        bench_indicators = ["benchmark", "perf", "latency", "throughput", "bench"]

        # Check key files
        for f in scope.key_files:
            path = f["path"].lower()
            if any(ind in path for ind in bench_indicators):
                return True

        # Check PR body for perf claims
        body = (bundle.metadata.body or "").lower()
        perf_keywords = ["p50", "p95", "p99", "latency", "throughput", "benchmark", "performance"]
        if any(kw in body for kw in perf_keywords):
            return True

        # Check labels
        for label in bundle.metadata.labels:
            if any(ind in label.lower() for ind in bench_indicators):
                return True

        return False

    def build_prompt(self, bundle: FactBundle) -> str:
        scope = bundle.scope

        # Find benchmark-related files
        bench_files = []
        for f in scope.key_files:
            path = f["path"].lower()
            if any(ind in path for ind in ["benchmark", "perf", "latency", "bench"]):
                bench_files.append(f["path"])

        # Extract potential metrics from body
        body = bundle.metadata.body or ""

        return f"""Audit performance measurement integrity for this PR:

## PR: {bundle.metadata.title}
URL: {bundle.metadata.url}

## PR Description
{body[:3000] if body else "(no description)"}

## Benchmark-Related Files
{chr(10).join(f"  - {f}" for f in bench_files) if bench_files else "  (none detected)"}

## Receipt Paths Found
{chr(10).join(f"  - {p}" for p in bundle.receipt_paths) if bundle.receipt_paths else "  (none)"}

## Diff Excerpt (benchmark files)
{bundle.diff[:20000] if bundle.diff else "(diff not available)"}

Analyze performance measurement integrity. Output your JSON."""
