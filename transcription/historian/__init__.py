"""
Historian pipeline for PR analysis and dossier generation.

This package provides tools for:
- Gathering PR data from GitHub into deterministic fact bundles
- Running LLM-powered subagent analyzers
- Synthesizing and validating PR dossiers
- Publishing dossiers and updating PR descriptions

Usage:
    # Programmatic API
    from transcription.historian import FactBundle, gather_pr_data
    from transcription.historian.estimation import compute_bounded_estimate
    from transcription.historian.analyzers import run_all_analyzers
    from transcription.historian.synthesis import synthesize_dossier
    from transcription.historian.publisher import publish_dossier

    # CLI
    python scripts/generate-pr-ledger.py --pr 123 --dump-bundle
    python scripts/generate-pr-ledger.py --pr 123 --analyze --llm claude
    python scripts/generate-pr-ledger.py --pr 123 --publish --llm claude
"""

from transcription.historian.bundle import (
    CheckRunData,
    CommentData,
    CommitData,
    FactBundle,
    PRMetadata,
    ReviewData,
    ScopeData,
    SessionData,
    gather_pr_data,
)
from transcription.historian.estimation import (
    BoundedEstimation,
    MachineTimeEstimate,
    compute_bounded_estimate,
    compute_devlt_split,
    compute_machine_time,
    compute_session_bounds,
)

__all__ = [
    # Bundle types
    "FactBundle",
    "PRMetadata",
    "ScopeData",
    "CommitData",
    "SessionData",
    "CommentData",
    "ReviewData",
    "CheckRunData",
    # Bundle gathering
    "gather_pr_data",
    # Estimation
    "BoundedEstimation",
    "MachineTimeEstimate",
    "compute_bounded_estimate",
    "compute_session_bounds",
    "compute_devlt_split",
    "compute_machine_time",
]
