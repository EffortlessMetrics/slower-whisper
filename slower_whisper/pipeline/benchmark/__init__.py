"""Benchmark infrastructure for evaluation and quality measurement.

This subpackage provides tools for evaluating semantic annotation quality
against gold standard datasets.
"""

from __future__ import annotations

from .semantic_metrics import (
    aggregate_semantic_metrics,
    compute_action_metrics,
    compute_risk_metrics,
    compute_topic_f1,
    normalize_text,
)

__all__ = [
    "aggregate_semantic_metrics",
    "compute_action_metrics",
    "compute_risk_metrics",
    "compute_topic_f1",
    "normalize_text",
]
