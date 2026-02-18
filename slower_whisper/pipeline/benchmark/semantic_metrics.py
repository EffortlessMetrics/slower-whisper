"""Semantic metrics for benchmark evaluation.

Provides deterministic metrics comparing predicted outputs from
KeywordSemanticAnnotator against gold labels.

Functions:
    compute_topic_f1: Micro-averaged precision/recall/F1 for topic labels
    compute_risk_metrics: Overall and per-severity metrics for risk tags
    compute_action_metrics: Accuracy/precision/recall for action items
    normalize_text: Text normalization helper for matching
    aggregate_semantic_metrics: Aggregate metrics across multiple samples
"""

from __future__ import annotations

import difflib
import re
import string
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize text for action matching.

    Performs:
    - casefold (aggressive lowercase)
    - collapse whitespace to single space
    - strip leading/trailing whitespace
    - remove punctuation

    Args:
        text: Input text to normalize

    Returns:
        Normalized text string

    Example:
        >>> normalize_text("  Hello,   World!  ")
        'hello world'
    """
    if not text:
        return ""

    # Casefold for aggressive lowercasing
    result = text.casefold()

    # Remove punctuation
    result = result.translate(str.maketrans("", "", string.punctuation))

    # Collapse whitespace to single space and strip
    result = re.sub(r"\s+", " ", result).strip()

    return result


def compute_topic_f1(predicted: list[str], gold: list[str] | None) -> dict[str, float] | None:
    """Compute micro-averaged precision, recall, and F1 for topic labels.

    Topic labels are treated as a set - order does not matter.
    Uses micro-averaging: counts true positives, false positives, and
    false negatives across all labels.

    Args:
        predicted: List of predicted topic labels
        gold: List of gold standard topic labels

    Returns:
        Dict with "precision", "recall", "f1" keys, or None if gold is empty/None

    Example:
        >>> compute_topic_f1(["pricing", "churn"], ["pricing", "escalation"])
        {'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
    """
    if gold is None:
        return None

    # Normalize to sets for comparison
    pred_set = set(predicted) if predicted else set()
    gold_set = set(gold) if gold else set()

    # Edge case: both empty
    if not gold_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # Edge case: gold is empty but predictions exist
    if not gold_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    # Compute metrics
    true_positives = len(pred_set & gold_set)
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_risk_metrics(
    predicted: list[dict[str, Any]], gold: list[dict[str, Any]] | None
) -> dict[str, Any] | None:
    """Compute precision/recall/F1 for risk tags.

    Matches are determined by type + segment_id. If segment_id is missing,
    it is treated as None.

    Args:
        predicted: List of predicted risk dicts with keys: type, severity, segment_id
        gold: List of gold standard risk dicts with same keys

    Returns:
        Dict with "overall" and "by_severity" metrics, or None if gold is None

    Example:
        >>> predicted = [{"type": "escalation", "severity": "high", "segment_id": 1}]
        >>> gold = [{"type": "escalation", "severity": "high", "segment_id": 1}]
        >>> result = compute_risk_metrics(predicted, gold)
        >>> result["overall"]["f1"]
        1.0
    """
    if gold is None:
        return None

    def make_key(item: dict[str, Any]) -> tuple[str | None, int | None]:
        """Create a matching key from type and segment_id."""
        return (item.get("type"), item.get("segment_id"))

    def compute_prf(pred_keys: set[Any], gold_keys: set[Any]) -> dict[str, float]:
        """Compute precision, recall, F1 from key sets."""
        if not gold_keys and not pred_keys:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not gold_keys:
            return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

        tp = len(pred_keys & gold_keys)
        fp = len(pred_keys - gold_keys)
        fn = len(gold_keys - pred_keys)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    # Build key sets
    pred_list = predicted if predicted else []
    gold_list = gold if gold else []

    pred_keys = {make_key(item) for item in pred_list}
    gold_keys = {make_key(item) for item in gold_list}

    overall = compute_prf(pred_keys, gold_keys)

    # Per-severity breakdown
    severities = ["low", "medium", "high"]
    by_severity: dict[str, dict[str, float]] = {}

    for severity in severities:
        pred_sev = {make_key(item) for item in pred_list if item.get("severity") == severity}
        gold_sev = {make_key(item) for item in gold_list if item.get("severity") == severity}
        by_severity[severity] = compute_prf(pred_sev, gold_sev)

    return {"overall": overall, "by_severity": by_severity}


def compute_action_metrics(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]] | None,
    similarity_threshold: float = 0.8,
) -> dict[str, Any] | None:
    """Compute accuracy/precision/recall for action items.

    Matching is based on text similarity using difflib.SequenceMatcher.
    An action is considered matched if its normalized text has similarity
    >= similarity_threshold with any gold action.

    Args:
        predicted: List of predicted action dicts with keys: text, speaker_id?, segment_ids?
        gold: List of gold standard action dicts with same keys
        similarity_threshold: Minimum similarity ratio for a match (default 0.8)

    Returns:
        Dict with accuracy, precision, recall, matched_count, gold_count,
        or None if gold is None

    Example:
        >>> predicted = [{"text": "I'll send the report"}]
        >>> gold = [{"text": "I will send the report"}]
        >>> result = compute_action_metrics(predicted, gold)
        >>> result["matched_count"]
        1
    """
    if gold is None:
        return None

    pred_list = predicted if predicted else []
    gold_list = gold if gold else []

    gold_count = len(gold_list)
    pred_count = len(pred_list)

    # Edge case: both empty
    if gold_count == 0 and pred_count == 0:
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "matched_count": 0,
            "gold_count": 0,
        }

    # Edge case: no gold but predictions exist
    if gold_count == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 1.0,
            "matched_count": 0,
            "gold_count": 0,
        }

    # Normalize gold texts
    gold_texts = [normalize_text(item.get("text", "")) for item in gold_list]

    # Track which gold items have been matched
    gold_matched = [False] * gold_count
    matched_count = 0

    for pred_item in pred_list:
        pred_text = normalize_text(pred_item.get("text", ""))
        if not pred_text:
            continue

        # Find best matching gold item
        best_score = 0.0
        best_idx = -1

        for idx, gold_text in enumerate(gold_texts):
            if gold_matched[idx]:
                continue  # Already matched
            if not gold_text:
                continue

            score = difflib.SequenceMatcher(None, pred_text, gold_text).ratio()
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score >= similarity_threshold and best_idx >= 0:
            gold_matched[best_idx] = True
            matched_count += 1

    # Compute metrics
    precision = matched_count / pred_count if pred_count > 0 else 0.0
    recall = matched_count / gold_count if gold_count > 0 else 0.0

    # Accuracy as the harmonic mean of precision and recall (same as F1)
    accuracy = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "matched_count": matched_count,
        "gold_count": gold_count,
    }


def aggregate_semantic_metrics(sample_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics across multiple samples.

    Computes macro and micro averages for topic, risk, and action metrics.
    Tracks counts of measured vs not_measured samples for each metric type.

    Args:
        sample_metrics: List of per-sample metric dicts. Each dict may contain:
            - "topic": result from compute_topic_f1
            - "risk": result from compute_risk_metrics
            - "action": result from compute_action_metrics

    Returns:
        Dict with aggregated metrics:
        {
            "topic": {
                "macro": {precision, recall, f1},
                "micro": {precision, recall, f1},
                "measured": int,
                "not_measured": int
            },
            "risk": {
                "overall": {macro: {...}, micro: {...}},
                "by_severity": {...},
                "measured": int,
                "not_measured": int
            },
            "action": {
                "macro": {accuracy, precision, recall},
                "micro": {accuracy, precision, recall},
                "total_matched": int,
                "total_gold": int,
                "measured": int,
                "not_measured": int
            }
        }

    Example:
        >>> samples = [
        ...     {"topic": {"precision": 1.0, "recall": 0.5, "f1": 0.67}},
        ...     {"topic": {"precision": 0.5, "recall": 1.0, "f1": 0.67}},
        ... ]
        >>> agg = aggregate_semantic_metrics(samples)
        >>> agg["topic"]["macro"]["precision"]
        0.75
    """
    if not sample_metrics:
        return {
            "topic": {"macro": None, "micro": None, "measured": 0, "not_measured": 0},
            "risk": {
                "overall": {"macro": None, "micro": None},
                "by_severity": {},
                "measured": 0,
                "not_measured": 0,
            },
            "action": {
                "macro": None,
                "micro": None,
                "total_matched": 0,
                "total_gold": 0,
                "measured": 0,
                "not_measured": 0,
            },
        }

    total_samples = len(sample_metrics)

    # Aggregate topic metrics
    topic_samples = [m["topic"] for m in sample_metrics if m.get("topic") is not None]
    topic_measured = len(topic_samples)
    topic_not_measured = total_samples - topic_measured

    topic_result: dict[str, Any] = {
        "macro": None,
        "micro": None,
        "measured": topic_measured,
        "not_measured": topic_not_measured,
    }

    if topic_samples:
        # Macro average: average of per-sample metrics
        topic_result["macro"] = {
            "precision": sum(s["precision"] for s in topic_samples) / topic_measured,
            "recall": sum(s["recall"] for s in topic_samples) / topic_measured,
            "f1": sum(s["f1"] for s in topic_samples) / topic_measured,
        }
        # For topics, micro = macro since we don't have TP/FP/FN counts
        # In a more sophisticated implementation, we'd track counts
        topic_result["micro"] = topic_result["macro"]

    # Aggregate risk metrics
    risk_samples = [m["risk"] for m in sample_metrics if m.get("risk") is not None]
    risk_measured = len(risk_samples)
    risk_not_measured = total_samples - risk_measured

    risk_result: dict[str, Any] = {
        "overall": {"macro": None, "micro": None},
        "by_severity": {},
        "measured": risk_measured,
        "not_measured": risk_not_measured,
    }

    if risk_samples:
        # Macro average overall
        overall_samples = [s["overall"] for s in risk_samples]
        risk_result["overall"]["macro"] = {
            "precision": sum(s["precision"] for s in overall_samples) / risk_measured,
            "recall": sum(s["recall"] for s in overall_samples) / risk_measured,
            "f1": sum(s["f1"] for s in overall_samples) / risk_measured,
        }
        risk_result["overall"]["micro"] = risk_result["overall"]["macro"]

        # Per-severity aggregation
        for severity in ["low", "medium", "high"]:
            sev_samples = [
                s["by_severity"][severity]
                for s in risk_samples
                if severity in s.get("by_severity", {})
            ]
            if sev_samples:
                risk_result["by_severity"][severity] = {
                    "macro": {
                        "precision": sum(s["precision"] for s in sev_samples) / len(sev_samples),
                        "recall": sum(s["recall"] for s in sev_samples) / len(sev_samples),
                        "f1": sum(s["f1"] for s in sev_samples) / len(sev_samples),
                    }
                }
            else:
                risk_result["by_severity"][severity] = {"macro": None}

    # Aggregate action metrics
    action_samples = [m["action"] for m in sample_metrics if m.get("action") is not None]
    action_measured = len(action_samples)
    action_not_measured = total_samples - action_measured

    action_result: dict[str, Any] = {
        "macro": None,
        "micro": None,
        "total_matched": 0,
        "total_gold": 0,
        "measured": action_measured,
        "not_measured": action_not_measured,
    }

    if action_samples:
        # Macro average
        action_result["macro"] = {
            "accuracy": sum(s["accuracy"] for s in action_samples) / action_measured,
            "precision": sum(s["precision"] for s in action_samples) / action_measured,
            "recall": sum(s["recall"] for s in action_samples) / action_measured,
        }

        # Micro average: compute from totals
        total_matched = sum(s["matched_count"] for s in action_samples)
        total_gold = sum(s["gold_count"] for s in action_samples)
        # For micro precision, we'd need total predictions
        # Approximate using matched/gold for recall
        micro_recall = total_matched / total_gold if total_gold > 0 else 0.0

        action_result["micro"] = {
            "accuracy": action_result["macro"]["accuracy"],  # Approx
            "precision": action_result["macro"]["precision"],  # Approx
            "recall": micro_recall,
        }
        action_result["total_matched"] = total_matched
        action_result["total_gold"] = total_gold

    return {
        "topic": topic_result,
        "risk": risk_result,
        "action": action_result,
    }
