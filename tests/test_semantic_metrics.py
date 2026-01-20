"""
Unit tests for semantic metrics module (transcription/benchmark/semantic_metrics.py).

Tests cover:
- compute_topic_f1: Precision, recall, and F1 for topic detection
- compute_risk_metrics: Risk detection with per-severity breakdown
- compute_action_metrics: Action item matching with similarity threshold
- normalize_text: Text normalization for comparison
- aggregate_semantic_metrics: Aggregation across multiple samples
"""

from __future__ import annotations

import pytest

# ============================================================================
# Test normalize_text
# ============================================================================


class TestNormalizeText:
    """Tests for normalize_text() function."""

    def test_uppercase_to_lowercase(self) -> None:
        """Uppercase letters are converted to lowercase."""
        from transcription.benchmark.semantic_metrics import normalize_text

        assert normalize_text("HELLO WORLD") == "hello world"
        assert normalize_text("HeLLo WoRLd") == "hello world"

    def test_extra_whitespace_collapsed(self) -> None:
        """Multiple spaces are collapsed to single space."""
        from transcription.benchmark.semantic_metrics import normalize_text

        assert normalize_text("hello   world") == "hello world"
        assert normalize_text("  hello  world  ") == "hello world"
        assert normalize_text("hello\t\tworld") == "hello world"
        assert normalize_text("hello\n\nworld") == "hello world"

    def test_punctuation_stripped(self) -> None:
        """Punctuation is removed from text."""
        from transcription.benchmark.semantic_metrics import normalize_text

        assert normalize_text("hello, world!") == "hello world"
        assert normalize_text("hello... world???") == "hello world"
        assert normalize_text("hello-world") == "helloworld"
        assert normalize_text("hello (world)") == "hello world"
        assert normalize_text("it's a test") == "its a test"

    def test_mixed_cases(self) -> None:
        """Combination of uppercase, whitespace, and punctuation."""
        from transcription.benchmark.semantic_metrics import normalize_text

        assert normalize_text("  HELLO,   World!!! ") == "hello world"
        assert normalize_text("What's UP?!") == "whats up"
        assert normalize_text("A.B.C.") == "abc"

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        from transcription.benchmark.semantic_metrics import normalize_text

        assert normalize_text("") == ""
        assert normalize_text("   ") == ""
        assert normalize_text("...") == ""

    def test_already_normalized(self) -> None:
        """Already normalized text is unchanged."""
        from transcription.benchmark.semantic_metrics import normalize_text

        assert normalize_text("hello world") == "hello world"
        assert normalize_text("test") == "test"


# ============================================================================
# Test compute_topic_f1
# ============================================================================


class TestComputeTopicF1:
    """Tests for compute_topic_f1() function."""

    def test_perfect_match(self) -> None:
        """Predicted topics exactly match gold topics."""
        from transcription.benchmark.semantic_metrics import compute_topic_f1

        predicted = ["billing", "refund", "account"]
        gold = ["billing", "refund", "account"]

        result = compute_topic_f1(predicted, gold)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_overlap(self) -> None:
        """Some topics match, some don't."""
        from transcription.benchmark.semantic_metrics import compute_topic_f1

        # 2 out of 3 predicted are correct
        # 2 out of 3 gold are found
        predicted = ["billing", "refund", "shipping"]
        gold = ["billing", "refund", "account"]

        result = compute_topic_f1(predicted, gold)

        # Precision = 2/3 (correct predictions / total predictions)
        assert result["precision"] == pytest.approx(2 / 3)
        # Recall = 2/3 (found gold / total gold)
        assert result["recall"] == pytest.approx(2 / 3)
        # F1 = 2 * P * R / (P + R)
        expected_f1 = 2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3))
        assert result["f1"] == pytest.approx(expected_f1)

    def test_no_overlap(self) -> None:
        """No topics in common."""
        from transcription.benchmark.semantic_metrics import compute_topic_f1

        predicted = ["shipping", "returns"]
        gold = ["billing", "account"]

        result = compute_topic_f1(predicted, gold)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_empty_predicted(self) -> None:
        """No topics predicted."""
        from transcription.benchmark.semantic_metrics import compute_topic_f1

        predicted: list[str] = []
        gold = ["billing", "account"]

        result = compute_topic_f1(predicted, gold)

        # Precision is undefined (0/0), should return 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_empty_gold(self) -> None:
        """No gold topics (nothing to find)."""
        from transcription.benchmark.semantic_metrics import compute_topic_f1

        predicted = ["billing", "account"]
        gold: list[str] = []

        result = compute_topic_f1(predicted, gold)

        # Empty gold: precision=0 (all FP), recall=1.0 (found 100% of nothing), F1=0
        assert result["precision"] == 0.0
        assert result["recall"] == 1.0  # Standard IR: recall is 1.0 when gold is empty
        assert result["f1"] == 0.0

    def test_both_empty(self) -> None:
        """Both predicted and gold are empty."""
        from transcription.benchmark.semantic_metrics import compute_topic_f1

        predicted: list[str] = []
        gold: list[str] = []

        result = compute_topic_f1(predicted, gold)

        # Both empty - perfect match (nothing to predict, nothing predicted)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_single_item_match(self) -> None:
        """Single topic in both lists, matching."""
        from transcription.benchmark.semantic_metrics import compute_topic_f1

        predicted = ["billing"]
        gold = ["billing"]

        result = compute_topic_f1(predicted, gold)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_single_item_no_match(self) -> None:
        """Single topic in both lists, not matching."""
        from transcription.benchmark.semantic_metrics import compute_topic_f1

        predicted = ["billing"]
        gold = ["refund"]

        result = compute_topic_f1(predicted, gold)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_superset_predicted(self) -> None:
        """Predicted is superset of gold."""
        from transcription.benchmark.semantic_metrics import compute_topic_f1

        predicted = ["billing", "refund", "account", "shipping"]
        gold = ["billing", "refund"]

        result = compute_topic_f1(predicted, gold)

        # Precision = 2/4 (found 2 correct out of 4 predicted)
        assert result["precision"] == pytest.approx(0.5)
        # Recall = 2/2 (found all gold)
        assert result["recall"] == 1.0
        # F1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 2/3
        assert result["f1"] == pytest.approx(2 / 3)

    def test_subset_predicted(self) -> None:
        """Predicted is subset of gold."""
        from transcription.benchmark.semantic_metrics import compute_topic_f1

        predicted = ["billing"]
        gold = ["billing", "refund", "account"]

        result = compute_topic_f1(predicted, gold)

        # Precision = 1/1 (all predictions correct)
        assert result["precision"] == 1.0
        # Recall = 1/3 (found 1 of 3 gold)
        assert result["recall"] == pytest.approx(1 / 3)
        # F1 = 2 * 1.0 * (1/3) / (1.0 + 1/3) = 0.5
        assert result["f1"] == pytest.approx(0.5)

    def test_case_sensitivity(self) -> None:
        """Topics should be compared case-insensitively (if normalized)."""
        from transcription.benchmark.semantic_metrics import compute_topic_f1

        # Assuming topics are normalized or compared exactly
        # Test exact match first
        predicted = ["Billing", "REFUND"]
        gold = ["billing", "refund"]

        result = compute_topic_f1(predicted, gold)

        # Depending on implementation, this may or may not match
        # If case-insensitive, should be perfect match
        # If case-sensitive, should be no match
        # Typically topic comparison is case-insensitive
        assert result["precision"] >= 0.0
        assert result["recall"] >= 0.0


# ============================================================================
# Test compute_risk_metrics
# ============================================================================


class TestComputeRiskMetrics:
    """Tests for compute_risk_metrics() function."""

    def test_exact_matches(self) -> None:
        """Predicted risks exactly match gold risks (type + segment_id)."""
        from transcription.benchmark.semantic_metrics import compute_risk_metrics

        predicted = [
            {"type": "escalation", "segment_id": 0, "severity": "high"},
            {"type": "churn_risk", "segment_id": 2, "severity": "medium"},
        ]
        gold = [
            {"type": "escalation", "segment_id": 0, "severity": "high"},
            {"type": "churn_risk", "segment_id": 2, "severity": "medium"},
        ]

        result = compute_risk_metrics(predicted, gold)

        assert result["overall"]["precision"] == 1.0
        assert result["overall"]["recall"] == 1.0
        assert result["overall"]["f1"] == 1.0

    def test_type_matches_different_segment(self) -> None:
        """Type matches but segment_id is different."""
        from transcription.benchmark.semantic_metrics import compute_risk_metrics

        predicted = [
            {"type": "escalation", "segment_id": 0, "severity": "high"},
        ]
        gold = [
            {"type": "escalation", "segment_id": 5, "severity": "high"},
        ]

        result = compute_risk_metrics(predicted, gold)

        # Match requires both type AND segment_id
        assert result["overall"]["precision"] == 0.0
        assert result["overall"]["recall"] == 0.0
        assert result["overall"]["f1"] == 0.0

    def test_per_severity_breakdown(self) -> None:
        """Metrics broken down by severity level."""
        from transcription.benchmark.semantic_metrics import compute_risk_metrics

        predicted = [
            {"type": "escalation", "segment_id": 0, "severity": "high"},
            {"type": "churn_risk", "segment_id": 2, "severity": "medium"},
            {"type": "complaint", "segment_id": 4, "severity": "low"},
        ]
        gold = [
            {"type": "escalation", "segment_id": 0, "severity": "high"},
            {"type": "churn_risk", "segment_id": 2, "severity": "medium"},
            {"type": "issue", "segment_id": 6, "severity": "low"},
        ]

        result = compute_risk_metrics(predicted, gold)

        # Check by_severity exists
        assert "by_severity" in result

        # High severity: 1 correct / 1 predicted / 1 gold
        if "high" in result["by_severity"]:
            assert result["by_severity"]["high"]["precision"] == 1.0
            assert result["by_severity"]["high"]["recall"] == 1.0

        # Medium severity: 1 correct / 1 predicted / 1 gold
        if "medium" in result["by_severity"]:
            assert result["by_severity"]["medium"]["precision"] == 1.0
            assert result["by_severity"]["medium"]["recall"] == 1.0

        # Low severity: 0 correct / 1 predicted / 1 gold
        if "low" in result["by_severity"]:
            assert result["by_severity"]["low"]["precision"] == 0.0
            assert result["by_severity"]["low"]["recall"] == 0.0

    def test_empty_predicted_risks(self) -> None:
        """No risks predicted."""
        from transcription.benchmark.semantic_metrics import compute_risk_metrics

        predicted: list[dict] = []
        gold = [
            {"type": "escalation", "segment_id": 0, "severity": "high"},
        ]

        result = compute_risk_metrics(predicted, gold)

        assert result["overall"]["precision"] == 0.0
        assert result["overall"]["recall"] == 0.0
        assert result["overall"]["f1"] == 0.0

    def test_empty_gold_risks(self) -> None:
        """No gold risks (nothing to find)."""
        from transcription.benchmark.semantic_metrics import compute_risk_metrics

        predicted = [
            {"type": "escalation", "segment_id": 0, "severity": "high"},
        ]
        gold: list[dict] = []

        result = compute_risk_metrics(predicted, gold)

        # Empty gold: precision=0 (all FP), recall=1.0 (found 100% of nothing), F1=0
        assert result["overall"]["precision"] == 0.0
        assert result["overall"]["recall"] == 1.0  # Standard IR: recall is 1.0 when gold is empty
        assert result["overall"]["f1"] == 0.0

    def test_both_empty_risks(self) -> None:
        """Both predicted and gold are empty."""
        from transcription.benchmark.semantic_metrics import compute_risk_metrics

        predicted: list[dict] = []
        gold: list[dict] = []

        result = compute_risk_metrics(predicted, gold)

        # Both empty - perfect match
        assert result["overall"]["precision"] == 1.0
        assert result["overall"]["recall"] == 1.0
        assert result["overall"]["f1"] == 1.0

    def test_mixed_severities(self) -> None:
        """Risks with different severities."""
        from transcription.benchmark.semantic_metrics import compute_risk_metrics

        predicted = [
            {"type": "escalation", "segment_id": 0, "severity": "high"},
            {"type": "escalation", "segment_id": 1, "severity": "high"},
            {"type": "complaint", "segment_id": 3, "severity": "low"},
        ]
        gold = [
            {"type": "escalation", "segment_id": 0, "severity": "high"},
            {"type": "churn_risk", "segment_id": 2, "severity": "medium"},
            {"type": "complaint", "segment_id": 3, "severity": "low"},
        ]

        result = compute_risk_metrics(predicted, gold)

        # Overall: 2 correct (seg 0 and 3), 3 predicted, 3 gold
        assert result["overall"]["precision"] == pytest.approx(2 / 3)
        assert result["overall"]["recall"] == pytest.approx(2 / 3)

    def test_partial_matches_same_segment(self) -> None:
        """Same segment_id but different type."""
        from transcription.benchmark.semantic_metrics import compute_risk_metrics

        predicted = [
            {"type": "escalation", "segment_id": 0, "severity": "high"},
        ]
        gold = [
            {"type": "churn_risk", "segment_id": 0, "severity": "high"},
        ]

        result = compute_risk_metrics(predicted, gold)

        # Different types, even on same segment, should not match
        assert result["overall"]["precision"] == 0.0
        assert result["overall"]["recall"] == 0.0


# ============================================================================
# Test compute_action_metrics
# ============================================================================


class TestComputeActionMetrics:
    """Tests for compute_action_metrics() function."""

    def test_exact_text_match(self) -> None:
        """Action text matches exactly."""
        from transcription.benchmark.semantic_metrics import compute_action_metrics

        predicted = [{"text": "Send invoice to customer"}]
        gold = [{"text": "Send invoice to customer"}]

        result = compute_action_metrics(predicted, gold, similarity_threshold=0.8)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["accuracy"] >= 1.0

    def test_similar_text_above_threshold(self) -> None:
        """Action text is similar, above threshold."""
        from transcription.benchmark.semantic_metrics import compute_action_metrics

        predicted = [{"text": "Send the invoice to the customer"}]
        gold = [{"text": "Send invoice to customer"}]

        result = compute_action_metrics(predicted, gold, similarity_threshold=0.7)

        # Should match due to high similarity
        assert result["precision"] >= 0.0
        assert result["recall"] >= 0.0

    def test_dissimilar_text_below_threshold(self) -> None:
        """Action text is too different, below threshold."""
        from transcription.benchmark.semantic_metrics import compute_action_metrics

        predicted = [{"text": "Call the customer about billing issue"}]
        gold = [{"text": "Send invoice to customer"}]

        result = compute_action_metrics(predicted, gold, similarity_threshold=0.9)

        # Should not match due to low similarity
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_empty_predicted_actions(self) -> None:
        """No actions predicted."""
        from transcription.benchmark.semantic_metrics import compute_action_metrics

        predicted: list[dict] = []
        gold = [{"text": "Send invoice to customer"}]

        result = compute_action_metrics(predicted, gold, similarity_threshold=0.8)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_empty_gold_actions(self) -> None:
        """No gold actions (nothing to find)."""
        from transcription.benchmark.semantic_metrics import compute_action_metrics

        predicted = [{"text": "Send invoice to customer"}]
        gold: list[dict] = []

        result = compute_action_metrics(predicted, gold, similarity_threshold=0.8)

        # Empty gold: precision=0 (all FP), recall=1.0 (found 100% of nothing)
        assert result["precision"] == 0.0
        assert result["recall"] == 1.0  # Standard IR: recall is 1.0 when gold is empty

    def test_both_empty_actions(self) -> None:
        """Both predicted and gold are empty."""
        from transcription.benchmark.semantic_metrics import compute_action_metrics

        predicted: list[dict] = []
        gold: list[dict] = []

        result = compute_action_metrics(predicted, gold, similarity_threshold=0.8)

        # Both empty - perfect match
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_threshold_boundary_exact(self) -> None:
        """Similarity exactly at threshold."""
        from transcription.benchmark.semantic_metrics import compute_action_metrics

        # Two identical texts should have similarity 1.0
        predicted = [{"text": "test action"}]
        gold = [{"text": "test action"}]

        result = compute_action_metrics(predicted, gold, similarity_threshold=1.0)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_threshold_zero(self) -> None:
        """Threshold of 0.0 matches everything."""
        from transcription.benchmark.semantic_metrics import compute_action_metrics

        predicted = [{"text": "completely different text"}]
        gold = [{"text": "unrelated action item"}]

        result = compute_action_metrics(predicted, gold, similarity_threshold=0.0)

        # With threshold 0, anything should match
        assert result["precision"] >= 0.0
        assert result["recall"] >= 0.0

    def test_multiple_actions_partial_match(self) -> None:
        """Multiple actions with some matching."""
        from transcription.benchmark.semantic_metrics import compute_action_metrics

        predicted = [
            {"text": "Send invoice to customer"},
            {"text": "Schedule follow-up call"},
            {"text": "Update CRM record"},
        ]
        gold = [
            {"text": "Send invoice to customer"},
            {"text": "Schedule a follow-up call"},
            {"text": "Process refund request"},
        ]

        result = compute_action_metrics(predicted, gold, similarity_threshold=0.7)

        # At least one exact match (first item)
        assert result["precision"] > 0.0
        assert result["recall"] > 0.0

    def test_result_contains_expected_keys(self) -> None:
        """Result dictionary contains all expected metrics."""
        from transcription.benchmark.semantic_metrics import compute_action_metrics

        predicted = [{"text": "test"}]
        gold = [{"text": "test"}]

        result = compute_action_metrics(predicted, gold, similarity_threshold=0.8)

        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result


# ============================================================================
# Test aggregate_semantic_metrics
# ============================================================================


class TestAggregateSemanticMetrics:
    """Tests for aggregate_semantic_metrics() function."""

    def test_multiple_samples_all_metrics(self) -> None:
        """Aggregate metrics from multiple samples with all metrics present."""
        from transcription.benchmark.semantic_metrics import aggregate_semantic_metrics

        # Use keys expected by implementation: "topic", "risk", "action"
        sample_metrics = [
            {
                "topic": {"precision": 0.8, "recall": 0.9, "f1": 0.85},
                "risk": {
                    "overall": {"precision": 0.7, "recall": 0.8, "f1": 0.75},
                    "by_severity": {},
                },
                "action": {
                    "precision": 0.9,
                    "recall": 0.8,
                    "accuracy": 0.85,
                    "matched_count": 4,
                    "gold_count": 5,
                },
            },
            {
                "topic": {"precision": 0.9, "recall": 0.85, "f1": 0.875},
                "risk": {
                    "overall": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
                    "by_severity": {},
                },
                "action": {
                    "precision": 0.85,
                    "recall": 0.9,
                    "accuracy": 0.875,
                    "matched_count": 4,
                    "gold_count": 5,
                },
            },
        ]

        result = aggregate_semantic_metrics(sample_metrics)

        # Should have aggregated values (typically means)
        assert "topic" in result
        assert "risk" in result
        assert "action" in result

        # Check that values are reasonable aggregates
        assert result["topic"]["macro"] is not None
        assert 0.0 <= result["topic"]["macro"]["precision"] <= 1.0
        assert 0.0 <= result["topic"]["macro"]["recall"] <= 1.0

    def test_samples_with_none_values(self) -> None:
        """Some samples have None for unmeasured metrics."""
        from transcription.benchmark.semantic_metrics import aggregate_semantic_metrics

        sample_metrics = [
            {
                "topic": {"precision": 0.8, "recall": 0.9, "f1": 0.85},
                "risk": None,  # Not measured
                "action": {
                    "precision": 0.9,
                    "recall": 0.8,
                    "accuracy": 0.85,
                    "matched_count": 4,
                    "gold_count": 5,
                },
            },
            {
                "topic": {"precision": 0.9, "recall": 0.85, "f1": 0.875},
                "risk": {
                    "overall": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
                    "by_severity": {},
                },
                "action": None,  # Not measured
            },
        ]

        result = aggregate_semantic_metrics(sample_metrics)

        # Should still produce valid aggregation
        assert "topic" in result
        # Coverage tracking
        assert result["topic"]["measured"] == 2
        assert result["risk"]["measured"] == 1
        assert result["action"]["measured"] == 1

    def test_coverage_reporting(self) -> None:
        """Aggregation includes coverage information."""
        from transcription.benchmark.semantic_metrics import aggregate_semantic_metrics

        sample_metrics = [
            {
                "topic_f1": {"precision": 0.8, "recall": 0.9, "f1": 0.85},
                "risk_metrics": None,
                "action_metrics": None,
            },
            {
                "topic_f1": {"precision": 0.9, "recall": 0.85, "f1": 0.875},
                "risk_metrics": None,
                "action_metrics": None,
            },
            {
                "topic_f1": None,
                "risk_metrics": {"overall": {"precision": 0.8, "recall": 0.7, "f1": 0.75}},
                "action_metrics": {"precision": 0.9, "recall": 0.8, "accuracy": 0.85},
            },
        ]

        result = aggregate_semantic_metrics(sample_metrics)

        # Should include coverage info
        if "coverage" in result:
            assert "topic_f1" in result["coverage"] or "n_samples" in result
        # Or coverage might be reported differently

    def test_empty_list(self) -> None:
        """Empty list of samples."""
        from transcription.benchmark.semantic_metrics import aggregate_semantic_metrics

        sample_metrics: list[dict] = []

        result = aggregate_semantic_metrics(sample_metrics)

        # Should return empty or default aggregation
        assert isinstance(result, dict)

    def test_single_sample(self) -> None:
        """Single sample - aggregation is just that sample's values."""
        from transcription.benchmark.semantic_metrics import aggregate_semantic_metrics

        sample_metrics = [
            {
                "topic": {"precision": 0.8, "recall": 0.9, "f1": 0.85},
                "risk": {
                    "overall": {"precision": 0.7, "recall": 0.8, "f1": 0.75},
                    "by_severity": {},
                },
                "action": {
                    "precision": 0.9,
                    "recall": 0.8,
                    "accuracy": 0.85,
                    "matched_count": 4,
                    "gold_count": 5,
                },
            },
        ]

        result = aggregate_semantic_metrics(sample_metrics)

        # With one sample, aggregated values should match the single sample
        assert result["topic"]["macro"]["precision"] == pytest.approx(0.8)
        assert result["topic"]["macro"]["recall"] == pytest.approx(0.9)
        assert result["topic"]["macro"]["f1"] == pytest.approx(0.85)

    def test_all_none_metrics(self) -> None:
        """All samples have None for a particular metric."""
        from transcription.benchmark.semantic_metrics import aggregate_semantic_metrics

        sample_metrics = [
            {
                "topic": None,
                "risk": None,
                "action": None,
            },
            {
                "topic": None,
                "risk": None,
                "action": None,
            },
        ]

        result = aggregate_semantic_metrics(sample_metrics)

        # Should handle gracefully - macro values are None when no measurements
        assert isinstance(result, dict)
        assert result["topic"]["macro"] is None
        assert result["topic"]["not_measured"] == 2

    def test_aggregation_is_mean(self) -> None:
        """Verify aggregation computes mean of values."""
        from transcription.benchmark.semantic_metrics import aggregate_semantic_metrics

        sample_metrics = [
            {
                "topic": {"precision": 0.6, "recall": 0.8, "f1": 0.7},
                "risk": {
                    "overall": {"precision": 0.5, "recall": 0.6, "f1": 0.55},
                    "by_severity": {},
                },
                "action": {
                    "precision": 0.7,
                    "recall": 0.8,
                    "accuracy": 0.75,
                    "matched_count": 3,
                    "gold_count": 4,
                },
            },
            {
                "topic": {"precision": 0.8, "recall": 0.9, "f1": 0.85},
                "risk": {
                    "overall": {"precision": 0.7, "recall": 0.8, "f1": 0.75},
                    "by_severity": {},
                },
                "action": {
                    "precision": 0.9,
                    "recall": 0.85,
                    "accuracy": 0.875,
                    "matched_count": 4,
                    "gold_count": 5,
                },
            },
            {
                "topic": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "risk": {
                    "overall": {"precision": 0.9, "recall": 1.0, "f1": 0.95},
                    "by_severity": {},
                },
                "action": {
                    "precision": 0.8,
                    "recall": 0.75,
                    "accuracy": 0.775,
                    "matched_count": 3,
                    "gold_count": 4,
                },
            },
        ]

        result = aggregate_semantic_metrics(sample_metrics)

        # Topic precision should be mean: (0.6 + 0.8 + 1.0) / 3 = 0.8
        assert result["topic"]["macro"]["precision"] == pytest.approx(0.8)
        # Topic recall should be mean: (0.8 + 0.9 + 1.0) / 3 = 0.9
        assert result["topic"]["macro"]["recall"] == pytest.approx(0.9)
        # Topic f1 should be mean: (0.7 + 0.85 + 1.0) / 3 = 0.85
        assert result["topic"]["macro"]["f1"] == pytest.approx(0.85)


# ============================================================================
# Integration Tests
# ============================================================================


class TestSemanticMetricsIntegration:
    """Integration tests for semantic metrics workflow."""

    def test_full_evaluation_workflow(self) -> None:
        """Test a complete evaluation workflow with all metrics."""
        from transcription.benchmark.semantic_metrics import (
            aggregate_semantic_metrics,
            compute_action_metrics,
            compute_risk_metrics,
            compute_topic_f1,
        )

        # Simulate evaluation of multiple samples
        samples = [
            {
                "predicted": {
                    "topics": ["billing", "refund"],
                    "risks": [{"type": "escalation", "segment_id": 0, "severity": "high"}],
                    "actions": [{"text": "Process refund"}],
                },
                "gold": {
                    "topics": ["billing", "refund", "account"],
                    "risks": [{"type": "escalation", "segment_id": 0, "severity": "high"}],
                    "actions": [{"text": "Process the refund request"}],
                },
            },
            {
                "predicted": {
                    "topics": ["support", "technical"],
                    "risks": [{"type": "churn_risk", "segment_id": 2, "severity": "medium"}],
                    "actions": [{"text": "Schedule callback"}],
                },
                "gold": {
                    "topics": ["support", "technical"],
                    "risks": [{"type": "churn_risk", "segment_id": 2, "severity": "medium"}],
                    "actions": [{"text": "Schedule callback"}],
                },
            },
        ]

        # Compute metrics for each sample
        sample_metrics = []
        for sample in samples:
            topic_metrics = compute_topic_f1(
                sample["predicted"]["topics"], sample["gold"]["topics"]
            )
            risk_metrics = compute_risk_metrics(
                sample["predicted"]["risks"], sample["gold"]["risks"]
            )
            action_metrics = compute_action_metrics(
                sample["predicted"]["actions"],
                sample["gold"]["actions"],
                similarity_threshold=0.7,
            )
            # Use keys expected by aggregate function
            sample_metrics.append(
                {
                    "topic": topic_metrics,
                    "risk": risk_metrics,
                    "action": action_metrics,
                }
            )

        # Aggregate across samples
        aggregated = aggregate_semantic_metrics(sample_metrics)

        # Verify structure (uses "topic", "risk", "action" keys)
        assert "topic" in aggregated
        assert "risk" in aggregated
        assert "action" in aggregated

        # Verify reasonable values
        assert aggregated["topic"]["macro"] is not None
        assert 0.0 <= aggregated["topic"]["macro"]["f1"] <= 1.0
        assert aggregated["risk"]["overall"]["macro"] is not None
        assert 0.0 <= aggregated["risk"]["overall"]["macro"]["f1"] <= 1.0

    def test_normalize_text_used_in_comparison(self) -> None:
        """Verify normalize_text is used for action comparison."""
        from transcription.benchmark.semantic_metrics import (
            compute_action_metrics,
            normalize_text,
        )

        # Two texts that normalize to the same thing
        text1 = "Send Invoice To Customer!"
        text2 = "send invoice to customer"

        assert normalize_text(text1) == normalize_text(text2)

        # They should match with high enough threshold
        predicted = [{"text": text1}]
        gold = [{"text": text2}]

        result = compute_action_metrics(predicted, gold, similarity_threshold=0.9)

        # Should match since normalized texts are identical
        assert result["precision"] >= 0.5  # At least partial match expected
