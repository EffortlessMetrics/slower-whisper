"""
Integration tests for SemanticBenchmarkRunner (transcription/benchmark_cli.py).

Tests cover:
- Gold label loading matches sample IDs
- _evaluate_tags produces non-empty metrics for samples with gold
- Proper delegation to semantic_metrics module
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


class TestSemanticBenchmarkRunnerGoldLoading:
    """Tests for gold label loading in SemanticBenchmarkRunner."""

    def test_gold_labels_load_from_correct_path(self, tmp_path: Path) -> None:
        """Gold labels load from benchmarks/gold/semantic/<meeting_id>.json."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner

        # Create a mock gold directory with a sample gold file
        gold_dir = tmp_path / "gold" / "semantic"
        gold_dir.mkdir(parents=True)

        gold_data = {
            "schema_version": 1,
            "meeting_id": "test_meeting",
            "topics": [{"label": "pricing", "segment_ids": [0, 1]}],
            "risks": [{"type": "escalation", "severity": "high", "segment_id": 1}],
            "actions": [{"text": "Send proposal", "speaker_id": "spk_1"}],
        }
        gold_file = gold_dir / "test_meeting.json"
        gold_file.write_text(json.dumps(gold_data))

        # Create runner with mocked gold_dir
        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )
        runner._gold_dir = gold_dir

        # Load gold labels
        result = runner._load_gold_labels("test_meeting")

        assert result is not None
        assert result["schema_version"] == 1
        assert result["meeting_id"] == "test_meeting"
        assert len(result["topics"]) == 1
        assert result["topics"][0]["label"] == "pricing"
        assert len(result["risks"]) == 1
        assert result["risks"][0]["type"] == "escalation"
        assert len(result["actions"]) == 1
        assert result["actions"][0]["text"] == "Send proposal"

    def test_gold_labels_return_none_for_missing_file(self, tmp_path: Path) -> None:
        """Return None when gold file doesn't exist."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner

        gold_dir = tmp_path / "gold" / "semantic"
        gold_dir.mkdir(parents=True)

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )
        runner._gold_dir = gold_dir

        result = runner._load_gold_labels("nonexistent_meeting")

        assert result is None


class TestSemanticBenchmarkRunnerEvaluateTags:
    """Tests for _evaluate_tags method."""

    def test_evaluate_tags_with_gold_produces_metrics(self, tmp_path: Path) -> None:
        """_evaluate_tags produces non-empty metrics when gold labels exist."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        # Create sample with transcript containing triggering keywords
        sample = EvalSample(
            dataset="ami",
            id="test_sample",
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="I need to escalate this to a manager. We should send the proposal.",
        )

        gold = {
            "schema_version": 1,
            "meeting_id": "test_sample",
            "topics": [{"label": "escalation", "segment_ids": [0]}],
            "risks": [{"type": "escalation", "severity": "high", "segment_id": 0}],
            "actions": [{"text": "Send the proposal", "speaker_id": "spk_1"}],
        }

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )

        result = runner._evaluate_tags(sample, gold)

        # Should have all expected metric keys
        assert "topic_precision" in result
        assert "topic_recall" in result
        assert "topic_f1" in result
        assert "risk_precision" in result
        assert "risk_recall" in result
        assert "risk_f1" in result
        assert "action_precision" in result
        assert "action_recall" in result
        assert "action_f1" in result

        # tags_reason should be None (not a failure case)
        assert result.get("tags_reason") is None

        # At least some metrics should be non-None (transcript has triggering content)
        non_none_metrics = [
            k for k in ["topic_f1", "risk_f1", "action_f1"] if result.get(k) is not None
        ]
        assert len(non_none_metrics) > 0, "Expected at least some metrics to be computed"

    def test_evaluate_tags_without_gold_returns_none_metrics(self) -> None:
        """_evaluate_tags returns None metrics with reason when no gold."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        sample = EvalSample(
            dataset="ami",
            id="test_sample",
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="Some transcript text.",
        )

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )

        result = runner._evaluate_tags(sample, gold=None)

        # All metrics should be None
        assert result["topic_precision"] is None
        assert result["topic_recall"] is None
        assert result["topic_f1"] is None
        assert result["risk_precision"] is None
        assert result["risk_recall"] is None
        assert result["risk_f1"] is None
        assert result["action_precision"] is None
        assert result["action_recall"] is None
        assert result["action_f1"] is None

        # Should have reason
        assert result["tags_reason"] == "no_gold_labels"

    def test_evaluate_tags_without_transcript_returns_none_metrics(self) -> None:
        """_evaluate_tags returns None metrics when sample has no transcript."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        sample = EvalSample(
            dataset="ami",
            id="test_sample",
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="",  # Empty transcript
        )

        gold = {
            "schema_version": 1,
            "meeting_id": "test_sample",
            "topics": [{"label": "pricing"}],
            "risks": [],
            "actions": [],
        }

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )

        result = runner._evaluate_tags(sample, gold)

        # All metrics should be None
        assert result["topic_f1"] is None
        assert result["risk_f1"] is None
        assert result["action_f1"] is None

        # Should have reason
        assert result["tags_reason"] == "no_transcript"


class TestSemanticBenchmarkRunnerMetricsIntegration:
    """Integration tests verifying semantic_metrics module is used."""

    def test_evaluate_tags_uses_compute_topic_f1(self) -> None:
        """_evaluate_tags delegates topic scoring to compute_topic_f1."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        sample = EvalSample(
            dataset="ami",
            id="test_sample",
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="We need to discuss pricing and costs for the budget.",
        )

        gold = {
            "schema_version": 1,
            "meeting_id": "test_sample",
            "topics": [{"label": "pricing"}],  # Matches annotator's output
            "risks": [],
            "actions": [],
        }

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )

        result = runner._evaluate_tags(sample, gold)

        # Topic metrics should be computed (not None)
        assert result["topic_precision"] is not None
        assert result["topic_recall"] is not None
        assert result["topic_f1"] is not None

    def test_evaluate_tags_uses_compute_risk_metrics(self) -> None:
        """_evaluate_tags delegates risk scoring to compute_risk_metrics."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        sample = EvalSample(
            dataset="ami",
            id="test_sample",
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="I need to speak to a manager about this escalation.",
        )

        gold = {
            "schema_version": 1,
            "meeting_id": "test_sample",
            "topics": [],
            "risks": [{"type": "escalation", "severity": "high", "segment_id": 0}],
            "actions": [],
        }

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )

        result = runner._evaluate_tags(sample, gold)

        # Risk metrics should be computed (not None)
        assert result["risk_precision"] is not None
        assert result["risk_recall"] is not None
        assert result["risk_f1"] is not None

        # Should note that severity matching isn't supported
        assert result.get("risk_note") == "severity_not_measured"

    def test_evaluate_tags_uses_compute_action_metrics(self) -> None:
        """_evaluate_tags delegates action scoring to compute_action_metrics."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        sample = EvalSample(
            dataset="ami",
            id="test_sample",
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="I'll send you the proposal by tomorrow.",
        )

        gold = {
            "schema_version": 1,
            "meeting_id": "test_sample",
            "topics": [],
            "risks": [],
            "actions": [{"text": "Send the proposal", "speaker_id": "spk_1"}],
        }

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )

        result = runner._evaluate_tags(sample, gold)

        # Action metrics should be computed (not None)
        assert result["action_precision"] is not None
        assert result["action_recall"] is not None
        assert result["action_f1"] is not None

        # Should include action counts
        assert "action_matched" in result
        assert "action_gold" in result


class TestSemanticBenchmarkRunnerTagsMode:
    """Tests for tags mode behavior."""

    def test_tags_mode_does_not_include_summary_metrics(self) -> None:
        """In tags mode, summary metrics (faithfulness/coverage/clarity) are not included."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        sample = EvalSample(
            dataset="ami",
            id="test_sample",
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="Test transcript",
        )

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",  # Tags mode only
        )

        # Mock gold loading to return None
        with patch.object(runner, "_load_gold_labels", return_value=None):
            result = runner.evaluate_sample(sample)

        # Summary metrics should NOT be present
        assert "faithfulness" not in result
        assert "coverage" not in result
        assert "clarity" not in result

        # Tags metrics should be present (even if None)
        assert "topic_f1" in result
        assert "risk_f1" in result
        assert "action_f1" in result

    def test_aggregate_metrics_tags_mode_excludes_summary(self) -> None:
        """In tags mode, aggregate_metrics only includes tag metrics."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )

        sample_results = [
            {
                "id": "sample1",
                "topic_precision": 0.8,
                "topic_recall": 0.9,
                "topic_f1": 0.85,
                "risk_precision": 0.7,
                "risk_recall": 0.8,
                "risk_f1": 0.75,
                "action_precision": 0.9,
                "action_recall": 0.85,
                "action_f1": 0.87,
            }
        ]

        metrics = runner.aggregate_metrics(sample_results)

        # Should have 9 tag metrics
        metric_names = [m.name for m in metrics]

        assert "topic_precision" in metric_names
        assert "topic_recall" in metric_names
        assert "topic_f1" in metric_names
        assert "risk_precision" in metric_names
        assert "risk_recall" in metric_names
        assert "risk_f1" in metric_names
        assert "action_precision" in metric_names
        assert "action_recall" in metric_names
        assert "action_f1" in metric_names

        # Should NOT have summary metrics
        assert "faithfulness" not in metric_names
        assert "coverage" not in metric_names
        assert "clarity" not in metric_names


class TestGoldSchemaAlignment:
    """Tests verifying gold schema field alignment."""

    def test_gold_schema_fields_read_correctly(self) -> None:
        """_evaluate_tags reads topics/risks/actions from gold (not keywords/risk_tags)."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        sample = EvalSample(
            dataset="ami",
            id="test_sample",
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="I need to escalate this and we should send the proposal.",
        )

        # Gold with correct schema fields (topics, risks, actions)
        # NOT the old keywords/risk_tags fields
        gold = {
            "schema_version": 1,
            "meeting_id": "test_sample",
            "topics": [
                {"label": "escalation", "segment_ids": [0]},
                {"label": "pricing", "segment_ids": [1]},
            ],
            "risks": [
                {"type": "escalation", "severity": "high", "segment_id": 0, "evidence": "escalate"}
            ],
            "actions": [{"text": "Send the proposal", "speaker_id": "spk_1", "segment_ids": [1]}],
        }

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )

        result = runner._evaluate_tags(sample, gold)

        # Should not fail and should produce some metrics
        assert result.get("tags_reason") is None

        # Topic, risk, and action metrics should be computed
        assert result["topic_f1"] is not None
        assert result["risk_f1"] is not None
        # Action might or might not match depending on fuzzy matching

    def test_old_gold_format_not_used(self) -> None:
        """_evaluate_tags does NOT read old keywords/risk_tags fields."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        sample = EvalSample(
            dataset="ami",
            id="test_sample",
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="I need to escalate this to a manager.",
        )

        # Old format gold (should NOT be read)
        old_format_gold = {
            "schema_version": 1,
            "meeting_id": "test_sample",
            "keywords": ["escalate", "manager"],  # OLD format
            "risk_tags": ["escalation"],  # OLD format
            "actions": [],  # Still using correct actions format
            # Missing: topics, risks (new format)
        }

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )

        result = runner._evaluate_tags(sample, old_format_gold)

        # topics field is empty/missing, so gold_topics will be empty
        # This means compute_topic_f1 will run with empty gold
        # The test passes if it doesn't crash and doesn't try to read keywords
        assert result.get("tags_reason") is None


class TestSampleIdGoldFilenameAlignment:
    """Guardrail tests: sample IDs must match gold filenames."""

    def test_sample_id_mismatch_returns_no_gold_labels(self, tmp_path: Path) -> None:
        """When sample.id doesn't match any gold filename, tags_reason='no_gold_labels'."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        # Create gold dir with a gold file for a DIFFERENT meeting_id
        gold_dir = tmp_path / "gold" / "semantic"
        gold_dir.mkdir(parents=True)

        gold_data = {
            "schema_version": 1,
            "meeting_id": "design_review",  # This is the gold file name
            "topics": [{"label": "pricing"}],
            "risks": [],
            "actions": [],
        }
        (gold_dir / "design_review.json").write_text(json.dumps(gold_data))

        # Sample has a different ID (mimics AMI meeting ID like ES2002a)
        sample = EvalSample(
            dataset="ami",
            id="ES2002a",  # Doesn't match "design_review"
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="Some meeting transcript.",
        )

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )
        runner._gold_dir = gold_dir

        # Evaluate via evaluate_sample (which loads gold via ID)
        result = runner.evaluate_sample(sample)

        # No gold file for ES2002a, so should get no_gold_labels
        assert result["tags_reason"] == "no_gold_labels"
        assert result["topic_f1"] is None
        assert result["risk_f1"] is None
        assert result["action_f1"] is None

    def test_sample_id_matches_gold_filename_produces_metrics(self, tmp_path: Path) -> None:
        """When sample.id matches gold filename, metrics are computed."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        gold_dir = tmp_path / "gold" / "semantic"
        gold_dir.mkdir(parents=True)

        gold_data = {
            "schema_version": 1,
            "meeting_id": "ES2002a",
            "topics": [{"label": "escalation"}],
            "risks": [{"type": "escalation", "severity": "high", "segment_id": 0}],
            "actions": [],
        }
        (gold_dir / "ES2002a.json").write_text(json.dumps(gold_data))

        sample = EvalSample(
            dataset="ami",
            id="ES2002a",  # Matches gold filename
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="I need to escalate this to a manager.",
        )

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )
        runner._gold_dir = gold_dir

        result = runner.evaluate_sample(sample)

        # Should have metrics computed (not tags_reason)
        assert result.get("tags_reason") is None
        # Should produce some non-None metrics
        assert result["risk_f1"] is not None


class TestSyntheticSegmentsNote:
    """Tests for synthetic_transcript_segments note."""

    def test_synthetic_segments_note_added(self) -> None:
        """When using synthetic single-segment transcript, tags_note is set."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        sample = EvalSample(
            dataset="ami",
            id="test_sample",
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="I need to escalate this to a manager.",
        )

        gold = {
            "schema_version": 1,
            "meeting_id": "test_sample",
            "topics": [],
            "risks": [{"type": "escalation", "severity": "high", "segment_id": 0}],
            "actions": [],
        }

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )

        result = runner._evaluate_tags(sample, gold)

        # Should have synthetic segments note
        assert result.get("tags_note") == "synthetic_transcript_segments"

    def test_synthetic_segments_note_present_even_with_other_notes(self) -> None:
        """Synthetic segments note and other notes can coexist."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        sample = EvalSample(
            dataset="ami",
            id="test_sample",
            audio_path=Path("/fake/audio.wav"),
            reference_transcript="I need to escalate this to a manager.",
        )

        gold = {
            "schema_version": 1,
            "meeting_id": "test_sample",
            "topics": [{"label": "completely_unrelated_topic"}],  # Will cause vocabulary_mismatch
            "risks": [{"type": "escalation", "severity": "high", "segment_id": 0}],
            "actions": [],
        }

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )

        result = runner._evaluate_tags(sample, gold)

        # Should have both notes
        assert result.get("tags_note") == "synthetic_transcript_segments"
        assert result.get("topic_note") == "vocabulary_mismatch"
        assert result.get("risk_note") == "severity_not_measured"


class TestBenchmarkReceipt:
    """Tests for BenchmarkReceipt generation."""

    def test_receipt_generated_in_run(self) -> None:
        """BenchmarkResult includes receipt after run."""
        from unittest.mock import MagicMock

        from transcription.benchmark_cli import SemanticBenchmarkRunner

        runner = SemanticBenchmarkRunner(
            track="semantic",
            dataset="ami",
            split="test",
            mode="tags",
        )
        # Mock get_samples to return empty list (no samples to evaluate)
        runner.get_samples = MagicMock(return_value=[])

        result = runner.run(limit=0)

        assert result.receipt is not None
        assert result.receipt.tool_version != ""
        assert result.receipt.config_hash != ""
        assert result.receipt.mode == "tags"

    def test_receipt_config_hash_includes_mode(self) -> None:
        """Config hash differs between modes."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner

        runner_tags = SemanticBenchmarkRunner("semantic", "ami", "test", "tags")
        runner_summary = SemanticBenchmarkRunner("semantic", "ami", "test", "summary")

        receipt_tags = runner_tags._generate_receipt()
        receipt_summary = runner_summary._generate_receipt()

        # Config hashes should be different because mode is included
        assert receipt_tags.config_hash != receipt_summary.config_hash
        assert receipt_tags.mode == "tags"
        assert receipt_summary.mode == "summary"

    def test_receipt_includes_git_commit(self) -> None:
        """Receipt includes git commit if in a git repo."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner

        runner = SemanticBenchmarkRunner("semantic", "ami", "test", "tags")
        receipt = runner._generate_receipt()

        # In a git repo, git_commit should be a short hash
        # (this test assumes we're running in a git repo)
        if receipt.git_commit is not None:
            # Should be a short hash (7-12 chars)
            assert 7 <= len(receipt.git_commit) <= 12

    def test_receipt_serializes_to_dict(self) -> None:
        """BenchmarkResult.to_dict() includes receipt."""
        from unittest.mock import MagicMock

        from transcription.benchmark_cli import SemanticBenchmarkRunner

        runner = SemanticBenchmarkRunner("semantic", "ami", "test", "tags")
        runner.get_samples = MagicMock(return_value=[])

        result = runner.run(limit=0)
        result_dict = result.to_dict()

        assert "receipt" in result_dict
        assert result_dict["receipt"]["tool_version"] != ""
        assert result_dict["receipt"]["config_hash"] != ""
        assert result_dict["receipt"]["mode"] == "tags"


class TestTranscriptLoading:
    """Tests for transcript JSON loading from speaker_analytics_samples."""

    def test_loads_transcript_from_samples_dir(self) -> None:
        """_load_transcript_json loads from speaker_analytics_samples directory."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner

        runner = SemanticBenchmarkRunner("semantic", "ami", "test", "tags")

        # project_retro is a sample that should exist
        transcript = runner._load_transcript_json("project_retro")

        # If samples dir exists, should load
        if runner._samples_dir.exists():
            assert transcript is not None
            assert len(transcript.segments) > 0
        else:
            # Samples not available - that's OK
            assert transcript is None

    def test_returns_none_for_missing_transcript(self) -> None:
        """_load_transcript_json returns None for missing files."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner

        runner = SemanticBenchmarkRunner("semantic", "ami", "test", "tags")

        result = runner._load_transcript_json("nonexistent_sample_xyz123")

        assert result is None

    def test_evaluate_tags_uses_real_transcript_when_available(self) -> None:
        """_evaluate_tags uses loaded transcript with proper segments."""
        from transcription.benchmark_cli import SemanticBenchmarkRunner
        from transcription.benchmarks import EvalSample

        runner = SemanticBenchmarkRunner("semantic", "ami", "test", "tags")

        # Check if project_retro sample files exist
        gold = runner._load_gold_labels("project_retro")
        transcript = runner._load_transcript_json("project_retro")

        if gold is None or transcript is None:
            # Skip if test data not available
            return

        sample = EvalSample(
            dataset="ami",
            id="project_retro",
            audio_path=Path("/fake/path.wav"),
            reference_transcript="Fallback text that should not be used",
        )

        result = runner._evaluate_tags(sample, gold)

        # Should have evaluated with real transcript (no synthetic segments note)
        # Note: This may still have synthetic_transcript_segments note if the
        # transcript loading failed for some reason
        assert result.get("tags_reason") is None  # Evaluation succeeded
