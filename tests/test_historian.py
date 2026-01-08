"""
Tests for the historian PR analysis pipeline.

This module tests:
- TemporalAnalyzer: Deterministic temporal topology analysis
- Synthesis: Merge functions and pipeline reporting
- Estimation: Fallback decision candidates
- Integration: Full pipeline flow
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from transcription.historian.analyzers.temporal import (
    Hotspot,
    InflectionPoint,
    Oscillation,
    Phase,
    TemporalAnalyzer,
)
from transcription.historian.bundle import (
    CheckRunData,
    CommitData,
    FactBundle,
    PRMetadata,
    ScopeData,
    SessionData,
)
from transcription.historian.estimation import (
    DecisionEvent,
    compute_control_plane_devlt,
    generate_fallback_decision_candidates,
)
from transcription.historian.synthesis import (
    PipelineReport,
    _merge_temporal,
    finalize_costs,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def make_commit(
    sha: str,
    message: str,
    offset_hours: int = 0,
    is_fix: bool = False,
    is_refactor: bool = False,
    is_test: bool = False,
    is_doc: bool = False,
    is_revert: bool = False,
    is_wip: bool = False,
) -> CommitData:
    """Helper to create CommitData for tests."""
    base_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    return CommitData(
        sha=sha,
        message=message,
        author="test-author",
        committed_at=base_time + timedelta(hours=offset_hours),
        is_fix=is_fix,
        is_refactor=is_refactor,
        is_test=is_test,
        is_doc=is_doc,
        is_revert=is_revert,
        is_wip=is_wip,
    )


def make_check_run(
    name: str,
    conclusion: str = "success",
    offset_hours: int = 0,
) -> CheckRunData:
    """Helper to create CheckRunData for tests."""
    base_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    return CheckRunData(
        name=name,
        status="completed",
        conclusion=conclusion,
        started_at=base_time + timedelta(hours=offset_hours),
        completed_at=base_time + timedelta(hours=offset_hours, minutes=5),
    )


def make_key_file(path: str, additions: int = 10, deletions: int = 5) -> dict[str, Any]:
    """Helper to create key file dict for tests."""
    return {
        "path": path,
        "additions": additions,
        "deletions": deletions,
        "total_changes": additions + deletions,
    }


def make_bundle(
    commits: list[CommitData] | None = None,
    check_runs: list[CheckRunData] | None = None,
    key_files: list[dict[str, Any]] | None = None,
    blast_radius: str = "medium",
) -> FactBundle:
    """Helper to create minimal FactBundle for tests."""
    base_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)

    if commits is None:
        commits = [
            make_commit("abc1234", "feat: Initial feature implementation"),
            make_commit("def5678", "fix: Fix bug in feature", offset_hours=1, is_fix=True),
            make_commit("ghi9012", "test: Add tests", offset_hours=2, is_test=True),
        ]

    if check_runs is None:
        check_runs = [
            make_check_run("ruff", "success"),
            make_check_run("pytest", "success"),
        ]

    if key_files is None:
        key_files = [
            make_key_file("src/feature.py", 100, 20),
            make_key_file("tests/test_feature.py", 50, 0),
        ]

    return FactBundle(
        pr_number=138,
        metadata=PRMetadata(
            number=138,
            title="Test PR",
            url="https://github.com/test/repo/pull/138",
            state="merged",
            author="test-author",
            created_at=base_time,
            merged_at=base_time + timedelta(hours=3),
            closed_at=base_time + timedelta(hours=3),
            merged_by="reviewer",
            base_branch="main",
            head_branch="feature-branch",
            labels=["enhancement"],
            body="Test PR body",
        ),
        scope=ScopeData(
            files_changed=len(key_files),
            insertions=sum(f["additions"] for f in key_files),
            deletions=sum(f["deletions"] for f in key_files),
            top_directories=["src", "tests"],
            key_files=key_files,
            file_categories={
                "source": ["src/feature.py"],
                "tests": ["tests/test_feature.py"],
            },
            blast_radius=blast_radius,
        ),
        commits=commits,
        sessions=[
            SessionData(
                session_id=1,
                start=base_time,
                end=base_time + timedelta(hours=2),
                commits=[c.sha for c in commits],
                comments_in_window=[],
                reviews_in_window=[],
                is_author_session=True,
            )
        ],
        comments=[],
        reviews=[],
        check_runs=check_runs,
        receipt_paths=[],
        diff=None,
    )


# =============================================================================
# TemporalAnalyzer Tests
# =============================================================================


class TestTemporalAnalyzer:
    """Tests for TemporalAnalyzer deterministic analysis."""

    def test_analyzer_spec(self) -> None:
        """Test analyzer spec is correctly defined."""
        analyzer = TemporalAnalyzer()
        spec = analyzer.spec

        assert spec.name == "Temporal"
        assert "deterministic" in spec.description.lower()
        assert spec.required_bundle_fields == ["commits", "check_runs", "scope"]

    def test_single_commit_analysis(self) -> None:
        """Test analysis with single commit."""
        bundle = make_bundle(
            commits=[make_commit("abc1234", "feat: Add new feature")],
        )

        analyzer = TemporalAnalyzer()
        output = analyzer._analyze(bundle)

        assert output["method_id"] == "temporal-v1.1"
        assert output["confidence"] == "low"  # Single commit = low confidence
        assert len(output["phases"]) == 1
        assert output["phases"][0]["name"] == "exploration"

    def test_multi_commit_phases(self) -> None:
        """Test phase detection with multiple commits."""
        commits = [
            make_commit("c1", "feat: Start feature", offset_hours=0),
            make_commit("c2", "fix: Fix issue", offset_hours=1, is_fix=True),
            make_commit("c3", "wip: Work in progress", offset_hours=2, is_wip=True),
            make_commit("c4", "test: Add tests", offset_hours=3, is_test=True),
            make_commit("c5", "docs: Update README", offset_hours=4, is_doc=True),
        ]
        # Need 3+ files for high confidence (5+ commits AND 3+ files AND 1+ phases)
        key_files = [
            make_key_file("src/feature.py", 100, 20),
            make_key_file("tests/test_feature.py", 50, 0),
            make_key_file("docs/README.md", 30, 10),
        ]
        bundle = make_bundle(commits=commits, key_files=key_files)

        analyzer = TemporalAnalyzer()
        output = analyzer._analyze(bundle)

        assert output["confidence"] == "high"  # 5 commits + 3 files
        assert output["convergence_type"] in ["linear", "cyclical", "chaotic"]
        assert len(output["phases"]) >= 1

    def test_oscillation_detection_revert(self) -> None:
        """Test oscillation detection for revert patterns."""
        commits = [
            make_commit("c1", "feat: Add feature"),
            make_commit("c2", "revert: Revert add feature", offset_hours=1, is_revert=True),
            make_commit("c3", "feat: Re-add feature differently", offset_hours=2),
        ]
        bundle = make_bundle(commits=commits)

        analyzer = TemporalAnalyzer()
        output = analyzer._analyze(bundle)

        # Should detect revert_pattern oscillation
        osc_types = [o["type"] for o in output["oscillations"]]
        assert "revert_pattern" in osc_types

    def test_oscillation_detection_approach_flip(self) -> None:
        """Test oscillation detection for refactor patterns."""
        commits = [
            make_commit("c1", "feat: Initial implementation"),
            make_commit("c2", "refactor: Rewrite implementation", offset_hours=1, is_refactor=True),
            make_commit("c3", "fix: Fix refactor issue", offset_hours=2, is_fix=True),
            make_commit("c4", "fix: Another fix", offset_hours=3, is_fix=True),
        ]
        bundle = make_bundle(commits=commits)

        analyzer = TemporalAnalyzer()
        output = analyzer._analyze(bundle)

        osc_types = [o["type"] for o in output["oscillations"]]
        assert "approach_flip" in osc_types

    def test_hotspot_computation(self) -> None:
        """Test hotspot file detection."""
        key_files = [
            make_key_file("src/hot_file.py", 200, 100),  # High churn
            make_key_file("src/cold_file.py", 5, 2),  # Low churn
        ]
        bundle = make_bundle(key_files=key_files)

        analyzer = TemporalAnalyzer()
        output = analyzer._analyze(bundle)

        assert len(output["hotspots"]) > 0
        # Hotspots sorted by churn - hot_file should be first
        assert output["hotspots"][0]["file"] == "src/hot_file.py"
        assert output["hotspots"][0]["churn_sum"] == 300

    def test_inflection_point_detection(self) -> None:
        """Test inflection point (lock-in commit) detection."""
        commits = [
            make_commit("c1", "feat: Add feature"),
            make_commit("c2", "fix: Fix bug", offset_hours=1, is_fix=True),
            make_commit("c3", "test: Add tests", offset_hours=2, is_test=True),
            make_commit("c4", "docs: Update docs", offset_hours=3, is_doc=True),
        ]
        bundle = make_bundle(commits=commits)

        analyzer = TemporalAnalyzer()
        output = analyzer._analyze(bundle)

        assert output["inflection_point"] is not None
        assert output["inflection_point"]["commit"] is not None
        assert output["inflection_point"]["rationale"] is not None

    def test_convergence_classification(self) -> None:
        """Test convergence type classification."""
        # Linear: simple progression
        linear_commits = [
            make_commit("c1", "feat: Add feature"),
            make_commit("c2", "test: Add tests", offset_hours=1, is_test=True),
        ]
        bundle = make_bundle(commits=linear_commits)
        analyzer = TemporalAnalyzer()
        output = analyzer._analyze(bundle)
        assert output["convergence_type"] == "linear"

    def test_empty_commits(self) -> None:
        """Test handling of empty commit list."""
        # Pass empty key_files too since hotspots come from key_files
        bundle = make_bundle(commits=[], key_files=[])

        analyzer = TemporalAnalyzer()
        output = analyzer._analyze(bundle)

        assert output["phases"] == []
        assert output["hotspots"] == []
        assert output["inflection_point"] is None


# =============================================================================
# Estimation Tests
# =============================================================================


class TestEstimation:
    """Tests for estimation and fallback decision candidates."""

    def test_control_plane_devlt_computation(self) -> None:
        """Test control-plane DevLT computation from decision events."""
        events = [
            DecisionEvent(
                type="design",
                description="Schema change",
                anchor="commit:abc1234",
                confidence="high",
                minutes_lb=6,
                minutes_ub=20,
            ),
            DecisionEvent(
                type="debug",
                description="Fix bug",
                anchor="commit:def5678",
                confidence="medium",
                minutes_lb=8,
                minutes_ub=30,
            ),
        ]

        result = compute_control_plane_devlt(events, "github_only")

        assert result.lb_minutes == 14  # 6 + 8
        assert result.ub_minutes == 50  # 20 + 30
        assert result.decision_count == 2
        assert result.method == "decision-weighted-v1"
        assert result.coverage == "github_only"

    def test_fallback_candidates_blast_radius(self) -> None:
        """Test fallback candidates include blast radius context."""
        bundle = make_bundle(blast_radius="high")
        candidates = generate_fallback_decision_candidates(bundle)

        scope_candidates = [c for c in candidates if c.type == "scope"]
        assert len(scope_candidates) >= 1
        assert "high" in scope_candidates[0].description.lower()

    def test_fallback_candidates_check_failures(self) -> None:
        """Test fallback candidates include check failure decisions."""
        check_runs = [
            make_check_run("pytest", "failure"),
            make_check_run("ruff", "failure"),
        ]
        bundle = make_bundle(check_runs=check_runs)
        candidates = generate_fallback_decision_candidates(bundle)

        # Should have debug (test) and quality (lint) decisions
        types = [c.type for c in candidates]
        assert "debug" in types or "quality" in types

    def test_fallback_candidates_fix_loop(self) -> None:
        """Test fallback candidates detect fix-loop patterns."""
        commits = [
            make_commit("c1", "feat: Add feature"),
            make_commit("c2", "fix: Fix issue 1", offset_hours=1, is_fix=True),
            make_commit("c3", "fix: Fix issue 2", offset_hours=2, is_fix=True),
            make_commit("c4", "fix: Fix issue 3", offset_hours=3, is_fix=True),
        ]
        bundle = make_bundle(commits=commits)
        candidates = generate_fallback_decision_candidates(bundle)

        # Should detect fix-loop pattern
        debug_candidates = [c for c in candidates if c.type == "debug"]
        fix_loop_found = any("fix" in c.description.lower() for c in debug_candidates)
        assert fix_loop_found

    def test_fallback_candidates_temporal_oscillations(self) -> None:
        """Test fallback candidates incorporate temporal oscillations."""
        bundle = make_bundle()
        temporal_output = {
            "oscillations": [
                {
                    "type": "approach_flip",
                    "files": ["src/feature.py"],
                    "commits": ["c1", "c2"],
                    "evidence": "Detected refactor pattern",
                }
            ]
        }
        candidates = generate_fallback_decision_candidates(bundle, temporal_output)

        # Should include oscillation-based decisions
        assert len(candidates) >= 2  # At least scope + oscillation

    def test_fallback_candidates_contract_files(self) -> None:
        """Test fallback candidates detect contract/schema file touches."""
        key_files = [
            make_key_file("transcription/schemas/pr-dossier-v2.schema.json", 50, 10),
            make_key_file("CHANGELOG.md", 20, 0),
        ]
        bundle = make_bundle(key_files=key_files)
        candidates = generate_fallback_decision_candidates(bundle)

        # Should detect contract touches
        types = [c.type for c in candidates]
        assert "design" in types or "publish" in types


# =============================================================================
# Synthesis Tests
# =============================================================================


class TestSynthesis:
    """Tests for dossier synthesis and merge functions."""

    def test_merge_temporal_field_mapping(self) -> None:
        """Test _merge_temporal correctly maps TemporalAnalyzer output fields."""
        dossier: dict[str, Any] = {}
        temporal_output = {
            "method_id": "temporal-v1.1",
            "confidence": "high",
            "convergence_type": "linear",
            "phases": [
                {
                    "name": "exploration",
                    "start_commit": "abc1234",
                    "end_commit": "def5678",
                    "evidence": "Test evidence",
                    "session_type": "build-loop",
                }
            ],
            "hotspots": [
                {
                    "file": "src/feature.py",  # TemporalAnalyzer outputs "file"
                    "touch_count": 5,
                    "churn_sum": 150,
                    "is_oscillating": False,
                }
            ],
            "oscillations": [
                {
                    "type": "approach_flip",
                    "files": ["src/feature.py"],
                    "commits": ["abc1234"],
                    "evidence": "Test oscillation",
                    "keywords": ["refactor"],
                }
            ],
            "inflection_point": {
                "commit": "ghi9012",  # TemporalAnalyzer outputs "commit"
                "rationale": "Last commit touching core logic",
            },
            "notes": "Test notes",
        }

        _merge_temporal(dossier, temporal_output)

        temporal = dossier["process"]["temporal"]

        # Verify field mappings
        assert temporal["method_id"] == "temporal-v1.1"
        assert temporal["confidence"] == "high"
        assert temporal["convergence_type"] == "linear"

        # Verify phase mapping includes session_type
        assert temporal["phases"][0]["session_type"] == "build-loop"

        # Verify hotspot field mapping (file, not path)
        assert temporal["hotspots"][0]["file"] == "src/feature.py"

        # Verify oscillation mapping includes evidence and keywords
        assert temporal["oscillations"][0]["evidence"] == "Test oscillation"
        assert temporal["oscillations"][0]["keywords"] == ["refactor"]

        # Verify inflection_point mapping (commit, not commit_sha)
        assert temporal["inflection_point"]["commit"] == "ghi9012"

        # Verify notes
        assert temporal["notes"] == "Test notes"

    def test_merge_temporal_skipped(self) -> None:
        """Test _merge_temporal handles skipped output."""
        dossier: dict[str, Any] = {}
        _merge_temporal(dossier, {"skipped": True})
        assert "process" not in dossier

    def test_merge_temporal_null_inflection(self) -> None:
        """Test _merge_temporal handles null inflection point."""
        dossier: dict[str, Any] = {}
        temporal_output = {
            "convergence_type": "linear",
            "phases": [],
            "hotspots": [],
            "oscillations": [],
            "inflection_point": None,
        }

        _merge_temporal(dossier, temporal_output)

        temporal = dossier["process"]["temporal"]
        assert temporal["inflection_point"] is None

    def test_finalize_costs_with_decisions(self) -> None:
        """Test finalize_costs computes control-plane DevLT."""
        dossier: dict[str, Any] = {
            "decision_events": [
                {
                    "type": "design",
                    "description": "Schema change",
                    "anchor": "commit:abc",
                    "confidence": "high",
                    "minutes_lb": 6,
                    "minutes_ub": 20,
                }
            ],
            "inputs": {"coverage": "github_only"},
            "evidence": {},
        }

        finalize_costs(dossier)

        assert "cost" in dossier
        assert "devlt" in dossier["cost"]
        assert "control_plane" in dossier["cost"]["devlt"]
        assert dossier["cost"]["devlt"]["control_plane"]["decision_count"] == 1

    def test_finalize_costs_without_decisions(self) -> None:
        """Test finalize_costs handles empty decision events."""
        dossier: dict[str, Any] = {
            "decision_events": [],
            "inputs": {"coverage": "github_only"},
            "evidence": {},
            "cost": {"devlt": {}},
        }

        finalize_costs(dossier)

        # Should set no-decision-events method
        control_plane = dossier["cost"]["devlt"]["control_plane"]
        assert control_plane["method"] == "no-decision-events"
        assert control_plane["decision_count"] == 0


# =============================================================================
# Pipeline Report Tests
# =============================================================================


class TestPipelineReport:
    """Tests for pipeline report generation."""

    def test_pipeline_report_to_dict(self) -> None:
        """Test PipelineReport serialization."""
        from transcription.historian.synthesis import AnalyzerStatus

        report = PipelineReport(
            analyzers=[
                AnalyzerStatus(
                    name="Temporal",
                    attempted=True,
                    succeeded=True,
                    merged=True,
                    duration_ms=50,
                ),
                AnalyzerStatus(
                    name="DiffScout",
                    attempted=True,
                    succeeded=False,
                    merged=False,
                    error="LLM error",
                ),
            ],
            total_attempted=2,
            total_succeeded=1,
            total_merged=1,
            coverage="partial",
            notes=["Used fallback for DevLT"],
        )

        result = report.to_dict()

        assert result["total_attempted"] == 2
        assert result["total_succeeded"] == 1
        assert result["total_merged"] == 1
        assert result["coverage"] == "partial"
        assert "Used fallback" in result["notes"][0]
        assert result["analyzers"][0]["name"] == "Temporal"
        assert result["analyzers"][1]["error"] == "LLM error"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full pipeline flow."""

    @pytest.mark.asyncio
    async def test_temporal_analyzer_full_run(self) -> None:
        """Test TemporalAnalyzer.run() method end-to-end."""
        from unittest.mock import MagicMock

        bundle = make_bundle()
        analyzer = TemporalAnalyzer()

        # TemporalAnalyzer doesn't use LLM, so mock is unused
        mock_llm = MagicMock()

        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True
        assert result.name == "Temporal"
        assert result.output is not None
        assert "phases" in result.output
        assert "hotspots" in result.output
        assert "oscillations" in result.output
        assert result.duration_ms >= 0

    def test_decision_event_dataclass(self) -> None:
        """Test DecisionEvent dataclass to_dict."""
        event = DecisionEvent(
            type="design",
            description="Test decision",
            anchor="commit:abc",
            confidence="high",
            minutes_lb=6,
            minutes_ub=20,
        )

        result = event.to_dict()

        assert result["type"] == "design"
        assert result["minutes_lb"] == 6
        assert result["minutes_ub"] == 20

    def test_temporal_dataclasses(self) -> None:
        """Test temporal analysis dataclasses."""
        phase = Phase(
            name="exploration",
            start_commit="abc",
            end_commit="def",
            evidence="Test",
            session_type="build-loop",
        )
        assert phase.name == "exploration"

        hotspot = Hotspot(
            file="src/test.py",
            touch_count=5,
            churn_sum=100,
            is_oscillating=True,
        )
        assert hotspot.is_oscillating is True

        oscillation = Oscillation(
            type="approach_flip",
            files=["a.py"],
            commits=["abc"],
            evidence="test",
            keywords=["refactor"],
        )
        assert oscillation.type == "approach_flip"

        inflection = InflectionPoint(
            commit="abc123",
            rationale="Test rationale",
        )
        assert inflection.commit == "abc123"
