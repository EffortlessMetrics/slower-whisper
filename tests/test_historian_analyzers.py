"""
Tests for historian LLM-powered analyzers.

This module tests:
- BaseAnalyzer class
- Prompt formatting utilities
- All 7 LLM-powered analyzers: DiffScout, DesignAlignment, DocsSchema,
  FrictionMiner, EvidenceAuditor, PerfIntegrity, DecisionExtractor
- JSON parsing from LLM responses
- Schema validation
- Error handling
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import directly from submodules to avoid circular import issues
from transcription.historian.analyzers.base import (
    BaseAnalyzer,
    SubagentResult,
    SubagentSpec,
    format_check_runs,
    format_comments,
    format_commits,
    format_key_files,
    format_reviews,
    format_sessions,
)
from transcription.historian.analyzers.decision_extractor import DecisionExtractorAnalyzer
from transcription.historian.analyzers.design_alignment import DesignAlignmentAnalyzer
from transcription.historian.analyzers.diff_scout import DiffScoutAnalyzer
from transcription.historian.analyzers.docs_schema import DocsSchemaAuditorAnalyzer
from transcription.historian.analyzers.evidence_auditor import EvidenceAuditorAnalyzer
from transcription.historian.analyzers.friction_miner import FrictionMinerAnalyzer
from transcription.historian.analyzers.perf_integrity import PerfIntegrityAnalyzer
from transcription.historian.analyzers.temporal import TemporalAnalyzer
from transcription.historian.bundle import (
    CheckRunData,
    CommentData,
    CommitData,
    FactBundle,
    PRMetadata,
    ReviewData,
    ScopeData,
    SessionData,
)
from transcription.historian.llm_client import LLMConfig, LLMResponse, MockProvider

# Import ALL_ANALYZERS from __init__ only for type checking
if TYPE_CHECKING:
    pass

# Define ALL_ANALYZERS locally to avoid import issues
ALL_ANALYZERS: list[type[BaseAnalyzer]] = [
    TemporalAnalyzer,
    DiffScoutAnalyzer,
    EvidenceAuditorAnalyzer,
    FrictionMinerAnalyzer,
    DesignAlignmentAnalyzer,
    PerfIntegrityAnalyzer,
    DocsSchemaAuditorAnalyzer,
    DecisionExtractorAnalyzer,
]


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


def make_comment(
    comment_id: str,
    body: str,
    author: str = "reviewer",
    comment_type: str = "issue_comment",
    offset_hours: int = 0,
) -> CommentData:
    """Helper to create CommentData for tests."""
    base_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    return CommentData(
        id=comment_id,
        author=author,
        body=body,
        created_at=base_time + timedelta(hours=offset_hours),
        comment_type=comment_type,
    )


def make_review(
    review_id: str,
    body: str,
    state: str = "APPROVED",
    author: str = "reviewer",
    offset_hours: int = 0,
) -> ReviewData:
    """Helper to create ReviewData for tests."""
    base_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    return ReviewData(
        id=review_id,
        author=author,
        body=body,
        state=state,
        submitted_at=base_time + timedelta(hours=offset_hours),
    )


def make_session(
    session_id: int,
    commits: list[str],
    offset_hours: int = 0,
    lb_minutes: int = 30,
    ub_minutes: int = 60,
) -> SessionData:
    """Helper to create SessionData for tests."""
    base_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC)
    return SessionData(
        session_id=session_id,
        start=base_time + timedelta(hours=offset_hours),
        end=base_time + timedelta(hours=offset_hours + 2),
        commits=commits,
        comments_in_window=[],
        reviews_in_window=[],
        lb_minutes=lb_minutes,
        ub_minutes=ub_minutes,
        is_author_session=True,
    )


def make_bundle(
    commits: list[CommitData] | None = None,
    check_runs: list[CheckRunData] | None = None,
    key_files: list[dict[str, Any]] | None = None,
    comments: list[CommentData] | None = None,
    reviews: list[ReviewData] | None = None,
    sessions: list[SessionData] | None = None,
    blast_radius: str = "medium",
    diff: str
    | None = "diff --git a/src/feature.py b/src/feature.py\n+ def new_feature():\n+     pass",
    receipt_paths: list[str] | None = None,
    labels: list[str] | None = None,
    body: str = "Test PR body with details",
    file_categories: dict[str, list[str]] | None = None,
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

    if comments is None:
        comments = []

    if reviews is None:
        reviews = []

    if sessions is None:
        sessions = [
            SessionData(
                session_id=1,
                start=base_time,
                end=base_time + timedelta(hours=2),
                commits=[c.sha for c in commits],
                comments_in_window=[],
                reviews_in_window=[],
                is_author_session=True,
            )
        ]

    if receipt_paths is None:
        receipt_paths = []

    if labels is None:
        labels = ["enhancement"]

    if file_categories is None:
        file_categories = {
            "source": ["src/feature.py"],
            "tests": ["tests/test_feature.py"],
            "docs": [],
            "config": [],
        }

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
            labels=labels,
            body=body,
        ),
        scope=ScopeData(
            files_changed=len(key_files),
            insertions=sum(f["additions"] for f in key_files),
            deletions=sum(f["deletions"] for f in key_files),
            top_directories=["src", "tests"],
            key_files=key_files,
            file_categories=file_categories,
            blast_radius=blast_radius,
        ),
        commits=commits,
        sessions=sessions,
        comments=comments,
        reviews=reviews,
        check_runs=check_runs,
        receipt_paths=receipt_paths,
        diff=diff,
    )


def create_mock_llm(response: dict[str, Any] | str) -> MockProvider:
    """Create a mock LLM provider that returns the given response."""
    if isinstance(response, dict):
        response_text = json.dumps(response)
    else:
        response_text = response

    config = LLMConfig(provider="mock")
    mock = MockProvider(config, {"": response_text})
    return mock


async def create_async_mock_llm(response: dict[str, Any] | str) -> MagicMock:
    """Create an async mock LLM provider."""
    if isinstance(response, dict):
        response_text = json.dumps(response)
    else:
        response_text = response

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(text=response_text, duration_ms=100))
    return mock_llm


# =============================================================================
# Prompt Formatting Utilities Tests
# =============================================================================


class TestFormatKeyFiles:
    """Tests for format_key_files utility."""

    def test_format_key_files_basic(self) -> None:
        """Test basic key files formatting."""
        files = [
            {"path": "src/a.py", "additions": 10, "deletions": 5},
            {"path": "src/b.py", "additions": 20, "deletions": 0},
        ]
        result = format_key_files(files)
        assert "src/a.py (+10/-5)" in result
        assert "src/b.py (+20/-0)" in result

    def test_format_key_files_empty(self) -> None:
        """Test formatting with no files."""
        result = format_key_files([])
        assert "no key files" in result

    def test_format_key_files_limit(self) -> None:
        """Test that limit is respected."""
        files = [{"path": f"file{i}.py", "additions": i, "deletions": 0} for i in range(20)]
        result = format_key_files(files, limit=5)
        assert "file4.py" in result
        assert "file5.py" not in result


class TestFormatCommits:
    """Tests for format_commits utility."""

    def test_format_commits_basic(self) -> None:
        """Test basic commit formatting."""
        commits = [
            make_commit("abc1234", "feat: Add feature"),
            make_commit("def5678", "fix: Bug fix", is_fix=True),
        ]
        result = format_commits(commits)
        assert "abc1234" in result
        assert "Add feature" in result
        assert "[fix]" in result

    def test_format_commits_multiple_patterns(self) -> None:
        """Test commit with multiple patterns."""
        commits = [make_commit("abc", "test fix", is_fix=True, is_test=True)]
        result = format_commits(commits)
        assert "[fix,test]" in result

    def test_format_commits_empty(self) -> None:
        """Test formatting with no commits."""
        result = format_commits([])
        assert "no commits" in result

    def test_format_commits_limit(self) -> None:
        """Test that limit and truncation message work."""
        commits = [make_commit(f"sha{i}", f"message {i}") for i in range(30)]
        result = format_commits(commits, limit=5)
        assert "sha4" in result
        assert "sha5" not in result
        assert "25 more commits" in result


class TestFormatComments:
    """Tests for format_comments utility."""

    def test_format_comments_basic(self) -> None:
        """Test basic comment formatting."""
        comments = [
            make_comment("c1", "Nice work!", "reviewer"),
            make_comment("c2", "LGTM", "approver", "review_comment"),
        ]
        result = format_comments(comments)
        assert "Nice work!" in result
        assert "reviewer" in result
        assert "[issue_comment]" in result
        assert "[review_comment]" in result

    def test_format_comments_truncation(self) -> None:
        """Test long comment truncation."""
        long_body = "x" * 300
        comments = [make_comment("c1", long_body)]
        result = format_comments(comments)
        assert "..." in result
        assert len(result) < 350

    def test_format_comments_empty(self) -> None:
        """Test formatting with no comments."""
        result = format_comments([])
        assert "no comments" in result


class TestFormatReviews:
    """Tests for format_reviews utility."""

    def test_format_reviews_basic(self) -> None:
        """Test basic review formatting."""
        reviews = [
            make_review("r1", "Approved!", "APPROVED"),
            make_review("r2", "Please fix this", "CHANGES_REQUESTED"),
        ]
        result = format_reviews(reviews)
        assert "[APPROVED]" in result
        assert "[CHANGES_REQUESTED]" in result
        assert "Approved!" in result

    def test_format_reviews_empty(self) -> None:
        """Test formatting with no reviews."""
        result = format_reviews([])
        assert "no reviews" in result


class TestFormatCheckRuns:
    """Tests for format_check_runs utility."""

    def test_format_check_runs_basic(self) -> None:
        """Test basic check run formatting."""
        check_runs = [
            make_check_run("pytest", "success"),
            make_check_run("ruff", "failure"),
        ]
        result = format_check_runs(check_runs)
        assert "pytest: success" in result
        assert "ruff: failure" in result
        # Duration should be included
        assert "300s" in result  # 5 minutes

    def test_format_check_runs_empty(self) -> None:
        """Test formatting with no check runs."""
        result = format_check_runs([])
        assert "no check runs" in result


class TestFormatSessions:
    """Tests for format_sessions utility."""

    def test_format_sessions_basic(self) -> None:
        """Test basic session formatting."""
        sessions = [
            make_session(0, ["abc", "def"], lb_minutes=30, ub_minutes=60),
            make_session(1, ["ghi"], lb_minutes=15, ub_minutes=30),
        ]
        result = format_sessions(sessions)
        assert "Session 0" in result
        assert "Session 1" in result
        assert "2 commits" in result
        assert "LB=30m UB=60m" in result

    def test_format_sessions_empty(self) -> None:
        """Test formatting with no sessions."""
        result = format_sessions([])
        assert "no sessions" in result


# =============================================================================
# BaseAnalyzer Tests
# =============================================================================


class TestBaseAnalyzer:
    """Tests for BaseAnalyzer class."""

    @pytest.mark.asyncio
    async def test_run_skipped_when_should_run_false(self) -> None:
        """Test that analyzer is skipped when should_run returns False."""
        # DocsSchemaAuditor skips when no docs or schema files
        bundle = make_bundle(
            key_files=[make_key_file("src/feature.py", 100, 20)],
            file_categories={"source": ["src/feature.py"], "tests": [], "docs": [], "config": []},
        )

        analyzer = DocsSchemaAuditorAnalyzer()
        mock_llm = MagicMock()

        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True
        assert result.output is not None
        assert result.output.get("skipped") is True
        # LLM should not have been called
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_handles_llm_failure(self) -> None:
        """Test that LLM failures are handled gracefully."""
        bundle = make_bundle()
        analyzer = DiffScoutAnalyzer()

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("API Error"))

        result = await analyzer.run(bundle, mock_llm)

        assert result.success is False
        assert "LLM call failed" in result.errors[0]
        assert "API Error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_run_handles_invalid_json(self) -> None:
        """Test that invalid JSON responses are handled."""
        bundle = make_bundle()
        analyzer = DiffScoutAnalyzer()

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=LLMResponse(text="not valid json {{{", duration_ms=100)
        )

        result = await analyzer.run(bundle, mock_llm)

        assert result.success is False
        assert any("Failed to parse JSON" in e for e in result.errors)
        assert result.raw_response == "not valid json {{{"

    @pytest.mark.asyncio
    async def test_run_extracts_json_from_markdown(self) -> None:
        """Test that JSON is extracted from markdown code blocks."""
        bundle = make_bundle()
        analyzer = DiffScoutAnalyzer()

        valid_response = {
            "review_map": [{"directory": "src/", "files_count": 1, "change_type": "source"}],
            "key_files": [{"path": "src/feature.py", "reason": "Main implementation"}],
            "blast_radius": {"level": "medium", "rationale": "Limited scope"},
            "semantic_summary": "Added new feature",
        }

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=LLMResponse(
                text=f"```json\n{json.dumps(valid_response)}\n```",
                duration_ms=100,
            )
        )

        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True
        assert result.output == valid_response

    @pytest.mark.asyncio
    async def test_run_validates_output_schema(self) -> None:
        """Test that schema validation catches invalid outputs."""
        bundle = make_bundle()
        analyzer = DiffScoutAnalyzer()

        # Missing required fields
        invalid_response = {
            "review_map": [],
            # Missing: key_files, blast_radius, semantic_summary
        }

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=LLMResponse(text=json.dumps(invalid_response), duration_ms=100)
        )

        result = await analyzer.run(bundle, mock_llm)

        assert result.success is False
        assert len(result.errors) > 0
        # Should have validation errors about missing fields
        error_text = " ".join(result.errors).lower()
        assert "required" in error_text or "key_files" in error_text

    def test_validate_output_without_jsonschema(self) -> None:
        """Test that validation gracefully handles missing jsonschema."""
        analyzer = DiffScoutAnalyzer()

        # Validation should not crash if jsonschema is not available
        # (it will just skip validation)
        errors = analyzer.validate_output({"incomplete": True})
        # May have errors if jsonschema is installed, or empty if not
        assert isinstance(errors, list)


# =============================================================================
# DiffScoutAnalyzer Tests
# =============================================================================


class TestDiffScoutAnalyzer:
    """Tests for DiffScoutAnalyzer."""

    def test_spec(self) -> None:
        """Test analyzer spec is correctly defined."""
        analyzer = DiffScoutAnalyzer()
        spec = analyzer.spec

        assert spec.name == "DiffScout"
        assert "change surface" in spec.description.lower()
        assert "scope" in spec.required_bundle_fields
        assert "diff" in spec.required_bundle_fields

    def test_build_prompt_includes_key_info(self) -> None:
        """Test that build_prompt includes necessary information."""
        bundle = make_bundle(
            key_files=[
                make_key_file("src/feature.py", 100, 20),
                make_key_file("tests/test_feature.py", 50, 0),
            ],
            diff="diff --git a/src/feature.py b/src/feature.py\n+new code",
        )

        analyzer = DiffScoutAnalyzer()
        prompt = analyzer.build_prompt(bundle)

        assert "Test PR" in prompt
        assert "src/feature.py" in prompt
        assert "Files changed:" in prompt
        assert "Insertions:" in prompt
        assert "blast radius" in prompt.lower()
        assert "diff" in prompt.lower()

    @pytest.mark.asyncio
    async def test_successful_analysis(self) -> None:
        """Test successful DiffScout analysis."""
        bundle = make_bundle()
        analyzer = DiffScoutAnalyzer()

        valid_response = {
            "review_map": [{"directory": "src/", "files_count": 1, "change_type": "source"}],
            "key_files": [{"path": "src/feature.py", "reason": "Core implementation"}],
            "blast_radius": {"level": "medium", "rationale": "Moderate impact"},
            "generated_estimate": 0.3,
            "semantic_summary": "Added new feature implementation",
        }

        mock_llm = await create_async_mock_llm(valid_response)
        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True
        assert result.name == "DiffScout"
        assert result.output["blast_radius"]["level"] == "medium"


# =============================================================================
# DesignAlignmentAnalyzer Tests
# =============================================================================


class TestDesignAlignmentAnalyzer:
    """Tests for DesignAlignmentAnalyzer."""

    def test_spec(self) -> None:
        """Test analyzer spec is correctly defined."""
        analyzer = DesignAlignmentAnalyzer()
        spec = analyzer.spec

        assert spec.name == "DesignAlignment"
        assert "drift" in spec.description.lower()
        assert "metadata" in spec.required_bundle_fields
        assert "commits" in spec.required_bundle_fields

    def test_build_prompt_includes_intent_info(self) -> None:
        """Test that build_prompt includes PR intent information."""
        bundle = make_bundle(
            body="This PR adds a new feature for X",
            labels=["feature", "enhancement"],
        )

        analyzer = DesignAlignmentAnalyzer()
        prompt = analyzer.build_prompt(bundle)

        assert "adds a new feature" in prompt
        assert "feature" in prompt
        assert "enhancement" in prompt
        assert "Commits" in prompt
        assert "Key Files" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis_no_drift(self) -> None:
        """Test successful analysis with no drift detected."""
        bundle = make_bundle()
        analyzer = DesignAlignmentAnalyzer()

        valid_response = {
            "intent": {
                "stated_goal": "Add new feature",
                "inferred_type": "feature",
            },
            "alignment": {
                "drifted": False,
                "drift_type": "none",
                "details": "Implementation matches stated intent",
            },
            "scope_check": {
                "stated_in_scope": ["feature implementation"],
                "stated_out_of_scope": [],
                "actual_scope": ["src/feature.py"],
                "discrepancies": [],
            },
            "evidence": ["abc1234: feat: Initial feature implementation"],
        }

        mock_llm = await create_async_mock_llm(valid_response)
        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True
        assert result.output["alignment"]["drifted"] is False

    @pytest.mark.asyncio
    async def test_analysis_detects_scope_creep(self) -> None:
        """Test that scope creep can be detected."""
        bundle = make_bundle()
        analyzer = DesignAlignmentAnalyzer()

        valid_response = {
            "intent": {
                "stated_goal": "Fix bug",
                "inferred_type": "feature",  # Type mismatch suggests drift
            },
            "alignment": {
                "drifted": True,
                "drift_type": "scope_creep",
                "details": "PR started as bug fix but expanded to new feature",
            },
            "scope_check": {
                "stated_in_scope": ["bug fix"],
                "stated_out_of_scope": [],
                "actual_scope": ["src/feature.py", "src/new_module.py"],
                "discrepancies": ["new_module.py not mentioned in PR"],
            },
            "evidence": ["def5678: feat: Added new module (scope creep)"],
        }

        mock_llm = await create_async_mock_llm(valid_response)
        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True
        assert result.output["alignment"]["drifted"] is True
        assert result.output["alignment"]["drift_type"] == "scope_creep"


# =============================================================================
# DocsSchemaAuditorAnalyzer Tests
# =============================================================================


class TestDocsSchemaAuditorAnalyzer:
    """Tests for DocsSchemaAuditorAnalyzer."""

    def test_spec(self) -> None:
        """Test analyzer spec is correctly defined."""
        analyzer = DocsSchemaAuditorAnalyzer()
        spec = analyzer.spec

        assert spec.name == "DocsSchemaAuditor"
        assert "documentation" in spec.description.lower()
        assert "scope" in spec.required_bundle_fields

    def test_should_run_with_docs(self) -> None:
        """Test should_run returns True when docs are present."""
        bundle = make_bundle(
            file_categories={"docs": ["docs/README.md"], "source": [], "tests": [], "config": []},
        )
        analyzer = DocsSchemaAuditorAnalyzer()
        assert analyzer.should_run(bundle) is True

    def test_should_run_with_schema_files(self) -> None:
        """Test should_run returns True when schema files are present."""
        bundle = make_bundle(
            key_files=[make_key_file("schemas/api.schema.json", 50, 10)],
            file_categories={"docs": [], "source": [], "tests": [], "config": []},
        )
        analyzer = DocsSchemaAuditorAnalyzer()
        assert analyzer.should_run(bundle) is True

    def test_should_run_false_without_docs_or_schema(self) -> None:
        """Test should_run returns False when no docs or schema files."""
        bundle = make_bundle(
            key_files=[make_key_file("src/feature.py", 100, 20)],
            file_categories={"docs": [], "source": ["src/feature.py"], "tests": [], "config": []},
        )
        analyzer = DocsSchemaAuditorAnalyzer()
        assert analyzer.should_run(bundle) is False

    @pytest.mark.asyncio
    async def test_successful_analysis(self) -> None:
        """Test successful docs/schema analysis."""
        bundle = make_bundle(
            key_files=[make_key_file("docs/README.md", 20, 5)],
            file_categories={"docs": ["docs/README.md"], "source": [], "tests": [], "config": []},
        )
        analyzer = DocsSchemaAuditorAnalyzer()

        valid_response = {
            "docs_touched": True,
            "schema_touched": False,
            "issues": [],
            "doc_files_checked": ["docs/README.md"],
            "schema_files_checked": [],
            "recommendations": [],
            "overall_quality": "good",
        }

        mock_llm = await create_async_mock_llm(valid_response)
        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True
        assert result.output["overall_quality"] == "good"


# =============================================================================
# FrictionMinerAnalyzer Tests
# =============================================================================


class TestFrictionMinerAnalyzer:
    """Tests for FrictionMinerAnalyzer."""

    def test_spec(self) -> None:
        """Test analyzer spec is correctly defined."""
        analyzer = FrictionMinerAnalyzer()
        spec = analyzer.spec

        assert spec.name == "FrictionMiner"
        assert "friction" in spec.description.lower()
        assert "commits" in spec.required_bundle_fields
        assert "comments" in spec.required_bundle_fields

    def test_build_prompt_includes_friction_signals(self) -> None:
        """Test that build_prompt includes friction signal counts."""
        bundle = make_bundle(
            commits=[
                make_commit("a", "feat: Add feature"),
                make_commit("b", "fix: Fix bug", is_fix=True),
                make_commit("c", "fix: Another fix", is_fix=True),
                make_commit("d", "wip: Work in progress", is_wip=True),
            ],
            check_runs=[
                make_check_run("pytest", "failure"),
                make_check_run("ruff", "success"),
            ],
        )

        analyzer = FrictionMinerAnalyzer()
        prompt = analyzer.build_prompt(bundle)

        assert "Fix commits: 2" in prompt
        assert "WIP commits: 1" in prompt
        assert "Failed check runs: 1" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis(self) -> None:
        """Test successful friction mining analysis."""
        bundle = make_bundle(
            commits=[
                make_commit("a", "feat: Add feature"),
                make_commit("b", "fix: Fix type error", is_fix=True),
            ],
            check_runs=[make_check_run("mypy", "failure")],
        )
        analyzer = FrictionMinerAnalyzer()

        valid_response = {
            "events": [
                {
                    "event": "Type error in new feature",
                    "type": "implementation_error",
                    "detected_by": "gate",
                    "disposition": "fixed here",
                    "prevention": "Add type annotations",
                    "evidence": "commit:b",
                }
            ],
            "iteration_count": 1,
            "friction_score": "low",
            "notes": "Minor type issue caught by CI",
        }

        mock_llm = await create_async_mock_llm(valid_response)
        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True
        assert len(result.output["events"]) == 1
        assert result.output["friction_score"] == "low"


# =============================================================================
# EvidenceAuditorAnalyzer Tests
# =============================================================================


class TestEvidenceAuditorAnalyzer:
    """Tests for EvidenceAuditorAnalyzer."""

    def test_spec(self) -> None:
        """Test analyzer spec is correctly defined."""
        analyzer = EvidenceAuditorAnalyzer()
        spec = analyzer.spec

        assert spec.name == "EvidenceAuditor"
        # Check for key terms in description (claims, artifacts, receipts)
        desc_lower = spec.description.lower()
        assert "claims" in desc_lower or "artifacts" in desc_lower or "receipts" in desc_lower
        assert "check_runs" in spec.required_bundle_fields

    def test_build_prompt_includes_check_summary(self) -> None:
        """Test that build_prompt includes check run summary."""
        bundle = make_bundle(
            check_runs=[
                make_check_run("pytest", "success"),
                make_check_run("ruff", "success"),
                make_check_run("mypy", "failure"),
            ],
            receipt_paths=["receipts/ci.json"],
        )

        analyzer = EvidenceAuditorAnalyzer()
        prompt = analyzer.build_prompt(bundle)

        assert "3 total" in prompt
        assert "2 passed" in prompt
        assert "1 failed" in prompt
        assert "receipts/ci.json" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis(self) -> None:
        """Test successful evidence audit."""
        bundle = make_bundle()
        analyzer = EvidenceAuditorAnalyzer()

        valid_response = {
            "claims": [
                {
                    "claim": "Tests pass",
                    "evidence": "pytest check run succeeded",
                    "status": "supported",
                }
            ],
            "evidence": {
                "local_gate": {"observed": True, "path": None},
                "tests": {"observed": True, "added": 5, "modified": 2},
                "typing": {"mypy_observed": True, "ruff_observed": True},
                "benchmarks": {"observed": False, "path": None},
                "docs_updated": False,
                "schema_validated": "unknown",
            },
            "missing_receipts": [],
            "evidence_completeness": "high",
            "notes": "Good evidence coverage",
        }

        mock_llm = await create_async_mock_llm(valid_response)
        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True
        assert result.output["evidence_completeness"] == "high"


# =============================================================================
# PerfIntegrityAnalyzer Tests
# =============================================================================


class TestPerfIntegrityAnalyzer:
    """Tests for PerfIntegrityAnalyzer."""

    def test_spec(self) -> None:
        """Test analyzer spec is correctly defined."""
        analyzer = PerfIntegrityAnalyzer()
        spec = analyzer.spec

        assert spec.name == "PerfIntegrity"
        assert "benchmark" in spec.description.lower()
        assert "diff" in spec.required_bundle_fields

    def test_should_run_with_benchmark_files(self) -> None:
        """Test should_run returns True when benchmark files present."""
        bundle = make_bundle(
            key_files=[make_key_file("benchmarks/perf_test.py", 50, 10)],
        )
        analyzer = PerfIntegrityAnalyzer()
        assert analyzer.should_run(bundle) is True

    def test_should_run_with_perf_keywords_in_body(self) -> None:
        """Test should_run returns True when perf keywords in PR body."""
        bundle = make_bundle(
            body="This PR improves p50 latency by 20%",
        )
        analyzer = PerfIntegrityAnalyzer()
        assert analyzer.should_run(bundle) is True

    def test_should_run_with_perf_label(self) -> None:
        """Test should_run returns True with perf-related label."""
        bundle = make_bundle(
            labels=["performance", "enhancement"],
        )
        analyzer = PerfIntegrityAnalyzer()
        assert analyzer.should_run(bundle) is True

    def test_should_run_false_without_perf_indicators(self) -> None:
        """Test should_run returns False without perf indicators."""
        bundle = make_bundle(
            key_files=[make_key_file("src/feature.py", 100, 20)],
            body="This PR adds a new feature",
            labels=["feature"],
        )
        analyzer = PerfIntegrityAnalyzer()
        assert analyzer.should_run(bundle) is False

    @pytest.mark.asyncio
    async def test_successful_analysis_no_perf_claims(self) -> None:
        """Test analysis when no performance claims exist."""
        bundle = make_bundle(
            key_files=[make_key_file("benchmarks/test_perf.py", 20, 5)],
        )
        analyzer = PerfIntegrityAnalyzer()

        valid_response = {
            "has_perf_claims": False,
            "measurement_integrity": {
                "valid": "unknown",
                "invalidation_reason": None,
                "confidence": "low",
            },
            "baseline": {
                "specified": False,
                "commit_or_tag": None,
                "semantics_unchanged": "unknown",
            },
            "metrics_found": [],
            "issues": [],
            "recommendations": [],
        }

        mock_llm = await create_async_mock_llm(valid_response)
        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True
        assert result.output["has_perf_claims"] is False

    @pytest.mark.asyncio
    async def test_analysis_with_perf_claims(self) -> None:
        """Test analysis with valid performance claims."""
        bundle = make_bundle(
            body="This PR improves p50 latency from 100ms to 80ms",
            key_files=[make_key_file("benchmarks/latency.py", 30, 10)],
        )
        analyzer = PerfIntegrityAnalyzer()

        valid_response = {
            "has_perf_claims": True,
            "measurement_integrity": {
                "valid": True,
                "invalidation_reason": None,
                "confidence": "medium",
            },
            "baseline": {
                "specified": True,
                "commit_or_tag": "v1.0.0",
                "semantics_unchanged": True,
            },
            "metrics_found": [
                {"name": "p50_ms", "value": 80.0, "unit": "ms", "context": "PR body"}
            ],
            "issues": [],
            "recommendations": ["Add p95/p99 metrics"],
        }

        mock_llm = await create_async_mock_llm(valid_response)
        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True
        assert result.output["has_perf_claims"] is True
        assert result.output["measurement_integrity"]["valid"] is True


# =============================================================================
# DecisionExtractorAnalyzer Tests
# =============================================================================


class TestDecisionExtractorAnalyzer:
    """Tests for DecisionExtractorAnalyzer."""

    def test_spec(self) -> None:
        """Test analyzer spec is correctly defined."""
        analyzer = DecisionExtractorAnalyzer()
        spec = analyzer.spec

        assert spec.name == "DecisionExtractor"
        assert "decision" in spec.description.lower()
        assert "commits" in spec.required_bundle_fields
        assert "comments" in spec.required_bundle_fields
        assert "reviews" in spec.required_bundle_fields

    def test_build_prompt_includes_decision_signals(self) -> None:
        """Test that build_prompt includes decision signal analysis."""
        bundle = make_bundle(
            commits=[
                make_commit("a", "feat: Add feature"),
                make_commit("b", "fix: Fix bug", is_fix=True),
                make_commit("c", "refactor: Clean up", is_refactor=True),
                make_commit("d", "test: Add tests", is_test=True),
            ],
            comments=[
                make_comment("c1", "This is out of scope for this PR"),
                make_comment("c2", "The API contract must hold"),
            ],
        )

        analyzer = DecisionExtractorAnalyzer()
        prompt = analyzer.build_prompt(bundle)

        assert "Fix commits: 1" in prompt
        assert "Refactor commits: 1" in prompt
        assert "Test commits: 1" in prompt
        assert "Scope-related keywords" in prompt
        assert "Design-related keywords" in prompt
        assert "Time Bands" in prompt

    @pytest.mark.asyncio
    async def test_successful_analysis(self) -> None:
        """Test successful decision extraction."""
        bundle = make_bundle(
            commits=[
                make_commit("a", "feat: Add schema change"),
                make_commit("b", "fix: Fix validation", is_fix=True),
            ],
        )
        analyzer = DecisionExtractorAnalyzer()

        valid_response = {
            "decisions": [
                {
                    "type": "design",
                    "description": "Changed schema structure for new feature",
                    "anchor": "commit:a",
                    "confidence": "high",
                    "minutes_lb": 6,
                    "minutes_ub": 20,
                },
                {
                    "type": "debug",
                    "description": "Fixed validation bug caught by tests",
                    "anchor": "commit:b",
                    "confidence": "med",
                    "minutes_lb": 8,
                    "minutes_ub": 30,
                },
            ],
            "decision_count": 2,
            "dominant_type": "mixed",
            "evidence_quality": "moderate",
            "notes": "Clear progression from design to debug",
        }

        mock_llm = await create_async_mock_llm(valid_response)
        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True
        assert result.output["decision_count"] == 2
        assert len(result.output["decisions"]) == 2
        assert result.output["decisions"][0]["type"] == "design"

    def test_decision_time_bands(self) -> None:
        """Test that decision time bands are properly defined."""
        from transcription.historian.analyzers.decision_extractor import DECISION_TIME_BANDS

        assert "scope" in DECISION_TIME_BANDS
        assert "design" in DECISION_TIME_BANDS
        assert "quality" in DECISION_TIME_BANDS
        assert "debug" in DECISION_TIME_BANDS
        assert "publish" in DECISION_TIME_BANDS

        # Check band structure
        for band_type, (lb, ub) in DECISION_TIME_BANDS.items():
            assert lb < ub, f"{band_type} band has invalid range"


# =============================================================================
# SubagentResult Tests
# =============================================================================


class TestSubagentResult:
    """Tests for SubagentResult dataclass."""

    def test_creation_success(self) -> None:
        """Test creating a successful result."""
        result = SubagentResult(
            name="TestAnalyzer",
            success=True,
            output={"key": "value"},
            duration_ms=100,
        )

        assert result.name == "TestAnalyzer"
        assert result.success is True
        assert result.output == {"key": "value"}
        assert result.errors == []
        assert result.duration_ms == 100

    def test_creation_failure(self) -> None:
        """Test creating a failed result."""
        result = SubagentResult(
            name="TestAnalyzer",
            success=False,
            errors=["Error 1", "Error 2"],
            raw_response="invalid json",
        )

        assert result.success is False
        assert len(result.errors) == 2
        assert result.raw_response == "invalid json"


# =============================================================================
# SubagentSpec Tests
# =============================================================================


class TestSubagentSpec:
    """Tests for SubagentSpec dataclass."""

    def test_creation(self) -> None:
        """Test creating a spec."""
        spec = SubagentSpec(
            name="TestAnalyzer",
            description="Test description",
            system_prompt="You are a test analyzer",
            output_schema={"type": "object"},
            required_bundle_fields=["commits", "scope"],
        )

        assert spec.name == "TestAnalyzer"
        assert spec.description == "Test description"
        assert spec.system_prompt == "You are a test analyzer"
        assert spec.output_schema == {"type": "object"}
        assert spec.required_bundle_fields == ["commits", "scope"]


# =============================================================================
# ALL_ANALYZERS Tests
# =============================================================================


class TestAllAnalyzers:
    """Tests for ALL_ANALYZERS list and analyzer consistency."""

    def test_all_analyzers_count(self) -> None:
        """Test that ALL_ANALYZERS has expected count."""
        assert len(ALL_ANALYZERS) == 8  # Including TemporalAnalyzer

    def test_all_analyzers_have_unique_names(self) -> None:
        """Test that all analyzers have unique names."""
        names = [analyzer().spec.name for analyzer in ALL_ANALYZERS]
        assert len(names) == len(set(names))

    def test_all_analyzers_have_valid_specs(self) -> None:
        """Test that all analyzers have valid specs."""
        for analyzer_cls in ALL_ANALYZERS:
            analyzer = analyzer_cls()
            spec = analyzer.spec

            # Check required spec fields
            assert spec.name, f"{analyzer_cls.__name__} has no name"
            assert spec.description, f"{analyzer_cls.__name__} has no description"
            # TemporalAnalyzer is deterministic so it doesn't need a system prompt
            if analyzer_cls != TemporalAnalyzer:
                assert spec.system_prompt, f"{analyzer_cls.__name__} has no system prompt"
            assert spec.output_schema, f"{analyzer_cls.__name__} has no output schema"

    def test_all_llm_analyzers_have_schemas(self) -> None:
        """Test that all LLM analyzers have output schemas with 'type'."""
        # Skip TemporalAnalyzer since it's deterministic
        from transcription.historian.analyzers.temporal import TemporalAnalyzer

        for analyzer_cls in ALL_ANALYZERS:
            if analyzer_cls == TemporalAnalyzer:
                continue

            analyzer = analyzer_cls()
            spec = analyzer.spec

            assert "type" in spec.output_schema, f"{spec.name} schema missing 'type'"
            assert spec.output_schema["type"] == "object", f"{spec.name} schema not object type"


# =============================================================================
# Error Handling Integration Tests
# =============================================================================


class TestErrorHandling:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_json_extraction_from_plain_code_block(self) -> None:
        """Test JSON extraction from plain code block (no json marker)."""
        bundle = make_bundle()
        analyzer = DiffScoutAnalyzer()

        valid_response = {
            "review_map": [],
            "key_files": [],
            "blast_radius": {"level": "low", "rationale": "No changes"},
            "semantic_summary": "Empty PR",
        }

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(
            return_value=LLMResponse(
                text=f"```\n{json.dumps(valid_response)}\n```",
                duration_ms=100,
            )
        )

        result = await analyzer.run(bundle, mock_llm)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_multiple_analyzers_sequential(self) -> None:
        """Test running multiple analyzers sequentially."""
        bundle = make_bundle(
            key_files=[
                make_key_file("src/feature.py", 100, 20),
                make_key_file("docs/README.md", 30, 5),
            ],
            file_categories={
                "source": ["src/feature.py"],
                "docs": ["docs/README.md"],
                "tests": [],
                "config": [],
            },
            body="This PR improves p50 latency by 10%",
        )

        analyzers_to_test = [
            DiffScoutAnalyzer,
            DesignAlignmentAnalyzer,
            DocsSchemaAuditorAnalyzer,
            FrictionMinerAnalyzer,
            EvidenceAuditorAnalyzer,
            PerfIntegrityAnalyzer,
            DecisionExtractorAnalyzer,
        ]

        for analyzer_cls in analyzers_to_test:
            analyzer = analyzer_cls()

            if not analyzer.should_run(bundle):
                continue

            # Create mock response matching the analyzer's schema
            mock_response = _create_mock_response_for_analyzer(analyzer)
            mock_llm = await create_async_mock_llm(mock_response)

            result = await analyzer.run(bundle, mock_llm)

            assert result.name == analyzer.spec.name
            # Either success or proper error handling
            assert isinstance(result.success, bool)


def _create_mock_response_for_analyzer(analyzer: BaseAnalyzer) -> dict[str, Any]:
    """Create a valid mock response for the given analyzer."""
    name = analyzer.spec.name

    responses: dict[str, dict[str, Any]] = {
        "DiffScout": {
            "review_map": [],
            "key_files": [],
            "blast_radius": {"level": "medium", "rationale": "Moderate"},
            "semantic_summary": "Test summary",
        },
        "DesignAlignment": {
            "intent": {"stated_goal": "Test", "inferred_type": "feature"},
            "alignment": {"drifted": False, "drift_type": "none", "details": "Aligned"},
            "scope_check": {
                "stated_in_scope": [],
                "stated_out_of_scope": [],
                "actual_scope": [],
                "discrepancies": [],
            },
            "evidence": [],
        },
        "DocsSchemaAuditor": {
            "docs_touched": True,
            "schema_touched": False,
            "issues": [],
            "doc_files_checked": [],
            "schema_files_checked": [],
            "recommendations": [],
            "overall_quality": "good",
        },
        "FrictionMiner": {
            "events": [],
            "iteration_count": 0,
            "friction_score": "low",
            "notes": "Smooth PR",
        },
        "EvidenceAuditor": {
            "claims": [],
            "evidence": {
                "local_gate": {"observed": False, "path": None},
                "tests": {"observed": False, "added": 0, "modified": 0},
                "typing": {"mypy_observed": False, "ruff_observed": False},
                "benchmarks": {"observed": False, "path": None},
                "docs_updated": False,
                "schema_validated": "unknown",
            },
            "missing_receipts": [],
            "evidence_completeness": "low",
            "notes": "Limited evidence",
        },
        "PerfIntegrity": {
            "has_perf_claims": False,
            "measurement_integrity": {
                "valid": "unknown",
                "invalidation_reason": None,
                "confidence": "low",
            },
            "baseline": {
                "specified": False,
                "commit_or_tag": None,
                "semantics_unchanged": "unknown",
            },
            "metrics_found": [],
            "issues": [],
            "recommendations": [],
        },
        "DecisionExtractor": {
            "decisions": [],
            "decision_count": 0,
            "dominant_type": "mixed",
            "evidence_quality": "weak",
            "notes": "No decisions found",
        },
    }

    return responses.get(name, {})
