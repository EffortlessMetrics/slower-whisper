"""
Fact bundle generation for PR analysis.

This module gathers all deterministic PR data from GitHub into a structured
FactBundle that can be used for LLM analysis or stored as a snapshot.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PRMetadata:
    """Core PR metadata."""

    number: int
    title: str
    url: str
    state: str
    author: str
    created_at: datetime
    merged_at: datetime | None
    closed_at: datetime | None
    merged_by: str | None
    base_branch: str
    head_branch: str
    labels: list[str]
    body: str


@dataclass
class ScopeData:
    """PR change scope information."""

    files_changed: int
    insertions: int
    deletions: int
    top_directories: list[str]
    key_files: list[dict[str, Any]]  # {path, additions, deletions, total_changes}
    file_categories: dict[str, list[str]]  # source/tests/docs/config/other
    blast_radius: str  # low/medium/high


@dataclass
class CommitData:
    """Individual commit information."""

    sha: str
    message: str
    author: str
    committed_at: datetime
    # Patterns detected
    is_fix: bool = False
    is_refactor: bool = False
    is_test: bool = False
    is_doc: bool = False
    is_revert: bool = False
    is_wip: bool = False


@dataclass
class SessionData:
    """A work session derived from commit bursts."""

    session_id: int
    start: datetime
    end: datetime
    commits: list[str]  # SHA list
    comments_in_window: list[str]  # comment IDs
    reviews_in_window: list[str]  # review IDs

    # Bounded estimation (set by estimation module)
    lb_minutes: int = 0  # Lower bound
    ub_minutes: int = 0  # Upper bound

    # Classification
    is_author_session: bool = True  # Has commits
    is_review_session: bool = False  # Post-last-commit


@dataclass
class CommentData:
    """PR comment information."""

    id: str
    author: str
    body: str
    created_at: datetime
    comment_type: str  # 'issue_comment' or 'review_comment'


@dataclass
class ReviewData:
    """PR review information."""

    id: str
    author: str
    body: str
    state: str  # APPROVED, CHANGES_REQUESTED, COMMENTED
    submitted_at: datetime


@dataclass
class CheckRunData:
    """CI check run with timing intervals."""

    name: str
    status: str  # queued, in_progress, completed
    conclusion: str | None  # success, failure, neutral, cancelled, skipped, timed_out
    started_at: datetime | None
    completed_at: datetime | None

    @property
    def duration_seconds(self) -> float | None:
        """Compute duration if timing available."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class FactBundle:
    """Complete deterministic fact bundle for a PR."""

    pr_number: int
    metadata: PRMetadata
    scope: ScopeData
    commits: list[CommitData]
    sessions: list[SessionData]
    comments: list[CommentData]
    reviews: list[ReviewData]
    check_runs: list[CheckRunData]
    receipt_paths: list[str]  # Discovered from PR body/comments
    diff: str | None

    # Computed fields (set by estimation module)
    estimation: Any = None  # BoundedEstimation
    machine_time: Any = None  # MachineTimeEstimate

    # Bundle metadata
    bundle_version: int = 1
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "bundle_version": self.bundle_version,
            "generated_at": self.generated_at.isoformat(),
            "pr_number": self.pr_number,
            "metadata": {
                "number": self.metadata.number,
                "title": self.metadata.title,
                "url": self.metadata.url,
                "state": self.metadata.state,
                "author": self.metadata.author,
                "created_at": self.metadata.created_at.isoformat(),
                "merged_at": self.metadata.merged_at.isoformat()
                if self.metadata.merged_at
                else None,
                "closed_at": self.metadata.closed_at.isoformat()
                if self.metadata.closed_at
                else None,
                "merged_by": self.metadata.merged_by,
                "base_branch": self.metadata.base_branch,
                "head_branch": self.metadata.head_branch,
                "labels": self.metadata.labels,
                "body": self.metadata.body,
            },
            "scope": {
                "files_changed": self.scope.files_changed,
                "insertions": self.scope.insertions,
                "deletions": self.scope.deletions,
                "top_directories": self.scope.top_directories,
                "key_files": self.scope.key_files,
                "file_categories": self.scope.file_categories,
                "blast_radius": self.scope.blast_radius,
            },
            "commits": [
                {
                    "sha": c.sha,
                    "message": c.message,
                    "author": c.author,
                    "committed_at": c.committed_at.isoformat(),
                    "patterns": {
                        "is_fix": c.is_fix,
                        "is_refactor": c.is_refactor,
                        "is_test": c.is_test,
                        "is_doc": c.is_doc,
                        "is_revert": c.is_revert,
                        "is_wip": c.is_wip,
                    },
                }
                for c in self.commits
            ],
            "sessions": [
                {
                    "session_id": s.session_id,
                    "start": s.start.isoformat(),
                    "end": s.end.isoformat(),
                    "commits": s.commits,
                    "comments_in_window": s.comments_in_window,
                    "reviews_in_window": s.reviews_in_window,
                    "lb_minutes": s.lb_minutes,
                    "ub_minutes": s.ub_minutes,
                    "is_author_session": s.is_author_session,
                    "is_review_session": s.is_review_session,
                }
                for s in self.sessions
            ],
            "comments": [
                {
                    "id": c.id,
                    "author": c.author,
                    "body": c.body[:1000],  # Truncate
                    "created_at": c.created_at.isoformat(),
                    "type": c.comment_type,
                }
                for c in self.comments
            ],
            "reviews": [
                {
                    "id": r.id,
                    "author": r.author,
                    "body": r.body[:1000],  # Truncate
                    "state": r.state,
                    "submitted_at": r.submitted_at.isoformat(),
                }
                for r in self.reviews
            ],
            "check_runs": [
                {
                    "name": cr.name,
                    "status": cr.status,
                    "conclusion": cr.conclusion,
                    "started_at": cr.started_at.isoformat() if cr.started_at else None,
                    "completed_at": cr.completed_at.isoformat() if cr.completed_at else None,
                    "duration_seconds": cr.duration_seconds,
                }
                for cr in self.check_runs
            ],
            "receipt_paths": self.receipt_paths,
            "diff": self.diff,
            "estimation": self.estimation.to_dict() if self.estimation else None,
            "machine_time": self.machine_time.to_dict() if self.machine_time else None,
        }


# --- GitHub CLI helpers ---


def _run_gh(args: list[str], check: bool = True) -> dict[str, Any] | list[Any] | str | None:
    """Run gh CLI command and return parsed JSON or raw output."""
    cmd = ["gh"] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running gh command: {' '.join(cmd)}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("Error: gh CLI not found. Install from https://cli.github.com/", file=sys.stderr)
        raise


def _parse_datetime(dt_str: str | None) -> datetime | None:
    """Parse ISO datetime string."""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _get_pr_metadata(pr_number: int) -> dict[str, Any]:
    """Fetch comprehensive PR metadata."""
    fields = [
        "number",
        "title",
        "url",
        "state",
        "createdAt",
        "mergedAt",
        "closedAt",
        "additions",
        "deletions",
        "changedFiles",
        "body",
        "labels",
        "author",
        "mergedBy",
        "baseRefName",
        "headRefName",
    ]
    data = _run_gh(["pr", "view", str(pr_number), "--json", ",".join(fields)])
    return data if isinstance(data, dict) else {}


def _get_pr_files(pr_number: int) -> list[dict[str, Any]]:
    """Fetch changed files with diff stats."""
    data = _run_gh(["pr", "view", str(pr_number), "--json", "files"])
    if isinstance(data, dict) and "files" in data:
        return data["files"]
    return []


def _get_pr_commits(pr_number: int) -> list[dict[str, Any]]:
    """Fetch commits with messages."""
    data = _run_gh(["pr", "view", str(pr_number), "--json", "commits"])
    if isinstance(data, dict) and "commits" in data:
        return data["commits"]
    return []


def _get_pr_comments(pr_number: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch PR comments and reviews separately."""
    data = _run_gh(["pr", "view", str(pr_number), "--json", "comments,reviews"])
    comments = []
    reviews = []
    if isinstance(data, dict):
        if "comments" in data:
            comments = data["comments"]
        if "reviews" in data:
            reviews = data["reviews"]
    return comments, reviews


def _get_pr_diff(pr_number: int, max_size: int = 50000) -> str:
    """Fetch the PR diff."""
    result = _run_gh(["pr", "diff", str(pr_number)], check=False)
    diff = result if isinstance(result, str) else ""
    if len(diff) > max_size:
        diff = diff[:max_size] + f"\n\n[... diff truncated at {max_size} chars ...]"
    return diff


def _get_check_runs(pr_number: int) -> list[dict[str, Any]]:
    """Fetch CI check runs with timing information."""
    # Get the head SHA for detailed check info
    pr_data = _run_gh(["pr", "view", str(pr_number), "--json", "headRefOid,statusCheckRollup"])
    if not isinstance(pr_data, dict):
        return []

    head_sha = pr_data.get("headRefOid")
    rollup = pr_data.get("statusCheckRollup") or []

    # Try to get detailed check runs with timing
    if head_sha:
        detailed = _run_gh(
            ["api", f"repos/{{owner}}/{{repo}}/commits/{head_sha}/check-runs"],
            check=False,
        )
        if isinstance(detailed, dict) and "check_runs" in detailed:
            return detailed["check_runs"]

    # Fallback to rollup (less detailed)
    return rollup


# --- Pattern detection ---


def _detect_commit_patterns(message: str) -> dict[str, bool]:
    """Detect patterns in commit message."""
    msg_lower = message.lower()

    fix_patterns = [r"\bfix\b", r"\bbug\b", r"\bissue\b", r"\bcorrect\b", r"\bresolve\b"]
    refactor_patterns = [r"\brefactor\b", r"\bclean\b", r"\brestructure\b"]
    test_patterns = [r"\btest\b", r"\bspec\b", r"\bcoverage\b"]
    doc_patterns = [r"\bdoc\b", r"\breadme\b", r"\bcomment\b"]
    revert_patterns = [r"\brevert\b", r"\bundo\b", r"\brollback\b"]
    wip_patterns = [r"\bwip\b", r"\bwork in progress\b", r"\bwip:\b"]

    return {
        "is_fix": any(re.search(p, msg_lower) for p in fix_patterns),
        "is_refactor": any(re.search(p, msg_lower) for p in refactor_patterns),
        "is_test": any(re.search(p, msg_lower) for p in test_patterns),
        "is_doc": any(re.search(p, msg_lower) for p in doc_patterns),
        "is_revert": any(re.search(p, msg_lower) for p in revert_patterns),
        "is_wip": any(re.search(p, msg_lower) for p in wip_patterns),
    }


def _categorize_files(files: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Categorize files by type."""
    categories: dict[str, list[str]] = {
        "source": [],
        "tests": [],
        "docs": [],
        "config": [],
        "other": [],
    }

    for f in files:
        path = f.get("path", "")
        if "/test" in path or path.startswith("test") or "_test." in path or "test_" in path:
            categories["tests"].append(path)
        elif path.endswith((".md", ".rst", ".txt")) or "/docs/" in path or "/doc/" in path:
            categories["docs"].append(path)
        elif path.endswith((".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".lock")):
            categories["config"].append(path)
        elif path.endswith((".py", ".rs", ".ts", ".tsx", ".js", ".jsx")):
            categories["source"].append(path)
        else:
            categories["other"].append(path)

    return categories


def _get_top_directories(files: list[dict[str, Any]], limit: int = 5) -> list[str]:
    """Extract top directories from changed files."""
    dir_counts: dict[str, int] = {}
    for f in files:
        path = f.get("path", "")
        parts = path.split("/")
        if len(parts) > 1:
            dir_path = "/".join(parts[:-1]) + "/"
            dir_counts[dir_path] = dir_counts.get(dir_path, 0) + 1

    sorted_dirs = sorted(dir_counts.items(), key=lambda x: -x[1])
    return [d for d, _ in sorted_dirs[:limit]]


def _get_key_files(files: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    """Extract key files with change stats."""
    scored = []
    for f in files:
        path = f.get("path", "")
        additions = f.get("additions", 0)
        deletions = f.get("deletions", 0)
        scored.append(
            {
                "path": path,
                "additions": additions,
                "deletions": deletions,
                "total_changes": additions + deletions,
            }
        )

    return sorted(scored, key=lambda x: -x["total_changes"])[:limit]


def _extract_receipt_paths(text: str) -> list[str]:
    """Extract potential receipt file paths from text."""
    if not text:
        return []

    paths = []
    # Look for common receipt patterns
    patterns = [
        r"receipts?/[\w/-]+\.(?:txt|json|log)",
        r"\.runs/[\w/-]+\.(?:txt|json|log)",
        r"benchmarks?/[\w/-]+\.(?:txt|json)",
        r"docs/audit/[\w/-]+\.(?:json|md)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        paths.extend(matches)

    return list(set(paths))


def _detect_sessions(
    commits: list[CommitData],
    comments: list[CommentData],
    reviews: list[ReviewData],
    gap_minutes: int = 45,
) -> list[SessionData]:
    """Detect work sessions from commit bursts."""
    if not commits:
        return []

    # Sort commits by time
    sorted_commits = sorted(commits, key=lambda c: c.committed_at)

    sessions: list[SessionData] = []
    session_start = sorted_commits[0].committed_at
    session_end = sorted_commits[0].committed_at
    session_commits = [sorted_commits[0].sha]
    session_id = 0

    for commit in sorted_commits[1:]:
        gap = (commit.committed_at - session_end).total_seconds() / 60
        if gap <= gap_minutes:
            session_end = commit.committed_at
            session_commits.append(commit.sha)
        else:
            # Save current session
            sessions.append(
                SessionData(
                    session_id=session_id,
                    start=session_start,
                    end=session_end,
                    commits=session_commits,
                    comments_in_window=[],
                    reviews_in_window=[],
                    is_author_session=True,
                )
            )
            session_id += 1
            session_start = commit.committed_at
            session_end = commit.committed_at
            session_commits = [commit.sha]

    # Save final session
    sessions.append(
        SessionData(
            session_id=session_id,
            start=session_start,
            end=session_end,
            commits=session_commits,
            comments_in_window=[],
            reviews_in_window=[],
            is_author_session=True,
        )
    )

    # Associate comments/reviews with sessions
    last_commit_time = sorted_commits[-1].committed_at if sorted_commits else None

    for comment in comments:
        # Find which session this comment falls in
        for session in sessions:
            if session.start <= comment.created_at <= session.end:
                session.comments_in_window.append(comment.id)
                break
        else:
            # Comment is after all commit sessions - mark as review session
            if last_commit_time and comment.created_at > last_commit_time:
                # Create or extend review session
                if sessions and sessions[-1].is_review_session:
                    sessions[-1].comments_in_window.append(comment.id)
                    sessions[-1].end = max(sessions[-1].end, comment.created_at)
                else:
                    sessions.append(
                        SessionData(
                            session_id=len(sessions),
                            start=comment.created_at,
                            end=comment.created_at,
                            commits=[],
                            comments_in_window=[comment.id],
                            reviews_in_window=[],
                            is_author_session=False,
                            is_review_session=True,
                        )
                    )

    for review in reviews:
        # Find which session this review falls in
        for session in sessions:
            if session.start <= review.submitted_at <= session.end:
                session.reviews_in_window.append(review.id)
                break
        else:
            # Review is after all commit sessions
            if last_commit_time and review.submitted_at > last_commit_time:
                if sessions and sessions[-1].is_review_session:
                    sessions[-1].reviews_in_window.append(review.id)
                    sessions[-1].end = max(sessions[-1].end, review.submitted_at)
                else:
                    sessions.append(
                        SessionData(
                            session_id=len(sessions),
                            start=review.submitted_at,
                            end=review.submitted_at,
                            commits=[],
                            comments_in_window=[],
                            reviews_in_window=[review.id],
                            is_author_session=False,
                            is_review_session=True,
                        )
                    )

    return sessions


def gather_pr_data(pr_number: int, include_diff: bool = True) -> FactBundle:
    """
    Gather all PR data into a FactBundle.

    Args:
        pr_number: The GitHub PR number
        include_diff: Whether to include the diff (can be large)

    Returns:
        FactBundle with all PR data
    """
    print(f"Gathering data for PR #{pr_number}...", file=sys.stderr)

    # Fetch raw data
    meta = _get_pr_metadata(pr_number)
    if not meta:
        raise ValueError(f"Could not fetch PR #{pr_number}")

    files = _get_pr_files(pr_number)
    raw_commits = _get_pr_commits(pr_number)
    raw_comments, raw_reviews = _get_pr_comments(pr_number)
    raw_check_runs = _get_check_runs(pr_number)
    diff = _get_pr_diff(pr_number) if include_diff else None

    # Parse metadata
    metadata = PRMetadata(
        number=meta.get("number", pr_number),
        title=meta.get("title", ""),
        url=meta.get("url", ""),
        state=meta.get("state", ""),
        author=meta.get("author", {}).get("login", "unknown"),
        created_at=_parse_datetime(meta.get("createdAt")) or datetime.now(),
        merged_at=_parse_datetime(meta.get("mergedAt")),
        closed_at=_parse_datetime(meta.get("closedAt")),
        merged_by=meta.get("mergedBy", {}).get("login") if meta.get("mergedBy") else None,
        base_branch=meta.get("baseRefName", "main"),
        head_branch=meta.get("headRefName", ""),
        labels=[label.get("name", "") for label in meta.get("labels", [])],
        body=meta.get("body", "") or "",
    )

    # Parse scope
    changed = meta.get("changedFiles", 0)
    if changed <= 5:
        blast_radius = "low"
    elif changed <= 20:
        blast_radius = "medium"
    else:
        blast_radius = "high"

    scope = ScopeData(
        files_changed=changed,
        insertions=meta.get("additions", 0),
        deletions=meta.get("deletions", 0),
        top_directories=_get_top_directories(files),
        key_files=_get_key_files(files),
        file_categories=_categorize_files(files),
        blast_radius=blast_radius,
    )

    # Parse commits
    commits = []
    for c in raw_commits:
        committed_date = c.get("committedDate") or c.get("commit", {}).get("committedDate")
        message = c.get("messageHeadline", "") or c.get("commit", {}).get("messageHeadline", "")
        sha = c.get("oid", "")[:7] or c.get("commit", {}).get("oid", "")[:7]
        author = (
            c.get("authors", [{}])[0].get("login", "unknown") if c.get("authors") else "unknown"
        )

        patterns = _detect_commit_patterns(message)
        commits.append(
            CommitData(
                sha=sha,
                message=message,
                author=author,
                committed_at=_parse_datetime(committed_date) or datetime.now(),
                **patterns,
            )
        )

    # Parse comments
    comments = []
    for i, c in enumerate(raw_comments):
        author = c.get("author", {})
        if isinstance(author, dict):
            author_name = author.get("login", "unknown")
        else:
            author_name = str(author) if author else "unknown"

        comments.append(
            CommentData(
                id=f"comment-{i}",
                author=author_name,
                body=c.get("body", ""),
                created_at=_parse_datetime(c.get("createdAt")) or datetime.now(),
                comment_type="issue_comment",
            )
        )

    # Parse reviews
    reviews = []
    for i, r in enumerate(raw_reviews):
        author = r.get("author", {})
        if isinstance(author, dict):
            author_name = author.get("login", "unknown")
        else:
            author_name = str(author) if author else "unknown"

        reviews.append(
            ReviewData(
                id=f"review-{i}",
                author=author_name,
                body=r.get("body", ""),
                state=r.get("state", ""),
                submitted_at=_parse_datetime(r.get("submittedAt")) or datetime.now(),
            )
        )

    # Parse check runs
    check_runs = []
    for cr in raw_check_runs:
        check_runs.append(
            CheckRunData(
                name=cr.get("name") or cr.get("context", "unknown"),
                status=cr.get("status") or cr.get("state", "unknown"),
                conclusion=cr.get("conclusion"),
                started_at=_parse_datetime(cr.get("started_at") or cr.get("startedAt")),
                completed_at=_parse_datetime(cr.get("completed_at") or cr.get("completedAt")),
            )
        )

    # Detect sessions
    sessions = _detect_sessions(commits, comments, reviews)

    # Extract receipt paths from body and comments
    receipt_paths = _extract_receipt_paths(metadata.body)
    for comment in comments:
        receipt_paths.extend(_extract_receipt_paths(comment.body))
    receipt_paths = list(set(receipt_paths))

    return FactBundle(
        pr_number=pr_number,
        metadata=metadata,
        scope=scope,
        commits=commits,
        sessions=sessions,
        comments=comments,
        reviews=reviews,
        check_runs=check_runs,
        receipt_paths=receipt_paths,
        diff=diff,
    )
