"""
Bounded estimation model for PR development time and machine costs.

This module implements the "bounded-estimate" approach where every metric
has explicit lower and upper bounds, along with the method used to compute them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transcription.historian.bundle import CheckRunData, FactBundle, SessionData


# Constants for estimation (can be overridden via config)
ESTIMATION_CONSTANTS = {
    "commit_floor_minutes": 3,  # Minimum time per commit
    "comment_floor_minutes": 2,  # Minimum time per comment
    "review_floor_minutes": 6,  # Minimum time per review submission
    "session_floor_minutes": 10,  # Minimum time per session
    "session_slack_minutes": 15,  # Slack time added to session span
    "session_cap_minutes": 240,  # Maximum time per session
    "gap_threshold_minutes": 45,  # Gap threshold for session detection
}

# Decision event weight constants (min, max minutes)
DECISION_WEIGHTS = {
    "scope": (2, 8),
    "design": (6, 20),
    "quality": (4, 15),
    "debug": (8, 30),
    "publish": (3, 12),
}

# Fallback decision candidate weights (more conservative ranges)
# Used when LLM extraction fails or is skipped
FALLBACK_WEIGHTS = {
    # Base context load by blast radius
    "context_low": (5, 10),
    "context_medium": (10, 25),
    "context_high": (20, 45),
    # Per-oscillation cluster
    "oscillation": (6, 20),
    # Per-check failure category
    "check_failure": (4, 12),
    # Contract/doc/schema touch
    "contract_touch": (3, 10),
    # Fix-loop (consecutive fix commits)
    "fix_loop": (5, 15),
}

# DevLT bands
DEVLT_BANDS = [
    (0, 10, "0-10m"),
    (10, 20, "10-20m"),
    (20, 45, "20-45m"),
    (45, 90, "45-90m"),
    (90, float("inf"), ">90m"),
]

# Machine spend bands
MACHINE_SPEND_BANDS = [
    (0, 2, "$0-$2"),
    (2, 10, "$2-$10"),
    (10, 25, "$10-$25"),
    (25, float("inf"), ">$25"),
]


def _minutes_to_band(minutes: int, bands: Sequence[tuple[float, float, str]]) -> str:
    """Convert minutes to a band string."""
    for low, high, label in bands:
        if low <= minutes < high:
            return label
    return bands[-1][2]  # Default to last band


@dataclass
class DecisionEvent:
    """A decision event that contributes to control-plane DevLT."""

    type: str  # scope, design, quality, debug, publish
    description: str
    anchor: str  # commit SHA, comment URL, issue link
    confidence: str  # high, med, low
    minutes_lb: int
    minutes_ub: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "type": self.type,
            "description": self.description,
            "anchor": self.anchor,
            "confidence": self.confidence,
            "minutes_lb": self.minutes_lb,
            "minutes_ub": self.minutes_ub,
        }


@dataclass
class ControlPlaneDevLT:
    """Control-plane DevLT estimation based on decision events."""

    lb_minutes: int
    ub_minutes: int
    band: str
    method: str
    coverage: str  # github_only | github_plus_claude
    decision_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "lb_minutes": self.lb_minutes,
            "ub_minutes": self.ub_minutes,
            "band": self.band,
            "method": self.method,
            "coverage": self.coverage,
            "decision_count": self.decision_count,
        }


def compute_control_plane_devlt(
    decision_events: list[DecisionEvent],
    coverage: str,
) -> ControlPlaneDevLT:
    """
    Compute control-plane DevLT from decision events.

    Args:
        decision_events: List of decision events with time bounds
        coverage: Coverage level (github_only | github_plus_claude)

    Returns:
        ControlPlaneDevLT with aggregated bounds and band
    """
    lb_total = sum(e.minutes_lb for e in decision_events)
    ub_total = sum(e.minutes_ub for e in decision_events)
    midpoint = (lb_total + ub_total) // 2

    return ControlPlaneDevLT(
        lb_minutes=lb_total,
        ub_minutes=ub_total,
        band=_minutes_to_band(midpoint, DEVLT_BANDS),
        method="decision-weighted-v1",
        coverage=coverage,
        decision_count=len(decision_events),
    )


def generate_fallback_decision_candidates(
    bundle: FactBundle,
    temporal_output: dict[str, Any] | None = None,
) -> list[DecisionEvent]:
    """
    Generate deterministic decision candidates when LLM extraction fails.

    This is the fallback hierarchy for DevLT estimation:
    1. LLM extracted decision events (preferred)
    2. Deterministic decision candidates (this function)
    3. Never "unknown" - always produce bounded estimates

    Decision candidates are generated from observable signals:
    - Blast radius → base context load
    - Temporal oscillations → debug/design decisions
    - Check failures → quality decisions
    - Contract/doc/schema touches → publish/design decisions
    - Fix-loop patterns → debug decisions

    Args:
        bundle: The fact bundle with commits, check runs, scope
        temporal_output: Optional temporal analyzer output for oscillation data

    Returns:
        List of DecisionEvent candidates with conservative time bounds
    """
    candidates: list[DecisionEvent] = []

    # 1. Base context load by blast radius
    blast_radius = bundle.scope.blast_radius
    if blast_radius == "high":
        lb, ub = FALLBACK_WEIGHTS["context_high"]
        candidates.append(
            DecisionEvent(
                type="scope",
                description="High blast radius requires significant context load",
                anchor=f"scope: {bundle.scope.files_changed} files, {bundle.scope.insertions}+ {bundle.scope.deletions}-",
                confidence="medium",
                minutes_lb=lb,
                minutes_ub=ub,
            )
        )
    elif blast_radius == "medium":
        lb, ub = FALLBACK_WEIGHTS["context_medium"]
        candidates.append(
            DecisionEvent(
                type="scope",
                description="Medium blast radius requires moderate context load",
                anchor=f"scope: {bundle.scope.files_changed} files",
                confidence="medium",
                minutes_lb=lb,
                minutes_ub=ub,
            )
        )
    else:  # low
        lb, ub = FALLBACK_WEIGHTS["context_low"]
        candidates.append(
            DecisionEvent(
                type="scope",
                description="Low blast radius - focused change",
                anchor=f"scope: {bundle.scope.files_changed} files",
                confidence="medium",
                minutes_lb=lb,
                minutes_ub=ub,
            )
        )

    # 2. Oscillation clusters from temporal analysis
    if temporal_output:
        oscillations = temporal_output.get("oscillations", [])
        for osc in oscillations:
            osc_type = osc.get("type", "unknown")
            files = osc.get("files", [])
            lb, ub = FALLBACK_WEIGHTS["oscillation"]
            candidates.append(
                DecisionEvent(
                    type="debug" if osc_type in ("revert_pattern", "approach_flip") else "design",
                    description=f"Oscillation pattern: {osc_type}",
                    anchor=f"temporal: {', '.join(files[:3])}{'...' if len(files) > 3 else ''}",
                    confidence="low",
                    minutes_lb=lb,
                    minutes_ub=ub,
                )
            )

    # 3. Check failures indicate quality/debug decisions
    failure_categories: set[str] = set()
    for check in bundle.check_runs:
        if check.conclusion in ("failure", "timed_out", "cancelled"):
            # Categorize by check name patterns
            name_lower = check.name.lower()
            if "lint" in name_lower or "format" in name_lower or "ruff" in name_lower:
                failure_categories.add("lint")
            elif "test" in name_lower or "pytest" in name_lower:
                failure_categories.add("test")
            elif "type" in name_lower or "mypy" in name_lower:
                failure_categories.add("type")
            elif "build" in name_lower or "compile" in name_lower:
                failure_categories.add("build")
            else:
                failure_categories.add("other")

    for category in failure_categories:
        lb, ub = FALLBACK_WEIGHTS["check_failure"]
        candidates.append(
            DecisionEvent(
                type="quality" if category in ("lint", "type") else "debug",
                description=f"Check failure: {category} category",
                anchor=f"check: {category} failures detected",
                confidence="medium",
                minutes_lb=lb,
                minutes_ub=ub,
            )
        )

    # 4. Contract/doc/schema touches indicate publish/design decisions
    contract_files = []
    for kf in bundle.scope.key_files:
        path = kf["path"].lower()
        if any(
            indicator in path
            for indicator in [
                "schema",
                "changelog",
                "readme",
                "api",
                "interface",
                ".d.ts",
                "types",
                "openapi",
                "swagger",
            ]
        ):
            contract_files.append(kf["path"])

    if contract_files:
        lb, ub = FALLBACK_WEIGHTS["contract_touch"]
        candidates.append(
            DecisionEvent(
                type="publish"
                if any("changelog" in f.lower() for f in contract_files)
                else "design",
                description=f"Contract/schema files touched: {len(contract_files)}",
                anchor=f"files: {', '.join(contract_files[:3])}{'...' if len(contract_files) > 3 else ''}",
                confidence="medium",
                minutes_lb=lb,
                minutes_ub=ub,
            )
        )

    # 5. Fix-loop patterns from commit sequence
    consecutive_fixes = 0
    max_fix_streak = 0
    for commit in bundle.commits:
        if commit.is_fix:
            consecutive_fixes += 1
            max_fix_streak = max(max_fix_streak, consecutive_fixes)
        else:
            consecutive_fixes = 0

    if max_fix_streak >= 2:
        lb, ub = FALLBACK_WEIGHTS["fix_loop"]
        # Scale by streak length
        multiplier = min(max_fix_streak - 1, 3)  # Cap at 3x
        candidates.append(
            DecisionEvent(
                type="debug",
                description=f"Fix-loop pattern: {max_fix_streak} consecutive fix commits",
                anchor=f"commits: {max_fix_streak} fix streak",
                confidence="medium",
                minutes_lb=lb * multiplier,
                minutes_ub=ub * multiplier,
            )
        )

    return candidates


@dataclass
class BoundedEstimation:
    """Bounded estimation results for a PR."""

    # Wall clock (deterministic)
    wall_clock_days: float
    wall_clock_hours: float
    created_at: datetime
    merged_at: datetime | None

    # Active work (session-based proxy)
    session_proxy_lb_hours: float
    session_proxy_ub_hours: float
    session_proxy_method: str
    session_count: int

    # DevLT split (session-based proxy)
    author_lb_minutes: int
    author_ub_minutes: int
    author_band: str
    review_lb_minutes: int
    review_ub_minutes: int
    review_band: str
    session_proxy_devlt_method: str

    # Control-plane DevLT (decision-weighted)
    control_plane: ControlPlaneDevLT | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict matching schema v2."""
        result: dict[str, Any] = {
            "wall_clock": {
                "days": self.wall_clock_days,
                "hours": self.wall_clock_hours,
                "created_at": self.created_at.isoformat(),
                "merged_at": self.merged_at.isoformat() if self.merged_at else None,
            },
            # Schema v2: active_work_proxy (legacy metric, kept for reference)
            "active_work_proxy": {
                "lb_hours": self.session_proxy_lb_hours,
                "ub_hours": self.session_proxy_ub_hours,
                "method": self.session_proxy_method,
                "session_count": self.session_count,
            },
            # Schema v2: devlt at root level (not nested in active_work_proxy)
            "devlt": {
                "author": {
                    "lb_minutes": self.author_lb_minutes,
                    "ub_minutes": self.author_ub_minutes,
                    "band": self.author_band,
                },
                "review": {
                    "lb_minutes": self.review_lb_minutes,
                    "ub_minutes": self.review_ub_minutes,
                    "band": self.review_band,
                },
                "method": self.session_proxy_devlt_method,
            },
        }
        # Schema v2: control_plane goes inside devlt
        if self.control_plane is not None:
            result["devlt"]["control_plane"] = self.control_plane.to_dict()
        return result


@dataclass
class MachineTimeEstimate:
    """Machine time and spend estimation."""

    # Check run timing
    wall_minutes: float | None  # Union of intervals
    compute_minutes: float | None  # Sum of durations
    check_count: int
    checks_with_timing: int

    # Spend estimation
    spend_estimate_usd: float | None
    spend_band: str
    spend_method: str
    spend_notes: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict matching schema v2."""
        return {
            "check_runs": {
                "wall_minutes": self.wall_minutes,
                "compute_minutes": self.compute_minutes,
                "check_count": self.check_count,
                "checks_with_timing": self.checks_with_timing,
            },
            # Schema v2: machine_spend (not just "spend")
            "machine_spend": {
                "estimate_usd": self.spend_estimate_usd,
                "band": self.spend_band,
                "method": self.spend_method,
                "notes": self.spend_notes,
            },
        }


def compute_session_bounds(
    session: SessionData,
    constants: dict[str, int] | None = None,
) -> tuple[int, int]:
    """
    Compute LB/UB for a single session.

    Args:
        session: The session data
        constants: Optional override for estimation constants

    Returns:
        (lower_bound_minutes, upper_bound_minutes)
    """
    c = constants or ESTIMATION_CONSTANTS

    n_commits = len(session.commits)
    n_comments = len(session.comments_in_window)
    n_reviews = len(session.reviews_in_window)

    # Lower bound: activity-based floor
    activity_lb = (
        n_commits * c["commit_floor_minutes"]
        + n_comments * c["comment_floor_minutes"]
        + n_reviews * c["review_floor_minutes"]
    )
    lb = max(c["session_floor_minutes"], activity_lb)

    # Upper bound: span-based cap
    span_minutes = (session.end - session.start).total_seconds() / 60
    ub = min(c["session_cap_minutes"], span_minutes + c["session_slack_minutes"])

    # LB can't exceed UB
    lb = min(lb, int(ub))

    return (int(lb), int(ub))


def compute_devlt_split(
    sessions: list[SessionData],
    constants: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Split DevLT into author vs review time.

    Author sessions: sessions with commits
    Review sessions: sessions without commits (post-last-commit)

    Returns dict with:
        author: {lb_minutes, ub_minutes, band}
        review: {lb_minutes, ub_minutes, band}
        method: str
    """
    author_lb = author_ub = 0
    review_lb = review_ub = 0

    for session in sessions:
        # Compute bounds if not already set
        if session.lb_minutes == 0 and session.ub_minutes == 0:
            lb, ub = compute_session_bounds(session, constants)
            session.lb_minutes = lb
            session.ub_minutes = ub

        lb, ub = session.lb_minutes, session.ub_minutes

        if session.is_author_session:
            author_lb += lb
            author_ub += ub

        if session.is_review_session:
            review_lb += lb
            review_ub += ub

    return {
        "author": {
            "lb_minutes": author_lb,
            "ub_minutes": author_ub,
            "band": _minutes_to_band((author_lb + author_ub) // 2, DEVLT_BANDS),
        },
        "review": {
            "lb_minutes": review_lb,
            "ub_minutes": review_ub,
            "band": _minutes_to_band((review_lb + review_ub) // 2, DEVLT_BANDS),
        },
        "method": "bounded session estimation (commit/comment/review floors + span caps)",
    }


def _compute_interval_union(intervals: list[tuple[datetime, datetime]]) -> float:
    """
    Compute the union of time intervals in minutes.

    This accounts for overlapping check runs.
    """
    if not intervals:
        return 0.0

    # Sort by start time
    sorted_intervals = sorted(intervals)

    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping - extend
            merged[-1] = (last_start, max(last_end, end))
        else:
            # Non-overlapping - add new
            merged.append((start, end))

    # Sum the merged intervals
    total_seconds = sum((end - start).total_seconds() for start, end in merged)
    return total_seconds / 60


def compute_machine_time(
    check_runs: list[CheckRunData],
) -> MachineTimeEstimate:
    """
    Compute machine time from check runs.

    Returns:
        MachineTimeEstimate with wall time (union) and compute time (sum)
    """
    intervals: list[tuple[datetime, datetime]] = []
    compute_sum_seconds = 0.0
    checks_with_timing = 0

    for check in check_runs:
        if check.started_at and check.completed_at:
            intervals.append((check.started_at, check.completed_at))
            duration = check.duration_seconds
            if duration is not None:
                compute_sum_seconds += duration
            checks_with_timing += 1

    if not intervals:
        return MachineTimeEstimate(
            wall_minutes=None,
            compute_minutes=None,
            check_count=len(check_runs),
            checks_with_timing=0,
            spend_estimate_usd=None,
            spend_band="unknown",
            spend_method="no check timing data available",
            spend_notes="Check runs did not include started_at/completed_at",
        )

    wall_minutes = _compute_interval_union(intervals)
    compute_minutes = compute_sum_seconds / 60

    return MachineTimeEstimate(
        wall_minutes=round(wall_minutes, 1),
        compute_minutes=round(compute_minutes, 1),
        check_count=len(check_runs),
        checks_with_timing=checks_with_timing,
        spend_estimate_usd=None,  # Can't estimate without cost data
        spend_band="unknown",
        spend_method="check run timing only (no cost data)",
        spend_notes=f"Wall: {wall_minutes:.1f}m (union), Compute: {compute_minutes:.1f}m (sum)",
    )


def compute_bounded_estimate(
    bundle: FactBundle,
    decision_events: list[DecisionEvent] | None = None,
    coverage: str | None = None,
) -> tuple[BoundedEstimation, MachineTimeEstimate]:
    """
    Compute all bounded estimates for a PR bundle.

    This updates the bundle's sessions with LB/UB and returns
    the full estimation results.

    Args:
        bundle: The fact bundle to analyze
        decision_events: Optional list of decision events for control-plane DevLT
        coverage: Coverage level for control-plane (github_only | github_plus_claude)

    Returns:
        (BoundedEstimation, MachineTimeEstimate)
    """
    # Wall clock (deterministic)
    if bundle.metadata.merged_at:
        wall_clock_delta = bundle.metadata.merged_at - bundle.metadata.created_at
        wall_clock_days = wall_clock_delta.total_seconds() / 86400
        wall_clock_hours = wall_clock_delta.total_seconds() / 3600
    else:
        wall_clock_days = 0.0
        wall_clock_hours = 0.0

    # Session-based estimation
    total_lb = 0
    total_ub = 0

    for session in bundle.sessions:
        lb, ub = compute_session_bounds(session)
        session.lb_minutes = lb
        session.ub_minutes = ub
        total_lb += lb
        total_ub += ub

    # DevLT split
    devlt = compute_devlt_split(bundle.sessions)

    # Machine time
    machine_time = compute_machine_time(bundle.check_runs)

    # Control-plane DevLT (decision-weighted)
    control_plane: ControlPlaneDevLT | None = None
    if decision_events and coverage:
        control_plane = compute_control_plane_devlt(decision_events, coverage)

    estimation = BoundedEstimation(
        wall_clock_days=round(wall_clock_days, 2),
        wall_clock_hours=round(wall_clock_hours, 1),
        created_at=bundle.metadata.created_at,
        merged_at=bundle.metadata.merged_at,
        session_proxy_lb_hours=round(total_lb / 60, 1),
        session_proxy_ub_hours=round(total_ub / 60, 1),
        session_proxy_method=f"bounded session estimation (gap ≤{ESTIMATION_CONSTANTS['gap_threshold_minutes']}min = same session)",
        session_count=len(bundle.sessions),
        author_lb_minutes=devlt["author"]["lb_minutes"],
        author_ub_minutes=devlt["author"]["ub_minutes"],
        author_band=devlt["author"]["band"],
        review_lb_minutes=devlt["review"]["lb_minutes"],
        review_ub_minutes=devlt["review"]["ub_minutes"],
        review_band=devlt["review"]["band"],
        session_proxy_devlt_method=devlt["method"],
        control_plane=control_plane,
    )

    # Update bundle
    bundle.estimation = estimation
    bundle.machine_time = machine_time

    return estimation, machine_time
