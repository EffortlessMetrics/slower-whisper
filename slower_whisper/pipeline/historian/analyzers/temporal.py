"""
TemporalAnalyzer - deterministic temporal topology from PR data.

This analyzer computes convergence phases, hotspots, oscillations, and
inflection points WITHOUT requiring LLM calls. All analysis is derived
from the commit sequence, check runs, and file change patterns.

Output:
- temporal.phases[]: Commit burst sessions with phase classification
- temporal.hotspots[]: Per-file activity analysis with churn metrics
- temporal.oscillations[]: Detected oscillation patterns with evidence
- temporal.inflection_point: The stabilization point in commit history
- method_id: Algorithm version for reproducibility
- confidence: Overall confidence in the analysis
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .base import BaseAnalyzer, SubagentResult, SubagentSpec

if TYPE_CHECKING:
    from ..bundle import CommitData, FactBundle
    from ..llm_client import LLMProvider

# Method version for reproducibility
METHOD_ID = "temporal-v1.1"

# Oscillation keywords for pattern detection
OSCILLATION_KEYWORDS = {
    "revert_pattern": ["revert", "undo", "rollback", "back out", "restore"],
    "approach_flip": ["refactor", "rewrite", "redo", "rework", "overhaul", "redesign"],
    "dependency_flip": [
        "dep",
        "dependency",
        "package",
        "require",
        "import",
        "lock",
        "upgrade",
        "downgrade",
    ],
    "schema_rename": [
        "schema",
        "rename",
        "config",
        "model",
        "type",
        "interface",
        "field",
        "column",
    ],
}


@dataclass
class Phase:
    """A detected convergence phase."""

    name: str  # "exploration", "oscillation", "lock-in"
    start_commit: str  # SHA
    end_commit: str  # SHA
    evidence: str  # Human-readable evidence string
    session_type: str = "build-loop"  # "build-loop", "publish-loop", "docs-loop"


@dataclass
class Hotspot:
    """A file with high activity."""

    file: str  # path to the file
    touch_count: int
    churn_sum: int  # additions + deletions across all commits
    is_oscillating: bool


@dataclass
class Oscillation:
    """A detected oscillation pattern."""

    type: str  # "dependency_flip", "schema_rename", "approach_flip", "revert_pattern"
    files: list[str]
    commits: list[str]  # SHAs involved
    evidence: str  # Human-readable description of what was detected
    keywords: list[str]  # Actual keywords that triggered detection


@dataclass
class InflectionPoint:
    """The lock-in commit where core logic stabilized."""

    commit: str  # SHA
    rationale: str


class TemporalAnalyzer(BaseAnalyzer):
    """
    Deterministic analyzer for temporal topology.

    Computes convergence phases, hotspots, oscillations, and inflection
    points from PR data without requiring LLM calls.
    """

    @property
    def spec(self) -> SubagentSpec:
        return SubagentSpec(
            name="Temporal",
            description="Deterministic temporal topology analysis",
            system_prompt="",  # Not used - deterministic analyzer
            output_schema={
                "type": "object",
                "required": [
                    "phases",
                    "hotspots",
                    "oscillations",
                    "inflection_point",
                    "convergence_type",
                    "method_id",
                    "confidence",
                ],
                "properties": {
                    "method_id": {
                        "type": "string",
                        "description": "Algorithm version for reproducibility",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Overall confidence in the analysis",
                    },
                    "phases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "name",
                                "start_commit",
                                "end_commit",
                                "evidence",
                                "session_type",
                            ],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": ["exploration", "oscillation", "lock-in"],
                                },
                                "start_commit": {"type": "string"},
                                "end_commit": {"type": "string"},
                                "evidence": {"type": "string"},
                                "session_type": {
                                    "type": "string",
                                    "enum": ["build-loop", "publish-loop", "docs-loop"],
                                },
                            },
                        },
                    },
                    "hotspots": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "file",
                                "touch_count",
                                "churn_sum",
                                "is_oscillating",
                            ],
                            "properties": {
                                "file": {"type": "string"},
                                "touch_count": {"type": "integer"},
                                "churn_sum": {"type": "integer"},
                                "is_oscillating": {"type": "boolean"},
                            },
                        },
                    },
                    "oscillations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["type", "files", "commits", "evidence", "keywords"],
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "dependency_flip",
                                        "schema_rename",
                                        "approach_flip",
                                        "revert_pattern",
                                    ],
                                },
                                "files": {"type": "array", "items": {"type": "string"}},
                                "commits": {"type": "array", "items": {"type": "string"}},
                                "evidence": {"type": "string"},
                                "keywords": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                    "inflection_point": {
                        "type": ["object", "null"],
                        "properties": {
                            "commit": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                    },
                    "convergence_type": {
                        "type": "string",
                        "enum": ["linear", "cyclical", "chaotic"],
                    },
                    "notes": {"type": "string"},
                },
            },
            required_bundle_fields=["commits", "check_runs", "scope"],
        )

    def build_prompt(self, bundle: FactBundle) -> str:
        """Not used - this analyzer is deterministic."""
        return ""

    async def run(
        self,
        bundle: FactBundle,
        llm_provider: LLMProvider,
    ) -> SubagentResult:
        """
        Run deterministic temporal analysis.

        This overrides the base run() to avoid LLM calls entirely.
        All analysis is computed from the bundle data.
        """
        name = self.spec.name
        start_time = time.time()

        try:
            output = self._analyze(bundle)
            duration_ms = int((time.time() - start_time) * 1000)

            # Validate output
            validation_errors = self.validate_output(output)
            if validation_errors:
                return SubagentResult(
                    name=name,
                    success=False,
                    output=output,
                    errors=validation_errors,
                    duration_ms=duration_ms,
                )

            return SubagentResult(
                name=name,
                success=True,
                output=output,
                duration_ms=duration_ms,
            )
        except Exception as e:
            return SubagentResult(
                name=name,
                success=False,
                errors=[f"Analysis failed: {e}"],
                duration_ms=int((time.time() - start_time) * 1000),
            )

    def _analyze(self, bundle: FactBundle) -> dict[str, Any]:
        """Perform the actual temporal analysis."""
        commits = bundle.commits
        check_runs = bundle.check_runs
        key_files = bundle.scope.key_files

        # Compute phases
        phases = self._detect_phases(commits, check_runs, key_files)

        # Compute hotspots from key_files (final state) and commit patterns
        hotspots = self._compute_hotspots(commits, key_files)

        # Detect oscillations from commit patterns
        oscillations = self._detect_oscillations(commits, key_files)

        # Find the inflection point (lock-in commit)
        inflection = self._find_inflection_point(commits, phases)

        # Determine overall convergence type
        convergence_type = self._classify_convergence(phases, oscillations)

        # Compute confidence based on data quality
        confidence = self._compute_confidence(commits, key_files, phases)

        # Generate notes
        notes = self._generate_notes(phases, hotspots, oscillations, convergence_type)

        return {
            "method_id": METHOD_ID,
            "confidence": confidence,
            "phases": [
                {
                    "name": p.name,
                    "start_commit": p.start_commit,
                    "end_commit": p.end_commit,
                    "evidence": p.evidence,
                    "session_type": p.session_type,
                }
                for p in phases
            ],
            "hotspots": [
                {
                    "file": h.file,
                    "touch_count": h.touch_count,
                    "churn_sum": h.churn_sum,
                    "is_oscillating": h.is_oscillating,
                }
                for h in hotspots
            ],
            "oscillations": [
                {
                    "type": o.type,
                    "files": o.files,
                    "commits": o.commits,
                    "evidence": o.evidence,
                    "keywords": o.keywords,
                }
                for o in oscillations
            ],
            "inflection_point": (
                {
                    "commit": inflection.commit,
                    "rationale": inflection.rationale,
                }
                if inflection
                else None
            ),
            "convergence_type": convergence_type,
            "notes": notes,
        }

    def _compute_confidence(
        self,
        commits: list[CommitData],
        key_files: list[dict[str, Any]],
        phases: list[Phase],
    ) -> str:
        """
        Compute overall confidence in the analysis.

        Confidence factors:
        - high: 5+ commits, good file coverage, clear phase boundaries
        - medium: 3+ commits OR sparse data but clear patterns
        - low: <3 commits, no file data, or ambiguous patterns
        """
        commit_count = len(commits)
        file_count = len(key_files)
        phase_count = len(phases)

        # High confidence: sufficient data and clear patterns
        if commit_count >= 5 and file_count >= 3 and phase_count >= 1:
            return "high"

        # Medium confidence: some data to work with
        if commit_count >= 3 or (commit_count >= 2 and file_count >= 2):
            return "medium"

        # Low confidence: sparse data
        return "low"

    def _detect_phases(
        self,
        commits: list[CommitData],
        check_runs: list[Any],
        key_files: list[dict[str, Any]],
    ) -> list[Phase]:
        """
        Detect convergence phases from commit trajectory.

        Phases:
        - exploration: high churn, wide file spread, lots of fix/wip, failing checks
        - oscillation: repeated edits to same hotspots, alternating changes
        - lock-in: last N commits are tests/docs/cleanup; core logic untouched

        Session types (inferred from files touched + commit verbs):
        - build-loop: Source code changes, fix commits, feature work
        - publish-loop: Changelog, version bumps, release prep
        - docs-loop: Documentation, README, comments
        """
        if not commits:
            return []

        # Sort commits by time
        sorted_commits = sorted(commits, key=lambda c: c.committed_at)
        n = len(sorted_commits)

        if n == 1:
            # Single commit - classify based on its patterns
            c = sorted_commits[0]
            phase_name = self._classify_single_commit(c)
            session_type = self._infer_session_type([c], key_files)
            return [
                Phase(
                    name=phase_name,
                    start_commit=c.sha,
                    end_commit=c.sha,
                    evidence=f"Single commit: {c.message[:50]}",
                    session_type=session_type,
                )
            ]

        # Check for failing checks (indicator of exploration)
        has_failures = any(
            cr.conclusion in ("failure", "timed_out", "cancelled") for cr in check_runs
        )

        phases: list[Phase] = []
        window_size = max(3, n // 3)  # Use 1/3 of commits or min 3

        # Analyze commit windows
        i = 0
        while i < n:
            window_end = min(i + window_size, n)
            window = sorted_commits[i:window_end]

            phase_name, evidence = self._classify_window(window, has_failures, i == 0)
            session_type = self._infer_session_type(window, key_files)

            # Extend phase if next window has same classification
            while window_end < n:
                next_end = min(window_end + window_size, n)
                next_window = sorted_commits[window_end:next_end]
                next_name, _ = self._classify_window(next_window, has_failures, False)

                if next_name == phase_name:
                    window_end = next_end
                    window = sorted_commits[i:window_end]
                else:
                    break

            # Re-infer session type for extended window
            session_type = self._infer_session_type(window, key_files)

            phases.append(
                Phase(
                    name=phase_name,
                    start_commit=window[0].sha,
                    end_commit=window[-1].sha,
                    evidence=evidence,
                    session_type=session_type,
                )
            )

            i = window_end

        # Check if final commits indicate lock-in
        if n >= 3:
            final_window = sorted_commits[-3:]
            if self._is_lockin_window(final_window):
                # Either update last phase or add lock-in phase
                if phases and phases[-1].name != "lock-in":
                    session_type = self._infer_session_type(final_window, key_files)
                    phases.append(
                        Phase(
                            name="lock-in",
                            start_commit=final_window[0].sha,
                            end_commit=final_window[-1].sha,
                            evidence="Final commits are tests/docs/cleanup",
                            session_type=session_type,
                        )
                    )

        return phases

    def _infer_session_type(
        self,
        commits: list[CommitData],
        key_files: list[dict[str, Any]],
    ) -> str:
        """
        Infer session type from commits and files touched.

        Session types:
        - build-loop: Source code changes, fix commits, feature work
        - publish-loop: Changelog, version bumps, release prep
        - docs-loop: Documentation, README, comments
        """
        # Count commit patterns
        doc_count = sum(1 for c in commits if c.is_doc)
        total = len(commits)

        # Check commit messages for publish indicators
        publish_keywords = ["changelog", "version", "release", "bump", "publish", "ship"]
        publish_count = 0
        for c in commits:
            msg_lower = c.message.lower()
            if any(kw in msg_lower for kw in publish_keywords):
                publish_count += 1

        # Check key files for patterns
        doc_files = 0
        publish_files = 0
        source_files = 0
        for kf in key_files:
            path = kf["path"].lower()
            if path.endswith((".md", ".rst", ".txt")) or "/docs/" in path or "/doc/" in path:
                doc_files += 1
            elif "changelog" in path or "version" in path or "release" in path:
                publish_files += 1
            elif path.endswith((".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".go", ".java")):
                source_files += 1

        # Classify based on dominant pattern
        if total > 0:
            # If majority are doc commits, it's docs-loop
            if doc_count / total >= 0.5:
                return "docs-loop"
            # If publish keywords dominate, it's publish-loop
            if publish_count / total >= 0.3 or publish_files > 0:
                return "publish-loop"

        # Default to build-loop for source work
        return "build-loop"

    def _classify_single_commit(self, commit: CommitData) -> str:
        """Classify a single commit into a phase type."""
        if commit.is_test or commit.is_doc:
            return "lock-in"
        if commit.is_fix or commit.is_wip:
            return "exploration"
        return "exploration"  # Default for new features

    def _classify_window(
        self,
        window: list[CommitData],
        has_failures: bool,
        is_first: bool,
    ) -> tuple[str, str]:
        """Classify a window of commits into a phase."""
        evidence_parts: list[str] = []

        # Count patterns in window
        fix_count = sum(1 for c in window if c.is_fix)
        wip_count = sum(1 for c in window if c.is_wip)
        test_count = sum(1 for c in window if c.is_test)
        doc_count = sum(1 for c in window if c.is_doc)
        revert_count = sum(1 for c in window if c.is_revert)
        refactor_count = sum(1 for c in window if c.is_refactor)

        n = len(window)
        churn_ratio = (fix_count + wip_count) / n if n > 0 else 0
        cleanup_ratio = (test_count + doc_count + refactor_count) / n if n > 0 else 0

        # Exploration: high fix/wip ratio, or first window with failures
        if churn_ratio >= 0.5 or (is_first and has_failures):
            evidence_parts.append(f"{fix_count} fix commits, {wip_count} wip commits")
            if has_failures:
                evidence_parts.append("CI failures present")
            return "exploration", "; ".join(evidence_parts)

        # Oscillation: reverts or repeated fix patterns
        if revert_count >= 1 or (fix_count >= 2 and n >= 3):
            evidence_parts.append(f"{revert_count} reverts, {fix_count} fixes in window")
            return "oscillation", "; ".join(evidence_parts)

        # Lock-in: mostly tests/docs/refactoring
        if cleanup_ratio >= 0.6:
            evidence_parts.append(f"{test_count} test commits, {doc_count} doc commits")
            return "lock-in", "; ".join(evidence_parts)

        # Default to exploration for feature work
        evidence_parts.append("Feature development commits")
        return "exploration", "; ".join(evidence_parts)

    def _is_lockin_window(self, window: list[CommitData]) -> bool:
        """Check if a window represents lock-in (cleanup/tests/docs only)."""
        for c in window:
            # If any commit is NOT test/doc/refactor, it's not lock-in
            if not (c.is_test or c.is_doc or c.is_refactor):
                # Check if message suggests cleanup
                msg_lower = c.message.lower()
                cleanup_indicators = [
                    "cleanup",
                    "clean up",
                    "lint",
                    "format",
                    "typo",
                    "readme",
                    "changelog",
                ]
                if not any(ind in msg_lower for ind in cleanup_indicators):
                    return False
        return True

    def _compute_hotspots(
        self,
        commits: list[CommitData],
        key_files: list[dict[str, Any]],
    ) -> list[Hotspot]:
        """
        Compute hotspots from key files and commit patterns.

        Since we don't have per-commit file changes, we use:
        - key_files for churn data (total add+del across commits, not just final diff)
        - Commit message patterns to estimate touch counts
        """
        hotspots: list[Hotspot] = []

        # Build file touch estimates from commit messages
        file_touches: dict[str, int] = defaultdict(int)
        for c in commits:
            msg_lower = c.message.lower()
            # Try to extract file references from commit messages
            for kf in key_files:
                path = kf["path"]
                filename = path.split("/")[-1].lower()
                dirname = path.rsplit("/", 1)[0].lower() if "/" in path else ""

                # Check if commit message references this file/directory
                if filename in msg_lower or (dirname and dirname.split("/")[-1] in msg_lower):
                    file_touches[path] += 1

        # If no file references found, estimate from commit patterns
        if not file_touches and commits:
            # Distribute touches based on change size
            total_changes = sum(f.get("total_changes", 0) for f in key_files)
            for kf in key_files:
                if total_changes > 0:
                    weight = kf.get("total_changes", 0) / total_changes
                    file_touches[kf["path"]] = max(1, int(len(commits) * weight))
                else:
                    file_touches[kf["path"]] = 1

        # Create hotspots
        for kf in key_files:
            path = kf["path"]
            # churn_sum is total additions + deletions (approximation of total churn)
            churn = kf.get("additions", 0) + kf.get("deletions", 0)
            touch_count = file_touches.get(path, 1)

            # Determine if oscillating:
            # - High touch count relative to commits (touched in many commits)
            # - OR high churn relative to file size suggests rewrites
            is_oscillating = len(commits) > 3 and touch_count >= len(commits) * 0.5

            hotspots.append(
                Hotspot(
                    file=path,
                    touch_count=touch_count,
                    churn_sum=churn,
                    is_oscillating=is_oscillating,
                )
            )

        # Sort by churn
        hotspots.sort(key=lambda h: h.churn_sum, reverse=True)

        return hotspots[:10]  # Top 10 hotspots

    def _detect_oscillations(
        self,
        commits: list[CommitData],
        key_files: list[dict[str, Any]],
    ) -> list[Oscillation]:
        """
        Detect oscillation patterns from commit sequence.

        Oscillation types:
        - revert_pattern: Explicit reverts detected
        - approach_flip: Same file rewritten repeatedly, refactor keywords
        - dependency_flip: Dependency add/remove/upgrade/downgrade patterns
        - schema_rename: Schema/config/model changes
        """
        oscillations: list[Oscillation] = []

        if len(commits) < 2:
            return oscillations

        # Sort commits
        sorted_commits = sorted(commits, key=lambda c: c.committed_at)

        # Helper to extract files that might be affected by a commit
        def get_likely_files(commit: CommitData) -> list[str]:
            """Try to match commit to files based on message content."""
            files: list[str] = []
            msg_lower = commit.message.lower()
            for kf in key_files:
                path = kf["path"]
                filename = path.split("/")[-1].lower()
                dirname = path.rsplit("/", 1)[0].lower() if "/" in path else ""
                if filename in msg_lower or (dirname and dirname.split("/")[-1] in msg_lower):
                    files.append(path)
            return files

        # Look for revert patterns
        revert_commits = [c for c in sorted_commits if c.is_revert]
        if revert_commits:
            detected_keywords = []
            for c in revert_commits:
                msg_lower = c.message.lower()
                for kw in OSCILLATION_KEYWORDS["revert_pattern"]:
                    if kw in msg_lower and kw not in detected_keywords:
                        detected_keywords.append(kw)

            affected_files: list[str] = []
            for c in revert_commits:
                affected_files.extend(get_likely_files(c))

            oscillations.append(
                Oscillation(
                    type="revert_pattern",
                    files=list(set(affected_files)),
                    commits=[c.sha for c in revert_commits],
                    evidence=f"Detected {len(revert_commits)} revert commit(s)",
                    keywords=detected_keywords or ["revert"],
                )
            )

        # Look for approach_flip: refactor keywords or fix-after-fix patterns
        approach_commits: list[CommitData] = []
        approach_keywords: list[str] = []
        for c in sorted_commits:
            msg_lower = c.message.lower()
            for kw in OSCILLATION_KEYWORDS["approach_flip"]:
                if kw in msg_lower:
                    if c not in approach_commits:
                        approach_commits.append(c)
                    if kw not in approach_keywords:
                        approach_keywords.append(kw)

        # Also detect fix-after-fix patterns
        fix_sequences: list[CommitData] = []
        consecutive_fixes = 0
        for c in sorted_commits:
            if c.is_fix:
                consecutive_fixes += 1
                if consecutive_fixes >= 2:
                    fix_sequences.append(c)
            else:
                consecutive_fixes = 0

        if approach_commits or len(fix_sequences) >= 2:
            # Deduplicate by SHA since CommitData isn't hashable
            seen_shas: set[str] = set()
            all_commits: list[CommitData] = []
            for c in approach_commits + fix_sequences:
                if c.sha not in seen_shas:
                    seen_shas.add(c.sha)
                    all_commits.append(c)
            all_commits.sort(key=lambda c: c.committed_at)
            affected_files = []
            for c in all_commits:
                affected_files.extend(get_likely_files(c))

            evidence_parts = []
            if approach_commits:
                evidence_parts.append(f"{len(approach_commits)} refactor/rewrite commits")
            if len(fix_sequences) >= 2:
                evidence_parts.append(f"{len(fix_sequences)} consecutive fix commits")

            oscillations.append(
                Oscillation(
                    type="approach_flip",
                    files=list(set(affected_files)),
                    commits=[c.sha for c in all_commits],
                    evidence="; ".join(evidence_parts),
                    keywords=approach_keywords or ["fix"],
                )
            )

        # Look for dependency_flip patterns
        dep_commits: list[CommitData] = []
        dep_keywords: list[str] = []
        for c in sorted_commits:
            msg_lower = c.message.lower()
            for kw in OSCILLATION_KEYWORDS["dependency_flip"]:
                if kw in msg_lower:
                    if c not in dep_commits:
                        dep_commits.append(c)
                    if kw not in dep_keywords:
                        dep_keywords.append(kw)

        if len(dep_commits) >= 2:
            affected_files = []
            for c in dep_commits:
                affected_files.extend(get_likely_files(c))
            # Also check for dependency-related files
            for kf in key_files:
                path = kf["path"].lower()
                if any(
                    ind in path
                    for ind in ["requirements", "pyproject", "package.json", "cargo", "lock"]
                ):
                    if kf["path"] not in affected_files:
                        affected_files.append(kf["path"])

            oscillations.append(
                Oscillation(
                    type="dependency_flip",
                    files=list(set(affected_files)),
                    commits=[c.sha for c in dep_commits],
                    evidence=f"Detected {len(dep_commits)} dependency-related commits",
                    keywords=dep_keywords,
                )
            )

        # Look for schema_rename patterns
        schema_commits: list[CommitData] = []
        schema_keywords: list[str] = []
        for c in sorted_commits:
            msg_lower = c.message.lower()
            for kw in OSCILLATION_KEYWORDS["schema_rename"]:
                if kw in msg_lower:
                    if c not in schema_commits:
                        schema_commits.append(c)
                    if kw not in schema_keywords:
                        schema_keywords.append(kw)

        if len(schema_commits) >= 2:
            affected_files = []
            for c in schema_commits:
                affected_files.extend(get_likely_files(c))
            # Also check for schema-related files
            for kf in key_files:
                path = kf["path"].lower()
                if any(
                    ind in path
                    for ind in ["schema", "model", "types", "interface", ".json", ".yaml"]
                ):
                    if kf["path"] not in affected_files:
                        affected_files.append(kf["path"])

            oscillations.append(
                Oscillation(
                    type="schema_rename",
                    files=list(set(affected_files)),
                    commits=[c.sha for c in schema_commits],
                    evidence=f"Detected {len(schema_commits)} schema/config-related commits",
                    keywords=schema_keywords,
                )
            )

        return oscillations

    def _find_inflection_point(
        self,
        commits: list[CommitData],
        phases: list[Phase],
    ) -> InflectionPoint | None:
        """
        Find the inflection point (lock-in commit).

        The inflection point is the stabilization point where core logic
        stopped changing and remaining commits are tests/docs/cleanup.
        """
        if not commits:
            return None

        sorted_commits = sorted(commits, key=lambda c: c.committed_at)

        # Look for start of lock-in phase
        for phase in phases:
            if phase.name == "lock-in":
                return InflectionPoint(
                    commit=phase.start_commit,
                    rationale="First commit of lock-in phase; remaining commits are tests/docs/cleanup",
                )

        # Find last commit that touched core logic (not test/doc/refactor)
        last_core_commit = None
        for c in sorted_commits:
            if not (c.is_test or c.is_doc or c.is_refactor):
                last_core_commit = c

        if last_core_commit:
            return InflectionPoint(
                commit=last_core_commit.sha,
                rationale="Last commit touching core logic",
            )

        # Default to last commit
        return InflectionPoint(
            commit=sorted_commits[-1].sha,
            rationale="Final commit in PR",
        )

    def _classify_convergence(
        self,
        phases: list[Phase],
        oscillations: list[Oscillation],
    ) -> str:
        """Classify overall convergence pattern."""
        if not phases:
            return "linear"

        phase_names = [p.name for p in phases]

        # Chaotic: multiple phase transitions or many oscillations
        if len(phases) >= 4 or len(oscillations) >= 3:
            return "chaotic"

        # Cyclical: oscillation phase present or oscillation patterns
        if "oscillation" in phase_names or len(oscillations) >= 2:
            return "cyclical"

        # Linear: clean progression
        return "linear"

    def _generate_notes(
        self,
        phases: list[Phase],
        hotspots: list[Hotspot],
        oscillations: list[Oscillation],
        convergence_type: str,
    ) -> str:
        """Generate human-readable notes about the analysis."""
        notes_parts: list[str] = []

        # Convergence summary
        notes_parts.append(f"Convergence type: {convergence_type}.")

        # Phase summary with session types
        if phases:
            phase_info = [f"{p.name} ({p.session_type})" for p in phases]
            notes_parts.append(f"Phases detected: {' -> '.join(phase_info)}.")

        # Hotspot summary
        if hotspots:
            top_hotspot = hotspots[0]
            oscillating_count = sum(1 for h in hotspots if h.is_oscillating)
            notes_parts.append(
                f"Top hotspot: {top_hotspot.file} ({top_hotspot.churn_sum} lines changed)."
            )
            if oscillating_count > 0:
                notes_parts.append(f"{oscillating_count} file(s) show oscillating edit patterns.")

        # Oscillation summary
        if oscillations:
            osc_types = {o.type for o in oscillations}
            notes_parts.append(f"Oscillation patterns: {', '.join(sorted(osc_types))}.")

        return " ".join(notes_parts)
