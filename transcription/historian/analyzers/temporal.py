"""
TemporalAnalyzer - deterministic temporal topology from PR data.

This analyzer computes convergence phases, hotspots, oscillations, and
inflection points WITHOUT requiring LLM calls. All analysis is derived
from the commit sequence, check runs, and file change patterns.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from transcription.historian.analyzers.base import BaseAnalyzer, SubagentResult, SubagentSpec

if TYPE_CHECKING:
    from transcription.historian.bundle import CommitData, FactBundle
    from transcription.historian.llm_client import LLMProvider


@dataclass
class Phase:
    """A detected convergence phase."""

    name: str  # "exploration", "oscillation", "lock-in"
    start_commit: str  # SHA
    end_commit: str  # SHA
    evidence: list[str] = field(default_factory=list)


@dataclass
class Hotspot:
    """A file with high activity."""

    path: str
    touch_count: int
    churn_sum: int  # additions + deletions
    is_oscillating: bool


@dataclass
class Oscillation:
    """A detected oscillation pattern."""

    type: str  # "dep_flip", "approach_flip", "schema_flip"
    files: list[str]
    commits: list[str]  # SHAs involved


@dataclass
class InflectionPoint:
    """The lock-in commit where core logic stabilized."""

    commit_sha: str
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
                ],
                "properties": {
                    "phases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "start_commit", "end_commit", "evidence"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": ["exploration", "oscillation", "lock-in"],
                                },
                                "start_commit": {"type": "string"},
                                "end_commit": {"type": "string"},
                                "evidence": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                    "hotspots": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "path",
                                "touch_count",
                                "churn_sum",
                                "is_oscillating",
                            ],
                            "properties": {
                                "path": {"type": "string"},
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
                            "required": ["type", "files", "commits"],
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["dep_flip", "approach_flip", "schema_flip"],
                                },
                                "files": {"type": "array", "items": {"type": "string"}},
                                "commits": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                    "inflection_point": {
                        "type": ["object", "null"],
                        "properties": {
                            "commit_sha": {"type": "string"},
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
        phases = self._detect_phases(commits, check_runs)

        # Compute hotspots from key_files (final state) and commit patterns
        hotspots = self._compute_hotspots(commits, key_files)

        # Detect oscillations from commit patterns
        oscillations = self._detect_oscillations(commits)

        # Find the inflection point (lock-in commit)
        inflection = self._find_inflection_point(commits, phases)

        # Determine overall convergence type
        convergence_type = self._classify_convergence(phases, oscillations)

        # Generate notes
        notes = self._generate_notes(phases, hotspots, oscillations, convergence_type)

        return {
            "phases": [
                {
                    "name": p.name,
                    "start_commit": p.start_commit,
                    "end_commit": p.end_commit,
                    "evidence": p.evidence,
                }
                for p in phases
            ],
            "hotspots": [
                {
                    "path": h.path,
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
                }
                for o in oscillations
            ],
            "inflection_point": (
                {
                    "commit_sha": inflection.commit_sha,
                    "rationale": inflection.rationale,
                }
                if inflection
                else None
            ),
            "convergence_type": convergence_type,
            "notes": notes,
        }

    def _detect_phases(
        self,
        commits: list[CommitData],
        check_runs: list[Any],
    ) -> list[Phase]:
        """
        Detect convergence phases from commit trajectory.

        Phases:
        - exploration: high churn, wide file spread, lots of fix/wip, failing checks
        - oscillation: repeated edits to same hotspots, alternating changes
        - lock-in: last N commits are tests/docs/cleanup; core logic untouched
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
            return [
                Phase(
                    name=phase_name,
                    start_commit=c.sha,
                    end_commit=c.sha,
                    evidence=[f"Single commit: {c.message[:50]}"],
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

            phases.append(
                Phase(
                    name=phase_name,
                    start_commit=window[0].sha,
                    end_commit=window[-1].sha,
                    evidence=evidence,
                )
            )

            i = window_end

        # Check if final commits indicate lock-in
        if n >= 3:
            final_window = sorted_commits[-3:]
            if self._is_lockin_window(final_window):
                # Either update last phase or add lock-in phase
                if phases and phases[-1].name != "lock-in":
                    phases.append(
                        Phase(
                            name="lock-in",
                            start_commit=final_window[0].sha,
                            end_commit=final_window[-1].sha,
                            evidence=["Final commits are tests/docs/cleanup"],
                        )
                    )

        return phases

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
    ) -> tuple[str, list[str]]:
        """Classify a window of commits into a phase."""
        evidence: list[str] = []

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
            evidence.append(f"{fix_count} fix commits, {wip_count} wip commits")
            if has_failures:
                evidence.append("CI failures present")
            return "exploration", evidence

        # Oscillation: reverts or repeated fix patterns
        if revert_count >= 1 or (fix_count >= 2 and n >= 3):
            evidence.append(f"{revert_count} reverts, {fix_count} fixes in window")
            return "oscillation", evidence

        # Lock-in: mostly tests/docs/refactoring
        if cleanup_ratio >= 0.6:
            evidence.append(f"{test_count} test commits, {doc_count} doc commits")
            return "lock-in", evidence

        # Default to exploration for feature work
        evidence.append("Feature development commits")
        return "exploration", evidence

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
        - key_files for churn data
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
            churn = kf.get("additions", 0) + kf.get("deletions", 0)
            touch_count = file_touches.get(path, 1)

            # Determine if oscillating (high touch count relative to commits)
            is_oscillating = len(commits) > 3 and touch_count >= len(commits) * 0.5

            hotspots.append(
                Hotspot(
                    path=path,
                    touch_count=touch_count,
                    churn_sum=churn,
                    is_oscillating=is_oscillating,
                )
            )

        # Sort by churn
        hotspots.sort(key=lambda h: h.churn_sum, reverse=True)

        return hotspots[:10]  # Top 10 hotspots

    def _detect_oscillations(self, commits: list[CommitData]) -> list[Oscillation]:
        """
        Detect oscillation patterns from commit sequence.

        Looks for:
        - dep_flip: dependency add/remove patterns
        - approach_flip: same file rewritten repeatedly
        - schema_flip: schema/config changes back and forth
        """
        oscillations: list[Oscillation] = []

        if len(commits) < 3:
            return oscillations

        # Sort commits
        sorted_commits = sorted(commits, key=lambda c: c.committed_at)

        # Look for revert patterns
        revert_commits = [c for c in sorted_commits if c.is_revert]
        if revert_commits:
            oscillations.append(
                Oscillation(
                    type="approach_flip",
                    files=[],  # Can't determine files without per-commit diff
                    commits=[c.sha for c in revert_commits],
                )
            )

        # Look for fix-after-fix patterns (suggesting oscillation)
        fix_sequences: list[list[str]] = []
        current_sequence: list[str] = []
        for c in sorted_commits:
            if c.is_fix:
                current_sequence.append(c.sha)
            else:
                if len(current_sequence) >= 2:
                    fix_sequences.append(current_sequence)
                current_sequence = []
        if len(current_sequence) >= 2:
            fix_sequences.append(current_sequence)

        for seq in fix_sequences:
            oscillations.append(
                Oscillation(
                    type="approach_flip",
                    files=[],
                    commits=seq,
                )
            )

        # Look for dependency-related oscillations
        dep_indicators = ["dep", "dependency", "package", "require", "import", "lock"]
        dep_commits = []
        for c in sorted_commits:
            msg_lower = c.message.lower()
            if any(ind in msg_lower for ind in dep_indicators):
                dep_commits.append(c.sha)

        if len(dep_commits) >= 2:
            oscillations.append(
                Oscillation(
                    type="dep_flip",
                    files=[],
                    commits=dep_commits,
                )
            )

        # Look for schema/config oscillations
        schema_indicators = ["schema", "config", "model", "type", "interface"]
        schema_commits = []
        for c in sorted_commits:
            msg_lower = c.message.lower()
            if any(ind in msg_lower for ind in schema_indicators):
                schema_commits.append(c.sha)

        if len(schema_commits) >= 2:
            oscillations.append(
                Oscillation(
                    type="schema_flip",
                    files=[],
                    commits=schema_commits,
                )
            )

        return oscillations

    def _find_inflection_point(
        self,
        commits: list[CommitData],
        phases: list[Phase],
    ) -> InflectionPoint | None:
        """Find the inflection point (lock-in commit)."""
        if not commits:
            return None

        sorted_commits = sorted(commits, key=lambda c: c.committed_at)

        # Look for start of lock-in phase
        for phase in phases:
            if phase.name == "lock-in":
                return InflectionPoint(
                    commit_sha=phase.start_commit,
                    rationale="First commit of lock-in phase",
                )

        # Find last commit that touched core logic (not test/doc/refactor)
        last_core_commit = None
        for c in sorted_commits:
            if not (c.is_test or c.is_doc or c.is_refactor):
                last_core_commit = c

        if last_core_commit:
            return InflectionPoint(
                commit_sha=last_core_commit.sha,
                rationale="Last commit touching core logic",
            )

        # Default to last commit
        return InflectionPoint(
            commit_sha=sorted_commits[-1].sha,
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

        # Phase summary
        if phases:
            phase_names = [p.name for p in phases]
            notes_parts.append(f"Phases detected: {' -> '.join(phase_names)}.")

        # Hotspot summary
        if hotspots:
            top_hotspot = hotspots[0]
            oscillating_count = sum(1 for h in hotspots if h.is_oscillating)
            notes_parts.append(
                f"Top hotspot: {top_hotspot.path} ({top_hotspot.churn_sum} lines changed)."
            )
            if oscillating_count > 0:
                notes_parts.append(f"{oscillating_count} file(s) show oscillating edit patterns.")

        # Oscillation summary
        if oscillations:
            osc_types = {o.type for o in oscillations}
            notes_parts.append(f"Oscillation patterns: {', '.join(osc_types)}.")

        return " ".join(notes_parts)
