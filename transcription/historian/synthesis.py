"""
Dossier synthesis - merge subagent outputs into a validated dossier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transcription.historian.analyzers.base import SubagentResult
    from transcription.historian.bundle import FactBundle
    from transcription.historian.llm_client import LLMProvider


@dataclass
class SynthesisResult:
    """Result of dossier synthesis."""

    dossier: dict[str, Any]
    validation_errors: list[str]
    analyzer_results: dict[str, Any]
    synthesis_notes: list[str]


def _build_base_dossier(bundle: FactBundle) -> dict[str, Any]:
    """Build base dossier from bundle (deterministic fields only)."""
    return {
        "schema_version": 2,
        "pr_number": bundle.pr_number,
        "title": bundle.metadata.title,
        "url": bundle.metadata.url,
        "created_at": bundle.metadata.created_at.isoformat(),
        "merged_at": bundle.metadata.merged_at.isoformat() if bundle.metadata.merged_at else None,
        "scope": {
            "files_changed": bundle.scope.files_changed,
            "insertions": bundle.scope.insertions,
            "deletions": bundle.scope.deletions,
            "top_directories": bundle.scope.top_directories,
            "key_files": [f["path"] for f in bundle.scope.key_files[:5]],
            "blast_radius": bundle.scope.blast_radius,
        },
        "cost": bundle.estimation.to_dict() if bundle.estimation else {},
    }


def _merge_diff_scout(dossier: dict[str, Any], output: dict[str, Any]) -> None:
    """Merge DiffScout output into dossier."""
    if output.get("skipped"):
        return

    # Update scope with LLM-derived fields
    if "key_files" in output:
        dossier["scope"]["key_files"] = [
            f["path"] if isinstance(f, dict) else f for f in output["key_files"]
        ]

    if "blast_radius" in output:
        br = output["blast_radius"]
        if isinstance(br, dict):
            dossier["scope"]["blast_radius"] = br.get("level", dossier["scope"]["blast_radius"])

    if "semantic_summary" in output:
        dossier.setdefault("intent", {})["semantic_summary"] = output["semantic_summary"]


def _merge_evidence_auditor(dossier: dict[str, Any], output: dict[str, Any]) -> None:
    """Merge EvidenceAuditor output into dossier."""
    if output.get("skipped"):
        return

    evidence = output.get("evidence", {})
    dossier["evidence"] = {
        "local_gate": {
            "passed": evidence.get("local_gate", {}).get("observed", False),
            "command": "./scripts/ci-local.sh fast",
            "receipt_path": evidence.get("local_gate", {}).get("path"),
        },
        "tests": {
            "added": evidence.get("tests", {}).get("added", 0),
            "modified": evidence.get("tests", {}).get("modified", 0),
            "path": None,
            "coverage_delta": None,
        },
        "typing": {
            "mypy_passed": evidence.get("typing", {}).get("mypy_observed", False),
            "ruff_passed": evidence.get("typing", {}).get("ruff_observed", False),
        },
        "benchmarks": evidence.get("benchmarks", {}),
        "docs_updated": evidence.get("docs_updated", False),
        "schema_validated": evidence.get("schema_validated", "unknown"),
    }

    # Store claims for reference
    if "claims" in output:
        dossier.setdefault("_analysis", {})["claims"] = output["claims"]

    if "missing_receipts" in output:
        dossier.setdefault("_analysis", {})["missing_receipts"] = output["missing_receipts"]


def _merge_friction_miner(dossier: dict[str, Any], output: dict[str, Any]) -> None:
    """Merge FrictionMiner output into dossier."""
    if output.get("skipped"):
        return

    events = output.get("events", [])
    dossier.setdefault("process", {})["friction_events"] = [
        {
            "event": e.get("event"),
            "detected_by": e.get("detected_by"),
            "disposition": e.get("disposition"),
            "prevention": e.get("prevention"),
        }
        for e in events
    ]

    # Store findings (different format)
    dossier["findings"] = [
        {
            "type": e.get("type"),
            "description": e.get("event"),
            "detected_by": e.get("detected_by"),
            "disposition": e.get("disposition"),
            "commit": e.get("evidence"),
            "prevention_added": e.get("prevention"),
        }
        for e in events
    ]


def _merge_design_alignment(dossier: dict[str, Any], output: dict[str, Any]) -> None:
    """Merge DesignAlignment output into dossier."""
    if output.get("skipped"):
        return

    intent = output.get("intent", {})
    dossier.setdefault("intent", {}).update(
        {
            "goal": intent.get("stated_goal"),
            "type": intent.get("inferred_type"),
        }
    )

    alignment = output.get("alignment", {})
    dossier.setdefault("process", {})["design_alignment"] = {
        "drifted": alignment.get("drifted", False),
        "notes": alignment.get("details"),
    }


def _merge_perf_integrity(dossier: dict[str, Any], output: dict[str, Any]) -> None:
    """Merge PerfIntegrity output into dossier."""
    if output.get("skipped") or not output.get("has_perf_claims"):
        return

    integrity = output.get("measurement_integrity", {})
    dossier.setdefault("process", {})["measurement_integrity"] = {
        "valid": integrity.get("valid", "unknown"),
        "invalidation_reason": integrity.get("invalidation_reason"),
        "correction_link": None,
    }

    baseline = output.get("baseline", {})
    metrics = output.get("metrics_found", [])

    if metrics or baseline.get("specified"):
        dossier.setdefault("evidence", {})["benchmarks"] = {
            "results_path": None,
            "metrics": {m["name"]: m["value"] for m in metrics} if metrics else {},
            "baseline_commit": baseline.get("commit_or_tag"),
            "semantics_unchanged": baseline.get("semantics_unchanged", "unknown"),
        }


def _merge_docs_schema(dossier: dict[str, Any], output: dict[str, Any]) -> None:
    """Merge DocsSchemaAuditor output into dossier."""
    if output.get("skipped"):
        return

    dossier.setdefault("evidence", {})["docs_updated"] = output.get("docs_touched", False)

    # Add issues to findings if severe
    issues = output.get("issues", [])
    for issue in issues:
        if issue.get("severity") in ("high", "medium"):
            dossier.setdefault("findings", []).append(
                {
                    "type": "doc_drift" if "doc" in issue.get("type", "") else "other",
                    "description": issue.get("description"),
                    "detected_by": "DocsSchemaAuditor",
                    "disposition": "noted",
                    "commit": issue.get("file"),
                    "prevention_added": None,
                }
            )


def _merge_temporal(dossier: dict[str, Any], output: dict[str, Any]) -> None:
    """Merge TemporalAnalyzer output into dossier."""
    if output.get("skipped"):
        return

    temporal = {
        "convergence_type": output.get("convergence_type"),
        "phases": [
            {
                "name": p.get("name"),
                "start_commit": p.get("start_commit"),
                "end_commit": p.get("end_commit"),
                "evidence": p.get("evidence"),
            }
            for p in output.get("phases", [])
        ],
        "hotspots": [
            {
                "file": h.get("path"),  # Analyzer outputs "path", schema uses "file"
                "touch_count": h.get("touch_count"),
                "churn_sum": h.get("churn_sum"),
                "is_oscillating": h.get("is_oscillating"),
            }
            for h in output.get("hotspots", [])
        ],
        "oscillations": [
            {
                "type": o.get("type"),
                "files": o.get("files", []),
                "commits": o.get("commits", []),
            }
            for o in output.get("oscillations", [])
        ],
        "inflection_point": (
            {
                "commit": output["inflection_point"].get("commit_sha"),  # Analyzer uses commit_sha
                "rationale": output["inflection_point"].get("rationale"),
            }
            if output.get("inflection_point")
            else None
        ),
    }

    dossier.setdefault("process", {})["temporal"] = temporal


def _merge_decision_extractor(dossier: dict[str, Any], output: dict[str, Any]) -> None:
    """Merge DecisionExtractor output into dossier.

    Note: This function only merges raw decision events. Cost computation
    (control-plane DevLT) is handled by finalize_costs() after all merges complete.
    """
    if output.get("skipped"):
        return

    decisions = output.get("decisions", [])

    # Store decision events at top level (per schema v2)
    dossier["decision_events"] = [
        {
            "type": decision.get("type"),
            "description": decision.get("description"),
            "anchor": decision.get("anchor"),
            "confidence": decision.get("confidence"),
            "minutes_lb": decision.get("minutes_lb"),
            "minutes_ub": decision.get("minutes_ub"),
        }
        for decision in decisions
    ]

    # Store analysis metadata
    if output.get("notes"):
        dossier.setdefault("_analysis", {})["decision_notes"] = output["notes"]
    if output.get("evidence_quality"):
        dossier.setdefault("_analysis", {})["decision_evidence_quality"] = output[
            "evidence_quality"
        ]


def _compute_evidence_completeness(dossier: dict[str, Any]) -> str:
    """Compute evidence completeness score from dossier evidence fields.

    Returns:
        "high", "medium", or "low" based on evidence availability
    """
    evidence = dossier.get("evidence", {})
    score = 0
    max_score = 0

    # Local gate (weight: 2)
    max_score += 2
    local_gate = evidence.get("local_gate", {})
    if local_gate.get("passed"):
        score += 2
    elif local_gate.get("receipt_path"):
        score += 1

    # Tests (weight: 2)
    max_score += 2
    tests = evidence.get("tests", {})
    if tests.get("added", 0) > 0 or tests.get("modified", 0) > 0:
        score += 2
    elif tests.get("path"):
        score += 1

    # Typing (weight: 2)
    max_score += 2
    typing = evidence.get("typing", {})
    if typing.get("mypy_passed") and typing.get("ruff_passed"):
        score += 2
    elif typing.get("mypy_passed") or typing.get("ruff_passed"):
        score += 1

    # Benchmarks (weight: 1)
    max_score += 1
    benchmarks = evidence.get("benchmarks", {})
    if benchmarks.get("metrics") or benchmarks.get("results_path"):
        score += 1

    # Docs (weight: 1)
    max_score += 1
    if evidence.get("docs_updated"):
        score += 1

    # Schema validation (weight: 1)
    max_score += 1
    schema_validated = evidence.get("schema_validated")
    if schema_validated is True or schema_validated == "passed":
        score += 1

    # Compute ratio and return band
    if max_score == 0:
        return "low"

    ratio = score / max_score
    if ratio >= 0.7:
        return "high"
    elif ratio >= 0.4:
        return "medium"
    else:
        return "low"


def _compute_exhibit_score(dossier: dict[str, Any]) -> dict[str, Any]:
    """Compute exhibit score summarizing dossier quality metrics.

    Returns:
        Dict with score breakdown and overall rating
    """
    scores = {
        "evidence": 0,
        "decision_coverage": 0,
        "temporal_analysis": 0,
        "findings_anchored": 0,
    }

    # Evidence completeness (0-25 points)
    completeness = _compute_evidence_completeness(dossier)
    if completeness == "high":
        scores["evidence"] = 25
    elif completeness == "medium":
        scores["evidence"] = 15
    else:
        scores["evidence"] = 5

    # Decision coverage (0-25 points)
    decision_events = dossier.get("decision_events", [])
    if len(decision_events) >= 8:
        scores["decision_coverage"] = 25
    elif len(decision_events) >= 5:
        scores["decision_coverage"] = 20
    elif len(decision_events) >= 3:
        scores["decision_coverage"] = 15
    elif len(decision_events) >= 1:
        scores["decision_coverage"] = 10
    else:
        scores["decision_coverage"] = 0

    # Temporal analysis presence (0-25 points)
    temporal = dossier.get("process", {}).get("temporal", {})
    if temporal.get("convergence_type") and temporal.get("phases"):
        scores["temporal_analysis"] = 25
    elif temporal.get("convergence_type") or temporal.get("hotspots"):
        scores["temporal_analysis"] = 15
    else:
        scores["temporal_analysis"] = 5

    # Findings anchored (0-25 points)
    findings = dossier.get("findings", [])
    if not findings:
        scores["findings_anchored"] = 25  # No findings = nothing to anchor
    else:
        anchored = sum(1 for f in findings if f.get("commit") or f.get("detected_by"))
        ratio = anchored / len(findings) if findings else 1
        scores["findings_anchored"] = int(ratio * 25)

    total = sum(scores.values())

    if total >= 80:
        rating = "A"
    elif total >= 60:
        rating = "B"
    elif total >= 40:
        rating = "C"
    else:
        rating = "D"

    return {
        "total": total,
        "rating": rating,
        "breakdown": scores,
    }


def finalize_costs(dossier: dict[str, Any]) -> None:
    """Finalize all derived cost and quality metrics in the dossier.

    This is the ONLY place that computes derived metrics. It should be
    called after all merge functions complete but before validation.

    Handles:
    - control_plane DevLT: Computed from decision_events
    - evidence_completeness: Computed from evidence fields
    - exhibit_score: Quality metric for the dossier itself

    Args:
        dossier: The dossier dict to update in place
    """
    # --- Control-plane DevLT from decision events ---
    decision_events = dossier.get("decision_events", [])

    if decision_events:
        from transcription.historian.estimation import (
            DecisionEvent,
            compute_control_plane_devlt,
        )

        # Convert to DecisionEvent objects
        events = [
            DecisionEvent(
                type=d.get("type", "unknown"),
                description=d.get("description", ""),
                anchor=d.get("anchor", ""),
                confidence=d.get("confidence", "low"),
                minutes_lb=d.get("minutes_lb", 0),
                minutes_ub=d.get("minutes_ub", 0),
            )
            for d in decision_events
        ]

        # Compute control-plane DevLT (coverage is github_only by default)
        coverage = dossier.get("inputs", {}).get("coverage", "github_only")
        control_plane = compute_control_plane_devlt(events, coverage)

        # Update cost.devlt.control_plane
        dossier.setdefault("cost", {}).setdefault("devlt", {})["control_plane"] = (
            control_plane.to_dict()
        )

    # --- Evidence completeness ---
    evidence_completeness = _compute_evidence_completeness(dossier)
    dossier.setdefault("_analysis", {})["evidence_completeness"] = evidence_completeness

    # --- Exhibit score (dossier quality metric) ---
    exhibit_score = _compute_exhibit_score(dossier)
    dossier.setdefault("_analysis", {})["exhibit_score"] = exhibit_score

    # --- Ensure cost structure consistency ---
    # If devlt exists but control_plane wasn't computed, note why
    cost = dossier.get("cost", {})
    devlt = cost.get("devlt", {})
    if devlt and "control_plane" not in devlt and not decision_events:
        devlt["control_plane"] = {
            "lb_minutes": 0,
            "ub_minutes": 0,
            "band": "0-10m",
            "method": "no-decision-events",
            "coverage": dossier.get("inputs", {}).get("coverage", "github_only"),
            "decision_count": 0,
        }


async def run_all_analyzers(
    bundle: FactBundle,
    llm_provider: LLMProvider,
) -> list[SubagentResult]:
    """
    Run all analyzers sequentially on a bundle.

    Args:
        bundle: The fact bundle to analyze
        llm_provider: The LLM provider to use

    Returns:
        List of SubagentResult from each analyzer
    """
    from transcription.historian.analyzers import ALL_ANALYZERS

    results = []
    for analyzer_class in ALL_ANALYZERS:
        analyzer = analyzer_class()
        result = await analyzer.run(bundle, llm_provider)
        results.append(result)

    return results


def synthesize_dossier(
    bundle: FactBundle,
    analyzer_results: list[SubagentResult],
    *,
    strict: bool = True,
) -> SynthesisResult:
    """
    Synthesize analyzer outputs into a dossier.

    Args:
        bundle: The original fact bundle
        analyzer_results: Results from all analyzers
        strict: If True (default), missing schema raises SchemaNotFoundError.
                This is the recommended setting for analyze/publish pipelines.

    Returns:
        SynthesisResult with the dossier and metadata

    Raises:
        SchemaNotFoundError: If strict=True and schema file doesn't exist
    """
    from transcription.historian.validation import require_schema, validate_dossier

    # Fail fast if schema is missing - don't do work we can't validate
    require_schema(strict=strict)

    # Build base dossier from bundle
    dossier = _build_base_dossier(bundle)

    # Collect results by name
    outputs: dict[str, dict[str, Any]] = {}
    synthesis_notes: list[str] = []

    for result in analyzer_results:
        if result.success and result.output:
            outputs[result.name] = result.output
        elif not result.success:
            synthesis_notes.append(f"{result.name} failed: {', '.join(result.errors)}")

    # Merge each analyzer's output
    if "DiffScout" in outputs:
        _merge_diff_scout(dossier, outputs["DiffScout"])
    if "EvidenceAuditor" in outputs:
        _merge_evidence_auditor(dossier, outputs["EvidenceAuditor"])
    if "FrictionMiner" in outputs:
        _merge_friction_miner(dossier, outputs["FrictionMiner"])
    if "DesignAlignment" in outputs:
        _merge_design_alignment(dossier, outputs["DesignAlignment"])
    if "PerfIntegrity" in outputs:
        _merge_perf_integrity(dossier, outputs["PerfIntegrity"])
    if "DocsSchemaAuditor" in outputs:
        _merge_docs_schema(dossier, outputs["DocsSchemaAuditor"])
    if "DecisionExtractor" in outputs:
        _merge_decision_extractor(dossier, outputs["DecisionExtractor"])
    if "Temporal" in outputs:
        _merge_temporal(dossier, outputs["Temporal"])

    # Fill in defaults for required fields
    dossier.setdefault("intent", {}).setdefault("goal", bundle.metadata.title)
    dossier.setdefault("intent", {}).setdefault("type", "unknown")
    dossier.setdefault("intent", {}).setdefault("issues", [])
    dossier.setdefault("intent", {}).setdefault("phase", "unknown")
    dossier.setdefault("intent", {}).setdefault("out_of_scope", [])

    dossier.setdefault("findings", [])
    dossier.setdefault("evidence", {})
    dossier.setdefault("process", {}).setdefault("friction_events", [])
    dossier.setdefault("process", {}).setdefault(
        "design_alignment", {"drifted": False, "notes": None}
    )
    dossier.setdefault("process", {}).setdefault(
        "measurement_integrity",
        {"valid": "unknown", "invalidation_reason": None, "correction_link": None},
    )

    dossier.setdefault("reflection", {"went_well": [], "could_be_better": []})
    dossier.setdefault("outcome", "shipped" if bundle.metadata.merged_at else "pending")
    dossier.setdefault(
        "factory_delta", {"gates_added": [], "contracts_added": [], "prevention_issues": []}
    )
    dossier.setdefault("followups", [])
    dossier.setdefault("decision_events", [])
    dossier.setdefault("inputs", {"coverage": "github_only", "missing_sources": []})

    # Finalize all derived cost metrics (after merges, before validation)
    finalize_costs(dossier)

    # Validate with strict mode (schema already checked above, so this won't raise)
    valid, errors = validate_dossier(dossier, strict=strict)
    if not valid:
        synthesis_notes.extend([f"Validation: {e}" for e in errors])

    return SynthesisResult(
        dossier=dossier,
        validation_errors=errors,
        analyzer_results=outputs,
        synthesis_notes=synthesis_notes,
    )
