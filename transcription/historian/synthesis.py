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
) -> SynthesisResult:
    """
    Synthesize analyzer outputs into a dossier.

    Args:
        bundle: The original fact bundle
        analyzer_results: Results from all analyzers

    Returns:
        SynthesisResult with the dossier and metadata
    """
    from transcription.historian.validation import validate_dossier

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

    # Validate
    valid, errors = validate_dossier(dossier)
    if not valid:
        synthesis_notes.extend([f"Validation: {e}" for e in errors])

    return SynthesisResult(
        dossier=dossier,
        validation_errors=errors,
        analyzer_results=outputs,
        synthesis_notes=synthesis_notes,
    )
