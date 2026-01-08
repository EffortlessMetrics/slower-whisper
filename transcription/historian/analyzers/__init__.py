"""
Subagent analyzers for PR analysis.

Each analyzer is a specialized module that analyzes a specific aspect of a PR
and returns structured JSON output. Most analyzers are LLM-powered, but some
(like TemporalAnalyzer) are deterministic.

Available analyzers:
- TemporalAnalyzer: Deterministic temporal topology (phases, hotspots, oscillations)
- DiffScoutAnalyzer: Maps change surface, key files, blast radius
- EvidenceAuditorAnalyzer: Maps claims to artifacts, identifies missing receipts
- FrictionMinerAnalyzer: Extracts friction events using FAILURE_MODES taxonomy
- DesignAlignmentAnalyzer: Detects design drift
- PerfIntegrityAnalyzer: Validates benchmark measurements (conditional)
- DocsSchemaAuditorAnalyzer: Checks doc/schema integrity
- DecisionExtractorAnalyzer: Extracts material decisions with anchored evidence
"""

from transcription.historian.analyzers.base import (
    BaseAnalyzer,
    SubagentResult,
    SubagentSpec,
)
from transcription.historian.analyzers.decision_extractor import DecisionExtractorAnalyzer
from transcription.historian.analyzers.design_alignment import DesignAlignmentAnalyzer
from transcription.historian.analyzers.diff_scout import DiffScoutAnalyzer
from transcription.historian.analyzers.docs_schema import DocsSchemaAuditorAnalyzer
from transcription.historian.analyzers.evidence_auditor import EvidenceAuditorAnalyzer
from transcription.historian.analyzers.friction_miner import FrictionMinerAnalyzer
from transcription.historian.analyzers.perf_integrity import PerfIntegrityAnalyzer
from transcription.historian.analyzers.temporal import TemporalAnalyzer

# All available analyzers in execution order
# TemporalAnalyzer is first since it's deterministic and others may use its output
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

__all__ = [
    # Base classes
    "SubagentSpec",
    "SubagentResult",
    "BaseAnalyzer",
    # Analyzers
    "TemporalAnalyzer",
    "DiffScoutAnalyzer",
    "EvidenceAuditorAnalyzer",
    "FrictionMinerAnalyzer",
    "DesignAlignmentAnalyzer",
    "PerfIntegrityAnalyzer",
    "DocsSchemaAuditorAnalyzer",
    "DecisionExtractorAnalyzer",
    # Collection
    "ALL_ANALYZERS",
]
