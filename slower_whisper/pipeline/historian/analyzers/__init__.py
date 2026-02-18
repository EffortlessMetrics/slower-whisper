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

from .base import (
    BaseAnalyzer,
    SubagentResult,
    SubagentSpec,
)
from .decision_extractor import DecisionExtractorAnalyzer
from .design_alignment import DesignAlignmentAnalyzer
from .diff_scout import DiffScoutAnalyzer
from .docs_schema import DocsSchemaAuditorAnalyzer
from .evidence_auditor import EvidenceAuditorAnalyzer
from .friction_miner import FrictionMinerAnalyzer
from .perf_integrity import PerfIntegrityAnalyzer
from .temporal import TemporalAnalyzer

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
