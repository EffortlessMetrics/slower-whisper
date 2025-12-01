"""Semantic annotator interfaces (v1.3 placeholder).

Provides a lightweight protocol for plugging in semantic tagging/annotation
without changing the enrichment pipeline behavior yet.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from .models import Transcript


class SemanticAnnotator(Protocol):
    """Annotate a transcript with semantic tags."""

    def annotate(self, transcript: Transcript) -> Transcript:
        """Return a transcript with semantic annotations attached."""


@dataclass(slots=True)
class NoOpSemanticAnnotator:
    """Placeholder annotator that returns the transcript unchanged."""

    def annotate(self, transcript: Transcript) -> Transcript:
        return transcript


@dataclass(slots=True)
class KeywordSemanticAnnotator:
    """Lightweight keyword tagger to prove the semantic hook works."""

    keyword_map: dict[str, Iterable[str]] = field(
        default_factory=lambda: {
            "pricing": ("price", "pricing", "quote", "budget", "discount"),
            "escalation": ("escalate", "supervisor", "manager", "complaint"),
            "churn_risk": ("cancel", "cancellation", "terminate", "churn"),
        }
    )

    def annotate(self, transcript: Transcript) -> Transcript:
        corpus = " ".join(
            seg.text.lower().strip() for seg in transcript.segments if getattr(seg, "text", "")
        )

        hits: list[dict[str, str]] = []
        tags: set[str] = set()

        for tag, keywords in self.keyword_map.items():
            for kw in keywords:
                if kw in corpus:
                    tags.add(tag)
                    hits.append({"tag": tag, "keyword": kw})
                    break

        existing = getattr(transcript, "annotations", None) or {}
        semantic = existing.get("semantic", {})
        merged_tags = set(semantic.get("tags", [])) | tags
        merged_hits = list(semantic.get("matches", [])) + hits

        semantic["tags"] = sorted(merged_tags)
        if merged_hits:
            semantic["matches"] = merged_hits

        existing["semantic"] = semantic
        transcript.annotations = existing
        return transcript
