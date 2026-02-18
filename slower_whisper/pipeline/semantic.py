"""Semantic annotator interfaces (v1.3).

Provides rule-based semantic tagging for transcripts, attaching
`annotations.semantic` with keywords, risk tags, and action items.
"""

from __future__ import annotations

import re
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
    """Rule-based annotator for semantic keywords, risks, and actions."""

    escalation_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "escalate",
            "escalation",
            "unacceptable",
            "supervisor",
            "manager",
            "complaint",
        )
    )
    churn_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "cancel",
            "cancellation",
            "terminate",
            "termination",
            "switch",
            "switching",
            "competitor",
            "leaving",
            "leave",
            "churn",
        )
    )
    pricing_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "price",
            "pricing",
            "cost",
            "budget",
            "expensive",
            "cheap",
        )
    )
    action_patterns: tuple[str, ...] = field(
        default_factory=lambda: (
            r"\b(i['’]?ll|i will)\s+",
            r"\bwe('?ll| will)\s+",
            r"\blet me\s+",
            r"\bwe should\s+",
            r"\bi can\s+",
            r"\bi will send\b",
            r"\bi['’]?ll send\b",
            r"\bi will follow up\b",
            r"\bi['’]?ll follow up\b",
        )
    )
    _escalation_patterns: tuple[tuple[str, re.Pattern[str]], ...] = field(init=False, repr=False)
    _churn_patterns: tuple[tuple[str, re.Pattern[str]], ...] = field(init=False, repr=False)
    _pricing_patterns: tuple[tuple[str, re.Pattern[str]], ...] = field(init=False, repr=False)
    _action_regexes: tuple[re.Pattern[str], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Pre-compile regexes for faster annotation."""
        object.__setattr__(
            self,
            "_escalation_patterns",
            tuple(
                (kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
                for kw in self.escalation_keywords
            ),
        )
        object.__setattr__(
            self,
            "_churn_patterns",
            tuple(
                (kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
                for kw in self.churn_keywords
            ),
        )
        object.__setattr__(
            self,
            "_pricing_patterns",
            tuple(
                (kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
                for kw in self.pricing_keywords
            ),
        )
        object.__setattr__(
            self,
            "_action_regexes",
            tuple(re.compile(pat, re.IGNORECASE) for pat in self.action_patterns),
        )

    def annotate(self, transcript: Transcript) -> Transcript:
        """Populate annotations.semantic with keywords, risk tags, and actions."""
        existing = getattr(transcript, "annotations", None) or {}
        semantic = existing.get("semantic") or {}

        keywords: set[str] = set(semantic.get("keywords", []))
        risk_tags: set[str] = set(semantic.get("risk_tags", []) or semantic.get("tags", []))
        matches: list[dict[str, object]] = list(semantic.get("matches", []))
        matches_seen = {(m.get("risk"), m.get("keyword"), m.get("segment_id")) for m in matches}

        actions = list(semantic.get("actions", []))
        actions_seen = {
            (
                act.get("text", ""),
                act.get("speaker_id"),
                tuple(act.get("segment_ids") or ()),
            )
            for act in actions
        }

        def record_match(risk: str, keyword: str, segment_id: int | None) -> None:
            key = (risk, keyword, segment_id)
            if key in matches_seen:
                return
            matches_seen.add(key)
            entry: dict[str, object] = {"risk": risk, "keyword": keyword}
            if segment_id is not None:
                entry["segment_id"] = segment_id
            matches.append(entry)

        for segment in transcript.segments:
            raw_text = getattr(segment, "text", "") or ""
            text = raw_text.strip()
            if not text:
                continue

            text_lower = text.lower()
            speaker_meta = getattr(segment, "speaker", None)
            speaker_id = speaker_meta.get("id") if isinstance(speaker_meta, dict) else None
            segment_id = getattr(segment, "id", None)
            segment_ids = [segment_id] if segment_id is not None else []

            for keyword, pattern in self._escalation_patterns:
                if pattern.search(text_lower):
                    keywords.add(keyword)
                    risk_tags.add("escalation")
                    record_match("escalation", keyword, segment_id)

            for keyword, pattern in self._churn_patterns:
                if pattern.search(text_lower):
                    keywords.add(keyword)
                    risk_tags.add("churn_risk")
                    record_match("churn_risk", keyword, segment_id)

            for keyword, pattern in self._pricing_patterns:
                if pattern.search(text_lower):
                    keywords.add(keyword)
                    risk_tags.add("pricing")
                    record_match("pricing", keyword, segment_id)

            for pattern in self._action_regexes:
                if pattern.search(text_lower):
                    action_key = (text, speaker_id, tuple(segment_ids))
                    if action_key not in actions_seen:
                        actions_seen.add(action_key)
                        actions.append(
                            {
                                "text": text,
                                "speaker_id": speaker_id,
                                "segment_ids": segment_ids,
                                "pattern": pattern.pattern,
                            }
                        )
                    break

        semantic["keywords"] = sorted(keywords)
        semantic["risk_tags"] = sorted(risk_tags)
        semantic["actions"] = actions

        # Backwards-compatible aliases for consumers still looking at tags/matches.
        semantic["tags"] = sorted(risk_tags)
        if matches:
            semantic["matches"] = matches

        existing["semantic"] = semantic or {"keywords": [], "risk_tags": [], "actions": []}
        transcript.annotations = existing
        return transcript
