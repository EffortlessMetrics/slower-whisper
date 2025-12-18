"""Turn-level enrichment utilities (v1.2 scaffolding).

These helpers work with the existing `transcription.turns` dict-based
representation used in the project and populate `metadata` on each turn.
"""

from __future__ import annotations

from typing import Any, cast

from .models import Segment, Transcript, TurnMeta
from .turn_helpers import turn_to_dict
from .turns import build_turns as build_turns_v1_1
from .types_audio import AudioState

_FILLERS = {
    "um",
    "uh",
    "erm",
    "hmm",
    "you",
    "know",
    "like",
    "sort",
    "of",
}


def _is_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    lowered = t.lower()
    for prefix in ("who ", "what ", "when ", "where ", "why ", "how ", "which "):
        if lowered.startswith(prefix):
            return True
    return False


def _estimate_disfluency_ratio(text: str) -> float:
    tokens = [t.strip(".,!?\"'`").lower() for t in text.split()]
    if not tokens:
        return 0.0
    fillers = sum(1 for tok in tokens if tok in _FILLERS)
    return fillers / max(len(tokens), 1)


def enrich_turns_metadata(transcript: Transcript) -> list[dict[str, Any]]:
    """Populate `metadata` for each turn in `transcript.turns`.

    If `transcript.turns` is empty or None this will call the v1.1
    `build_turns()` to construct them first.
    """
    # Ensure turns exist
    if transcript.turns is None:
        transcript = build_turns_v1_1(transcript)

    # Index segments
    seg_by_id = {seg.id: seg for seg in transcript.segments}

    enriched_turns: list[dict[str, Any]] = []

    for idx, turn in enumerate(transcript.turns or []):
        turn_dict = turn_to_dict(turn, copy=True)
        # turn is a dict with keys id, speaker_id, start, end, segment_ids, text
        seg_ids = list(turn_dict.get("segment_ids", []))
        segs: list[Segment] = [seg_by_id[sid] for sid in seg_ids if sid in seg_by_id]

        # Question count
        question_count = sum(1 for s in segs if _is_question(s.text or ""))

        # Disfluency: measure across concatenated turn text
        disfluency = _estimate_disfluency_ratio(turn_dict.get("text", ""))

        # Avg pause before first segment if available in audio_state
        # Prosody fields are nested: prosody.pauses.longest_ms
        avg_pause_ms: float | None = None
        if segs:
            first = segs[0]
            audio_state = cast(AudioState, getattr(first, "audio_state", None) or {})
            prosody = audio_state.get("prosody") or {}
            pauses = prosody.get("pauses") or {}
            longest_ms = pauses.get("longest_ms")
            if isinstance(longest_ms, int | float):
                avg_pause_ms = float(longest_ms)

        # Interruption heuristic: overlap with previous
        interruption_started_here = False
        if idx > 0 and transcript.turns:
            prev = transcript.turns[idx - 1]
            prev_dict = turn_to_dict(prev)
            if turn_dict.get("start", 0.0) < prev_dict.get("end", 0.0) - 0.15 and (
                turn_dict.get("speaker_id") != prev_dict.get("speaker_id")
            ):
                interruption_started_here = True

        meta = TurnMeta(
            question_count=question_count,
            interruption_started_here=interruption_started_here,
            avg_pause_ms=avg_pause_ms,
            disfluency_ratio=disfluency,
        )

        # attach metadata under 'metadata' key to be compatible with existing schema
        new_turn = dict(turn_dict)
        new_turn["metadata"] = {
            "question_count": meta.question_count,
            "interruption_started_here": meta.interruption_started_here,
            "avg_pause_ms": meta.avg_pause_ms,
            "disfluency_ratio": meta.disfluency_ratio,
        }
        enriched_turns.append(new_turn)

    transcript.turns = enriched_turns  # type: ignore[assignment]
    return enriched_turns
