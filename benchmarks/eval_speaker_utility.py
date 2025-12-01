"""Speaker analytics evaluation harness (v1.2 MVP).

Runs a tiny set of transcripts through:
1) Baseline prompt (no analytics)
2) Enriched prompt (speaker summary + turn view)
3) Judge (LLM if available, lexical fallback otherwise)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transcription import load_transcript
from transcription.llm_utils import render_conversation_for_llm, to_speaker_summary, to_turn_view
from transcription.speaker_stats import compute_speaker_stats
from transcription.turns import build_turns
from transcription.turns_enrich import enrich_turns_metadata

DEFAULT_SAMPLES = Path(__file__).resolve().parent / "data" / "speaker_analytics_samples.jsonl"
SUMMARY_INSTRUCTIONS = (
    "Summarize the conversation succinctly. Capture outcomes, objections and resolutions, and who "
    "committed to what. Use provided speaker labels for attribution and keep the tone factual."
)


@dataclass
class Sample:
    transcript_path: str
    reference_summary: str
    speaker_labels: dict[str, str] | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_samples(path: Path) -> list[Sample]:
    """Load sample definitions from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")

    samples: list[Sample] = []
    base = path.parent
    for row in _read_jsonl(path):
        t_path = Path(row["transcript_path"])
        if not t_path.is_absolute():
            t_path = (base / t_path).resolve()
        samples.append(
            Sample(
                transcript_path=str(t_path),
                reference_summary=row["reference_summary"],
                speaker_labels=row.get("speaker_labels"),
            )
        )
    return samples


def _call_openai(prompt: str, model: str) -> str | None:
    """Optional OpenAI helper; returns None when deps/keys are missing."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize conversations succinctly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] OpenAI call failed ({exc}); falling back to lexical summary", file=sys.stderr
        )
        return None


def _summarize(text: str, model: str | None = None, fallback_text: str | None = None) -> str:
    """Summarize with LLM when available; otherwise return trimmed extract."""
    if model:
        llm = _call_openai(text, model)
        if llm:
            return llm
    source = fallback_text if fallback_text is not None else text
    cleaned = " ".join(part.strip() for part in source.splitlines() if part.strip())
    words = cleaned.split()
    if len(words) <= 120:
        return cleaned
    return " ".join(words[:120]) + " ..."


def summarize_baseline(conversation_text: str, model: str | None = None) -> str:
    """Baseline summarizer using segment-level rendering."""
    prompt = f"{SUMMARY_INSTRUCTIONS}\n\nConversation:\n{conversation_text}"
    return _summarize(prompt, model=model, fallback_text=conversation_text)


def summarize_enriched(
    turn_view: str,
    speaker_summary: str,
    model: str | None = None,
) -> str:
    """Enriched summarizer combining speaker stats + turn view."""
    context = f"{speaker_summary}\n\nTurn cues:\n{turn_view}"
    prompt = f"{SUMMARY_INSTRUCTIONS}\n\nUse this context:\n{context}\n\nConversation:\n{turn_view}"
    return _summarize(prompt, model=model, fallback_text=turn_view)


def _overlap_score(reference: str, candidate: str) -> float:
    """Simple lexical overlap score for reference vs candidate."""
    ref_tokens = {tok.lower() for tok in reference.split() if tok}
    cand_tokens = {tok.lower() for tok in candidate.split() if tok}
    if not ref_tokens or not cand_tokens:
        return 0.0
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


def _judge_with_llm(reference: str, baseline: str, enriched: str, model: str) -> str | None:
    """Optional LLM judge; returns baseline/enriched/tie or None on failure."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    prompt = (
        "You are judging which summary better matches the reference. "
        "Prioritize: correct speaker attribution, objections/resolutions, and who committed to actions. "
        "Brevity alone should not win.\n\n"
        f"Reference:\n{reference}\n\n"
        f"Baseline summary:\n{baseline}\n\n"
        f"Enriched summary:\n{enriched}\n\n"
        "Respond with one token: 'baseline', 'enriched', or 'tie'."
    )
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=3,
        )
        answer = resp.choices[0].message.content.strip().lower()
        if answer in {"baseline", "enriched", "tie"}:
            return answer
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] OpenAI judge failed ({exc}); using lexical judge", file=sys.stderr)
    return None


def judge_pair(
    reference: str,
    baseline: str,
    enriched: str,
    judge_model: str | None = None,
) -> str:
    """Pick baseline/enriched/tie using LLM judge when available."""
    if judge_model:
        llm_choice = _judge_with_llm(reference, baseline, enriched, judge_model)
        if llm_choice:
            return llm_choice

    base_score = _overlap_score(reference, baseline)
    enriched_score = _overlap_score(reference, enriched)
    if abs(base_score - enriched_score) <= 0.02:
        return "tie"
    return "enriched" if enriched_score > base_score else "baseline"


def _ensure_analytics(transcript):
    """Populate turns + speaker stats so prompt helpers have metadata."""
    if not transcript.turns:
        transcript = build_turns(transcript)
    enrich_turns_metadata(transcript)
    compute_speaker_stats(transcript)
    return transcript


def run_eval(
    samples: Iterable[Sample],
    llm_model: str | None = None,
    judge_model: str | None = None,
    limit: int | None = None,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> None:
    wins: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        if limit is not None and idx >= limit:
            break

        transcript = load_transcript(sample.transcript_path)
        transcript = _ensure_analytics(transcript)

        render_mode = "turns" if transcript.turns else "segments"
        baseline_context = render_conversation_for_llm(
            transcript,
            mode=render_mode,
            include_audio_cues=False,
            include_timestamps=True,
            include_metadata=False,
            speaker_labels=sample.speaker_labels,
        )
        turn_view = to_turn_view(
            transcript,
            include_audio_state=True,
            include_timestamps=True,
            speaker_labels=sample.speaker_labels,
        )
        speaker_summary = to_speaker_summary(transcript, speaker_labels=sample.speaker_labels)

        baseline_summary = summarize_baseline(baseline_context, model=llm_model)
        enriched_summary = summarize_enriched(turn_view, speaker_summary, model=llm_model)

        winner = judge_pair(
            sample.reference_summary,
            baseline_summary,
            enriched_summary,
            judge_model=judge_model or llm_model,
        )
        wins[winner] += 1

        rows.append(
            {
                "transcript": sample.transcript_path,
                "reference": sample.reference_summary,
                "baseline_summary": baseline_summary,
                "enriched_summary": enriched_summary,
                "winner": winner,
                "speaker_labels": sample.speaker_labels or {},
            }
        )

    total = sum(wins.values()) or 1
    print("Results:")
    for k in ("enriched", "baseline", "tie"):
        count = wins.get(k, 0)
        print(f"- {k}: {count} ({100.0 * count / total:.1f}%)")

    if output_json:
        payload = {"wins": wins, "total": total, "results": rows}
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if output_md:
        lines = ["# Speaker analytics evaluation", ""]
        lines.append(f"Total samples: {total}")
        lines.append("")
        lines.append("| transcript | winner |")
        lines.append("| --- | --- |")
        for row in rows:
            lines.append(f"| {Path(row['transcript']).name} | {row['winner']} |")
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate speaker-aware prompts (MVP).")
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLES,
        help=f"JSONL with transcript_path + reference_summary (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--llm-model", type=str, default=None, help="LLM model for summaries (optional)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="LLM model for judging (defaults to --llm-model when provided)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--output-json", type=Path, default=None, help="Path to write JSON results")
    parser.add_argument(
        "--output-md", type=Path, default=None, help="Path to write Markdown results"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_rows = load_samples(args.samples)
    run_eval(
        sample_rows,
        llm_model=args.llm_model,
        judge_model=args.judge_model,
        limit=args.limit,
        output_json=args.output_json,
        output_md=args.output_md,
    )
