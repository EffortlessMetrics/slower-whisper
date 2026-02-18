"""
Lightweight PII redaction example for slower-whisper transcripts.

This is best-effort only and does NOT provide compliance guarantees.

Usage:
    python examples/redaction/redact_transcript.py transcript.json --output redacted.json
"""

from __future__ import annotations

import argparse
import re
from copy import deepcopy
from pathlib import Path

from slower_whisper.pipeline import load_transcript, save_transcript

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?){2}\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){12,16}\b")


def redact_text(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = CARD_RE.sub("[CARD]", text)
    return text


def redact_transcript(transcript):
    redacted = deepcopy(transcript)

    for seg in redacted.segments:
        seg.text = redact_text(seg.text)

    if redacted.turns:
        for turn in redacted.turns:
            if hasattr(turn, "text"):
                turn.text = redact_text(turn.text)  # type: ignore[attr-defined]
            elif isinstance(turn, dict):
                turn["text"] = redact_text(str(turn.get("text", "")))

    return redacted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redact PII from a transcript JSON.")
    parser.add_argument("transcript", type=Path, help="Path to transcript JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input>.redacted.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    transcript = load_transcript(args.transcript)
    redacted = redact_transcript(transcript)

    out_path = args.output or args.transcript.with_suffix(".redacted.json")
    save_transcript(redacted, out_path)
    print(f"[done] Wrote redacted transcript to {out_path}")
