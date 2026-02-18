"""Lightweight ASR WER harness.

Runs a small manifest of audio files through slower-whisper and computes WER
against provided reference transcripts.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slower_whisper.pipeline import TranscriptionConfig, transcribe_file

try:
    import jiwer
except Exception:  # noqa: BLE001
    jiwer = None


DEFAULT_MANIFEST = Path(__file__).parent / "data" / "asr" / "manifest.jsonl"

PROFILES: dict[str, dict[str, Any]] = {
    "call_center": {
        "model": "small.en",
        "device": "cpu",
        "compute_type": "int8",
        "language": "en",
        "vad_min_silence_ms": 400,
        "beam_size": 5,
    },
    "meeting": {
        "model": "medium",
        "device": "cpu",
        "compute_type": "int8",
        "language": "en",
        "vad_min_silence_ms": 600,
        "beam_size": 5,
    },
    "podcast": {
        "model": "large-v3",
        "language": None,
        "vad_min_silence_ms": 800,
        "beam_size": 5,
    },
}


@dataclass
class Sample:
    id: str
    audio_path: Path
    reference_text: str
    profile: str | None = None


def _normalize_text(text: str) -> str:
    if not jiwer:
        return text.strip()
    transform = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    return transform(text)


def compute_wer(reference: str, hypothesis: str) -> float:
    if not jiwer:
        raise RuntimeError(
            "jiwer is required for WER calculation. Install with `uv pip install jiwer`."
        )
    ref = _normalize_text(reference)
    hyp = _normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.wer(ref, hyp)


def load_manifest(path: Path) -> list[Sample]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    samples: list[Sample] = []
    base = path.parent
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = (base / audio_path).resolve()
        samples.append(
            Sample(
                id=row.get("id") or audio_path.stem,
                audio_path=audio_path,
                reference_text=row.get("reference_text", ""),
                profile=row.get("profile"),
            )
        )
    return samples


def _transcript_text(transcript) -> str:
    return " ".join(seg.text for seg in transcript.segments)


def _config_for_profile(name: str | None) -> TranscriptionConfig:
    base = TranscriptionConfig()
    if not name:
        return base
    overrides = PROFILES.get(name, {})
    for field, value in overrides.items():
        if hasattr(base, field):
            setattr(base, field, value)
    return base


def run_eval(
    manifest_path: Path,
    profile: str | None = None,
    output_md: Path | None = None,
    output_json: Path | None = None,
    limit: int | None = None,
) -> None:
    samples = load_manifest(manifest_path)
    if limit is not None:
        samples = samples[:limit]

    if not samples:
        print(f"No samples found in {manifest_path}")
        return

    total_wer = 0.0
    counted = 0
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="asr_eval_") as tmpdir:
        root = Path(tmpdir)
        for sample in samples:
            if not sample.audio_path.exists():
                print(f"[skip] Missing audio: {sample.audio_path}")
                continue
            if not sample.reference_text.strip():
                print(f"[skip] Missing reference text for {sample.id}")
                continue

            cfg = _config_for_profile(sample.profile or profile)
            start = time.time()
            transcript = transcribe_file(sample.audio_path, root=root, config=cfg)
            elapsed = time.time() - start

            hyp_text = _transcript_text(transcript)
            wer = compute_wer(sample.reference_text, hyp_text)

            rows.append(
                {
                    "id": sample.id,
                    "audio_path": str(sample.audio_path),
                    "profile": sample.profile or profile or "default",
                    "reference_text": sample.reference_text,
                    "hypothesis_text": hyp_text,
                    "wer": wer,
                    "duration_sec": transcript.segments[-1].end if transcript.segments else 0.0,
                    "runtime_sec": elapsed,
                }
            )
            total_wer += wer
            counted += 1

    if counted == 0:
        print("No samples evaluated (missing audio or references).")
        return

    avg_wer = total_wer / counted
    print(f"Samples: {counted} | Avg WER: {avg_wer:.3f}")

    if output_json:
        payload = {"average_wer": avg_wer, "samples": rows}
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if output_md:
        lines = [
            "# ASR Benchmark",
            "",
            f"- manifest: `{manifest_path}`",
            f"- samples: {counted}",
            f"- avg WER: {avg_wer:.3f}",
            "",
            "| id | WER | profile |",
            "| --- | --- | --- |",
        ]
        for row in rows:
            lines.append(f"| {row['id']} | {row['wer']:.3f} | {row['profile']} |")
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR WER on a tiny manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to manifest JSONL (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()) + ["default"],
        default="default",
        help="Named profile to use for decoding (default: default).",
    )
    parser.add_argument("--output-md", type=Path, default=None, help="Markdown report path.")
    parser.add_argument("--output-json", type=Path, default=None, help="JSON results path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile_arg = None if args.profile == "default" else args.profile
    run_eval(
        manifest_path=args.manifest,
        profile=profile_arg,
        output_md=args.output_md,
        output_json=args.output_json,
        limit=args.limit,
    )
