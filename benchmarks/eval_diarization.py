#!/usr/bin/env python3
"""Tiny diarization evaluation harness.

Loads a manifest of audio files + reference RTTM, runs the slower-whisper
diarization pipeline, and computes basic DER/speaker-count metrics. Results can
be emitted as JSON and Markdown for quick sharing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from slower_whisper.pipeline.diarization import Diarizer, SpeakerTurn
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from slower_whisper.pipeline.diarization import Diarizer, SpeakerTurn

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DiarizationSample:
    """Single evaluation item sourced from the manifest."""

    id: str
    audio_path: Path
    reference_rttm: Path
    expected_speaker_count: int | None = None
    notes: str | None = None


@dataclass
class DerMetrics:
    """DER and (optional) breakdown components."""

    value: float | None
    missed: float | None = None
    false_alarm: float | None = None
    confusion: float | None = None
    detail: str | None = None


@dataclass
class SampleResult:
    """Result for one evaluated sample."""

    id: str
    der: DerMetrics
    speaker_count: int | None
    expected_speaker_count: int | None
    speaker_count_delta: int | None
    runtime_seconds: float | None
    error: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["der"] = asdict(self.der)
        return data


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def read_manifest(manifest_path: Path) -> list[DiarizationSample]:
    """Parse a JSONL manifest into strongly typed samples."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    samples: list[DiarizationSample] = []
    base = manifest_path.parent
    with manifest_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            try:
                sample_id = row["id"]
                audio_path = Path(row["audio_path"])
                reference_rttm = Path(row["reference_rttm"])
            except KeyError as exc:
                raise ValueError(
                    f"Missing required field in manifest line {lineno}: {exc}"
                ) from exc

            if not audio_path.is_absolute():
                audio_path = (base / audio_path).resolve()
            if not reference_rttm.is_absolute():
                reference_rttm = (base / reference_rttm).resolve()

            samples.append(
                DiarizationSample(
                    id=str(sample_id),
                    audio_path=audio_path,
                    reference_rttm=reference_rttm,
                    expected_speaker_count=row.get("expected_speaker_count"),
                    notes=row.get("notes"),
                )
            )
    return samples


def load_rttm_turns(path: Path) -> list[SpeakerTurn]:
    """Load reference speaker turns from RTTM file."""
    if not path.exists():
        raise FileNotFoundError(f"Reference RTTM not found: {path}")

    turns: list[SpeakerTurn] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                raise ValueError(f"Invalid RTTM line {lineno} in {path}: {line.strip()}")
            start = float(parts[3])
            duration = float(parts[4])
            speaker_id = parts[7]
            turns.append(
                SpeakerTurn(
                    start=start,
                    end=start + duration,
                    speaker_id=speaker_id,
                    confidence=None,
                )
            )
    return turns


def _turns_to_annotation(turns: Iterable[SpeakerTurn]):
    """Convert turns into pyannote Annotation; returns None when dependency is missing."""
    try:
        from pyannote.core import Annotation, Segment  # type: ignore
    except Exception:
        return None

    annotation = Annotation()
    for turn in turns:
        annotation[Segment(turn.start, turn.end)] = turn.speaker_id
    return annotation


def compute_der(reference: list[SpeakerTurn], hypothesis: list[SpeakerTurn]) -> DerMetrics:
    """Compute DER using pyannote when available; otherwise return a stub result."""
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate  # type: ignore
    except Exception as exc:
        return DerMetrics(value=None, detail=f"pyannote.metrics not available ({exc})")

    ref_ann = _turns_to_annotation(reference)
    hyp_ann = _turns_to_annotation(hypothesis)
    if ref_ann is None or hyp_ann is None:
        return DerMetrics(value=None, detail="pyannote.core dependency is missing")

    metric = DiarizationErrorRate()
    try:
        der_output = metric(ref_ann, hyp_ann, detailed=True)
    except TypeError:
        # Older pyannote versions may not support detailed=True
        try:
            der_value = float(metric(ref_ann, hyp_ann))
        except Exception as exc:  # noqa: BLE001
            return DerMetrics(value=None, detail=f"DER computation failed ({exc})")
        return DerMetrics(value=der_value)
    except Exception as exc:  # noqa: BLE001
        return DerMetrics(value=None, detail=f"DER computation failed ({exc})")

    if isinstance(der_output, tuple) and len(der_output) == 2:
        der_value, components = der_output
        missed = _maybe_float(getattr(components, "missed_detection", None))
        false_alarm = _maybe_float(getattr(components, "false_alarm", None))
        confusion = _maybe_float(getattr(components, "confusion", None))
        if isinstance(components, dict):
            if missed is None:
                missed = _maybe_float(
                    components.get("missed detection") or components.get("missed_detection")
                )
            if false_alarm is None:
                false_alarm = _maybe_float(
                    components.get("false alarm") or components.get("false_alarm")
                )
            if confusion is None:
                confusion = _maybe_float(components.get("confusion"))
        return DerMetrics(
            value=_maybe_float(der_value),
            missed=missed,
            false_alarm=false_alarm,
            confusion=confusion,
        )

    if isinstance(der_output, dict):
        der_value = _maybe_float(
            der_output.get("diarization error rate")
            or der_output.get("diarization_error_rate")
            or der_output.get("der")
        )
        missed = _maybe_float(
            der_output.get("missed detection") or der_output.get("missed_detection")
        )
        false_alarm = _maybe_float(der_output.get("false alarm") or der_output.get("false_alarm"))
        confusion = _maybe_float(der_output.get("confusion"))
        detail = None if der_value is not None else "DER dict missing diarization error rate"
        return DerMetrics(
            value=der_value,
            missed=missed,
            false_alarm=false_alarm,
            confusion=confusion,
            detail=detail,
        )

    return DerMetrics(value=_maybe_float(der_output))


def evaluate_sample(sample: DiarizationSample, diarizer: Diarizer) -> SampleResult:
    """Run diarization on one sample and compute metrics."""
    started = perf_counter()
    try:
        reference_turns = load_rttm_turns(sample.reference_rttm)
    except Exception as exc:  # noqa: BLE001
        return SampleResult(
            id=sample.id,
            der=DerMetrics(value=None, detail="reference load failed"),
            speaker_count=None,
            expected_speaker_count=sample.expected_speaker_count,
            speaker_count_delta=None,
            runtime_seconds=None,
            error=str(exc),
            notes=sample.notes,
        )

    try:
        hypothesis_turns = diarizer.run(sample.audio_path)
    except Exception as exc:  # noqa: BLE001
        return SampleResult(
            id=sample.id,
            der=DerMetrics(value=None, detail="diarization failed"),
            speaker_count=None,
            expected_speaker_count=sample.expected_speaker_count,
            speaker_count_delta=None,
            runtime_seconds=perf_counter() - started,
            error=str(exc),
            notes=sample.notes,
        )

    der = compute_der(reference_turns, hypothesis_turns)
    runtime = perf_counter() - started

    speaker_count = len({turn.speaker_id for turn in hypothesis_turns})
    speaker_count_delta = (
        None
        if sample.expected_speaker_count is None
        else speaker_count - sample.expected_speaker_count
    )

    return SampleResult(
        id=sample.id,
        der=der,
        speaker_count=speaker_count,
        expected_speaker_count=sample.expected_speaker_count,
        speaker_count_delta=speaker_count_delta,
        runtime_seconds=runtime,
        notes=sample.notes,
    )


def aggregate_results(results: list[SampleResult]) -> dict[str, Any]:
    ders = [r.der.value for r in results if r.der.value is not None]
    speaker_acc = []
    for r in results:
        if r.expected_speaker_count is None or r.speaker_count is None:
            continue
        speaker_acc.append(1.0 if r.expected_speaker_count == r.speaker_count else 0.0)

    return {
        "num_samples": len(results),
        "num_with_der": len(ders),
        "avg_der": sum(ders) / len(ders) if ders else None,
        "min_der": min(ders) if ders else None,
        "max_der": max(ders) if ders else None,
        "speaker_count_accuracy": sum(speaker_acc) / len(speaker_acc) if speaker_acc else None,
        "total_runtime": sum(r.runtime_seconds for r in results if r.runtime_seconds is not None),
    }


def write_json(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def render_markdown(
    output: Path,
    results: list[SampleResult],
    aggregate: dict[str, Any],
    args: argparse.Namespace,
    manifest_path: Path,
    manifest_hash: str,
) -> None:
    lines: list[str] = []
    lines.append("# Diarization Benchmark")
    lines.append("")
    lines.append("## Run config")
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- manifest: `{manifest_path}` (sha256: {manifest_hash})")
    lines.append(f"- diarizer device: `{args.device}`")
    lines.append(f"- min/max speakers: {args.min_speakers}/{args.max_speakers}")
    mode = os.getenv("SLOWER_WHISPER_PYANNOTE_MODE")
    if mode:
        lines.append(f"- SLOWER_WHISPER_PYANNOTE_MODE: `{mode}`")
    lines.append("")

    lines.append("## Aggregate")
    lines.append(f"- samples: {aggregate['num_samples']}")
    lines.append(
        f"- avg DER: {aggregate['avg_der'] if aggregate['avg_der'] is not None else 'n/a'}"
    )
    lines.append(
        f"- speaker count accuracy: "
        f"{aggregate['speaker_count_accuracy'] if aggregate['speaker_count_accuracy'] is not None else 'n/a'}"
    )
    lines.append(f"- total runtime (s): {aggregate['total_runtime']:.2f}")
    lines.append("")

    lines.append("## Per-sample")
    lines.append(
        "| sample | DER | hyp speakers | exp speakers | runtime (s) | notes | error/detail |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        der_str = "n/a" if r.der.value is None else f"{r.der.value:.4f}"
        hyp = r.speaker_count if r.speaker_count is not None else "n/a"
        exp = r.expected_speaker_count if r.expected_speaker_count is not None else "n/a"
        runtime = f"{r.runtime_seconds:.2f}" if r.runtime_seconds is not None else "n/a"
        detail = r.error or r.der.detail or ""
        lines.append(
            f"| {r.id} | {der_str} | {hyp} | {exp} | {runtime} | {r.notes or ''} | {detail} |"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_dataset = ROOT / "benchmarks" / "data" / "diarization"
    parser = argparse.ArgumentParser(
        description="Evaluate diarization quality on a small manifest-driven dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Directory containing manifest + audio/RTTM files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest.jsonl (defaults to <dataset>/manifest.jsonl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit of samples to evaluate.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "diarization_eval.json",
        help="Where to write structured results.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "diarization_eval.md",
        help="Where to write Markdown summary.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Diarization device hint (auto/cuda/cpu).",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers (hint to diarizer).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers (hint to diarizer).",
    )
    parser.add_argument(
        "--pyannote-mode",
        choices=["auto", "stub", "missing"],
        default=None,
        help="Force pyannote mode (sets SLOWER_WHISPER_PYANNOTE_MODE).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.pyannote_mode:
        os.environ["SLOWER_WHISPER_PYANNOTE_MODE"] = args.pyannote_mode

    manifest_path = args.manifest or (args.dataset / "manifest.jsonl")
    samples = read_manifest(manifest_path)
    if args.limit is not None:
        samples = samples[: args.limit]

    diarizer = Diarizer(
        device=args.device,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    results = [evaluate_sample(sample, diarizer) for sample in samples]
    aggregate = aggregate_results(results)

    payload = {
        "config": {
            "dataset": str(args.dataset),
            "manifest": str(manifest_path),
            "device": args.device,
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers,
            "pyannote_mode": os.getenv("SLOWER_WHISPER_PYANNOTE_MODE"),
        },
        "manifest_hash": hash_file(manifest_path),
        "aggregate": aggregate,
        "results": [r.to_dict() for r in results],
    }

    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_json} exists. Pass --overwrite to replace.")
    if args.output_md.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_md} exists. Pass --overwrite to replace.")

    write_json(args.output_json, payload)
    render_markdown(
        args.output_md, results, aggregate, args, manifest_path, payload["manifest_hash"]
    )

    print(f"Wrote JSON to {args.output_json}")
    print(f"Wrote Markdown to {args.output_md}")
    if aggregate["avg_der"] is None:
        print("DER could not be computed (pyannote.metrics missing?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
