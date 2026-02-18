from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

DATASET_PATH = Path("benchmarks/data/diarization")
MANIFEST_PATH = DATASET_PATH / "manifest.jsonl"


def _is_finite(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except Exception:
        return False
    return math.isfinite(number)


def main() -> int:
    if not DATASET_PATH.exists() or not MANIFEST_PATH.exists():
        print("[check_diarization_stub] missing diarization smoke fixtures")
        print(f"  expected dataset: {DATASET_PATH}")
        print(f"  expected manifest: {MANIFEST_PATH}")
        return 1

    with tempfile.TemporaryDirectory(prefix="diar_stub_eval_") as tmp:
        output_json = Path(tmp) / "diarization_stub_eval.json"
        output_md = Path(tmp) / "diarization_stub_eval.md"
        cmd = [
            sys.executable,
            "benchmarks/eval_diarization.py",
            "--dataset",
            str(DATASET_PATH),
            "--manifest",
            str(MANIFEST_PATH),
            "--device",
            "cpu",
            "--pyannote-mode",
            "stub",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--overwrite",
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=300)
        if proc.returncode != 0:
            print(f"[check_diarization_stub] eval_diarization failed (exit={proc.returncode})")
            if proc.stdout:
                print(proc.stdout.strip())
            if proc.stderr:
                print(proc.stderr.strip())
            return 1

        if not output_json.exists():
            print("[check_diarization_stub] eval_diarization did not produce output JSON")
            return 1

        data = json.loads(output_json.read_text())

    aggregate = data.get("aggregate") or {}
    config = data.get("config") or {}

    avg_der = aggregate.get("avg_der")
    speaker_acc = aggregate.get("speaker_count_accuracy")
    num_samples = aggregate.get("num_samples")
    pyannote_mode = config.get("pyannote_mode")

    errors: list[str] = []

    if not _is_finite(avg_der) or not (0.0 <= float(avg_der) <= 1.5):
        errors.append(f"avg_der out of range: {avg_der}")

    if not _is_finite(speaker_acc):
        errors.append(f"speaker_count_accuracy not finite: {speaker_acc}")
    elif float(speaker_acc) < 1.0:
        errors.append(f"speaker_count_accuracy dropped: {speaker_acc}")

    if not _is_finite(num_samples) or float(num_samples) < 1.0:
        errors.append(f"num_samples invalid: {num_samples}")

    if pyannote_mode != "stub":
        errors.append(f"unexpected pyannote_mode: {pyannote_mode}")

    if errors:
        for err in errors:
            print(f"[check_diarization_stub] {err}")
        return 1

    manifest_hash = data.get("manifest_hash")
    mode_note = f", pyannote_mode={pyannote_mode}" if pyannote_mode else ""
    print(
        "[check_diarization_stub] OK "
        f"(avg_der={float(avg_der):.4f}, speaker_count_accuracy={float(speaker_acc):.3f}, "
        f"manifest={manifest_hash}{mode_note})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
