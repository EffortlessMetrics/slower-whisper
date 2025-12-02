from __future__ import annotations

import json
import math
from pathlib import Path

REPORT_PATH = Path("benchmarks/DIARIZATION_REPORT.json")


def _is_finite(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except Exception:
        return False
    return math.isfinite(number)


def main() -> int:
    if not REPORT_PATH.exists():
        print("[check_diarization_stub] missing benchmarks/DIARIZATION_REPORT.json")
        print("  Regenerate with SLOWER_WHISPER_PYANNOTE_MODE=stub benchmarks/eval_diarization.py")
        return 1

    data = json.loads(REPORT_PATH.read_text())
    aggregate = data.get("aggregate") or {}
    config = data.get("config") or {}

    avg_der = aggregate.get("avg_der")
    speaker_acc = aggregate.get("speaker_count_accuracy")
    pyannote_mode = config.get("pyannote_mode")

    errors: list[str] = []

    if not _is_finite(avg_der) or not (0.0 <= float(avg_der) <= 1.5):
        errors.append(f"avg_der out of range: {avg_der}")

    if not _is_finite(speaker_acc):
        errors.append(f"speaker_count_accuracy not finite: {speaker_acc}")
    elif float(speaker_acc) < 1.0:
        errors.append(f"speaker_count_accuracy dropped: {speaker_acc}")

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
