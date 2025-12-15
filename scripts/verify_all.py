#!/usr/bin/env python3
"""
Master verification script for slower-whisper.

Runs a suite of checks to validate:

- Code quality (ruff)
- Unit/fast tests (pytest)
- BDD acceptance scenarios (pytest-bdd)
- Docker image build + CLI smoke tests
- Kubernetes manifest validation (kubectl dry-run)

Usage:
    uv run python scripts/verify_all.py --quick
    uv run python scripts/verify_all.py --api
    uv run slower-whisper-verify --quick  # if installed as script
    uv run slower-whisper-verify --api
    uv run slower-whisper-verify  # full suite (includes Docker/K8s)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    allow_failure: bool = False,
    env: dict[str, str] | None = None,
) -> int:
    """Run a subprocess command and stream output."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        check=False,
        env=env,
    )
    if result.returncode != 0 and not allow_failure:
        raise SystemExit(result.returncode)
    return result.returncode


def check_ruff() -> None:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("1️⃣  Code quality (ruff)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    run(["uv", "run", "ruff", "check", "transcription/", "tests/"])


def check_mypy() -> None:
    """Run mypy on transcription/ and strategic test modules."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("1.5️⃣  Type checking (mypy)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    run(
        [
            "uv",
            "run",
            "mypy",
            "transcription/",
            "tests/test_llm_utils.py",
            "tests/test_writers.py",
            "tests/test_turn_helpers.py",
            "tests/test_audio_state_schema.py",
        ]
    )


def run_tests_fast() -> None:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("2️⃣  Unit / fast tests (pytest)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    run(
        [
            "uv",
            "run",
            "pytest",
            "-m",
            "not slow and not requires_gpu",
            "--cov=transcription",
            "--cov-report=term-missing",
        ]
    )


def validate_schema_samples() -> None:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("3️⃣  Schema validation (transcripts)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    sample_paths = [
        ROOT / "benchmarks/data/samples/sample_transcript.json",
    ]
    existing = [p for p in sample_paths if p.exists()]
    if not existing:
        print("⚠️  No sample transcripts found; skipping schema validation")
        return

    try:
        from transcription.validation import DEFAULT_SCHEMA_PATH, validate_many
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Skipping schema validation (dependency missing: {exc})")
        return

    failures = validate_many(existing, schema_path=DEFAULT_SCHEMA_PATH)
    if failures:
        print("❌ Schema validation failed:")
        for err in failures:
            print(f"- {err}")
        raise SystemExit(1)

    for path in existing:
        print(f"Schema validation: OK ({path})")
    print(f"✅ {len(existing)} transcript(s) valid against {DEFAULT_SCHEMA_PATH}")


def check_diarization_stub() -> None:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("3.5️⃣  Diarization stub regression check")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    run(
        [
            "uv",
            "run",
            "python",
            "benchmarks/check_diarization_stub.py",
        ]
    )


def eval_diarization_real() -> None:
    """Optional: run real diarization eval (pyannote) without gating builds."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("3.9️⃣  Diarization evaluation (real backend, optional)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    has_hf_token = bool(os.getenv("HF_TOKEN"))
    has_pyannote = importlib.util.find_spec("pyannote.audio") is not None
    dataset = ROOT / "benchmarks/data/diarization"
    manifest = dataset / "manifest.jsonl"
    output_md = ROOT / "benchmarks/DIARIZATION_REPORT_REAL.md"
    output_json = ROOT / "benchmarks/DIARIZATION_REPORT_REAL.json"

    if not has_pyannote:
        print("⚠️  pyannote.audio not installed; skipping real diarization eval")
        return
    if not has_hf_token:
        print("⚠️  HF_TOKEN not set; skipping real diarization eval")
        return
    if not dataset.exists() or not manifest.exists():
        print(f"⚠️  Missing diarization fixtures at {dataset}; skipping")
        return

    previous: dict[str, float] | None = None
    if output_json.exists():
        try:
            import json

            prev_data = json.loads(output_json.read_text())
            previous = prev_data.get("aggregate") or {}  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Unable to read previous DIARIZATION_REPORT_REAL.json: {exc}")

    env = os.environ.copy()
    env.setdefault("SLOWER_WHISPER_PYANNOTE_MODE", "auto")

    rc = run(
        [
            "uv",
            "run",
            "python",
            "benchmarks/eval_diarization.py",
            "--dataset",
            str(dataset),
            "--manifest",
            str(manifest),
            "--device",
            "cpu",
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
            "--overwrite",
        ],
        allow_failure=True,
        env=env,
    )
    if rc != 0:
        print(f"⚠️  Real diarization eval exited with {rc} (non-blocking)")
        return

    if not output_json.exists():
        print("⚠️  Real diarization output JSON missing; nothing to report")
        return

    try:
        import json

        data = json.loads(output_json.read_text())
        aggregate = data.get("aggregate") or {}
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Unable to read DIARIZATION_REPORT_REAL.json: {exc}")
        return

    avg_der = aggregate.get("avg_der")
    speaker_acc = aggregate.get("speaker_count_accuracy")
    num_samples = aggregate.get("num_samples")
    manifest_hash = data.get("manifest_hash")
    print(
        f"Real diarization eval: avg_der={avg_der}, "
        f"speaker_count_accuracy={speaker_acc}, samples={num_samples}, "
        f"manifest={manifest_hash}"
    )

    if previous:
        try:
            prev_der = previous.get("avg_der")
            prev_acc = previous.get("speaker_count_accuracy")
            delta_der = (
                float(avg_der) - float(prev_der)
                if avg_der is not None and prev_der is not None
                else None
            )
            delta_acc = (
                float(speaker_acc) - float(prev_acc)
                if speaker_acc is not None and prev_acc is not None
                else None
            )
            if delta_der is not None or delta_acc is not None:
                print("Diff vs previous DIARIZATION_REPORT_REAL.json:")
                if delta_der is not None:
                    print(f"  • avg_der: {prev_der} -> {avg_der} (Δ {delta_der:+.4f})")
                if delta_acc is not None:
                    print(
                        f"  • speaker_count_accuracy: {prev_acc} -> {speaker_acc} "
                        f"(Δ {delta_acc:+.4f})"
                    )
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Unable to diff against previous report: {exc}")


def verify_bdd() -> None:
    """Run BDD scenarios (Gherkin) via pytest-bdd."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("4️⃣  BDD acceptance scenarios (library)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print(f"✅ ffmpeg found at {ffmpeg}")
    else:
        print("⚠️  ffmpeg not found; BDD scenarios that require audio may xfail/skip")

    # Run BDD scenarios from tests/steps/ directory
    run(["uv", "run", "pytest", "tests/steps/", "-v"])


def verify_api_surface() -> None:
    """Run API-focused unit/integration tests and BDD scenarios."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("API surface (FastAPI + clients)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    missing = []
    for dep in ("fastapi", "httpx", "uvicorn"):
        if importlib.util.find_spec(dep) is None:
            missing.append(dep)
    if missing:
        deps = ", ".join(sorted(missing))
        print(f"⚠️  Missing API dependencies: {deps}; skipping API checks")
        print("   Install with: uv sync --extra api --extra dev")
        return

    print("\n🧪 Running API unit/integration tests...")
    run(
        [
            "uv",
            "run",
            "pytest",
            "tests/test_api_service.py",
            "tests/test_api_integration.py",
            "-m",
            "not slow and not requires_gpu",
            "-v",
        ]
    )

    verify_api_bdd()


def verify_api_bdd() -> None:
    """Run API BDD scenarios (REST service black-box tests)."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("5️⃣  BDD acceptance scenarios (API service)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Ensure FastAPI is available before attempting to start the service.
    try:
        import fastapi  # noqa: F401

        print("✅ fastapi available")
    except ImportError as exc:
        print(f"⚠️  fastapi import failed ({exc}); skipping API BDD tests")
        print("   Install with: uv sync --extra api --extra dev")
        return

    # Check for httpx (required for API tests)
    try:
        import httpx  # noqa: F401

        print("✅ httpx available")
    except ImportError:
        print("⚠️  httpx not installed; skipping API BDD tests")
        print("   Install with: uv sync --extra api --extra dev")
        return

    # Check for uvicorn (required to run the service)
    if shutil.which("uvicorn") is None:
        print("⚠️  uvicorn not found; skipping API BDD tests")
        print("   Install with: uv sync --extra api --extra dev")
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print(f"✅ ffmpeg found at {ffmpeg}")
    else:
        print("⚠️  ffmpeg not found; API transcription scenarios may skip")

    # Run API BDD scenarios from features/ directory
    print("\n🚀 Starting API service for BDD tests...")
    print("   (service will auto-start/stop via pytest fixtures)")
    run(["uv", "run", "pytest", "features/", "-v", "-m", "api"])


def docker_smoke() -> None:
    """Build and smoke-test Docker images."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("6️⃣  Docker smoke tests")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if shutil.which("docker") is None:
        print("⚠️  docker not found; skipping Docker smoke tests")
        return

    # CPU image
    print("\n🐋 Building CPU image...")
    run(
        [
            "docker",
            "build",
            "-t",
            "slower-whisper:test-cpu",
            "-f",
            "Dockerfile",
            ".",
        ]
    )
    print("✅ CPU image built")

    print("\nTesting CLI in CPU image...")
    run(
        [
            "docker",
            "run",
            "--rm",
            "slower-whisper:test-cpu",
            "--help",
        ]
    )
    print("✅ CLI help works in CPU image")

    # API image
    print("\n🐋 Building API image...")
    run(
        [
            "docker",
            "build",
            "-t",
            "slower-whisper:test-api",
            "-f",
            "Dockerfile.api",
            ".",
        ]
    )
    print("✅ API image built (endpoint-level tests are optional)")

    print("\n✅ Docker smoke tests complete")


def validate_k8s() -> None:
    """Validate Kubernetes manifests via kubectl dry-run=client."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("7️⃣  Kubernetes manifest validation")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if shutil.which("kubectl") is None:
        print("⚠️  kubectl not found; skipping K8s validation")
        return

    context_rc = run(
        [
            "kubectl",
            "config",
            "current-context",
        ],
        allow_failure=True,
    )
    if context_rc != 0:
        print("⚠️  kubectl context unavailable; skipping K8s validation")
        return

    manifests = sorted(ROOT.glob("k8s/*.yaml"))
    if not manifests:
        print("⚠️  No k8s/*.yaml manifests found; skipping")
        return

    for manifest in manifests:
        print(f"Validating {manifest.name}...")
        rc = run(
            [
                "kubectl",
                "apply",
                "--dry-run=client",
                "--validate=false",
                "-f",
                str(manifest),
            ],
            allow_failure=True,
        )
        if rc != 0:
            print(f"❌ {manifest.name} failed validation")
            raise SystemExit(rc)
        print(f"  ✅ {manifest.name} OK")

    print("\n✅ All Kubernetes manifests validate")


def feature_summary() -> None:
    """Print a short status summary for analytics + diarization evidence."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Feature summary")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    try:
        from transcription.config import EnrichmentConfig, TranscriptionConfig
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Skipping feature summary (config import failed: {exc})")
        return

    enrich_cfg = EnrichmentConfig.from_env()
    analytics_enabled = enrich_cfg.enable_turn_metadata and enrich_cfg.enable_speaker_stats
    analytics_bits = []
    if enrich_cfg.enable_turn_metadata:
        analytics_bits.append("turn metadata")
    if enrich_cfg.enable_speaker_stats:
        analytics_bits.append("speaker_stats")
    analytics_detail = " + ".join(analytics_bits) if analytics_bits else "off"
    print(
        f"Speaker analytics: {'enabled' if analytics_enabled else 'disabled'} ({analytics_detail})"
    )

    trans_cfg = TranscriptionConfig.from_env()
    diar_requested = bool(trans_cfg.enable_diarization)
    try:
        has_pyannote = importlib.util.find_spec("pyannote.audio") is not None
    except ValueError:
        # Some mocked modules may not set __spec__; treat as missing
        has_pyannote = False
    has_hf_token = bool(os.getenv("HF_TOKEN"))
    if not diar_requested:
        diar_status = "disabled"
    elif has_pyannote and has_hf_token:
        diar_status = "ready"
    elif has_pyannote:
        diar_status = "installed (HF_TOKEN missing)"
    else:
        diar_status = "missing dependency"
    print(f"Diarization: requested={'yes' if diar_requested else 'no'}, status={diar_status}")

    speaker_report = Path("benchmarks/SPEAKER_ANALYTICS_MVP.md")
    diar_report = Path("benchmarks/DIARIZATION_REPORT.md")
    speaker_msg = (
        f"available at {speaker_report}"
        if speaker_report.exists()
        else "missing (benchmarks/SPEAKER_ANALYTICS_MVP.md)"
    )
    diar_msg = (
        f"available at {diar_report}"
        if diar_report.exists()
        else "missing (benchmarks/DIARIZATION_REPORT.md)"
    )
    print(f"Speaker analytics MVP harness: {speaker_msg}")
    print(f"Diarization harness: {diar_msg}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run slower-whisper verification checks.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="Skip Docker and K8s checks; run only code + tests + BDD.",
    )
    mode_group.add_argument(
        "--api",
        action="store_true",
        help="Run API-only checks (FastAPI unit/integration + BDD).",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip API BDD tests (useful if httpx/uvicorn not installed).",
    )
    parser.add_argument(
        "--eval-diarization",
        action="store_true",
        help="Optionally run real diarization eval (pyannote, requires HF_TOKEN); non-blocking.",
    )
    args = parser.parse_args(argv)

    if args.api and args.skip_api:
        parser.error("--api and --skip-api cannot be used together.")

    if args.api:
        if args.eval_diarization:
            eval_diarization_real()
        verify_api_surface()
        feature_summary()
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("✅ API verification completed")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return 0

    check_ruff()
    check_mypy()
    run_tests_fast()
    validate_schema_samples()
    check_diarization_stub()
    if args.eval_diarization:
        eval_diarization_real()
    verify_bdd()

    if not args.skip_api:
        verify_api_bdd()

    if not args.quick:
        docker_smoke()
        validate_k8s()

    feature_summary()
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✅ All verification steps completed")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
