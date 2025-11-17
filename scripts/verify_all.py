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
    uv run python scripts/verify_all.py --full
    uv run slower-whisper-verify --quick  # if installed as script
"""

from __future__ import annotations

import argparse
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
) -> int:
    """Run a subprocess command and stream output."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        check=False,
    )
    if result.returncode != 0 and not allow_failure:
        raise SystemExit(result.returncode)
    return result.returncode


def check_ruff() -> None:
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("1️⃣  Code quality (ruff)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    run(["uv", "run", "ruff", "check", "transcription/", "tests/"])


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


def verify_bdd() -> None:
    """Run BDD scenarios (Gherkin) via pytest-bdd."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("3️⃣  BDD acceptance scenarios")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print(f"✅ ffmpeg found at {ffmpeg}")
    else:
        print("⚠️  ffmpeg not found; BDD scenarios that require audio may xfail/skip")

    # Run BDD scenarios from tests/steps/ directory
    run(["uv", "run", "pytest", "tests/steps/", "-v"])


def docker_smoke() -> None:
    """Build and smoke-test Docker images."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("4️⃣  Docker smoke tests")
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
            "slower-whisper",
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
    print("5️⃣  Kubernetes manifest validation")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if shutil.which("kubectl") is None:
        print("⚠️  kubectl not found; skipping K8s validation")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run slower-whisper verification checks.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip Docker and K8s checks; run only code + tests + BDD.",
    )
    args = parser.parse_args(argv)

    check_ruff()
    run_tests_fast()
    verify_bdd()

    if not args.quick:
        docker_smoke()
        validate_k8s()

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✅ All verification steps completed")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
