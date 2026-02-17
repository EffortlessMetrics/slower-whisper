# Development Scripts

This directory contains maintenance and verification scripts used by maintainers and contributors.

Most users do not need these scripts for normal `slower-whisper` runtime usage.

## Primary Local Gates

| Script | Purpose |
|--------|---------|
| `ci-local.sh` | Canonical local gate (full/fast modes) |
| `verify_all.py` / `verify_all.sh` | Aggregate verification entry points |
| `verify_bdd.sh` / `verify_bdd_legacy.sh` | BDD-focused verification |

Typical usage:

```bash
./scripts/ci-local.sh
./scripts/ci-local.sh fast
```

## Packaging and Entry-Point Checks

| Script | Purpose |
|--------|---------|
| `test_entry_points.py` | Programmatic validation of installed entry points |
| `test_entry_points.sh` | Shell wrapper for entry-point checks |
| `regenerate-requirements.sh` | Regenerate pinned requirements files |

## Benchmark/Dataset Tooling

| Script | Purpose |
|--------|---------|
| `setup_benchmark_datasets.py` | Stage and validate benchmark datasets |
| `download_datasets.py` / `fetch_datasets.py` | Dataset download helpers |
| `download_librispeech.sh` | LibriSpeech convenience download |
| `validate_gold_labels.py` | Validate semantic gold labels |
| `verify_ami_setup.sh` / `fix_ami_annotations.py` | AMI setup and annotation repair helpers |
| `check-calibration.py` | Calibration checks for audit artifacts |
| `check_model_cache.py` | Verify local model cache state |

## Documentation Tooling

| Script | Purpose |
|--------|---------|
| `verify_code_examples.py` | Validate executable Python snippets in docs |
| `render-doc-snippets.py` | Regenerate snippet fixtures used in docs |
| `docs-build.sh` | Build documentation |
| `generate-roadmap-status.py` | Regenerate roadmap status summaries |
| `generate-pr-ledger.py` | Generate audit/ledger artifacts for PR analysis |

## Deployment and Ops Helpers

| Script | Purpose |
|--------|---------|
| `docker_smoke_test.sh` | Smoke test Docker build/runtime |
| `validate_k8s.sh` | Validate Kubernetes manifests |
| `setup-env.sh` | Local environment bootstrap |
| `dogfood.sh` | Dogfooding workflow runner |
| `collect-telemetry.py` | Collect telemetry artifacts |
| `ci-benchmarks.sh` | Benchmark-oriented CI helper |

## Notes

- Prefer `./scripts/ci-local.sh` as the default contributor gate.
- Many scripts assume execution from repo root.
- Some scripts require optional dependencies or external tools (`gh`, Docker, kubectl, dataset credentials).

For contributor policy and required checks, see [../CONTRIBUTING.md](../CONTRIBUTING.md) and [../CLAUDE.md](../CLAUDE.md).
