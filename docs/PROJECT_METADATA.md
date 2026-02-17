# Project Metadata and Governance

This document defines canonical metadata surfaces for packaging, citation, and repository governance.

## Canonical Metadata Files

| File | Purpose | Primary Consumers |
|------|---------|-------------------|
| [`pyproject.toml`](../pyproject.toml) | Python distribution metadata (name/version/extras/entry points/URLs) | PyPI, installers, tooling |
| [`MANIFEST.in`](../MANIFEST.in) | Source distribution include/exclude policy | `setuptools` sdist builds |
| [`CITATION.cff`](../CITATION.cff) | Citation metadata for software reuse | GitHub citation UI, researchers |
| [`README.md`](../README.md) | Long package description and top-level capability map | PyPI/GitHub users |
| [`CHANGELOG.md`](../CHANGELOG.md) | Release-level behavior and API changes | Users, maintainers |
| [`ROADMAP.md`](../ROADMAP.md) | Current status and planned work | Contributors, stakeholders |
| [`transcription/__init__.py`](../transcription/__init__.py) | Runtime version contract (`__version__` from package metadata) and public API exports | Runtime callers, tests |
| [`slower_whisper/__init__.py`](../slower_whisper/__init__.py) | Compatibility package metadata/export surface | `faster-whisper` migrators |
| [`.github/settings.yml`](../.github/settings.yml) | Repository settings as code (description/topics/labels/protection) | Maintainers, GitHub automation |
| [`.github/SUPPORT.md`](../.github/SUPPORT.md) | User support routing and escalation paths | Users, contributors |
| [`SECURITY.md`](../SECURITY.md) | Vulnerability reporting policy | Security reporters |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Contribution workflow and quality gates | Contributors |

## Python Surface Map (Crate-Equivalent View)

| Surface | Role |
|---------|------|
| `transcription` | Primary Python API package (transcription, enrichment, streaming, service helpers) |
| `slower_whisper` | Drop-in compatibility shim for `faster-whisper` imports |
| `transcription.integrations` + top-level `integrations` | RAG/framework adapters |
| `transcription.historian` | Audit/historian tooling and analyzers |
| `transcription.store` | Store schema/CLI/types for persistent conversation artifacts |
| `transcription.benchmark_cli` + `benchmarks/` | Benchmark tracks, baselines, and evaluation fixtures |
| `scripts/` | Maintainer verification, dataset, and release support scripts |

## Version and Metadata Sync Checklist

When preparing a release or metadata-only update, keep these aligned in the same PR:

1. `pyproject.toml`: `[project].version`, description/keywords/classifiers/URLs.
2. `CITATION.cff`: `version` and `date-released`.
3. `CHANGELOG.md`: release heading and date.
4. `README.md`: install matrix/docs links/feature claims.
5. `transcription.__version__` behavior: metadata-derived (no hardcoded release constant).
6. `slower_whisper.__version__` behavior: metadata-derived fallback contract.
7. `MANIFEST.in`: includes match real file layout.
8. `docs/INDEX.md` and any surface README files (`benchmarks/`, `scripts/`, `k8s/`) when behavior/contracts change.
9. `.github/settings.yml`: repository description/topics/status checks if governance changed.

## Update Workflow

1. Change metadata surfaces together in one PR so reviewers can verify consistency.
2. Run at least a fast local gate:

```bash
./scripts/ci-local.sh fast
```

3. For version/packaging changes, also run targeted checks:

```bash
pytest tests/test_versioning.py tests/test_compat_shim.py
```

4. Include command receipts in the PR description.

This keeps package metadata, runtime contracts, and public documentation synchronized.
