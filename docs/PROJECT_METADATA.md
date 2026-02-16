# Project Metadata and Governance

This document lists the canonical metadata files for repository governance, packaging, and contributor support.

---

## Canonical Metadata Files

| File | Purpose | Consumers |
|------|---------|-----------|
| [`.github/settings.yml`](../.github/settings.yml) | Repository settings as code (description, topics, labels, branch protection) | GitHub Settings automation, maintainers |
| [`.github/CODEOWNERS`](../.github/CODEOWNERS) | Code owner routing for review enforcement | GitHub pull request review rules |
| [`.github/FUNDING.yml`](../.github/FUNDING.yml) | Sponsorship links shown in GitHub UI | GitHub Sponsors UI |
| [`.github/SUPPORT.md`](../.github/SUPPORT.md) | User support channels and escalation paths | GitHub support links, contributors |
| [`CITATION.cff`](../CITATION.cff) | Citation metadata for software reuse | GitHub citation UI, researchers |
| [`pyproject.toml`](../pyproject.toml) | Package metadata (version, classifiers, URLs, extras) | PyPI, installers, downstream tooling |
| [`SECURITY.md`](../SECURITY.md) | Vulnerability reporting and response policy | Security reporters, users |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Contribution workflow and quality bar | Contributors |

---

## GitHub Settings Contract

The repository-level source of truth is [`.github/settings.yml`](../.github/settings.yml).

Current policy encoded there:

- Default branch is `main`
- Merge strategy is squash-only (`allow_merge_commit: false`, `allow_rebase_merge: false`)
- Branch deletion after merge is enabled
- `main` requires status checks:
  - `CI Success`
  - `Verify (quick)`
- `main` requires one approving review and CODEOWNERS review
- Labels are standardized for triage (`bug`, `documentation`, `dependencies`, `needs-receipts`, etc.)

When CI check names change, update this file in the same PR as the workflow change.

---

## Metadata Sync Checklist

When release or governance metadata changes, keep these aligned:

1. `pyproject.toml` project metadata (`version`, classifiers, URLs)
2. `CITATION.cff` release metadata (`version`)
3. `ROADMAP.md` headers (`Current Version`, `Last Updated`)
4. `CHANGELOG.md` release heading and date
5. `transcription.__version__` behavior (metadata-derived; no manual bump)
6. `README.md` badges and policy links
7. `docs/INDEX.md` navigation entries
8. `.github/settings.yml` status checks and labels

---

## Update Workflow

1. Edit metadata files in one PR so reviewers can validate consistency.
2. Run local checks relevant to changed surfaces:
   - `./scripts/ci-local.sh fast`
   - `pytest -m "not heavy"` (if behavior changed)
3. Include receipts in the PR description and link to changed metadata files.

This keeps project governance reproducible and auditable.
