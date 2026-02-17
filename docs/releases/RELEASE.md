# RELEASE.md

Concise release playbook for slower-whisper. Follow this alongside the detailed
checklists in `releases/RELEASE_CHECKLIST.md`, `releases/FINAL_VERIFICATION_CHECKLIST.md`,
and `../RELEASE_CHECKLIST_NIX.md`.

## Quick preflight

- Run sanity checks: `./scripts/ci-local.sh fast` (or `make verify-quick`)
- Ensure changelog is current: `CHANGELOG.md`
- Confirm version updates: `pyproject.toml`, `CITATION.cff`, and `ROADMAP.md`
- `transcription.__version__` is metadata-derived via `importlib.metadata`; do not bump it manually
- Docs and examples refreshed (README, docs/INDEX, examples/*)

## Release workflow

1. Prep a clean workspace

```bash
git status
make clean
```

1. Finalize version + changelog

- Set the new version in `pyproject.toml` and `CITATION.cff`.
- Update `ROADMAP.md` (`Current Version`, `Last Updated`) and release sections in `CHANGELOG.md`.
- Commit: `git commit -am "Release vX.Y.Z"` (or similar).

1. Run verification

```bash
make verify-quick               # Code quality + tests + BDD
# Optional: full suite with GPU/diarization + Docker/K8s
make verify
```

1. Build artifacts

```bash
rm -rf dist/ build/ *.egg-info
uv build
uv run twine check dist/*
```

1. Tag and push

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main vX.Y.Z
```

Pushing a `vX.Y.Z` tag triggers `.github/workflows/release.yml`, which publishes to PyPI and creates the GitHub release.

1. Optional: manual TestPyPI dry run (no production publish)

```bash
gh workflow run release.yml -f version=X.Y.Z-rc1
```

1. GitHub Release

- Create a release using tag `vX.Y.Z`.
- Title: `vX.Y.Z - <short headline>`.
- Paste highlights + link to `CHANGELOG.md`.
- Attach wheel/sdist if desired.

## Post-release

- Reopen `## [Unreleased]` in `CHANGELOG.md`.
- Update README badges if needed.
- Announce (GitHub Discussions, social, internal channels).
- Plan next milestone via `ROADMAP.md`.
