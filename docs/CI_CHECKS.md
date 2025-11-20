# CI Checks & Nix Integration

This document explains how the Nix flake–based CI checks map to the actual
commands under the hood, and when you should run which check locally.

The goal: if `nix flake check` passes on your laptop, you can assume the Nix CI
job in GitHub Actions will pass as well.

---

## Overview

CI is defined in two places:

- `flake.nix` — defines:
  - `devShells` (local dev environment)
  - `checks` (lint, tests, BDD, verify, dogfood, etc.)
  - `apps` (wrappers for dogfood/verify)
- `.github/workflows/ci-nix.yml` — runs `nix flake check` in CI, plus each
  check individually via a matrix, with `HF_TOKEN` and HuggingFace caching.

The flake uses a hybrid approach:

- **Nix** manages system dependencies: `ffmpeg`, `libsndfile`, Python, etc.
- **uv** manages Python dependencies: faster-whisper, torch, pytest, ruff, etc.

---

## Dev Shell & Global CI Command

### Enter the dev shell

From the repo root:

```bash
nix develop
```

Inside the shell:

```bash
# Install all Python deps (once per environment)
uv sync --extra full --extra diarization --extra dev

# Run full CI (all checks)
nix flake check
```

> If `nix flake check` passes here, `ci-nix.yml` should pass in GitHub Actions
> (assuming `HF_TOKEN` is configured in repo secrets).

---

## Checks: What They Do

The flake exposes the following checks under
`checks.<system>.<name>`, e.g. `checks.x86_64-linux.lint`.

You can build any check with:

```bash
nix build .#checks.x86_64-linux.<name>
```

### Summary Table

| Check name         | What it runs                                           | Cost / when to run                               |
| ------------------ | ------------------------------------------------------ | ------------------------------------------------ |
| `lint`             | `ruff` lint via `uv`                                   | Fast. Run before commits touching Python.        |
| `format`           | `ruff format --check` (no auto-fix)                    | Fast. Run before major PRs.                      |
| `type-check`       | `mypy` on `transcription/` and `tests/`                | Medium. Run when changing types/APIs.            |
| `test-fast`        | `pytest` fast suite (no slow/GPU marks)                | Medium. Run regularly during dev.                |
| `test-integration` | `pytest` integration tests                             | Medium/slow. Run before feature branches.        |
| `bdd-library`      | Library BDD (pytest-bdd scenarios)                     | Medium. Run when touching core behaviors.        |
| `bdd-api`          | API BDD smoke (REST service behavior)                  | Medium. Run before API-related changes.          |
| `verify`           | `slower-whisper-verify --quick` (verification spine)   | Medium/slow. Run before release / big refactors. |
| `dogfood-smoke`    | `slower-whisper-dogfood --sample synthetic --skip-llm` | Medium. Run when touching pipeline wiring.       |
| `ci-all`           | Aggregates all of the above via `test -f` dependencies | Slowest. Mirrors `nix flake check`.              |

### Examples

```bash
# Lint only
nix build .#checks.x86_64-linux.lint

# Formatting check only
nix build .#checks.x86_64-linux.format

# Fast tests
nix build .#checks.x86_64-linux.test-fast

# Full verification
nix build .#checks.x86_64-linux.verify

# Dogfood synthetic audio
nix build .#checks.x86_64-linux.dogfood-smoke

# Everything
nix flake check          # or:
nix build .#checks.x86_64-linux.ci-all
```

---

## Apps: `nix run` Helpers

The flake also exposes convenient apps under `apps` so you don't have to
remember the underlying `uv` commands.

### Dogfood

```bash
# Inside or outside nix develop (recommended inside)
nix run .#dogfood -- --sample synthetic --skip-llm
nix run .#dogfood -- --sample LS_TEST001 --skip-llm
```

These are thin wrappers around:

```bash
uv run slower-whisper-dogfood ...
```

They set `SLOWER_WHISPER_CACHE_ROOT` and use Nix's `uv` so you get the right
tooling automatically.

### Verify

```bash
# Quick verification
nix run .#verify -- --quick

# Full verification (if you define it in slower-whisper-verify)
nix run .#verify
```

These are thin wrappers for:

```bash
uv run slower-whisper-verify ...
```

---

## HF_TOKEN & HuggingFace Cache

Some checks (notably `verify` and `dogfood-smoke`) run diarization and require
access to pyannote models on HuggingFace.

### Local

Set `HF_TOKEN` in your environment:

```bash
export HF_TOKEN=hf_...
```

You should also accept the terms for:

* `pyannote/segmentation-3.0`
* `pyannote/speaker-diarization-3.1`
* `pyannote/speaker-diarization-community-1`

Then:

```bash
nix flake check
# or:
nix build .#checks.x86_64-linux.verify
nix build .#checks.x86_64-linux.dogfood-smoke
```

### CI

`.github/workflows/ci-nix.yml`:

* Passes `HF_TOKEN` from GitHub secrets into jobs.
* Caches `~/.cache/huggingface` keyed on `pyproject.toml`.

If you add or change model usage, update secrets as needed; no changes to the
flake are required.

---

## When to Run What (Practical Guidance)

### During normal feature work

* On small changes (docs, minor refactors):

  ```bash
  nix build .#checks.x86_64-linux.lint
  nix build .#checks.x86_64-linux.test-fast
  ```

* When touching CLI / API / core pipeline:

  ```bash
  nix build .#checks.x86_64-linux.lint
  nix build .#checks.x86_64-linux.test-fast
  nix build .#checks.x86_64-linux.bdd-library
  nix build .#checks.x86_64-linux.dogfood-smoke
  ```

### Before opening a PR

```bash
nix build .#checks.x86_64-linux.lint
nix build .#checks.x86_64-linux.format
nix build .#checks.x86_64-linux.test-fast
nix build .#checks.x86_64-linux.bdd-library
```

(Optionally add `test-integration` if you touched integration surfaces.)

### Before tagging a release / major refactor

```bash
export HF_TOKEN=hf_...

nix flake check          # or:
nix build .#checks.x86_64-linux.ci-all
```

This gives you:

* Lint + formatting
* Type-checking
* Fast + integration tests
* BDD (library + API)
* Verification suite
* Dogfood smoke

In other words: "this is as close to CI as you can get locally."

---

## FAQ

**Q: Do I still need to install uv manually?**
**A:** No. `uv` is provided by Nix in `systemDeps`. Inside `nix develop`, `uv` is on `PATH`.

**Q: Do I *have* to use Nix?**
**A:** No, but traditional setup is now treated as a fallback and may diverge from CI. For contributors and for anything CI-adjacent, Nix is the expected path.

**Q: Why do checks use temporary `$HOME`?**
**A:** To isolate CI runs and avoid leaking your personal cache into flake builds. For persistent speed locally, just run `uv sync` once in your dev shell and then call `pytest` / `ruff` directly.

---

This file is intentionally pragmatic: enough detail that you don't have to remember the flake internals, and clear mapping from "what I want to check" → "which `nix build` line to run."
