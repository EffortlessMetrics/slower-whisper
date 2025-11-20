# Release Checklist (Nix-First)

This checklist assumes:

- You're using **Nix** as the primary dev environment (`nix develop`).
- CI is wired through `flake.nix` + `.github/workflows/ci-nix.yml`.
- Evaluation harnesses for ASR/diarization are in place.

Goal: if you complete this checklist, you can cut a tag and publish without
surprises.

---

## 0. Preconditions

- [ ] Working on `main` (or a release branch) and **clean git status**:

```bash
git status
```

- [ ] No half-baked local experiments (use feature branches if needed).
- [ ] `HF_TOKEN` is set in your shell and in GitHub repo secrets:

```bash
export HF_TOKEN=hf_...
```

---

## 1. Environment & CI Sanity

### 1.1 Enter Nix dev shell

- [ ] Start from the repo root:

```bash
nix develop
```

### 1.2 Ensure Python deps are synced

- [ ] Install / refresh Python deps (once per environment):

```bash
uv sync --extra full --extra diarization --extra dev
```

### 1.3 Run full Nix CI locally

- [ ] Run the full flake checks:

```bash
nix flake check
```

This runs all of:

- `lint`
- `format`
- `type-check`
- `test-fast`
- `test-integration`
- `bdd-library`
- `bdd-api`
- `verify`
- `dogfood-smoke`

If this fails, fix it **before** going further.

---

## 2. Evaluation Gates (Quality Baselines)

This is the "does the engine behave as expected?" section.

### 2.1 LibriSpeech – ASR / WER Baseline

**Purpose:** ensure Whisper integration hasn't regressed on clean speech.

Prereq: `dev-clean` staged under
`~/.cache/slower-whisper/benchmarks/librispeech/LibriSpeech/dev-clean`.

- [ ] Run WER eval on a decent sample size (e.g. 50):

```bash
uv run python benchmarks/eval_asr_diarization.py \
  --dataset librispeech \
  --n 50 \
  --model base \
  --device cpu \
  --output benchmarks/results/asr_librispeech_dev_clean_50.json
```

- [ ] Inspect aggregate:

```bash
cat benchmarks/results/asr_librispeech_dev_clean_50.json | jq '.aggregate'
```

- [ ] Confirm:

  - [ ] `avg_WER` in a reasonable range for `base+cpu` (e.g. ~0.08–0.15).
  - [ ] No obvious blow-ups (e.g. `avg_WER` > 0.25 → investigate before release).

*(Optional)* If you want a "showpiece" metric for the release notes, you can also
run:

```bash
# GPU + large-v3, if available
uv run python benchmarks/eval_asr_diarization.py \
  --dataset librispeech \
  --n 50 \
  --model large-v3 \
  --device cuda \
  --output benchmarks/results/asr_librispeech_dev_clean_50_large_cuda.json
```

---

### 2.2 AMI – WER + DER (Diarization Smoke)

**Purpose:** ensure diarization + speaker assignment are working on real speech.

Prereq: at least one AMI-style fixture staged under
`~/.cache/slower-whisper/benchmarks/ami/audio` and
`annotations` (e.g. `LS_TEST001`, or real AMI meetings once available).

- [ ] Run AMI eval on 1–2 fixtures:

```bash
uv run python benchmarks/eval_asr_diarization.py \
  --dataset ami \
  --n 2 \
  --model base \
  --device cpu \
  --output benchmarks/results/asr_diar_ami_smoke.json
```

- [ ] Inspect:

```bash
cat benchmarks/results/asr_diar_ami_smoke.json | jq '.aggregate, .samples'
```

- [ ] Confirm for each evaluated sample:

  - [ ] `WER` is reasonable given the fixture (for clean Libri-like speech, expect
    ~5–15%; for real AMI, may be higher).
  - [ ] `num_speakers_detected` matches the annotation (e.g. 1 for LS_TEST001; 2+
    for real meetings).
  - [ ] `DER` is not pathological (e.g. <0.3 for simple fixtures; investigate if
    it's ~1.0 across the board).

If this gate fails, fix diarization / annotations before shipping.

---

### 2.3 Dogfood Smoke (Pipeline Wiring)

**Purpose:** ensure the end-to-end pipeline and CLI wiring behave sanely.

- [ ] From `nix develop`, run:

```bash
# Nix app
nix run .#dogfood -- --sample synthetic --skip-llm
```

Confirm:

- [ ] Command completes without error.
- [ ] Transcripts & JSON are produced under `whisper_json/` and `transcripts/`.
- [ ] `scripts/diarization_stats.py` output looks coherent (speaker counts, turns).

*(Optional)* Once HF fixtures are in place:

```bash
nix run .#dogfood -- --sample LS_TEST001 --skip-llm
```

---

## 3. Schema & Docs Consistency

### 3.1 Schema stability

- [ ] Confirm **JSON schema** version is correct and unchanged for patch/minor
  releases:

  - `transcription/schemas/transcript-v2.schema.json`

    - Check `schema_version` and diarization fields (`speakers`, `turns`,
      `segment.speaker`).

- [ ] If making a **breaking change** to the schema:

  - [ ] Bump schema version (e.g. v2 → v3).
  - [ ] Update schema docs (e.g. ARCHITECTURE.md, SPEAKER_DIARIZATION.md).
  - [ ] Note the breaking change in CHANGELOG.

### 3.2 Documentation

- [ ] Update `CHANGELOG.md` with a new section for this version:

  - Features added
  - Fixes
  - Notable metrics (e.g. "LibriSpeech dev-clean WER ≈ 0.11 for `base+cpu`")

- [ ] Ensure user-facing docs are consistent:

  - [ ] `README.md` (install paths, Nix-first story, CLI sections)
  - [ ] `docs/LOCAL_EVAL_WORKFLOW.md`
  - [ ] `docs/DEV_ENV_NIX.md`
  - [ ] `docs/CI_CHECKS.md`
  - [ ] Any evaluation result docs you want to reference (e.g.
    `docs/LIBRISPEECH_EVAL_RESULTS.md`, AMI eval notes).

---

## 4. Version Bump & Tagging

### 4.1 Bump version

- [ ] Update version in `pyproject.toml`:

```toml
[project]
version = "x.y.z"
```

- [ ] Update version in package:

```python
# transcription/__init__.py
__version__ = "x.y.z"
```

### 4.2 Final CI run

- [ ] Push your branch and confirm **all CI jobs**, including `ci-nix.yml`, are
  green on GitHub.

---

## 5. Release Artifacts

### 5.1 Git tag

- [ ] Create a signed (or annotated) tag:

```bash
git commit -am "release: vX.Y.Z"
git tag -a vX.Y.Z -m "slower-whisper vX.Y.Z"
git push origin main
git push origin vX.Y.Z
```

### 5.2 GitHub Release

- [ ] Create a GitHub Release for tag `vX.Y.Z`:

  - [ ] Title: `vX.Y.Z – <short one-liner>`
  - [ ] Body:

    - Highlights
    - Links to evaluation metrics (LibriSpeech/AMI)
    - Notes about any experimental features (e.g. diarization status)

### 5.3 PyPI (if publishing)

*(Only if you're ready for PyPI; optional for private pre-release)*

- [ ] Build:

```bash
uv build
```

- [ ] Check:

```bash
uv run twine check dist/*
```

- [ ] Upload:

```bash
uv run twine upload dist/*
```

---

## 6. Post-Release Notes

- [ ] Capture a one-pager of **what this release proves**:

  - E.g. "v1.1.0: Speaker diarization experimental, WER ≈ 0.11 on Libri dev-clean,
    AMI smoke tests passing on fixtures LS_TEST001/…".

- [ ] Copy key metrics into:

  - `RELEASE_NOTES.md` or GitHub Release body
  - Any talk/blog post prep you're doing

---

If you follow this checklist end-to-end, you've:

- Verified code quality via Nix-native CI.
- Re-confirmed ASR and diarization behavior against real data.
- Ensured schema and docs are consistent.
- Cut a tag with a clear story and known metrics.

That's "b" done in a way that Future-You (and any contributor) can execute reliably.
