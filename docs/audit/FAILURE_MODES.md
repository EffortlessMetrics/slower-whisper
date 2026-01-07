# Failure Modes Taxonomy

This document catalogs known failure modes — classes of problems that can slip through review and cause issues later. Each entry includes detection mechanisms and prevention strategies.

## How to Use This Document

When you encounter a failure, check if it matches an existing mode. If not, add a new entry. This creates a living record that helps prevent repeat failures.

---

## Measurement Drift

**Type:** measurement drift

**Symptom:** Performance claims don't match reality; comparisons are invalid

**Root cause:** Baseline and current measurement use different semantics (normalization, dataset, config)

**Detection:**
- Receipt mismatch: `dataset_manifest_sha256` differs between runs
- Config hash differs without explicit acknowledgment
- Manual review catches "apples to oranges" comparisons

**Prevention:**
- Benchmark result schema requires `dataset_manifest_sha256` in receipt
- Regression comparator validates semantic compatibility before comparing
- PR template prompts for measurement validity statement

**Example:** PR claimed "2x speedup" but baseline used different normalization

---

## Doc Drift

**Type:** doc drift

**Symptom:** Documentation claims features/APIs that don't exist, or describes behavior that changed

**Root cause:** Code changed but docs weren't updated; docs were written speculatively

**Detection:**
- Manual testing of doc examples fails
- Grep for API names that no longer exist
- Users report "example doesn't work"

**Prevention:**
- BDD scenarios as executable documentation
- Doc examples in `tests/` so they're validated
- PR checklist includes "update docs if behavior changed"

**Example:** README showed `nix flake check` but devshell required `nix-clean` wrapper

---

## Packaging Drift

**Type:** packaging drift

**Symptom:** Package installs but runtime fails due to missing deps or version conflicts

**Root cause:** Dependency added to code but not to pyproject.toml; version pinning too loose

**Detection:**
- Fresh install fails on CI or clean machine
- Import errors at runtime
- User reports "works on your machine, not mine"

**Prevention:**
- Nix devshell catches system dep mismatches
- CI runs on fresh environment
- `uv sync` validates lockfile consistency

**Example:** New feature used `httpx` but it wasn't added to dependencies

---

## Test Flake

**Type:** test flake

**Symptom:** Test passes sometimes, fails other times, with same code

**Root cause:** Non-deterministic behavior (timing, ordering, external deps, random seeds)

**Detection:**
- CI shows intermittent failures
- `pytest -x` passes but `pytest` fails
- Adding `time.sleep()` "fixes" the test

**Prevention:**
- Mark flaky tests with `@pytest.mark.flaky` and document root cause
- Use deterministic seeds in tests
- Mock external dependencies

**Example:** Streaming test failed when machine was under load

---

## Dependency Hazard

**Type:** dependency hazard

**Symptom:** Update to dependency breaks functionality without obvious error

**Root cause:** Relying on undocumented behavior; version range too wide

**Detection:**
- CI fails after renovate/dependabot update
- Manual `uv sync --upgrade` breaks tests
- Security scan triggers investigation that reveals breakage

**Prevention:**
- Pin major versions
- Test against dep upgrade in separate CI job
- Document version assumptions in comments

**Example:** numpy 2.0 changed dtype behavior, broke audio processing

---

## Process Mismatch

**Type:** process mismatch

**Symptom:** PR follows wrong workflow; review misses critical issues

**Root cause:** CONTRIBUTING.md unclear; reviewer assumed different process

**Detection:**
- Post-merge issues discovered
- PR merged without required receipts
- Behavioral contract broken without discussion

**Prevention:**
- PR template enforces checklist
- BDD scenarios block merge if broken
- CONTRIBUTING.md updated when process changes

**Example:** PR merged without local gate receipt; broke on CI later

---

## Template

Use this template to add new failure modes:

```markdown
## [Name]

**Type:** [measurement drift | doc drift | packaging drift | test flake | dependency hazard | process mismatch | other]

**Symptom:** [What was observed]

**Root cause:** [What actually happened]

**Detection:**
- [How it was/could be caught]

**Prevention:**
- [What gate/contract change prevents this]

**Example:** [Specific instance, with PR/issue link if available]
```
