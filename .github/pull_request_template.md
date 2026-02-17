<!--
Thanks for the PR!

House rules:
- If a section doesn't apply, write "N/A" (don't delete headings).
- CI may be off/rate-limited. The local gate is canonical.
- Prefer links/paths to receipts over narrative claims.
-->

## What changed
- …

## Why
- …

Closes #

## How to review
- **Focus:** <paths/modules>
- **Key files:** <2–5 files that carry the behavior>
- **Out of scope:** <explicit non-goals / deferred work / follow-up issues>

## How to validate (local)
```bash
./scripts/ci-local.sh fast
# optional (full):
./scripts/ci-local.sh
# optional (nix checks; use wrapper inside devshell):
nix-clean flake check
```

## Local Gate Receipts

<!-- REQUIRED: Paste output or link to receipt file -->

```
# Paste receipt here
```

## Known limits / follow-ups

- …

<details>
<summary><strong>Optional: audit addendum (for large/risky PRs)</strong></summary>

### Receipts (links/paths)

- Tests:
- Benchmarks:
- Security/Policy:
- Docs/schema:

### Cost & attention (estimate)

- **DevLT:** author ~__m, review ~__m
- **Machine spend:** ~$__ (or `unknown`)

### What was wrong / surprises

- None

<!-- or: -->
<!-- - [description] → [disposition: fixed here | fixed in #X | still open] -->

</details>

## Checklist

<!-- Complete before requesting review -->

- [ ] I have run `./scripts/ci-local.sh fast` and all checks pass locally
- [ ] My code follows the project's style guidelines
- [ ] I have updated documentation as needed
- [ ] This PR introduces no breaking changes to public API, CLI, or JSON schema
