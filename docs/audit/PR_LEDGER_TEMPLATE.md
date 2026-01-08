# PR Ledger Template

Use this template for the "initial comment" on PRs. The ledger serves as the **best-available record** of:
- Intent + scope
- What actually shipped
- What went wrong / what was fixed / what remained
- Evidence (receipts)
- Costs (DevLT + machine spend)
- What we learned

---

## Template (copy this into PR description)

```markdown
## PR Ledger (updated YYYY-MM-DD)

### What this PR was
- **Goal:** [1-2 sentences]
- **Scope:** [in/out explicit]
- **Out of scope / deferred:** [explicit exclusions]
- **Primary surfaces:** `path/to/file.py`, `path/to/other.py`
- **Type:** feature | hardening | mechanization | perf/bench | refactor

### What shipped
- [3-7 bullets of concrete outcomes]
- Links: #___ (issue), docs: `path/to/doc.md`

### Evidence (receipts)
- **Local gate:** `./scripts/ci-local.sh fast` ✓ / ✗
- **Tests:** [added/updated count, coverage if tracked]
- **Type/lint:** mypy ✓, ruff ✓
- **Schema/docs sanity:** [validated or n/a]
- **Bench/perf (if relevant):** p50=Xms, p95=Yms, p99=Zms (baseline: commit abc1234)

### Process + friction

| Event | Detected by | Disposition | Prevention |
|-------|-------------|-------------|------------|
| [what broke] | [gate/review/post-merge] | [fixed here / deferred to #X] | [new gate/test/doc or none] |

- **Design alignment:** [where it drifted / who caught it / how resolved]
- **Measurement integrity:** [valid / invalid + link to correction]

### Cost & time (best-effort estimates)

| Metric | Value | Method |
|--------|-------|--------|
| Wall-clock | X days | `created_at → merged_at` |
| Active work | ~Xh | commit bursts heuristic |
| DevLT (author) | ~Xm | [band: 10-20m / 20-45m / 45-90m / >90m] |
| DevLT (review) | ~Xm | [band] |
| Machine spend | ~$X.XX | [source: claude session / api billing / unknown] |

### What went well
- [2-5 bullets]

### What could be better
- [2-5 bullets, actionable]

### Follow-ups
- #___ [description]
- #___ [description]
```

---

## Section Guidance

### What this PR was

Be explicit about scope boundaries. "Out of scope" is just as important as what's in scope.

**Type definitions:**
- `feature` — New functionality
- `hardening` — Resilience, error handling, edge cases
- `mechanization` — Automation, tooling, scripts
- `perf/bench` — Performance work or benchmarking
- `refactor` — Code restructuring without behavior change

### Evidence (receipts)

**Only claim what you can point to.** Acceptable evidence:

- Local gate output (paste or link to receipt file)
- Test summary from pytest
- Benchmark results with absolute numbers and baseline reference
- Schema validation output
- Doc sanity check (links work, examples run)

### Process + friction

This is where honesty matters most. Document:

- What broke during development
- How it was detected (which gate, review comment, post-merge)
- Whether it was fixed in this PR or deferred
- What prevention was added (or why none was needed)

### Cost & time

**Two clocks:**
- **Wall-clock:** PR opened → merged (from GH timestamps)
- **Active work:** Derived from commit bursts (gaps ≤45min = same session)

**DevLT estimation:**
- Use bands unless you truly measured: `~10-20m`, `~20-45m`, `~45-90m`, `>90m`
- Author = human attention during development
- Review = reviewer attention

**Machine spend:**
- Exact if you have it
- Band (`$0-$2`, `$2-$10`, `$10-$25`) if approximating
- `unknown` if you don't know (don't fake it)

### What went well / What could be better

Keep it short and actionable. This isn't therapy; it's process improvement.

### Follow-ups

Link to issues created or updated. Include explicit "deferred" items with rationale.

---

## Layered Usage

### Layer A: Review UI (always present)

For quick review, the minimum viable ledger includes:
- What this PR was (goal + scope)
- What shipped (bullet points)
- Evidence (gate + tests)
- Follow-ups

### Layer B: Full Ledger (for exhibits / significant PRs)

Add the full sections for PRs worth deep analysis:
- Process + friction table
- Full cost breakdown
- What went well / what could be better

Use `<details>` to collapse Layer B if needed:

```markdown
<details>
<summary>Full ledger details</summary>

[Layer B content here]

</details>
```

---

## See Also

- [PR_DOSSIER_SCHEMA.md](PR_DOSSIER_SCHEMA.md) — JSON schema for structured analysis
- [EXHIBITS.md](EXHIBITS.md) — Annotated PR examples
- [FAILURE_MODES.md](FAILURE_MODES.md) — Taxonomy for friction events
