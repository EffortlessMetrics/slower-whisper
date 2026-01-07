# PR Exhibits

Annotated PRs demonstrating the audit workflow in practice. Each exhibit shows the trust loop in action: what changed, what went wrong, how it was caught, and what prevention was added.

---

## Criteria for Inclusion

PRs are selected as exhibits when they demonstrate:

- Non-trivial changes with documented receipts
- "What was wrong / surprises" captured honestly
- DevLT and machine spend recorded
- Failure modes filed when applicable
- Prevention added to avoid repeat failures

**Not every PR needs to be an exhibit.** Select PRs that teach something about the process.

---

## Exhibit Index

| PR | Type | Key Lesson | Dossier |
|----|------|------------|---------|
| *Exhibits will be added as qualifying PRs are merged* | | | |

<!--
Example row:
| [#123](https://github.com/EffortlessMetrics/slower-whisper/pull/123) | perf/bench | Baseline semantic validation prevents measurement drift | [123.json](pr/123.json) |
-->

---

## Exhibit Template

When adding a new exhibit, use this format:

### PR #NNN — Short descriptive title

**Type:** feature | hardening | mechanization | perf/bench | refactor

**What this PR did:**
- [1-3 bullet summary]

**What went wrong / surprises:**
- [Issue] → [Detected by: gate/review/post-merge] → [Disposition: fixed here/deferred to #X]

**Prevention added:**
- [Gate/contract/test added, or "none needed" with rationale]

**Cost:**
- **DevLT:** author ~Xm, review ~Ym
- **Machine spend:** ~$Z.ZZ (or "unknown")
- **Wall-clock:** X days

**Key lesson:**
[1-2 sentences on what this PR teaches about the process]

**Dossier:** [NNN.json](pr/NNN.json) (if detailed analysis exists)

---

## Exhibit Categories

### Category: Measurement Integrity

PRs that demonstrate how to handle benchmarks, baselines, and performance claims.

*No exhibits yet*

### Category: Failure Detection

PRs where the gate/review process caught issues before they shipped.

*No exhibits yet*

### Category: Post-Merge Learning

PRs where issues were discovered after merge, leading to prevention.

*No exhibits yet*

### Category: Process Improvement

PRs that improved the development process itself (gates, templates, tooling).

*No exhibits yet*

---

## How to Select Exhibit PRs

Use this scoring rubric to identify high-value exhibits:

| Factor | Weight | Score (0-3) |
|--------|--------|-------------|
| Novel failure mode discovered | 3x | |
| Prevention added to codebase | 2x | |
| Clear receipts/evidence | 2x | |
| Significant iteration/rework | 1x | |
| Architectural significance | 1x | |

**Score ≥8:** Strong exhibit candidate
**Score 5-7:** Consider if category underrepresented
**Score <5:** Skip unless uniquely instructive

---

## Subagent Checklist (for deep analysis)

When generating a detailed exhibit, run these auditors:

- [ ] **Diff Scout:** Change surface + hotspots + "what's actually changed"
- [ ] **Evidence Auditor:** Receipts present? Do claims map to artifacts?
- [ ] **Docs/Schema Auditor:** Doc claims vs reality (wrong keys, stale examples?)
- [ ] **Perf Integrity Auditor:** Absolute metrics, baseline pinned, any drift?
- [ ] **Process/Friction Miner:** Extract 2-5 real "scar" events
- [ ] **Design Alignment Auditor:** Issue DoD/constraints met? Drift noted?

Each auditor produces 5-15 lines. The exhibit summary synthesizes these notes.

---

## See Also

- [AUDIT_PATH.md](AUDIT_PATH.md) — How to use exhibits during cold-reader audit
- [FAILURE_MODES.md](FAILURE_MODES.md) — Taxonomy linked from exhibits
- [PR_DOSSIER_SCHEMA.md](PR_DOSSIER_SCHEMA.md) — Detailed dossier format (v2)
- [PR_LEDGER_TEMPLATE.md](PR_LEDGER_TEMPLATE.md) — Template for PR initial comments
