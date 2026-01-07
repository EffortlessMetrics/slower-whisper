# Audit Path

A 15-minute route for external reviewers to validate this repo's claims.

---

## 1. Run the local gate (~2 min)

```bash
./scripts/ci-local.sh fast   # quick check
./scripts/ci-local.sh        # full gate
```

The local gate is canonical. CI is additive and may be rate-limited.

---

## 2. Check failure mode taxonomy (~3 min)

Review [FAILURE_MODES.md](FAILURE_MODES.md) for documented failure classes:

- Each entry should have a discovery link
- Each should have detection mechanism
- Prevention patterns should exist or be marked "open"

---

## 3. Inspect 3-5 exhibit PRs (~5 min)

Open the linked PRs from [EXHIBITS.md](EXHIBITS.md) (when populated). For each:

- [ ] PR includes receipts (gate output, benchmarks)
- [ ] "What was wrong / surprises" section present (if non-trivial)
- [ ] DevLT and machine spend recorded (optional but expected)

---

## 4. Validate schema claims (~2 min)

```bash
# Generate fresh example from test fixtures
python scripts/render-doc-snippets.py

# Validate a real output
python -m jsonschema -i whisper_json/sample.json \
  transcription/schemas/transcript-v2.schema.json
```

---

## 5. Verify receipts location (~3 min)

Receipts should appear in:

- PR comments (CI artifacts link or inline)
- `docs/audit/pr/` (if deep analysis dossiers exist)
- Benchmark runs tagged with `meta.receipt` in result JSON

---

## What to look for

| Signal | Healthy | Unhealthy |
|--------|---------|-----------|
| Gate passes | Passes clean | Skipped tests, warnings ignored |
| Receipts present | Linked or inline | "trust me" claims |
| Failure modes filed | Linked to fix | Silent fixes, no taxonomy entry |
| Schema examples | Generated | Hand-written, stale |
| Numbers in docs | Anchored (date or script) | Floating claims |

---

## Reporting issues

If something looks wrong:

1. File an issue with `audit:` prefix
2. Link to the specific claim or PR
3. Propose disposition: fix, mark invalid, or defer with rationale

---

## See Also

- [README.md](README.md) — Audit infrastructure overview
- [FAILURE_MODES.md](FAILURE_MODES.md) — Failure taxonomy
- [PR_DOSSIER_SCHEMA.md](PR_DOSSIER_SCHEMA.md) — Structured PR analysis format
