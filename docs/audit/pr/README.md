# PR Dossiers

This directory contains structured JSON dossiers for significant PRs.

## When to Create a Dossier

Not every PR needs one. Good candidates:

- PRs that revealed new failure modes
- Large architectural changes
- Performance-sensitive changes
- PRs with significant iteration/rework
- Exhibit PRs selected for audit demonstration

## File Naming

```
docs/audit/pr/<pr-number>.json
```

## Schema

See [PR_DOSSIER_SCHEMA.md](../PR_DOSSIER_SCHEMA.md) for the full schema definition.

## Generating Dossiers

Use the helper script:

```bash
python scripts/generate-pr-ledger.py --pr 123 --output docs/audit/pr/123.json
```

Or manually copy the template from [PR_DOSSIER_SCHEMA.md](../PR_DOSSIER_SCHEMA.md).

## See Also

- [../EXHIBITS.md](../EXHIBITS.md) — Annotated PR exhibits
- [../FAILURE_MODES.md](../FAILURE_MODES.md) — Failure taxonomy
- [../PR_LEDGER_TEMPLATE.md](../PR_LEDGER_TEMPLATE.md) — PR comment template
