# Provenance and receipt contract

Every successful file, bytes, and batch REST transcription receives one
canonical receipt at `meta.receipt`. Failure responses do not manufacture a
receipt.

## Identity authority

Runtime code reads source and build identity only from
`transcription._build_info`, a Python module packaged into the wheel and sdist.
It never runs `git`, reads the caller's repository, or infers identity from the
current working directory or runtime environment.

Source checkouts and unlabelled local builds deliberately contain:

```python
SOURCE_COMMIT = None
BUILD_ID = None
```

Unknown fields are omitted from the receipt rather than guessed.

Official artifact construction writes explicit trusted values before building:

```bash
python scripts/write_build_info.py \
  --source-commit "$GITHUB_SHA" \
  --build-id "release-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
uv build
```

The generator validates both inputs, shortens the hexadecimal source revision
to the existing 12-character `git_commit` contract, and produces byte-identical
module content for identical inputs. The historical `git_commit` field name is
retained for compatibility; it now means the source commit embedded in the
installed package, not a runtime repository lookup.

## Runtime evidence

The canonical metadata path records actual selected values after any supported
fallback:

- package/tool version;
- transcript schema version;
- model and optional model revision;
- backend;
- selected device and compute type;
- ordered model-load attempts;
- canonical configuration hash;
- run ID and creation time;
- package source commit and build ID when embedded.

Ordered attempts contain only bounded public fields: `device`, `compute_type`,
`outcome`, and `reason_code`. Provider exception text and local paths remain in
local logs/exception chains and never enter the receipt.

## Determinism

`config_hash` is a SHA-256 projection of canonical JSON data. Mapping order,
set order, enums, dataclasses, and `Path` instances are normalized before
serialization. Non-finite floats and unsupported object types fail instead of
being stringified into platform-dependent evidence.

Only `run_id` and `created_at` are per-run volatile fields. Removing those two
fields from receipts produced by the same artifact and runtime configuration
yields the same stable projection.

## Schemas

- `transcription/schemas/receipt-v1.schema.json` is the strict receipt schema.
- `transcription/schemas/transcript-v2.schema.json` remains the transcript
  schema and accepts the `meta.receipt` extension.

Artifact acceptance validates both schemas from the installed package. The
receipt schema permits the legacy UUID run-ID form for reading old data while
newly generated receipts use `run-YYYYMMDD-HHMMSS-XXXXXX`.

## Artifact acceptance

The provenance gate builds both wheel and sdist with explicit source/build
identity, installs each outside the checkout, launches from an unrelated fake
git repository with a fake `git` executable on `PATH`, and proves:

1. the installed package reports the embedded identity;
2. cwd and `PATH` cannot alter it;
3. successful metadata attaches an automatic receipt;
4. selected fallback values and ordered attempts survive;
5. provider paths and caller paths do not enter the receipt;
6. receipt and transcript JSON validate against installed schemas;
7. wheel and sdist contain the intended generated build module.
