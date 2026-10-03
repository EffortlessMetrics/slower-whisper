# Installed JSON schemas

The `transcription.schemas` package contains the public JSON contracts shipped
inside every slower-whisper wheel and sdist. Runtime and artifact tests load
these files through `importlib.resources`; consumers should not depend on a
source-checkout-relative path.

## Active schemas

### `transcript-v2.schema.json`

The complete transcript document contract:

- `schema_version`, source file, language, and segments;
- segment and word timing;
- optional speaker, tone, prosody, emotion, semantic, and extraction state;
- open-ended `meta`, including the automatic `meta.receipt` extension.

A successful transcript remains valid against this schema after its receipt is
attached.

### `receipt-v1.schema.json`

The strict provenance/runtime evidence attached at `meta.receipt`:

- package version and transcript schema version;
- model, backend, optional model revision;
- actual selected device and compute type;
- ordered bounded model-load attempts;
- canonical configuration hash;
- run ID and creation time;
- trusted package source commit and build ID when embedded.

Unknown source/build identity is omitted. Runtime code never consults the
caller’s repository, current working directory, `PATH`, or environment to
manufacture those fields.

### `stream_event.schema.json`

The streaming event envelope used by the WebSocket protocol.

## Load schemas from the installed package

```python
import json
from importlib import resources

package_root = resources.files("transcription")
transcript_schema = json.loads(
    package_root.joinpath("schemas/transcript-v2.schema.json").read_text(
        encoding="utf-8"
    )
)
receipt_schema = json.loads(
    package_root.joinpath("schemas/receipt-v1.schema.json").read_text(
        encoding="utf-8"
    )
)
```

## Validate a transcript and its receipt

```python
from jsonschema import Draft7Validator, FormatChecker

receipt = transcript["meta"]["receipt"]
assert not list(
    Draft7Validator(
        receipt_schema,
        format_checker=FormatChecker(),
    ).iter_errors(receipt)
)
assert not list(
    Draft7Validator(
        transcript_schema,
        format_checker=FormatChecker(),
    ).iter_errors(transcript)
)
```

The CLI can validate complete transcript documents:

```bash
slower-whisper validate transcript.json
slower-whisper validate transcript.json \
  --schema transcription/schemas/transcript-v2.schema.json
```

## Versioning

Schema files have independent contract versions. Breaking changes require a new
schema filename or an explicit migration path; adding optional transcript
metadata does not change the top-level transcript schema version.

| Schema | Version | Status |
|---|---:|---|
| `transcript-v2.schema.json` | 2 | Active |
| `receipt-v1.schema.json` | 1 | Active |
| `stream_event.schema.json` | current protocol | Active |

When changing an installed schema:

1. update or add the schema file;
2. update this inventory and the relevant contract documentation;
3. add source and installed-artifact validation;
4. verify wheel and sdist contents;
5. add a migration when existing persisted documents would otherwise become
   unreadable.

See [`docs/PROVENANCE.md`](../../docs/PROVENANCE.md) for the receipt authority
and artifact-build transaction, and [`docs/SCHEMA.md`](../../docs/SCHEMA.md) for
transcript structure and compatibility guidance.
