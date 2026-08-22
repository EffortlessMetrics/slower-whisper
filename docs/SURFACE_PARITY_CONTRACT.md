# Non-streaming surface parity contract

A supported transcription surface may choose a different input boundary or
lifecycle owner. It may not produce different transcript truth from the same
audio, runtime selection, and transcription configuration.

## Canonical semantic projection

Parity is evaluated on the complete schema-valid transcript, with these
semantic fields compared directly:

- transcript schema version;
- source filename;
- detected language;
- ordered segments, words, and timing;
- actual ASR backend, model, device, and compute type;
- ordered bounded model-load attempts;
- the stable receipt projection.

The stable receipt projection removes only the declared per-run fields:
`run_id` and `created_at`. A comparison must not discard `config_hash`, source
commit, build ID, selected runtime, or model-load attempts to make divergent
surfaces appear equivalent.

Invocation-local metadata such as a temporary working root is outside the
receipt and may differ where the surface contract requires it. Internal random
temporary filenames are not a valid substitute for the caller's source
identity.

## Checkpoint A: canonical public APIs

The first executable checkpoint compares:

1. direct file transcription with an explicitly supplied engine;
2. bytes transcription with operation-owned engine construction;
3. batch REST transcription through one process-owned runtime.

The deterministic test engine requests CUDA/float16, records that attempt as
failed, then selects CPU/int8. Every surface must preserve the same selected
runtime and ordered attempts, attach a schema-valid receipt, and close only the
engine it owns.

This checkpoint establishes the comparison harness. It does not claim that the
older directory/batch or CLI orchestration paths are already equivalent.

## Checkpoint B: directory and CLI

The same fixture and semantic projection must be extended to the supported
batch and CLI entrypoints. When one diverges, repair the owning orchestration
path so it exits through the canonical metadata, receipt, and writer seams.
Do not normalize a divergent document only inside the test.

Expected lifecycle differences remain explicit:

- direct file calls do not close an injected engine;
- bytes and CLI operations close their operation-owned engines;
- service requests reuse and close one process runtime;
- directory work may reuse one operation-owned engine across files but must not
  create an engine per output format.

## Failure truth

Parity includes failure behavior, not only successful JSON:

- genuine silence is a successful empty transcript on every surface;
- runtime/model unavailability is not silence;
- inference and invalid-output failures retain stable reason codes;
- provider paths and raw exception text remain local;
- profile mismatch is rejected before upload or model work in service mode.

## Artifact acceptance

The installed-wheel lane runs from outside the checkout and loads schemas
through `importlib.resources`. This proves the comparison uses the packaged
public API, packaged schemas, and packaged provenance implementation rather
than source-tree-relative files.

## Deliberate boundary

WebSocket streaming uses replacement revisions and its own public event
contract; it is not reduced to a batch transcript comparison here. Optional
enrichment joins parity only after its own behavior and failure contracts are
stable.
