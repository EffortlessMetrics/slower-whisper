# Non-streaming surface parity contract

A supported transcription surface may choose a different input boundary or
lifecycle owner. It may not produce different transcript truth from the same
audio, runtime selection, and transcription configuration.

## Canonical semantic projection

Parity is evaluated on the complete schema-valid transcript, with these
semantic fields compared directly:

- transcript schema version;
- source filename in both top-level `file` and `meta.audio_file`;
- detected language;
- ordered segments, words, probabilities, and timing;
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

The direct file, bytes, and REST boundaries restore one shared safe
caller-facing basename to both `Transcript.file_name` and `meta.audio_file`.
REST still writes the upload to a randomized temporary filesystem path. The
temporary path cannot escape into the public transcript or receipt.

REST responses use the schema-authoritative `file` key. The legacy `file_name`
key remains temporarily as a compatibility alias and must equal `file` exactly.
The REST serializer also carries the supported optional annotations, speakers,
turns, speaker statistics, and chunks when present.

## Lifecycle ownership

Expected lifecycle differences remain explicit:

- direct file calls do not close an injected engine;
- direct file and bytes operations close their operation-owned engines;
- service requests reuse and close one process runtime;
- directory work reuses one operation-owned engine across the batch and closes
  it on success, inference failure, and writer failure;
- no surface constructs one engine per output format.

## Checkpoint B1: directory operation

The directory operation runs the checkpoint-A deterministic fixture through
`run_pipeline()` and compares the resulting JSON with the canonical direct-file
document. The batch must preserve:

- the original raw source filename rather than the normalized WAV name;
- the selected runtime and ordered fallback evidence;
- timed segment and word state;
- the complete stable receipt projection;
- one operation-owned engine across the batch.

Raw files whose names differ only by extension or case can map to the same
normalized WAV. The pipeline rejects those collisions before ffmpeg or model
construction. It does not allow concurrent normalization to overwrite one
source with another and attempt to reconstruct identity afterward.

The operation constructs no engine when there is no normalized work. Once an
engine is constructed, terminal cleanup runs after successful processing,
per-file inference failure, and output-writer failure. Cleanup failure is logged
locally and does not replace the operation's primary result or exception.

## Checkpoint B2: CLI

The same fixture and semantic projection still need to execute through:

1. `transcription.cli.main()` with the actual argparse `transcribe` command;
2. the wheel-installed `slower-whisper` executable from an unrelated working
   directory.

Call-graph discovery is not behavioral parity. CLI parity is earned only by
executing those entrypoints against the complete bundled schemas and stable
receipt projection. When the CLI diverges, repair its owning orchestration path;
do not normalize a divergent document inside the test.

## Failure truth

Parity includes failure behavior, not only successful JSON:

- genuine silence is a successful empty transcript on every surface;
- runtime/model unavailability is not silence;
- inference and invalid-output failures retain stable reason codes;
- provider paths and raw exception text remain local;
- profile mismatch is rejected before upload or model work in service mode;
- ambiguous raw-to-normalized identity fails before normalization and inference.

## Artifact acceptance

The installed-wheel lanes run from outside the checkout and load schemas
through `importlib.resources`. This proves the comparison uses the packaged
public API, packaged schemas, and packaged provenance implementation rather
than source-tree-relative files.

The batch lane copies its executable parity test outside the checkout and runs
`run_pipeline()` from the installed wheel on Python 3.12 and 3.13. The later CLI
lane must separately invoke the installed console script itself.

## Deliberate boundary

WebSocket streaming uses replacement revisions and its own public event
contract; it is not reduced to a batch transcript comparison here. Optional
enrichment joins parity only after its own behavior and failure contracts are
stable.
