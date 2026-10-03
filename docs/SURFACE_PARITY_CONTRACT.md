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
- directory and CLI work reuse one operation-owned engine across the batch and
  close it on success, inference failure, and writer failure;
- no surface constructs one engine per output format.

## Checkpoint B1: directory operation

The directory operation runs the checkpoint-A deterministic fixture through
`run_pipeline()` and compares the resulting JSON with the canonical direct-file
document. The batch preserves:

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

## Checkpoint B2: argparse and installed console

The deterministic fixture now executes through both CLI boundaries:

1. `transcription.cli.main()` with the actual argparse `transcribe` command;
2. the wheel-installed `slower-whisper transcribe` executable from an unrelated
   working directory.

Every material configuration value is supplied through the real command-line
surface: root, model, requested device, compute type, language, task, beam size,
VAD silence, word timestamps, existing-output policy, chunking, and
diarization. Device discovery is controlled only at the external hardware seam;
the argparse parser, config merge, command dispatch, summary/exit code, and
`run_pipeline()` call remain real.

The source command is compared directly with the canonical file document. The
installed executable is launched through the wheel-generated console script.
A Python startup injection supplies only the deterministic engine, audio
normalization, build identity, and device probe. A separate lifecycle file
proves the subprocess constructed one engine, called it once, and closed it
once.

The source and installed CLI documents validate against bundled transcript and
receipt schemas and preserve the complete semantic and stable-receipt
projection. The console test runs from a directory unrelated to the checkout
or project root, with no source-tree `PYTHONPATH`.

Call-graph discovery does not substitute for this transaction. CLI parity is
an executed source-and-artifact result, not an inference from the shared batch
owner.

## Failure truth

Parity includes failure behavior, not only successful JSON:

- genuine silence is a successful empty transcript on every surface;
- runtime/model unavailability is not silence;
- inference and invalid-output failures retain stable reason codes;
- provider paths and raw exception text remain local;
- profile mismatch is rejected before upload or model work in service mode;
- ambiguous raw-to-normalized identity fails before normalization and inference;
- the CLI returns a non-zero exit code when its batch result contains failures.

## Artifact acceptance

The installed-wheel lanes run from outside the checkout and load schemas
through `importlib.resources`. This proves the comparison uses the packaged
public API, packaged schemas, packaged provenance implementation, and generated
console script rather than source-tree-relative files.

Python 3.12 and 3.13 each execute:

- direct, bytes, REST, and batch Python surfaces from the installed wheel;
- the wheel-installed `slower-whisper transcribe` executable;
- root and `transcribe` help through that executable;
- installed receipt completion and provider-detail redaction.

## Deliberate boundary

WebSocket streaming uses replacement revisions and its own public event
contract; it is not reduced to a batch transcript comparison here. Optional
enrichment joins parity only after its own behavior and failure contracts are
stable.
