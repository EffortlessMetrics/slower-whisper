# Directory and CLI surface-parity contract

The directory operation and the installed `slower-whisper` command are public
transcription surfaces. They may own a different operation lifecycle from the
file, bytes, or service APIs. They may not produce different transcript truth
from the same audio, selected runtime, and transcription configuration.

## Executed transaction

The parity gate runs one deterministic fallback engine through:

1. the canonical direct file API with an injected engine;
2. `run_pipeline()` over a project raw-audio directory;
3. the actual Click/Typer `transcribe` command dispatch;
4. the installed `slower-whisper` executable from an unrelated working
   directory.

The engine requests CUDA/float16, records one failed load attempt, then selects
CPU/int8. Every output must retain that ordered evidence.

## Compared truth

Each emitted JSON document validates against the bundled transcript and receipt
schemas. Parity compares:

- transcript schema version;
- safe public source filename and `meta.audio_file`;
- detected language;
- ordered segments, words, probabilities, and timing;
- actual backend, model, device, and compute type;
- ordered bounded model-load attempts;
- the stable receipt projection.

The stable receipt projection removes only `run_id` and `created_at`. It does
not erase `config_hash`, source commit, build ID, selected runtime, or fallback
attempts to make divergent surfaces appear equal.

## Lifecycle ownership

- the direct file baseline does not close its injected engine;
- the directory operation constructs and closes one operation engine;
- the CLI delegates to the directory operation and closes one operation engine;
- one input file must not cause one engine per output format;
- the installed console lane must resolve the wheel-installed entrypoint, not a
  source-tree wrapper.

## Command discovery

The test resolves the `slower-whisper` console entrypoint from installed package
metadata, converts the Typer application to its Click command, and invokes its
real `transcribe` subcommand. Required path arguments and deterministic runtime
options are derived from the live command parameters. An unknown required
parameter fails the contract rather than being guessed silently.

## Failure truth

The directory and CLI surfaces must preserve the same typed behavior as the
canonical file operation:

- genuine silence remains a successful empty transcript;
- unavailable, load, inference, and output failures do not become silence;
- provider paths and raw exception text remain local;
- malformed or incomplete output does not receive a fabricated receipt;
- operation-owned engines close on both success and failure.

## Artifact acceptance

The installed lane builds the wheel, installs its console script into a clean
virtual environment, injects a deterministic engine at Python startup, and runs
`slower-whisper` from an unrelated directory. The resulting JSON must validate
against schemas loaded from the installed package and must record exactly one
engine construction and one close.

## Deliberate boundary

WebSocket streaming is excluded because its public truth is replacement
revisions rather than a completed batch document. Optional enrichment joins
surface parity only after its own behavior and failure contracts are stable.
