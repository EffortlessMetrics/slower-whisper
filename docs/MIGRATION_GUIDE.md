# Migration Guide: Project Structure and v2 CLI

The project moved scripts, documentation and deployment files into dedicated
folders during v1. Those file moves are historical: **v2 removed the legacy
transcription and enrichment scripts**, including their `scripts.*` module paths.
Use the unified CLI below. For all v2 changes, see [MIGRATION_V2.md](MIGRATION_V2.md).

## Current v2 Workflow

Install the package with the dependencies needed for transcription and enrichment
(see [INSTALL.md](INSTALL.md)). From your project root, place recordings in
`raw_audio/`, then run:

```bash
slower-whisper transcribe --root .
slower-whisper enrich --root .
```

Transcription prepares normalized audio in `input_audio/` and writes transcripts
to `whisper_json/`. Enrichment updates those JSON files in place. To operate on
another project, pass that directory to `--root` for both commands.

| Removed v1 interface | Current v2 replacement |
| --- | --- |
| `transcribe_pipeline.py` or `scripts/transcribe_pipeline.py` | `slower-whisper transcribe` |
| `audio_enrich.py` or `scripts/audio_enrich.py` | `slower-whisper enrich` |
| `python -m scripts.transcribe_pipeline` | `slower-whisper transcribe` |
| `python -m scripts.audio_enrich` | `slower-whisper enrich` |
| `slower-whisper-enrich` | `slower-whisper enrich` |
| `--enrich-config FILE` | `--config FILE` |

The CLI processes a project directory; do not carry over old single-file
`--audio` or `--file` arguments. For supported options, run:

```bash
slower-whisper transcribe --help
slower-whisper enrich --help
```

## Historical v1 File Moves

These locations explain older instructions; they do not restore removed v2
commands or imply that compatibility wrappers still exist.

### Scripts

- `audio_enrich.py` moved to `scripts/audio_enrich.py` in v1, then was removed in v2.
- `transcribe_pipeline.py` moved to `scripts/transcribe_pipeline.py` in v1, then was removed in v2.
- `verify_ami_setup.sh` moved to `scripts/verify_ami_setup.sh`.

### Documentation

Documentation was consolidated under `docs/`. Use the current
[documentation index](INDEX.md) rather than historical root-level paths.

### Deployment Configuration

Docker and Compose files were organized under `config/`. See
[config/README.md](../config/README.md) for current deployment commands.

## Checking Your Migration

Update automation to use the current commands above. Do not restore the old
wrappers or revert an unrelated commit to recover removed scripts. If an
example still points to a removed command, report its file and command in a
[GitHub issue](https://github.com/EffortlessMetrics/slower-whisper/issues).
