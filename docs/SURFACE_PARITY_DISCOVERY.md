# Directory and CLI surface-parity discovery

This generated report exposes actual owners and call edges. It does **not** claim behavioral parity.

## Installed console scripts

- `slower-whisper` → `transcription.cli:main`
- `slower-whisper-dogfood` → `transcription.dogfood:main`

## Directory and batch candidates

- `transcription.api::transcribe_directory` — engine=False; canonical=[]; writers=[]
- `transcription.cli::transcribe_directory` — engine=False; canonical=[]; writers=[]
- `transcription.pipeline::_build_meta` — engine=False; canonical=['build_generation_metadata']; writers=[]
- `transcription.pipeline::run_pipeline` — engine=True; canonical=['transcribe_file']; writers=['write_json', 'write_srt']
- `transcription.transcription_orchestrator::_transcribe_file_impl` — engine=True; canonical=['build_generation_metadata', 'transcribe_file']; writers=['write_json', 'write_srt']

## CLI candidates

- `transcription.cli::transcribe_directory` — engine=False; canonical=[]; writers=[]
- `transcription.cli::_handle_transcribe_command` — engine=False; canonical=[]; writers=[]
- `transcription.cli::_handle_enrich_command` — engine=False; canonical=[]; writers=['write_json']
- `transcription.cli::_handle_outcomes_extract` — engine=False; canonical=[]; writers=['write_text']
- `transcription.cli::main` — engine=False; canonical=[]; writers=[]
- `transcription.cli_legacy_transcribe::main` — engine=False; canonical=[]; writers=[]
- `transcription.store.cli::_handle_export` — engine=False; canonical=[]; writers=['write_text']

## Existing surface tests

- `tests/test_api.py`
- `tests/test_api_integration.py`
- `tests/test_api_service.py`
- `tests/test_cli.py`
- `tests/test_cli_device.py`
- `tests/test_cli_integration.py`
- `tests/test_compat_shim.py`
- `tests/test_config_merge_bug.py`
- `tests/test_config_sources.py`
- `tests/test_conversation_store.py`
- `tests/test_diarization_skeleton.py`
- `tests/test_error_handling.py`
- `tests/test_integrations_adapters.py`
- `tests/test_pipeline.py`
- `tests/test_service_health.py`
- `tests/test_smoke_pipeline.py`
- `tests/test_streaming_client.py`
- `tests/test_versioning.py`

## `slower-whisper transcribe --help`

```text
usage: slower-whisper transcribe [-h] [--root ROOT] [--config CONFIG]
                                 [--model MODEL] [--device {auto,cuda,cpu}]
                                 [--compute-type COMPUTE_TYPE]
                                 [--language LANGUAGE]
                                 [--task {transcribe,translate}]
                                 [--vad-min-silence-ms VAD_MIN_SILENCE_MS]
                                 [--beam-size BEAM_SIZE]
                                 [--word-timestamps | --no-word-timestamps]
                                 [--skip-existing-json | --no-skip-existing-json | --skip-existing | --no-skip-existing]
                                 [--progress]
                                 [--enable-chunking | --no-enable-chunking]
                                 [--chunk-target-duration-s CHUNK_TARGET_DURATION_S]
                                 [--chunk-max-duration-s CHUNK_MAX_DURATION_S]
                                 [--chunk-target-tokens CHUNK_TARGET_TOKENS]
                                 [--chunk-pause-split-threshold-s CHUNK_PAUSE_SPLIT_THRESHOLD_S]
                                 [--enable-diarization | --no-enable-diarization]
                                 [--diarization-device {auto,cuda,cpu}]
                                 [--min-speakers MIN_SPEAKERS]
                                 [--max-speakers MAX_SPEAKERS]
                                 [--overlap-threshold OVERLAP_THRESHOLD]
                                 [--telemetry]

options:
  -h, --help            show this help message and exit
  --root ROOT           Project root (contains raw_audio/, input_audio/,
                        whisper_json/, transcripts/).
  --config CONFIG       Path to TranscriptionConfig JSON file. Precedence: CLI
                        flags > config file > env vars > defaults.
  --model MODEL         Whisper model name (default: large-v3).
  --device {auto,cuda,cpu}
                        Device for ASR (Whisper) inference. 'auto' detects
                        CUDA availability (default: auto).
  --compute-type COMPUTE_TYPE
                        faster-whisper compute type: float16, float32, int8,
                        int8_float16, etc. (default: auto-selected based on
                        device).
  --language LANGUAGE   Force language (e.g. en). Leave empty for auto-detect.
  --task {transcribe,translate}
                        Whisper task (default: transcribe).
  --vad-min-silence-ms VAD_MIN_SILENCE_MS
                        Minimum silence duration in ms to split segments
                        (default: 500).
  --beam-size BEAM_SIZE
                        Beam size for decoding (default: 5).
  --word-timestamps, --no-word-timestamps
                        Extract word-level timestamps (v1.8+, default: False).
  --skip-existing-json, --no-skip-existing-json
                        Skip files with existing JSON in whisper_json/
                        (default: True).
  --skip-existing, --no-skip-existing
                        Alias for --skip-existing-json (for consistency with
                        enrich command).
  --progress            Show progress indicator during transcription (file
                        counter).
  --enable-chunking, --no-enable-chunking
                        Emit turn-aware chunks for RAG/export (default:
                        False).
  --chunk-target-duration-s CHUNK_TARGET_DURATION_S
                        Soft target chunk duration in seconds (default: 30).
  --chunk-max-duration-s CHUNK_MAX_DURATION_S
                        Hard max chunk duration in seconds (default: 45).
  --chunk-target-tokens CHUNK_TARGET_TOKENS
                        Approximate max tokens per chunk before splitting
                        (default: 400).
  --chunk-pause-split-threshold-s CHUNK_PAUSE_SPLIT_THRESHOLD_S
                        Split on pauses >= this length when near target size
                        (default: 1.5).
  --enable-diarization, --no-enable-diarization
                        Enable speaker diarization (experimental, default:
                        False). See docs/SPEAKER_DIARIZATION.md for setup.
  --diarization-device {auto,cuda,cpu}
                        Device for diarization. 'auto' selects cuda if
                        available, else cpu (default: auto).
  --min-speakers MIN_SPEAKERS
                        Minimum number of speakers expected (diarization hint,
                        optional).
  --max-speakers MAX_SPEAKERS
                        Maximum number of speakers expected (diarization hint,
                        optional).
  --overlap-threshold OVERLAP_THRESHOLD
                        Minimum overlap ratio (0.0-1.0) required to assign a
                        speaker to a segment (default: 0.3).
  --telemetry           Include timing telemetry in output summary.

```

## Next falsification

Run the checkpoint-A deterministic audio/runtime fixture through the actual batch and CLI owners above. Compare the complete bundled-schema-valid transcript and `receipt_stable_projection()` after removing only declared volatile fields. Repair the owner when it diverges; do not normalize a bad document inside the test.
