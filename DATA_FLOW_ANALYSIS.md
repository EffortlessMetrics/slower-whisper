# Complete Diarization Data Flow (v1.1)

This document traces how diarization flows from CLI flags through config, pipeline, orchestrator, and JSON output. Line numbers drift quickly, so this focuses on function boundaries and data contracts.

---

## 1) CLI → Config

- Flags (transcribe subcommand): `--enable-diarization`, `--diarization-device`, `--min-speakers`, `--max-speakers`, `--overlap-threshold`.
- Precedence: CLI > config file > environment (`SLOWER_WHISPER_*`) > defaults.
- `compute_type` is auto-derived when unset (`float16` on CUDA, `int8` on CPU) and can be overridden explicitly.
- Result: `TranscriptionConfig` carries diarization fields and final device/compute_type after precedence resolution.

---

## 2) Config → Pipeline

- `transcribe_directory()` converts `TranscriptionConfig` to internal `AppConfig` + `AsrConfig` and calls `run_pipeline(app_cfg, diarization_config=config if config.enable_diarization else None)`.
- `run_pipeline()` normalizes audio, transcribes with `TranscriptionEngine`, then conditionally runs diarization if `diarization_config.enable_diarization` is True.

---

## 3) Diarization Orchestrator

`_maybe_run_diarization(transcript, wav_path, config)`:
- Early exit if `config.enable_diarization` is False.
- On success:
  1. Instantiate `Diarizer(device, min_speakers, max_speakers)`.
  2. Run `diarizer.run(wav_path)` → list of `SpeakerTurn`.
  3. `assign_speakers(transcript, speaker_turns, overlap_threshold)` populates `segment.speaker` and `transcript.speakers`.
  4. `build_turns(transcript)` groups contiguous speaker-labeled segments into `transcript.turns`.
  5. `meta.diarization` set to `{status: "success", requested: True, backend: "pyannote.audio", num_speakers: ...}`.
- On failure:
  - Restore original segment speakers, speakers list, and turns to avoid partial state.
  - `meta.diarization` records `{status: "failed", requested: True, error, error_type}` with categorized `auth|missing_dependency|file_not_found|unknown`.
  - Transcript is returned unchanged aside from metadata.
- Tested in `tests/test_diarization_skeleton.py`.

---

## 4) Serialization

- `write_json()` outputs schema v2 with:
  - `meta`: generated_at, model_name, pipeline_version, runtime `device`/`compute_type` (prefers ASR-emitted `asr_device`/`asr_compute_type`), `asr_backend`, optional `asr_model_load_warnings` or `asr_fallback_reason`, plus `diarization` status.
  - `segments[]`: includes `speaker`, `tone`, `audio_state`.
  - Optional `speakers[]` and `turns[]` when diarization succeeds.
- Companion writers emit TXT/SRT using the same `Transcript`.

**Trimmed JSON example:**
```json
{
  "schema_version": 2,
  "file": "audio.wav",
  "language": "en",
  "meta": {
    "generated_at": "...",
    "model_name": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "asr_backend": "faster-whisper",
    "asr_device": "cuda",
    "asr_compute_type": "float16",
    "diarization": {
      "status": "success",
      "requested": true,
      "backend": "pyannote.audio",
      "num_speakers": 2
    }
  },
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello", "speaker": {"id": "spk_0", "confidence": 0.95}},
    {"id": 1, "start": 2.5, "end": 4.0, "text": "Hi there", "speaker": {"id": "spk_1", "confidence": 0.87}}
  ],
  "speakers": [
    {"id": "spk_0", "label": null, "total_speech_time": 5.2, "num_segments": 8},
    {"id": "spk_1", "label": null, "total_speech_time": 3.8, "num_segments": 6}
  ],
  "turns": [
    {"id": "turn_0", "speaker_id": "spk_0", "start": 0.0, "end": 2.5, "segment_ids": [0], "text": "Hello"},
    {"id": "turn_1", "speaker_id": "spk_1", "start": 2.5, "end": 4.0, "segment_ids": [1], "text": "Hi there"}
  ]
}
```

---

## 5) Error Handling Snapshot

- Failure leaves segments untouched and clears speakers/turns.
- `meta.diarization.error_type` categorizes auth, missing dependency, file-not-found, or unknown.
- Graceful degradation is asserted in `tests/test_diarization_skeleton.py::test_maybe_run_diarization_graceful_failure`.

---

## 6) Data Structures (at-a-glance)

- **SpeakerTurn** (from `Diarizer.run`): `start`, `end`, `speaker_id`, `confidence?`.
- **Transcript (post-ASR, pre-diarization)**: `segments[]` with `id/start/end/text`, no speakers by default.
- **Transcript (post-diarization success)**:
  - `segments[].speaker = {"id": "spk_0", "confidence": 0.95}`
  - `speakers[]` aggregate stats
  - `turns[]` grouped by speaker
  - `meta.diarization.status = "success"`
- **Transcript (post-diarization failure)**:
  - Speakers/turns remain `None`
  - `meta.diarization.status = "failed"` with error + type
