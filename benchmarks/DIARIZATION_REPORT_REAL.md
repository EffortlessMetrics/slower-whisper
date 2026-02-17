# Diarization Benchmark

## Run config
- dataset: `benchmarks/data/diarization`
- manifest: `benchmarks/data/diarization/manifest.jsonl` (sha256: c21b86b9971b0502f51eff00254819c45aeebf3cfd7f8c67d7a266a968dcd831)
- diarizer device: `cpu`
- min/max speakers: None/None
- SLOWER_WHISPER_PYANNOTE_MODE: `auto`

## Aggregate
- samples: 3
- avg DER: 0.12170176080380361
- speaker count accuracy: 1.0
- total runtime (s): 174.71

## Per-sample
| sample | DER | hyp speakers | exp speakers | runtime (s) | notes | error/detail |
|---|---|---|---|---|---|---|
| meeting_dual_voice | 0.1193 | 2 | 2 | 137.03 | Alternating 2-speaker meeting recap with short pauses. |  |
| support_handoff_dual_voice | 0.1333 | 2 | 2 | 27.05 | Alternating support handoff with distinct speaker timbre. |  |
| planning_sync_dual_voice | 0.1125 | 2 | 2 | 10.63 | Alternating planning sync with concise turn boundaries. |  |