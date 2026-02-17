# Diarization Benchmark

## Run config
- dataset: `benchmarks/data/diarization`
- manifest: `benchmarks/data/diarization/manifest.jsonl` (sha256: c21b86b9971b0502f51eff00254819c45aeebf3cfd7f8c67d7a266a968dcd831)
- diarizer device: `cpu`
- min/max speakers: None/None
- SLOWER_WHISPER_PYANNOTE_MODE: `stub`

## Aggregate
- samples: 3
- avg DER: 0.541371059848541
- speaker count accuracy: 1.0
- total runtime (s): 1.59

## Per-sample
| sample | DER | hyp speakers | exp speakers | runtime (s) | notes | error/detail |
|---|---|---|---|---|---|---|
| meeting_dual_voice | 0.5500 | 2 | 2 | 1.59 | Alternating 2-speaker meeting recap with short pauses. |  |
| support_handoff_dual_voice | 0.5413 | 2 | 2 | 0.00 | Alternating support handoff with distinct speaker timbre. |  |
| planning_sync_dual_voice | 0.5328 | 2 | 2 | 0.00 | Alternating planning sync with concise turn boundaries. |  |