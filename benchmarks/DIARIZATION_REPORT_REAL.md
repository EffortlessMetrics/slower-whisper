# Diarization Benchmark

## Run config
- dataset: `/home/steven/code/Python/slower-whisper/benchmarks/data/diarization`
- manifest: `/home/steven/code/Python/slower-whisper/benchmarks/data/diarization/manifest.jsonl` (sha256: 34f8caa31589541c795dcc217df1688440bf25ee45d92669073eafdde0fe0120)
- diarizer device: `cpu`
- min/max speakers: None/None
- SLOWER_WHISPER_PYANNOTE_MODE: `auto`

## Aggregate
- samples: 3
- avg DER: 1.0
- speaker count accuracy: 0.0
- total runtime (s): 12.87

## Per-sample
| sample | DER | hyp speakers | exp speakers | runtime (s) | notes | error/detail |
|---|---|---|---|---|---|---|
| synthetic_2speaker | 1.0000 | 0 | 2 | 11.59 | Deterministic 2-speaker A/B tone pattern with 200ms gaps between turns. |  |
| overlap_tones | 1.0000 | 0 | 2 | 0.20 | Two-tone synthetic with overlapping exchanges at 2.0-2.5s and 5.0-5.6s. |  |
| call_mixed | 1.0000 | 0 | 2 | 1.07 | Longer alternating call-style pattern with short pauses; minimal overlap. |  |
