# Diarization Benchmark (2025-12-01)

Tiny harness run on synthetic fixtures to track diarization behavior.

## Run config
- dataset: `benchmarks/data/diarization`
- manifest sha256: `34f8caa31589541c795dcc217df1688440bf25ee45d92669073eafdde0fe0120`
- backend: `SLOWER_WHISPER_PYANNOTE_MODE=stub` (HF_TOKEN not set; pyannote.audio models not pulled)
- device: `auto`, min/max speakers: None/None
- metrics: `pyannote.metrics` 4.0.0 for DER

## Aggregate
- samples: 3
- avg DER: 0.674 (stub pipeline)
- speaker count accuracy: 3/3
- total runtime: 0.93s

## Per-file
| sample | DER | hyp speakers | exp speakers | notes |
| --- | --- | --- | --- | --- |
| synthetic_2speaker | 0.7833 | 2 | 2 | Deterministic A/B tone pattern with 200ms gaps. |
| overlap_tones | 0.4203 | 2 | 2 | Two-tone clip with brief overlaps at 2.0–2.5s and 5.0–5.6s. |
| call_mixed | 0.8182 | 2 | 2 | Longer alternating call-style tone pattern with short pauses. |

> Note: DERs reflect the stub diarization backend; real pyannote models will differ once HF_TOKEN-backed runs are available.
