# Diarization Benchmark

## Modes
- Stub (SLOWER_WHISPER_PYANNOTE_MODE=stub, cpu): avg DER **0.4511**, speaker-count accuracy **1.0**, total runtime **1.24s** on manifest sha256 `34f8caa31589541c795dcc217df1688440bf25ee45d92669073eafdde0fe0120` (details below; used for CI stub guard).
- Real pyannote (SLOWER_WHISPER_PYANNOTE_MODE=auto, HF_TOKEN required): generate `benchmarks/DIARIZATION_REPORT_REAL.{md,json}` with the real backend once a token + pyannote model are available. Example command (inside `nix develop`):
  ```bash
  export HF_TOKEN=hf_...
  export SLOWER_WHISPER_PYANNOTE_MODE=auto
  # Optional override if you need a different pipeline:
  # export SLOWER_WHISPER_PYANNOTE_MODEL=pyannote/speaker-diarization-3.1
  uv run python benchmarks/eval_diarization.py \
    --dataset benchmarks/data/diarization \
    --manifest benchmarks/data/diarization/manifest.jsonl \
    --device cpu \
    --output-md benchmarks/DIARIZATION_REPORT_REAL.md \
    --output-json benchmarks/DIARIZATION_REPORT_REAL.json \
    --overwrite
  ```
  (Real run not executed here because HF_TOKEN/model access was unavailable in this environment.)

## Run config
- dataset: `benchmarks/data/diarization`
- manifest: `benchmarks/data/diarization/manifest.jsonl` (sha256: 34f8caa31589541c795dcc217df1688440bf25ee45d92669073eafdde0fe0120)
- diarizer device: `cpu`
- min/max speakers: None/None
- SLOWER_WHISPER_PYANNOTE_MODE: `stub`

## Aggregate
- samples: 3
- avg DER: 0.45112834025877496
- speaker count accuracy: 1.0
- total runtime (s): 1.24

## Per-sample
| sample | DER | hyp speakers | exp speakers | runtime (s) | notes | error/detail |
|---|---|---|---|---|---|---|
| synthetic_2speaker | 0.5500 | 2 | 2 | 1.24 | Deterministic 2-speaker A/B tone pattern with 200ms gaps between turns. |  |
| overlap_tones | 0.3768 | 2 | 2 | 0.00 | Two-tone synthetic with overlapping exchanges at 2.0-2.5s and 5.0-5.6s. |  |
| call_mixed | 0.4266 | 2 | 2 | 0.00 | Longer alternating call-style pattern with short pauses; minimal overlap. |  |
