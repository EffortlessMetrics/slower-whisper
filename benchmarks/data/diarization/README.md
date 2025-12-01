# Diarization benchmark fixtures

Tiny deterministic fixtures used by `benchmarks/eval_diarization.py`.

- `manifest.jsonl`: manifest describing the samples.
- `synthetic_2speaker.wav`: 12.6s alternating A/B tone pattern with 200ms gaps.
- `synthetic_2speaker.rttm`: reference RTTM (A: 0-3.0 & 6.4-9.4, B: 3.2-6.2 & 9.6-12.6).
- `overlap_tones.wav`: 6s clip with two tone speakers overlapping briefly at 2.0-2.5s and 5.0-5.6s.
- `call_mixed.wav`: 15.5s alternating tone “call” with short pauses and light overlap.
