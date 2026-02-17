# Diarization benchmark fixtures

Deterministic fixtures used by `benchmarks/eval_diarization.py`.

Primary smoke fixtures (speech-based):

- `manifest.jsonl`: manifest describing the active speech smoke samples.
- `meeting_dual_voice.wav` + `meeting_dual_voice.rttm`
- `support_handoff_dual_voice.wav` + `support_handoff_dual_voice.rttm`
- `planning_sync_dual_voice.wav` + `planning_sync_dual_voice.rttm`

Legacy tone fixtures retained for protocol/mapping checks:

- `synthetic_2speaker.wav` + `synthetic_2speaker.rttm`
- `overlap_tones.wav` + `overlap_tones.rttm`
- `call_mixed.wav` + `call_mixed.rttm`

Regenerate the speech fixtures with:

```bash
uv run python benchmarks/scripts/generate_diarization_speech_smoke.py
```
