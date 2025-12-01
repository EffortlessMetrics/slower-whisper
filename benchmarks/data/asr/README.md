# ASR benchmark seed set

This folder now ships with three short, self-owned TTS clips under `audio/` plus a filled `manifest.jsonl`:
- `call_center_narrowband.wav` (8 kHz) — phone-style support reset.
- `team_sync_meeting.wav` (16 kHz) — clean meeting recap with ownership.
- `status_update_clean.wav` (16 kHz) — action-oriented wrap-up.

Add or swap clips (8–60s is fine) by appending rows to `manifest.jsonl` with `audio_path` + `reference_text`.

Run:

```bash
PYTHONPATH=. python benchmarks/eval_asr.py --manifest benchmarks/data/asr/manifest.jsonl --output-md benchmarks/ASR_REPORT.md --output-json benchmarks/ASR_REPORT.json
```

Dependencies:
- `jiwer` (`uv pip install jiwer`)
- Standard slower-whisper runtime (Whisper weights will be downloaded as needed).
