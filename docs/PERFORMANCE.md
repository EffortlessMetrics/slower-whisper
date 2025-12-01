# Performance (illustrative)

Quick way to sanity-check throughput on your own hardware.

## Run the probe

```bash
uv run python benchmarks/throughput.py \
  --audio raw_audio/test_sample.wav \
  --model tiny \
  --device cpu \
  --skip-enrich
```

Options:
- `--device cpu|cuda` – pick the execution device.
- `--model` – choose a Whisper size (`tiny`/`base`/etc.).
- `--enable-semantic-annotator` / `--enable-emotion` – include optional enrichers.
- `--skip-enrich` – measure ASR only.

## Example result (CPU, tiny model)

```
Audio duration: 10.00 s
ASR wall time:  4.53 s  (2.21x realtime)
Enrich wall:    skipped
```

Notes:
- Sample audio is short and mostly silence; use your own clips for realistic numbers.
- Enabling enrichment will add overhead; RTF will drop accordingly.
- These numbers are **indicative, not guarantees**. Hardware, model size, and audio content matter.
