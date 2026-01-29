# Performance Guide

This guide covers performance characteristics, memory requirements, and tuning recommendations for slower-whisper.

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Throughput Benchmarks](#throughput-benchmarks)
3. [Memory Requirements](#memory-requirements)
4. [Latency Characteristics](#latency-characteristics)
5. [Tuning Recommendations](#tuning-recommendations)
6. [Profiling Your Workload](#profiling-your-workload)

---

## Quick Reference

| Workload | Device | Model | Expected RTF | Memory |
|----------|--------|-------|--------------|--------|
| ASR only | GPU | large-v3 | 10-20x | ~10 GB VRAM |
| ASR only | GPU | base | 50-100x | ~1 GB VRAM |
| ASR only | CPU | large-v3 | 0.3-1x | ~4 GB RAM |
| ASR only | CPU | base | 5-10x | ~500 MB RAM |
| ASR + diarization | GPU | large-v3 | 5-10x | ~13 GB VRAM |
| ASR + diarization + enrichment | GPU | large-v3 | 3-5x | ~15 GB VRAM |

**RTF** = Real-Time Factor (10x means 10 minutes of audio processed per minute of wall time)

---

## Throughput Benchmarks

### Run the Benchmark

```bash
uv run python benchmarks/throughput.py \
  --audio raw_audio/test_sample.wav \
  --model base \
  --device auto
```

**Options:**
- `--device cpu|cuda|auto` — execution device
- `--model tiny|base|small|medium|large-v3` — Whisper model size
- `--enable-diarization` — include speaker identification
- `--enable-emotion` — include emotion extraction
- `--skip-enrich` — ASR-only (no prosody/emotion)

### Example Results

**GPU (RTX 3090, large-v3, 10-minute audio):**
```
Audio duration: 600.00 s
ASR wall time:   45.23 s  (13.3x realtime)
Diarization:     12.41 s
Enrichment:      28.92 s  (prosody + dimensional emotion)
Total:           86.56 s  (6.9x realtime)
```

**CPU (i9-13900K, base model, 10-minute audio):**
```
Audio duration: 600.00 s
ASR wall time:   82.14 s  (7.3x realtime)
Enrichment:     skipped
```

### Model-Specific Throughput

Tested on RTX 3090 with float16, single audio file:

| Model | Parameters | ASR RTF | + Diarization | + Full Enrichment |
|-------|------------|---------|---------------|-------------------|
| tiny | 39M | ~80x | ~40x | ~20x |
| base | 74M | ~50x | ~30x | ~15x |
| small | 244M | ~25x | ~15x | ~8x |
| medium | 769M | ~12x | ~8x | ~5x |
| large-v3 | 1.5B | ~15x | ~8x | ~5x |

**Note:** large-v3 uses optimized attention and can be faster than medium despite more parameters.

---

## Memory Requirements

### VRAM (GPU)

| Component | Memory | Notes |
|-----------|--------|-------|
| Whisper tiny | ~1 GB | float16 |
| Whisper base | ~1 GB | float16 |
| Whisper small | ~2 GB | float16 |
| Whisper medium | ~5 GB | float16 |
| Whisper large-v3 | ~10 GB | float16 |
| pyannote diarization | ~2.5 GB | Required for speaker ID |
| Dimensional emotion (wav2vec2) | ~1.5 GB | Optional enrichment |
| Categorical emotion | ~1.5 GB | Optional enrichment |

**Total for full pipeline (large-v3 + diarization + all enrichment):** ~15-16 GB VRAM

### RAM (CPU)

| Component | Memory | Notes |
|-----------|--------|-------|
| Whisper base (int8) | ~500 MB | Quantized for CPU |
| Whisper large-v3 (int8) | ~4 GB | Quantized for CPU |
| Audio buffer (1 hour) | ~350 MB | 16kHz mono float32 |
| pyannote diarization | ~2 GB | On CPU if no VRAM |
| Prosody extraction | ~200 MB | librosa + parselmouth |
| Emotion models (CPU) | ~3 GB | Falls back if no GPU |

**Peak RAM for full pipeline on CPU:** ~8-10 GB

### Reducing Memory Usage

```bash
# Minimal memory: small model, no enrichment
slower-whisper transcribe --model base --skip-enrich

# Medium memory: large model, prosody only (no emotion)
slower-whisper transcribe --model large-v3
slower-whisper enrich --no-enable-emotion

# Trade VRAM for speed: int8 quantization
slower-whisper transcribe --model large-v3 --compute-type int8
```

---

## Latency Characteristics

### Batch Processing Latency

For file transcription, latency depends on:
1. **Model loading** — 5-30 seconds first call (cached after)
2. **ASR inference** — proportional to audio length
3. **Diarization** — ~0.1x audio duration on GPU
4. **Enrichment** — ~0.05x audio duration (prosody), ~0.2x (emotion)

### Streaming Latency

For real-time streaming (WebSocket), target latencies:

| Stage | Target | Measured (P95) |
|-------|--------|----------------|
| VAD detection | < 100ms | ~80ms |
| ASR segment | < 500ms | ~300ms |
| Diarization update | < 200ms | ~150ms |
| End-to-end (partial) | < 800ms | ~600ms |
| End-to-end (final) | < 1.5s | ~1.2s |

**Note:** Streaming uses chunked inference. Latency varies with chunk size (default 2s chunks).

---

## Tuning Recommendations

### For Maximum Throughput

```bash
# GPU: larger batches, float16
slower-whisper transcribe \
  --model large-v3 \
  --device cuda \
  --compute-type float16 \
  --batch-size 16

# CPU: int8 quantization, smaller model
slower-whisper transcribe \
  --model small \
  --device cpu \
  --compute-type int8
```

### For Minimum Latency (Streaming)

```bash
# Smaller chunks, faster VAD
slower-whisper stream \
  --model base \
  --device cuda \
  --chunk-duration 1.0 \
  --vad-min-silence-ms 200
```

### For Minimum Memory

```bash
# Sequential processing, no parallel enrichment
slower-whisper transcribe \
  --model tiny \
  --device cpu \
  --compute-type int8

# Enrich separately (clears ASR model from memory first)
slower-whisper enrich --no-enable-emotion
```

### For Best Accuracy

```bash
# Largest model, beam search, word timestamps
slower-whisper transcribe \
  --model large-v3 \
  --device cuda \
  --beam-size 5 \
  --word-timestamps true \
  --language en  # Explicit language avoids detection errors
```

---

## Profiling Your Workload

### Basic Timing

```bash
# Time each stage
time slower-whisper transcribe --root . --model base
time slower-whisper enrich --root .
```

### Detailed Profiling

```python
from transcription.api import transcribe_file
import cProfile
import pstats

with cProfile.Profile() as pr:
    result = transcribe_file("audio.wav", model="base")

stats = pstats.Stats(pr)
stats.sort_stats("cumulative")
stats.print_stats(20)
```

### GPU Profiling

```bash
# Monitor GPU utilization during transcription
watch -n 0.5 nvidia-smi

# Detailed GPU profiling
nsys profile --stats=true python -m transcription.cli transcribe --root .
```

### Memory Profiling

```bash
# Peak memory usage
/usr/bin/time -v slower-whisper transcribe --model large-v3

# Python memory profiling
pip install memory-profiler
python -m memory_profiler transcription/cli.py transcribe --root .
```

---

## Performance by Audio Type

| Audio Type | Characteristics | Recommendations |
|------------|-----------------|-----------------|
| Clean speech | Single speaker, studio quality | `--model base` sufficient |
| Meetings | Multiple speakers, overlapping | `--model large-v3`, enable diarization |
| Phone calls | Compressed, background noise | `--model large-v3`, `--vad-min-silence-ms 400` |
| Podcasts | 2-3 speakers, good quality | `--model medium`, enable diarization |
| Noisy environments | Background music/noise | `--model large-v3`, may need preprocessing |

---

## Scaling Considerations

### Parallel File Processing

slower-whisper processes files sequentially by default. For parallel processing:

```bash
# Process multiple directories in parallel (separate processes)
slower-whisper transcribe --root /data/batch1 &
slower-whisper transcribe --root /data/batch2 &
wait
```

**Caution:** Each process loads its own model copy. Ensure sufficient GPU memory.

### Queue-Based Processing

For high-throughput production workloads, consider:
1. Pre-load model once, process queue of files
2. Use the Python API directly for fine-grained control
3. Deploy as a service (`slower-whisper serve`) for HTTP/WebSocket access

```python
from transcription.api import create_transcriber

# Load once
transcriber = create_transcriber(model="large-v3", device="cuda")

# Process many
for audio_file in audio_queue:
    result = transcriber.transcribe(audio_file)
    save_result(result)
```

---

## Expected Degradation

Performance degrades gracefully under resource pressure:

| Condition | Behavior |
|-----------|----------|
| Low VRAM | Falls back to CPU (with warning) |
| Slow disk | Bottleneck on audio I/O, not inference |
| High CPU load | Enrichment slows, ASR (GPU) unaffected |
| Memory pressure | May swap, significant slowdown |

---

## See Also

- [GPU_SETUP.md](GPU_SETUP.md) — GPU configuration and troubleshooting
- [CONFIGURATION.md](CONFIGURATION.md) — All configuration options
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues and solutions
- [docs/BENCHMARKS.md](BENCHMARKS.md) — Formal benchmark methodology
