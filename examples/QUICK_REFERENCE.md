# Quick Reference: API Examples

## At a Glance

| Script | Purpose | Use Case | Lines |
|--------|---------|----------|-------|
| `basic_transcription.py` | Simple batch transcription | Process all files in a directory | 152 |
| `single_file.py` | Single file processing | Quick transcription with detailed output | 186 |
| `custom_config.py` | Advanced configuration | Fine-tune quality/speed, use presets | 295 |
| `enrichment_workflow.py` | Two-stage pipeline | Transcribe + enrich with audio features | 352 |

---

## Quick Commands

```bash
# 1. Basic transcription of a directory
python examples/basic_transcription.py /path/to/project --language en

# 2. Transcribe a single file
python examples/single_file.py audio.mp3 ./output --show-segments

# 3. Use configuration preset
python examples/custom_config.py /path/to/project --preset balanced

# 4. Full workflow: transcribe + enrich
python examples/enrichment_workflow.py /path/to/project --transcribe --enrich
```

---

## Feature Matrix

| Feature | basic | single | custom | enrichment |
|---------|-------|--------|--------|------------|
| Batch processing | ✓ | - | ✓ | ✓ |
| Single file | - | ✓ | - | - |
| Custom config | Basic | Basic | Advanced | Advanced |
| Presets | - | - | ✓ | - |
| Interactive mode | - | - | ✓ | - |
| Prosody features | - | - | - | ✓ |
| Emotion detection | - | - | - | ✓ |
| Progress display | ✓ | ✓ | ✓ | ✓ |
| Error handling | ✓ | ✓ | ✓ | ✓ |

---

## When to Use Each Script

### Use `basic_transcription.py` when:
- You have multiple audio files to transcribe
- You want the simplest possible workflow
- You only need text output (JSON, TXT, SRT)
- Default settings are sufficient

### Use `single_file.py` when:
- You have one audio file to process
- You want detailed statistics and preview
- You're testing transcription settings
- You need to see segment-by-segment breakdown

### Use `custom_config.py` when:
- You need specific quality/speed trade-offs
- You're working with limited resources (CPU-only, small models)
- You want to compare different configurations
- You need multilingual or translation features

### Use `enrichment_workflow.py` when:
- You need audio features beyond text (prosody, emotion)
- You're doing sentiment/emotion analysis
- You're analyzing speech patterns (pitch, energy, rate)
- You want to separate transcription and enrichment stages

---

## Configuration Presets (custom_config.py)

```
fast_draft      → tiny model, minimal beam search (fastest)
balanced        → medium model, standard settings (recommended)
high_quality    → large-v3, thorough search (best quality)
cpu_fallback    → medium model, CPU-optimized (no GPU)
multilingual    → auto-detect + translate to English
english_only    → English-specific model (faster for English)
```

---

## Common Parameters

### Transcription
```python
--model <name>           # tiny, base, small, medium, large-v3
--device <cuda|cpu>      # GPU or CPU
--language <code>        # en, es, fr, de, zh, etc. (or auto-detect)
--task <type>            # transcribe or translate
```

### Enrichment
```python
--enrich                    # Enable enrichment stage
--enable-categorical        # Add categorical emotion
--skip-prosody              # Disable prosody features
--skip-emotion              # Disable emotion features
--enrich-device <cuda|cpu>  # Device for enrichment
```

---

## Expected Directory Structure

```
project_root/
├── raw_audio/       ← Put audio files here
├── input_audio/     ← Auto-created (normalized WAV)
├── whisper_json/    ← Auto-created (JSON transcripts)
└── transcripts/     ← Auto-created (TXT, SRT)
```

---

## Typical Workflows

### Workflow 1: Simple Batch Transcription
```bash
# Put files in raw_audio/
mkdir -p project/raw_audio
cp *.mp3 project/raw_audio/

# Transcribe
python examples/basic_transcription.py project --language en

# Results in project/transcripts/
```

### Workflow 2: High-Quality with Enrichment
```bash
# Step 1: Transcribe with best quality
python examples/enrichment_workflow.py project \
    --transcribe --model large-v3 --language en

# Step 2: Enrich with all features
python examples/enrichment_workflow.py project \
    --enrich --enable-categorical

# Step 3: Analyze (use complete_workflow.py)
python examples/complete_workflow.py project/whisper_json/file.json \
    --audio project/input_audio/file.wav \
    --output-csv analysis.csv
```

### Workflow 3: Fast Draft
```bash
# Quick transcription for review
python examples/custom_config.py project --preset fast_draft

# Refine later with higher quality
python examples/custom_config.py project --preset high_quality
```

### Workflow 4: CPU-Only Processing
```bash
# Transcribe on CPU
python examples/custom_config.py project --preset cpu_fallback

# Or single file
python examples/single_file.py audio.mp3 output --device cpu
```

---

## Output Files

Each transcribed audio file generates:

```
audio_file.mp3  →  whisper_json/audio_file.json    (structured data)
                   transcripts/audio_file.txt      (plain text)
                   transcripts/audio_file.srt      (subtitles)
```

---

## Loading & Using Results

```python
from transcription import load_transcript

# Load transcript
transcript = load_transcript("whisper_json/file.json")

# Access data
print(transcript.file_name)      # "file.wav"
print(transcript.language)       # "en"
print(len(transcript.segments))  # 42

# Iterate segments
for segment in transcript.segments:
    print(f"[{segment.start:.1f}s] {segment.text}")

    # If enriched
    if segment.audio_state:
        print(f"  Pitch: {segment.audio_state['pitch']['level']}")
        print(f"  Energy: {segment.audio_state['energy']['level']}")
```

---

## Performance Tips

| Scenario | Recommendation |
|----------|---------------|
| Best quality | `--preset high_quality` |
| Fastest speed | `--preset fast_draft` |
| No GPU | `--preset cpu_fallback` or `--device cpu` |
| Large batch | `basic_transcription.py` with `skip_existing_json=True` |
| One file | `single_file.py` |
| Out of memory | Use smaller model (`--model medium`) |
| Non-English | Don't set `--language` (auto-detect) or `--task translate` |

---

## Error Solutions

| Error | Solution |
|-------|----------|
| "No audio files found" | Check files are in `raw_audio/` subdirectory |
| "CUDA out of memory" | Use `--device cpu` or `--model medium` |
| "Model not found" | First run downloads model (requires internet) |
| Import errors | Install: `pip install -e ".[enrichment]"` |

---

## Next Steps

1. **Start simple:** Try `basic_transcription.py` on a test directory
2. **Single file:** Use `single_file.py` to verify quality
3. **Optimize:** Try presets in `custom_config.py`
4. **Enrich:** Add audio features with `enrichment_workflow.py`
5. **Analyze:** Use `complete_workflow.py` or `query_audio_features.py`

For full documentation, see `API_EXAMPLES_README.md`.
