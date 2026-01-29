# Migrating from faster-whisper

**slower-whisper** provides a drop-in replacement for **faster-whisper**. Change one import and get the same API, plus optional speaker diarization, audio enrichment, and RAG-ready transcripts.

---

## Quick Migration

```python
# Before
from faster_whisper import WhisperModel

# After (just change the import)
from slower_whisper import WhisperModel
```

That's it. Your existing code works unchanged.

---

## What You Get

### Same API, Same Results

The `slower_whisper` package exports the same types as `faster_whisper`:

| Type | Description |
|------|-------------|
| `WhisperModel` | Main transcription class |
| `Segment` | Transcribed segment (supports tuple unpacking) |
| `Word` | Word-level timestamp |
| `TranscriptionInfo` | Transcription metadata |

```python
from slower_whisper import WhisperModel

model = WhisperModel("base", device="auto", compute_type="int8")
segments, info = model.transcribe("audio.wav", word_timestamps=True)

# Same iteration patterns work
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

# Tuple unpacking still works (backwards compatibility)
for seg in segments:
    id_, seek, start, end, text, *rest = seg
```

### Plus: Enriched Transcripts

When you're ready, enable slower-whisper's enrichment features:

```python
from slower_whisper import WhisperModel

model = WhisperModel("base")
segments, info = model.transcribe(
    "meeting.wav",
    word_timestamps=True,
    diarize=True,   # Enable speaker diarization
    enrich=True,    # Enable audio enrichment (prosody, emotion)
)

# Access enriched transcript via last_transcript property
transcript = model.last_transcript

# Speaker turns
for turn in transcript.turns:
    print(f"Speaker {turn.speaker_id}: {turn.text}")

# Audio features per segment
for seg in transcript.segments:
    if seg.audio_state:
        print(f"Energy: {seg.audio_state.get('energy_db', 'N/A')} dB")
```

---

## API Compatibility

### WhisperModel Parameters

All `faster-whisper` constructor parameters are supported:

```python
model = WhisperModel(
    model_size_or_path="base",    # Model size or path
    device="auto",                 # "auto", "cpu", or "cuda"
    compute_type="int8",           # Compute precision
    device_index=0,                # GPU device index
    cpu_threads=4,                 # CPU threads
    num_workers=1,                 # Transcription workers
    download_root=None,            # Custom model cache
    local_files_only=False,        # Offline mode
)
```

### transcribe() Parameters

All `faster-whisper` transcribe parameters are supported:

```python
segments, info = model.transcribe(
    audio,                          # Path, file-like, or numpy array
    language="en",                  # Language code or None for auto-detect
    task="transcribe",              # "transcribe" or "translate"
    beam_size=5,                    # Beam search size
    word_timestamps=True,           # Enable word-level timestamps
    vad_filter=True,                # Voice activity detection
    vad_parameters={"min_silence_duration_ms": 500},
    # ... all other faster-whisper parameters

    # slower-whisper extensions:
    diarize=False,                  # Speaker diarization
    enrich=False,                   # Audio enrichment
)
```

### Segment Compatibility

The `Segment` type supports both modern attribute access and legacy tuple unpacking:

```python
# Attribute access (recommended)
segment.id
segment.start
segment.end
segment.text
segment.words  # List[Word] if word_timestamps=True

# Tuple unpacking (legacy compatibility)
id_, seek, start, end, text, tokens, avg_logprob, compression_ratio, no_speech_prob, words, temperature = segment

# Indexed access
segment[2]  # start time
segment[4]  # text

# Iteration
list(segment)  # [id, seek, start, end, text, ...]
len(segment)   # 11 fields
```

### Extended Attributes (slower-whisper only)

When using `diarize=True` or `enrich=True`, segments have additional attributes:

```python
segment.speaker      # {"id": "spk_0", "confidence": 0.9}
segment.audio_state  # {"energy_db": -20.0, ...}
```

---

## Key Differences

| Aspect | faster-whisper | slower-whisper |
|--------|----------------|----------------|
| Return type | Generator | List |
| Speaker diarization | Not included | Built-in (`diarize=True`) |
| Audio enrichment | Not included | Built-in (`enrich=True`) |
| Enriched transcript | N/A | `model.last_transcript` |
| Dependencies | CTranslate2 only | Modular (base → full) |

### Return Type: List vs Generator

`slower-whisper` returns a list of segments instead of a generator. This is intentional:

1. Post-processing (diarization, enrichment) requires all segments
2. Simpler usage patterns
3. Consistent behavior with or without enrichment

If you need streaming, use the [Streaming API](STREAMING_API.md) instead.

---

## Installation

### Base Install (transcription only)

```bash
pip install slower-whisper
```

### With Diarization

```bash
pip install slower-whisper[diarization]
```

### With Audio Enrichment

```bash
pip install slower-whisper[enrich-basic]
```

### Full Install

```bash
pip install slower-whisper[full]
```

---

## Common Migration Patterns

### Pattern 1: Simple Transcription

```python
# faster-whisper
from faster_whisper import WhisperModel
model = WhisperModel("base")
segments, info = model.transcribe("audio.wav")
for segment in segments:
    print(segment.text)

# slower-whisper (identical)
from slower_whisper import WhisperModel
model = WhisperModel("base")
segments, info = model.transcribe("audio.wav")
for segment in segments:
    print(segment.text)
```

### Pattern 2: Word Timestamps

```python
# faster-whisper
segments, info = model.transcribe("audio.wav", word_timestamps=True)
for segment in segments:
    for word in segment.words:
        print(f"{word.word} ({word.start:.2f}-{word.end:.2f})")

# slower-whisper (identical)
segments, info = model.transcribe("audio.wav", word_timestamps=True)
for segment in segments:
    for word in segment.words:
        print(f"{word.word} ({word.start:.2f}-{word.end:.2f})")
```

### Pattern 3: Adding Speaker Diarization

```python
# slower-whisper enhancement
from slower_whisper import WhisperModel

model = WhisperModel("base")
segments, info = model.transcribe("meeting.wav", diarize=True)

# Segments now have speaker info
for segment in segments:
    speaker = segment.speaker.get("id", "unknown") if segment.speaker else "unknown"
    print(f"[{speaker}] {segment.text}")

# Or use the enriched transcript
transcript = model.last_transcript
for turn in transcript.turns:
    print(f"[{turn.speaker_id}] {turn.text}")
```

### Pattern 4: LLM Pipeline Integration

```python
from slower_whisper import WhisperModel

model = WhisperModel("base")
segments, info = model.transcribe(
    "meeting.wav",
    diarize=True,
    enrich=True,
)

# Get RAG-ready chunks
transcript = model.last_transcript
for chunk in transcript.chunks:
    # Each chunk has speaker, text, timestamps, and audio features
    print(f"Speaker {chunk.speaker_id}: {chunk.text}")
    print(f"  Timestamps: {chunk.start:.2f}s - {chunk.end:.2f}s")
```

---

## Troubleshooting

### "Module not found: slower_whisper"

Install the package:
```bash
pip install slower-whisper
```

### "Diarization requested but pyannote.audio not available"

Install diarization dependencies:
```bash
pip install slower-whisper[diarization]
```

### "Enrichment requested but dependencies not available"

Install enrichment dependencies:
```bash
pip install slower-whisper[enrich-basic]
```

### Segments returned as list, not generator

This is intentional. If you need streaming, use the [Streaming API](STREAMING_API.md).

---

## Further Reading

- [API Quick Reference](API_QUICK_REFERENCE.md) — Full Python API documentation
- [Speaker Diarization](SPEAKER_DIARIZATION.md) — Diarization configuration
- [Audio Enrichment](AUDIO_ENRICHMENT.md) — Prosody and emotion features
- [Streaming API](STREAMING_API.md) — Real-time transcription
- [Schema Reference](SCHEMA.md) — JSON output format
