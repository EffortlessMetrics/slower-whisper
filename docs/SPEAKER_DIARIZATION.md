# Speaker Diarization in slower-whisper

**Status:** v1.1+ - Stable feature (requires extra dependencies)

**Released:** v1.1.0 (2025-11-18)

**Requirements:**
- Install diarization dependencies: `uv sync --extra diarization`
- ffmpeg (system dependency, required for all slower-whisper operations)
- HuggingFace token with access to pyannote models when using the real backend (`auto`): `export HF_TOKEN=...` (not needed for `stub`/`missing`)

**Backend modes (env: `SLOWER_WHISPER_PYANNOTE_MODE`, default `auto`):**
- `auto`: real pyannote if installed (requires HF_TOKEN)
- `stub`: lightweight fake diarization for tests (no HF_TOKEN needed; no model download)
- `missing`: simulate missing dependency/import error for graceful-failure paths

---

## Quality (synthetic fixtures)
- Dataset: `benchmarks/data/diarization` (manifest sha256 `34f8caa31589541c795dcc217df1688440bf25ee45d92669073eafdde0fe0120`).
- Stub backend: `SLOWER_WHISPER_PYANNOTE_MODE=stub`, device `cpu` → avg DER **0.451**, speaker-count accuracy **1.0** (3/3), total runtime **1.24s**. See `benchmarks/DIARIZATION_REPORT.{md,json}` (also used by the CI stub regression check).
- Real backend: `SLOWER_WHISPER_PYANNOTE_MODE=auto` with `HF_TOKEN` + pyannote model (default `pyannote/speaker-diarization-3.1`, override via `SLOWER_WHISPER_PYANNOTE_MODEL`). Generates `benchmarks/DIARIZATION_REPORT_REAL.{md,json}` when run; not executed in this workspace because HF_TOKEN/model access was unavailable.
- Optimized for 2–4 speakers with light overlap; stub DER is only for regression tracking (real pyannote numbers will differ).
- Per-file details live in `benchmarks/DIARIZATION_REPORT.md` (regenerated from `benchmarks/eval_diarization.py`).

| file                    | DER   | ref_speakers | pred_speakers | speaker_count_ok |
| ----------------------- | ----- | ------------ | ------------- | ---------------- |
| synthetic_2speaker      | 0.550 | 2            | 2             | yes              |
| overlap_tones           | 0.377 | 2            | 2             | yes              |
| call_mixed              | 0.427 | 2            | 2             | yes              |

To regenerate the stub run shown above:

```bash
SLOWER_WHISPER_PYANNOTE_MODE=stub \
  uv run python benchmarks/eval_diarization.py \
    --dataset benchmarks/data/diarization \
    --output-md benchmarks/DIARIZATION_REPORT.md \
    --output-json benchmarks/DIARIZATION_REPORT.json \
    --overwrite
```

To run with the real pyannote backend:

```bash
uv sync --extra diarization
export HF_TOKEN=hf_...
export SLOWER_WHISPER_PYANNOTE_MODE=auto
# Optional override if you need a different pipeline:
# export SLOWER_WHISPER_PYANNOTE_MODEL=pyannote/speaker-diarization-3.1
uv run python benchmarks/eval_diarization.py \
  --dataset benchmarks/data/diarization \
  --output-md benchmarks/DIARIZATION_REPORT_REAL.md \
  --output-json benchmarks/DIARIZATION_REPORT_REAL.json \
  --overwrite
```

---

## Overview

Speaker diarization answers the question **"who spoke when?"** by:

1. Identifying distinct speakers in an audio file
2. Attributing each ASR segment to a speaker
3. Building a global `speakers[]` table with metadata
4. Enabling downstream turn structure and speaker-aware analysis

**Design principle:** Never re-run ASR. Diarization operates on existing transcripts + audio.

---

## Design Decisions (v1.1)

These decisions are **locked contracts** for v1.1. Implementation must follow these rules.

### 1. Speaker ID Format

**Decision:** Normalize all speaker IDs to `spk_N` format (`spk_0`, `spk_1`, ...).

**Rationale:**
- Predictable, transport-safe, decoupled from backend (pyannote, future alternatives)
- Raw backend IDs (e.g. pyannote's `SPEAKER_00`) are implementation details, not part of the schema
- Debugging support via `meta.diarization.raw_speaker_ids` if needed (optional)

**Implementation:**
```python
def _normalize_speaker_id(raw_id: str, speaker_map: dict[str, int]) -> str:
    """Map backend speaker IDs to spk_N format."""
    if raw_id not in speaker_map:
        speaker_map[raw_id] = len(speaker_map)
    return f"spk_{speaker_map[raw_id]}"
```

---

### 2. Segment-to-Speaker Assignment Strategy

**Decision:** Max overlap duration wins, with 0.3 confidence threshold for unknown.

**Algorithm:**
1. For each ASR segment `[s_start, s_end]`:
   - Compute overlap duration with all `SpeakerTurn`s
   - Choose speaker with **max overlap duration**
   - Compute `overlap_ratio = overlap_duration / segment_duration`
2. If `overlap_ratio >= 0.3`: assign speaker
3. Else: `segment.speaker = None` (unknown)

**Edge cases:**
- **No overlap** (silence/pause): `segment.speaker = None`
- **Equal overlap** (unlikely): choose first speaker ID alphabetically (deterministic)
- **Partial overlap** (<30%): treat as unknown, don't guess

**Why 0.3?**
- Below 30% overlap, attribution is unreliable
- Better to say "unknown" than mislabel
- LLMs handle `null` speaker fields gracefully

**Configurability:**
- `overlap_threshold` is a parameter (default 0.3)
- Not exposed via CLI in v1.1 (may add `--min-speaker-confidence` in v1.2)

---

### 3. Confidence Semantics

**Decision:** Confidence = normalized overlap ratio (NOT backend's internal confidence).

**Formula:**
```python
segment.speaker.confidence = overlap_duration / segment_duration
# Range: [0.0, 1.0]
# 1.0 = segment fully contained in one speaker turn
# <0.3 = unknown (segment.speaker = None)
```

**Why not use pyannote/backend confidence?**
- Backend confidence is frame-level, not segment-level
- Overlap ratio is a clearer signal for segment attribution
- Easier to interpret, debug, and document

**Schema:**
```json
{
  "speaker": {
    "id": "spk_0",
    "confidence": 0.87
  }
}
```

**Note:** This is NOT a calibrated probability. It represents "fraction of segment dominated by this speaker."

---

### 4. Unknown Segment Handling

**Decision:** `segment.speaker = None` for low-confidence assignments.

**When this occurs:**
- Overlap ratio < 0.3
- Segment in silence/pause region (no diarization output)
- Diarization backend failure (graceful degradation)

**Turn building behavior:**
- Skip `None` speaker segments when building `turns[]`
- Log count of unknown segments for debugging

**Rationale:**
- `None` is cleaner than `"spk_unknown"` (known vs. unknown semantic)
- Easier to filter in downstream code
- Explicit "unknown" speaker ID would imply "someone outside the set"

---

### 5. Turn Structure (v1.1 Minimal)

**Decision:** Start new turn on speaker change; minimal metadata in v1.1.

**v1.1 Turn fields:**
```json
{
  "id": "turn_0",
  "speaker_id": "spk_0",
  "start": 0.0,
  "end": 4.2,
  "segment_ids": [0, 1, 2],
  "text": "concatenated segment text"
}
```

**Turn building rules:**
- Group contiguous segments with same `speaker_id`
- Ignore segments with `speaker = None`
- Sort by start time (should already be sorted from ASR)
- Text concatenation: `" ".join(seg.text.strip() for seg in turn_segments)`

**Deferred to v1.2:**
- `interruption_count`, `question_count`, `sentiment_summary`
- `pause_before_ms`, `pause_after_ms`
- Overlap detection and flagging

---

### 6. Per-Speaker Prosody Baselines

**Decision:** Compute after diarization, store in `meta.audio_enrichment.speaker_baselines`.

**Storage location:**
```json
{
  "meta": {
    "audio_enrichment": {
      "speaker_baselines": {
        "spk_0": {"pitch_median_hz": 120.5, "energy_median_db": -22.3},
        "spk_1": {"pitch_median_hz": 215.8, "energy_median_db": -20.1}
      }
    }
  }
}
```

**Behavior:**
- If `len(transcript.speakers) > 0`: compute per-speaker baselines
- Else: use global baseline (current v1.0 behavior)
- Enrichment still runs if diarization disabled (no breaking changes)

---

### 7. Error Handling & Graceful Degradation

**Decision:** Diarization failures don't block transcription.

**Failure modes:**
1. **Backend not installed** (pyannote missing): Log warning, proceed with `speakers = None`
2. **Diarization crashes**: Log error, proceed with `speakers = None`
3. **0 speakers detected**: Log warning, set `speakers = []` (valid but surprising)
4. **>10 speakers detected**: Log warning (likely incorrect), proceed anyway

**Logging:**
```python
logger.warning(
    f"Diarization failed for {wav_path}: {exc}. "
    "Proceeding with speakers=None."
)
```

**Metadata tracking:**
```json
{
  "meta": {
    "diarization": {
      "status": "success" | "skipped" | "failed",
      "backend": "pyannote.audio",
      "model_version": "3.1.1",
      "num_speakers": 2
    }
  }
}
```

---

## Architecture

### Components

**1. Diarization Engine** (`transcription/diarization.py`)
- **Model:** pyannote.audio 3.1+ (https://github.com/pyannote/pyannote-audio)
- **Input:** Normalized 16kHz mono WAV (from `input_audio/`)
- **Output:** List of `SpeakerTurn` objects with (start, end, speaker_id, confidence)

**2. Speaker-to-Segment Assignment** (`transcription/diarization.py::assign_speakers_to_segments()`)
- **Input:** ASR segments + diarization speaker turns
- **Logic:** Assign speaker with maximum overlap (IoU or duration-based)
- **Output:** Segments with `speaker` field populated + `speakers[]` metadata

**3. Turn Builder** (`transcription/turns.py::build_turns()`)
- **Input:** Speaker-attributed segments
- **Logic:** Group contiguous segments by same speaker
- **Output:** `turns[]` array with turn-level structure

**4. CLI Integration** (`transcription/cli.py`)
- **Flag:** `--enable-diarization` (experimental in v1.1)
- **Behavior:** Run diarization after ASR, before enrichment

### Data Flow

```
┌────────────────┐
│  ASR (Stage 1) │  → transcript.json (speaker = null)
└────────┬───────┘
         │
    --enable-diarization?
         │
         ▼
┌─────────────────────┐
│  Diarization (L2)   │  → SpeakerTurn[]
│  (pyannote.audio)   │     (start, end, speaker_id)
└────────┬────────────┘
         │
         ▼
┌──────────────────────────┐
│  Assign Speakers (L2)    │  → transcript.json
│  (overlap-based)         │     + speakers[] array
│                          │     + segment.speaker populated
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Build Turns (L2)        │  → transcript.json
│  (group by speaker)      │     + turns[] array
└──────────────────────────┘
```

---

## Schema Impact

### Before Diarization (v1.0 / v1.1 default)

```json
{
  "schema_version": 2,
  "file": "meeting.wav",
  "language": "en",
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello everyone.", "speaker": null},
    {"id": 1, "start": 2.7, "end": 5.0, "text": "Thanks for joining.", "speaker": null}
  ],
  "speakers": null,
  "turns": null
}
```

### After Diarization (v1.1 with --enable-diarization)

```json
{
  "schema_version": 2,
  "file": "meeting.wav",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Hello everyone.",
      "speaker": {"id": "spk_0", "confidence": 0.92}
    },
    {
      "id": 1,
      "start": 2.7,
      "end": 5.0,
      "text": "Thanks for joining.",
      "speaker": {"id": "spk_1", "confidence": 0.89}
    }
  ],
  "speakers": [
    {
      "id": "spk_0",
      "label": null,
      "total_speech_time": 2.5,
      "num_segments": 1
    },
    {
      "id": "spk_1",
      "label": null,
      "total_speech_time": 2.3,
      "num_segments": 1
    }
  ],
  "turns": [
    {
      "id": "turn_0",
      "speaker_id": "spk_0",
      "start": 0.0,
      "end": 2.5,
      "segment_ids": [0],
      "text": "Hello everyone."
    },
    {
      "id": "turn_1",
      "speaker_id": "spk_1",
      "start": 2.7,
      "end": 5.0,
      "segment_ids": [1],
      "text": "Thanks for joining."
    }
  ]
}
```

**Key changes:**
- `segment.speaker` populated with `{id, confidence}` (e.g. `{"id": "spk_0", "confidence": 0.92}`)
- `speakers[]` array with per-speaker aggregates
- `turns[]` array grouping segments by speaker

**Backward compatibility:**
- Schema version remains `2` (adding optional fields)
- v1.0 consumers ignore `speakers` and `turns` (graceful degradation)
- v1.0 transcripts load into v1.1+ (null speakers/turns)

---

## Implementation Plan (v1.1)

### Phase 1: Skeleton API ✅ Completed

**Completed:**
- [x] `transcription/diarization.py` with `Diarizer` class
- [x] `transcription/turns.py` with `build_turns()` function
- [x] `--enable-diarization` CLI flag (shows experimental warning)
- [x] BDD scenarios for nullable speaker fields
- [x] Schema updated with `speakers` and `turns` fields

### Phase 2: pyannote Integration ✅ Completed

**Completed:**
1. [x] pyannote.audio dependency added (optional extra: `diarization`)
2. [x] `Diarizer.run()` implemented
   - Lazy model loading with GPU/CPU device selection
   - Runs diarization on 16kHz mono WAV
   - Converts pyannote output to `List[SpeakerTurn]`
   - Handles min/max speaker constraints
3. [x] `assign_speakers_to_segments()` implemented
   - Overlap-based assignment with 0.3 threshold
   - Builds `SpeakerInfo` aggregates
   - Handles low-confidence cases (speaker = null)
4. [x] Wired into `transcribe_directory()` in `api.py`
   - Runs after ASR completes when diarization enabled
   - Assigns speakers to segments
   - Updates transcript with speakers and turns
5. [x] Tested on synthetic 2-speaker audio
   - `tests/fixtures/synthetic_2speaker.wav` generated
   - 14 assignment tests + 28 turn tests passing

### Phase 3: Turn Building ✅ Completed

**Completed:**
1. [x] `build_turns()` implemented
   - Groups contiguous segments by speaker
   - Handles segments with no speaker attribution
   - Populates basic turn metadata (id, speaker_id, start, end, segment_ids, text)
2. [x] Wired into pipeline after speaker assignment
3. [x] Tested on synthetic multi-turn conversation

### Phase 4: Real-World Validation (Future: v1.2+)

**Tasks:**
1. Acquire AMI Meeting Corpus subset (5-10 excerpts, 30-60s each)
2. Run diarization benchmark
3. Measure DER, speaker count accuracy, segment attribution F1
4. Document failure modes (overlaps, noise, etc.)
5. Tune overlap threshold if needed

---

## MVP Testbed

### Synthetic 2-Speaker Test

**File:** `tests/fixtures/synthetic_2speaker.wav`

**Structure:**
- **0.0–3.0s:** Speaker A (male voice, low pitch ~120 Hz)
  - Text: "Hello, this is the first speaker."
- **3.2–6.0s:** Speaker B (female voice, high pitch ~220 Hz)
  - Text: "And this is the second speaker responding."
- **6.2–9.0s:** Speaker A
  - Text: "I am speaking again now."
- **9.2–12.0s:** Speaker B
  - Text: "I will conclude this conversation."

**Expected Output:**
- `speakers`: 2 speakers (`spk_0`, `spk_1`) — normalized IDs, not raw backend labels
- `segments`: 4 segments, alternating A-B-A-B
- `turns`: 4 turns
- **DER = 0.00** (perfect on synthetic)

**Note on Speaker IDs:**
- pyannote.audio emits raw labels like `"SPEAKER_00"`, `"SPEAKER_01"` (internal only)
- `assign_speakers()` normalizes these to canonical `spk_N` IDs for schema consistency
- All schema fields (`segment.speaker.id`, `speakers[].id`, `turns[].speaker_id`) use `spk_N`

**Generation:**
```python
# tests/fixtures/generate_synthetic_2speaker.py
# Use TTS (gTTS, pyttsx3, or similar) with distinct voices
# Concatenate with 200ms pauses between speakers
# Save as 16kHz mono WAV
```

### AMI Meeting Corpus Subset (Real-World)

**Dataset:** AMI Meeting Corpus (http://groups.inf.ed.ac.uk/ami/corpus/)

**Subset Selection:**
- 5-10 meetings, 30-60s excerpts
- Ground truth speaker labels
- Mix of 2-4 speaker scenarios
- Includes some overlapping speech

**Quality Targets:**
- **DER < 0.25** (good enough for conversation intelligence)
- **Speaker count accuracy > 80%**
- **Segment attribution F1 > 0.75**

**Benchmark Script:**
```bash
# benchmarks/eval_diarization.py
uv run python benchmarks/eval_diarization.py \
  --dataset ami_subset \
  --metric DER \
  --threshold 0.25 \
  --output results/diarization_v1.1.json
```

---

## Configuration

### TranscriptionConfig Extension (Future)

```python
@dataclass
class TranscriptionConfig:
    # ... existing fields ...

    # v1.1+ diarization settings
    enable_diarization: bool = False
    diarization_device: str = "auto"  # "cuda", "cpu", or "auto"
    min_speakers: int | None = None   # None = auto-detect
    max_speakers: int | None = None   # None = auto-detect
    overlap_threshold: float = 0.5    # Minimum overlap to assign speaker
```

### CLI Usage

```bash
# Transcribe with diarization (v1.1 experimental)
uv run slower-whisper transcribe --enable-diarization

# Transcribe with speaker count hint
uv run slower-whisper transcribe \
  --enable-diarization \
  --min-speakers 2 \
  --max-speakers 4

# Diarization on GPU (faster)
uv run slower-whisper transcribe \
  --enable-diarization \
  --diarization-device cuda
```

---

## Caching Strategy

**Goal:** Never re-run diarization on same audio + config.

**Cache key:**
```python
cache_key = hash(
    audio_file_hash,
    diarization_config.enable_diarization,
    diarization_config.min_speakers,
    diarization_config.max_speakers,
    diarization_config.overlap_threshold,
    DIARIZATION_MODEL_VERSION,  # e.g., "pyannote-3.1"
)
```

**Storage:**
- Diarization results stored in `whisper_json/` alongside transcripts
- File: `{basename}_diarization.json`
- Contains: `SpeakerTurn[]` + metadata (model, timestamp, config)

**Invalidation:**
- Audio file changed (hash mismatch)
- Config changed (different thresholds, speaker counts)
- Model upgraded (DIARIZATION_MODEL_VERSION bumped)

---

## Performance Expectations

**Processing speed:**
- pyannote.audio diarization: ~1-2x realtime on CPU, ~0.3-0.5x on GPU
- Speaker assignment: near-instant (< 0.1s per file)
- Turn building: near-instant (< 0.1s per file)

**Memory:**
- pyannote models: ~1.5GB VRAM (GPU) or RAM (CPU)
- No significant increase to overall pipeline memory

**End-to-end:**
- 10-minute audio: ~15-30s diarization overhead (GPU)
- 60-minute audio: ~2-4 minutes diarization overhead (GPU)

---

## Error Handling

**Diarization failures:**
- pyannote model download fails → log warning, continue with speaker = null
- Audio format invalid → raise error (should never happen after normalization)
- GPU OOM → fallback to CPU (log warning)
- No speakers detected → empty `speakers[]`, all segments have speaker = null

**Graceful degradation:**
- If diarization fails for any reason, transcript is still usable (speaker fields null)
- Never block transcription on diarization errors
- Always log detailed error messages with traceback

---

## Future Enhancements (v1.2+)

**v1.2: Speaker Analytics**
- Turn-level metadata (questions, interruptions, pauses, disfluency)
- Per-speaker sentiment/emotion aggregation
- `speaker_stats[]` with interaction patterns

**v1.3: Advanced Diarization**
- Word-level speaker alignment (not just segment-level)
- Overlapping speech detection and handling
- Speaker embedding extraction for re-identification

**v2.0: Speaker Identification**
- Optional speaker identification (not just diarization)
- User-provided speaker embeddings or voice samples
- "Speaker A is John" vs "SPEAKER_00 is John"

---

## Testing Checklist (v1.1)

Before releasing v1.1 with diarization:

- [ ] `Diarizer.run()` implemented and tested on synthetic 2-speaker
- [ ] `assign_speakers_to_segments()` implemented with overlap logic
- [ ] `build_turns()` implemented for basic grouping
- [ ] BDD scenarios pass (speaker fields, turns structure)
- [ ] Synthetic 2-speaker test: DER = 0.00, 2 speakers, A-B-A-B pattern
- [ ] Single-speaker test: No false multi-speaker detection
- [ ] AMI subset: DER < 0.25 (if dataset acquired)
- [ ] Documentation updated (README, ARCHITECTURE, CLAUDE.md)
- [ ] CLI flag works end-to-end
- [ ] Cache invalidation tested
- [ ] Error handling tested (GPU OOM, missing model, invalid audio)

**Quality thresholds and test scenarios:** See [`docs/TESTING_STRATEGY.md`](TESTING_STRATEGY.md#layer-2-speaker-diarization-v11) for detailed DER thresholds, BDD scenarios, and acceptance criteria that must pass before promoting v1.1 to stable.

---

## Questions or Contributions?

- **Implementation questions:** Open GitHub issue with `diarization` label
- **Bug reports:** Include audio file, config, and error traceback
- **Feature requests:** Open discussion for v1.2+ enhancements

---

**Document History:**
- 2025-11-17: Initial speaker diarization design (v1.1 skeleton)
