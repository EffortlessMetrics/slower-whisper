# Diarization Data Flow - Quick Reference

## TL;DR: All Connected ✅

The complete diarization pipeline from CLI to JSON is **fully connected and functional**. All pieces work together seamlessly.

---

## Quick Flow

```
CLI Flag
  ↓ argparse
TranscriptionConfig (config.py:78-107)
  ↓ transcribe_directory(root, config)
API Layer (api.py:151-223)
  ↓ run_pipeline(..., diarization_config)
Pipeline (pipeline.py:51-145)
  ↓ if diarization_config.enable_diarization
_maybe_run_diarization() (api.py:39-148)
  ├─ Diarizer.run() → speaker_turns (diarization.py:78-211)
  ├─ assign_speakers() → segment.speaker + speakers[] (diarization.py:248-355)
  ├─ build_turns() → turns[] (turns.py:62-138)
  └─ Record meta.diarization (api.py:105-115)
  ↓
write_json() (writers.py:7-43)
  ↓
JSON Output (whisper_json/*.json)
```

---

## File Map

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| CLI definition | `transcription/cli.py` | 97-113 | ✅ |
| Config builder | `transcription/cli.py` | 360-410 | ✅ |
| Config class | `transcription/config.py` | 78-107 | ✅ |
| API entry | `transcription/api.py` | 151-223 | ✅ |
| Diarization orchestrator | `transcription/api.py` | 39-148 | ✅ |
| Diarizer class | `transcription/diarization.py` | 78-211 | ✅ |
| assign_speakers() | `transcription/diarization.py` | 248-355 | ✅ |
| build_turns() | `transcription/turns.py` | 62-138 | ✅ |
| Pipeline | `transcription/pipeline.py` | 51-145 | ✅ |
| JSON writer | `transcription/writers.py` | 7-43 | ✅ |
| Models | `transcription/models.py` | 1-72 | ✅ |

---

## Integration Points (11 Total)

### 1. CLI → Config ✅
```python
# cli.py:402-407
enable_diarization=args.enable_diarization if args.enable_diarization is not None else config.enable_diarization,
min_speakers=args.min_speakers if args.min_speakers is not None else config.min_speakers,
max_speakers=args.max_speakers if args.max_speakers is not None else config.max_speakers,
```

### 2. Config → API ✅
```python
# api.py:200
run_pipeline(app_cfg, diarization_config=config if config.enable_diarization else None)
```

### 3. API → Pipeline ✅
```python
# pipeline.py:119-126
if diarization_config and diarization_config.enable_diarization:
    from .api import _maybe_run_diarization
    transcript = _maybe_run_diarization(transcript, wav, config=diarization_config)
```

### 4. Pipeline → Orchestrator ✅
Calls `_maybe_run_diarization(transcript, wav_path, config)` with full config

### 5. Diarizer Instantiation ✅
```python
# api.py:74-78
diarizer = Diarizer(
    device=config.diarization_device,
    min_speakers=config.min_speakers,
    max_speakers=config.max_speakers,
)
```

### 6. Diarizer.run() ✅
```python
# diarization.py:166-211
speaker_turns = diarizer.run(wav_path)  # Returns list[SpeakerTurn]
```

### 7. assign_speakers() ✅
```python
# api.py:95-99
transcript = assign_speakers(
    transcript,
    speaker_turns,
    overlap_threshold=config.overlap_threshold,
)
```
Populates: `segment.speaker`, `transcript.speakers`

### 8. build_turns() ✅
```python
# api.py:102
transcript = build_turns(transcript)
```
Populates: `transcript.turns`

### 9. Metadata Success ✅
```python
# api.py:105-115
transcript.meta["diarization"] = {
    "status": "success",
    "requested": True,
    "backend": "pyannote.audio",
    "num_speakers": len(transcript.speakers),
}
```

### 10. Metadata Failure ✅
```python
# api.py:140-147
transcript.meta["diarization"] = {
    "status": "failed",
    "requested": True,
    "error": error_msg,
    "error_type": error_type,  # "auth" | "missing_dependency" | "file_not_found" | "unknown"
}
```

### 11. JSON Serialization ✅
```python
# writers.py:35-38
if transcript.speakers is not None:
    data["speakers"] = transcript.speakers
if transcript.turns is not None:
    data["turns"] = transcript.turns
# segments[].speaker already serialized at line 26
```

---

## Graceful Degradation ✅

**On Failure:**
- ✅ Transcript returned unchanged (no data loss)
- ✅ segment.speaker remains None
- ✅ transcript.speakers remains None
- ✅ transcript.turns remains None
- ✅ meta.diarization.status = "failed"
- ✅ meta.diarization.error contains error message
- ✅ meta.diarization.error_type categorized
- ✅ Pipeline continues (no crash)

**Test Coverage:**
- `test_maybe_run_diarization_disabled()` - Verifies early exit
- `test_maybe_run_diarization_graceful_failure()` - Verifies error handling
- `test_transcribe_file_with_diarization_skeleton()` - Integration test

---

## Data Structure Snapshot

### Input
```python
TranscriptionConfig(
    enable_diarization=True,
    diarization_device="auto",
    min_speakers=2,
    max_speakers=4,
    overlap_threshold=0.3,
)
```

### Output
```python
Transcript(
    segments=[
        Segment(
            id=0,
            start=0.0, end=2.5,
            text="Hello",
            speaker={"id": "spk_0", "confidence": 0.95},
            audio_state=None,
            tone=None,
        ),
        ...
    ],
    speakers=[
        {
            "id": "spk_0",
            "label": None,
            "total_speech_time": 10.5,
            "num_segments": 5,
        },
        ...
    ],
    turns=[
        {
            "id": "turn_0",
            "speaker_id": "spk_0",
            "start": 0.0,
            "end": 2.5,
            "segment_ids": [0],
            "text": "Hello",
        },
        ...
    ],
    meta={
        "diarization": {
            "status": "success",
            "requested": True,
            "backend": "pyannote.audio",
            "num_speakers": 2,
        },
        ...
    }
)
```

### JSON Output
```json
{
  "schema_version": 2,
  "file": "audio.wav",
  "language": "en",
  "meta": {
    "diarization": {
      "status": "success|failed",
      "requested": true,
      "backend": "pyannote.audio",
      "num_speakers": 2,
      "error": null,
      "error_type": null
    }
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Hello",
      "speaker": {"id": "spk_0", "confidence": 0.95},
      "tone": null,
      "audio_state": null
    }
  ],
  "speakers": [
    {
      "id": "spk_0",
      "label": null,
      "total_speech_time": 10.5,
      "num_segments": 5
    }
  ],
  "turns": [
    {
      "id": "turn_0",
      "speaker_id": "spk_0",
      "start": 0.0,
      "end": 2.5,
      "segment_ids": [0],
      "text": "Hello"
    }
  ]
}
```

---

## Error Handling

| Error Type | Trigger | Metadata |
|------------|---------|----------|
| `auth` | "HF_TOKEN" in error message | Shows auth failure |
| `missing_dependency` | "pyannote.audio" not installed | Shows missing package |
| `file_not_found` | Audio file doesn't exist | Shows file path issue |
| `unknown` | Any other exception | Generic error handling |

All errors result in:
- Graceful degradation (transcript returned unchanged)
- Detailed error message in metadata
- Pipeline continues without crash

---

## Testing

### Unit Tests ✅
- `test_maybe_run_diarization_disabled()` - Verify flag disables diarization
- `test_maybe_run_diarization_graceful_failure()` - Verify error handling
- `test_transcription_config_diarization_fields()` - Verify config accepts fields
- `test_transcription_config_diarization_defaults()` - Verify defaults are correct

### Integration Test ✅
- `test_synthetic_2speaker_diarization()` - Real pyannote.audio test with fixture

---

## Backward Compatibility ✅

✅ Diarization fields are optional (all default to None/False)
✅ Schema v2 includes speakers/turns as optional
✅ JSON writer checks for None before serializing
✅ Readers gracefully handle missing fields
✅ v1.0 transcripts still load correctly

---

## How It Works (Step by Step)

### Normal Case (Enabled)
1. User runs: `slower-whisper transcribe --enable-diarization`
2. CLI parses flag → TranscriptionConfig.enable_diarization = True
3. API calls pipeline with diarization_config
4. Pipeline checks flag, calls _maybe_run_diarization()
5. Diarizer runs pyannote.audio → SpeakerTurn list
6. assign_speakers() maps to segments → segment.speaker populated
7. build_turns() groups segments → turns array populated
8. Metadata recorded: status = "success"
9. write_json() serializes everything
10. Output: JSON with speakers[], turns[], segment.speaker

### Error Case (Missing Dependency)
1. User runs: `slower-whisper transcribe --enable-diarization`
2. Config assembled with enable_diarization = True
3. Pipeline calls _maybe_run_diarization()
4. Exception: ImportError("pyannote.audio not installed")
5. Caught in except block
6. error_type = "missing_dependency"
7. Metadata recorded: status = "failed", error message
8. Original transcript returned unchanged
9. write_json() serializes with failure metadata
10. Output: JSON without speakers/turns, but with error info

### Disabled Case
1. User runs: `slower-whisper transcribe` (no --enable-diarization)
2. Config assembled with enable_diarization = False (default)
3. Pipeline checks flag, skips diarization call
4. Transcript returned as-is from ASR
5. write_json() serializes without speakers/turns
6. Output: Standard v1.0 JSON (backward compatible)

---

## Verification Checklist

- ✅ CLI flag defined and documented
- ✅ Config field accepted and validated
- ✅ Config passed through all layers
- ✅ Diarizer instantiated with config
- ✅ Diarizer.run() produces speaker turns
- ✅ assign_speakers() populates segment.speaker
- ✅ assign_speakers() builds speakers array
- ✅ build_turns() creates turns array
- ✅ Metadata recorded on success
- ✅ Metadata recorded on failure
- ✅ JSON serializes all fields
- ✅ Graceful degradation works
- ✅ Backward compatibility maintained
- ✅ Tests pass
- ✅ No broken links in chain

---

## Conclusion

**The diarization data flow is complete, connected, and production-ready for v1.1.**

All 11 integration points are verified. Error handling is comprehensive. Graceful degradation works as designed. Backward compatibility is maintained. Tests cover both success and failure paths.

See `DATA_FLOW_ANALYSIS.md` for detailed documentation.
See `DIARIZATION_FLOWCHART.txt` for visual flowchart.
