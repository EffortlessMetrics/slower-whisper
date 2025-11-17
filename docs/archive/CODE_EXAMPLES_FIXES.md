# Code Examples - Issues and Suggested Fixes

## Summary

✓ **All 22 Python examples are syntactically correct and use proper imports/signatures**
⚠ **15 bash examples use deprecated CLI patterns** (but still work - backward compatible)
✗ **No breaking errors found**

---

## Issues Requiring Fixes

### Issue 1: Deprecated `slower-whisper-enrich` Command

**Location:** README.md, line ~417-419

**Current Code:**
```bash
# Enrich using the CLI tool
uv run slower-whisper-enrich whisper_json/meeting1.json input_audio/meeting1.wav
```

**Problem:** The `slower-whisper-enrich` command is deprecated. The unified CLI uses `slower-whisper enrich` instead.

**Fix:**
```bash
# Enrich using the unified CLI tool
uv run slower-whisper enrich

# Or for a specific file, use the example script:
uv run python examples/emotion_integration.py enrich whisper_json/meeting1.json input_audio/meeting1.wav
```

---

### Issue 2: Legacy CLI in "Usage Example" Section

**Location:** README.md, line ~406

**Current Code:**
```bash
# Enrich existing transcript with emotion analysis
uv run python examples/emotion_integration.py enrich whisper_json/meeting1.json input_audio/meeting1.wav
```

**Problem:** While this works, it's mixing the example script usage with the CLI. Should clarify the difference.

**Fix:**
```bash
# Stage 2: Enrich with audio features
uv run slower-whisper enrich

# Or run the example script for detailed emotion analysis:
uv run python examples/emotion_integration.py enrich whisper_json/meeting1.json input_audio/meeting1.wav
```

---

### Issue 3: Legacy CLI Examples in ARCHITECTURE.md

**Location:** ARCHITECTURE.md, lines ~196-211 (Section: "7. CLI Tool")

**Current Code:**
```bash
# Enrich all transcripts
python audio_enrich.py

# Enrich specific file
python audio_enrich.py --file whisper_json/meeting.json

# Skip already-enriched files
python audio_enrich.py --skip-existing

# CPU-only mode
python audio_enrich.py --device cpu

# Disable specific features
python audio_enrich.py --no-enable-emotion
```

**Problem:** These examples use the legacy `audio_enrich.py` script. Users should use the unified CLI.

**Fix:**
```bash
# Unified CLI (Recommended)
# -------------------------

# Enrich all transcripts
uv run slower-whisper enrich

# Enrich with custom options
uv run slower-whisper enrich --root /path/to/project

# Skip already-enriched files (default behavior)
uv run slower-whisper enrich --skip-existing

# CPU-only mode
uv run slower-whisper enrich --device cpu

# Disable specific features
uv run slower-whisper enrich --no-enable-emotion

# Legacy Script (Deprecated, but still supported)
# -----------------------------------------------
python audio_enrich.py
python audio_enrich.py --file whisper_json/meeting.json
```

---

### Issue 4: Legacy CLI in "Basic Two-Stage Pipeline"

**Location:** ARCHITECTURE.md, lines ~447-457

**Current Code:**
```bash
# Stage 1: Transcribe
python transcribe_pipeline.py --language en

# Stage 2: Enrich
python audio_enrich.py

# Verify
cat whisper_json/meeting.json | jq '.segments[0].audio_state'
```

**Problem:** Uses legacy script names instead of unified CLI.

**Fix:**
```bash
# Stage 1: Transcribe
uv run slower-whisper transcribe --language en

# Stage 2: Enrich
uv run slower-whisper enrich

# Verify
cat whisper_json/meeting.json | jq '.segments[0].audio_state'
```

---

### Issue 5: Legacy CLI in "Advanced Usage"

**Location:** ARCHITECTURE.md, lines ~459-472

**Current Code:**
```bash
# Prosody only (fast, no GPU)
python audio_enrich.py --no-enable-emotion

# Full emotional analysis (slower)
python audio_enrich.py --enable-categorical-emotion

# Skip already-enriched files
python audio_enrich.py --skip-existing

# Single file
python audio_enrich.py --file whisper_json/meeting.json
```

**Problem:** Uses legacy script names.

**Fix:**
```bash
# Prosody only (fast, no GPU)
uv run slower-whisper enrich --no-enable-emotion

# Full emotional analysis (slower)
uv run slower-whisper enrich --enable-categorical-emotion

# Skip already-enriched files (default)
uv run slower-whisper enrich --skip-existing

# For single-file processing, use the Python API:
from transcription import enrich_transcript, load_transcript, EnrichmentConfig

transcript = load_transcript("whisper_json/meeting.json")
config = EnrichmentConfig(enable_prosody=True, enable_emotion=True)
enriched = enrich_transcript(transcript, "input_audio/meeting.wav", config)
```

---

### Issue 6: Inconsistent CLI Example in README "Legacy CLI" Section

**Location:** README.md, lines ~287-293

**Current Code:**
```bash
# Use a lighter model and quantized weights
uv run slower-whisper --model medium --compute-type int8_float16

# Skip files that already have JSON output
uv run slower-whisper --skip-existing-json
```

**Problem:** This is in the "Legacy CLI" section but uses modern syntax **without the subcommand** (`transcribe`).

**Fix:**
```bash
# Use a lighter model and quantized weights
uv run slower-whisper transcribe --model medium --compute-type int8_float16

# Skip files that already have JSON output
uv run slower-whisper transcribe --skip-existing-json
```

**OR** if keeping for legacy documentation:
```bash
# Legacy style (deprecated)
python transcribe_pipeline.py --model medium --compute-type int8_float16
python transcribe_pipeline.py --skip-existing-json
```

---

## Non-Issues (Correctly Documented as Legacy)

### README.md lines 279-282

**Code:**
```bash
# Old style (still supported)
uv run python transcribe_pipeline.py
uv run python audio_enrich.py

# New unified style (recommended)
uv run slower-whisper transcribe
uv run slower-whisper enrich
```

**Status:** ✓ **Correct** - This is in the "Legacy CLI (Backward Compatibility)" section and properly contrasts old vs new styles.

---

## All Valid Python Examples (No Changes Needed)

The following Python examples are **perfect** and require **no changes**:

### ✓ README.md Examples
1. **Basic transcription** (lines 214-224)
2. **Single file transcription** (lines 228-241)
3. **Audio enrichment** (lines 244-261)
4. **Load and save transcripts** (lines 264-273)

### ✓ ARCHITECTURE.md Examples
1. **Schema version check** (lines 103-112)
2. **Safe audio state access** (lines 115-121)
3. **Audio rendering input** (lines 176-182)
4. **Complete workflow functions** (lines 355-367)

### ✓ API_QUICK_REFERENCE.md Examples
All 14 Python examples are valid:
1. Import statement (lines 5-21)
2. Directory transcription (lines 27-35)
3. Single file transcription (lines 39-47)
4. Directory enrichment (lines 53-61)
5. Single transcript enrichment (lines 65-73)
6. I/O operations (lines 77-83)
7. TranscriptionConfig (lines 89-100)
8. EnrichmentConfig (lines 104-112)
9. Transcript structure (lines 118-123)
10. Segment structure (lines 127-134)
11. Audio state access (lines 138-154)
12. Full pipeline (lines 190-198)
13. Custom processing (lines 202-215)
14. Error handling (lines 219-227)

---

## Recommended Changes Summary

### High Priority (User-Facing Documentation)

1. **README.md line 417**: Change `slower-whisper-enrich` → `slower-whisper enrich`
2. **README.md lines 287-293**: Add `transcribe` subcommand to legacy examples or clarify they're wrong

### Medium Priority (Internal Documentation)

3. **ARCHITECTURE.md section 7**: Add deprecation notice and show modern CLI equivalents
4. **ARCHITECTURE.md "Basic Two-Stage Pipeline"**: Update to use unified CLI
5. **ARCHITECTURE.md "Advanced Usage"**: Update to use unified CLI

### Low Priority (Historical Reference)

ARCHITECTURE.md is partly historical documentation. Consider adding a note at the top:

```markdown
> **Note:** This document includes historical implementation details.
> For current CLI usage, see README.md or API_QUICK_REFERENCE.md.
```

---

## Testing Recommendations

All Python examples should be tested with:

```bash
# Test imports work
python -c "from transcription import transcribe_directory, TranscriptionConfig"

# Test config creation
python -c "from transcription import TranscriptionConfig; c = TranscriptionConfig(model='base')"

# Test all public API imports
python -c "from transcription import (
    transcribe_directory, transcribe_file,
    enrich_directory, enrich_transcript,
    load_transcript, save_transcript,
    TranscriptionConfig, EnrichmentConfig,
    Transcript, Segment
)"
```

All should execute without errors ✓

---

## Conclusion

### Python API: Perfect ✓
- 22/22 examples are valid
- All imports correct
- All signatures correct
- Best practices demonstrated

### CLI Examples: Needs Updates ⚠
- Modern CLI examples are correct
- 15 legacy CLI examples should be updated or marked as deprecated
- No breaking changes (legacy still works)

### Recommended Action
Update the 6 issues listed above to guide users toward the modern unified CLI while keeping legacy examples clearly marked for backward compatibility reference.
