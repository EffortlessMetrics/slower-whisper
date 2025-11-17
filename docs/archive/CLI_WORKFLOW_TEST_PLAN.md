# CLI Workflow Test Plan

**Version:** 1.0.0
**Date:** 2025-11-15
**Project:** slower-whisper

This document provides comprehensive test procedures for the slower-whisper CLI, covering both successful workflows and error conditions. Tests can be executed manually or automated using pytest.

---

## Table of Contents

1. [Test Environment Setup](#test-environment-setup)
2. [Workflow 1: Transcribe Audio Files](#workflow-1-transcribe-audio-files)
3. [Workflow 2: Enrich Transcripts](#workflow-2-enrich-transcripts)
4. [Workflow 3: Combined Transcribe + Enrich](#workflow-3-combined-transcribe--enrich)
5. [Error Handling Tests](#error-handling-tests)
6. [Help Text and Usage Tests](#help-text-and-usage-tests)
7. [Advanced Scenarios](#advanced-scenarios)
8. [Automated Test Execution](#automated-test-execution)

---

## Test Environment Setup

### Prerequisites

```bash
# 1. Install system dependencies
# Linux/WSL:
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS:
brew install ffmpeg

# Windows (PowerShell, elevated):
choco install ffmpeg -y

# 2. Install Python dependencies
uv sync              # Base install (transcription only)
# OR
uv sync --extra full # Full install (transcription + enrichment)

# 3. Verify installation
uv run slower-whisper --help
ffmpeg -version
```

### Test Data Preparation

Create a test project structure:

```bash
# Create test project directory
mkdir -p test_project/{raw_audio,input_audio,whisper_json,transcripts}

# Verify structure
ls -la test_project/
```

**Required test files:**
- `sample.wav` - A valid WAV audio file (2-10 seconds recommended)
- `sample.mp3` - A valid MP3 audio file
- `sample.m4a` - A valid M4A audio file
- `corrupt.wav` - A corrupted/invalid audio file
- `empty.wav` - A zero-byte file

Place test audio files in `test_project/raw_audio/`.

---

## Workflow 1: Transcribe Audio Files

### Test 1.1: Basic Transcription with Defaults

**Objective:** Verify basic transcription with default settings.

**Setup:**
```bash
cd test_project
cp /path/to/sample.wav raw_audio/
```

**Command:**
```bash
uv run slower-whisper transcribe
```

**Expected Behavior:**
- Normalizes `raw_audio/sample.wav` to `input_audio/sample.wav` (16 kHz mono)
- Creates `whisper_json/sample.json` with transcript
- Creates `transcripts/sample.txt` with plaintext transcript
- Creates `transcripts/sample.srt` with subtitle format
- Prints progress messages and completion summary

**Success Criteria:**
1. Exit code: 0
2. All output files exist
3. JSON file contains valid schema (schema_version, file, language, meta, segments)
4. TXT file contains timestamped text
5. SRT file contains valid subtitle formatting
6. Console output shows: `[done] Transcribed 1 files`

**Verification:**
```bash
# Check files exist
ls -lh input_audio/sample.wav
ls -lh whisper_json/sample.json
ls -lh transcripts/sample.txt
ls -lh transcripts/sample.srt

# Validate JSON structure
cat whisper_json/sample.json | jq '.schema_version, .file, .language, .segments | length'

# View transcript
cat transcripts/sample.txt
```

**Expected Output Files:**

`whisper_json/sample.json`:
```json
{
  "schema_version": 2,
  "file": "sample.wav",
  "language": "en",
  "meta": {
    "generated_at": "2025-11-15T...",
    "audio_file": "sample.wav",
    "audio_duration_sec": 5.2,
    "model_name": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "beam_size": 5,
    "vad_min_silence_ms": 500,
    "language_hint": null,
    "task": "transcribe",
    "pipeline_version": "1.0.0",
    "root": "/path/to/test_project"
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "This is a test recording.",
      "speaker": null,
      "tone": null,
      "audio_state": null
    }
  ]
}
```

---

### Test 1.2: Transcription with Custom Model

**Objective:** Verify transcription with a different model size.

**Command:**
```bash
uv run slower-whisper transcribe --model base --device cpu
```

**Expected Behavior:**
- Uses "base" model instead of "large-v3"
- Runs on CPU instead of CUDA
- Faster but potentially less accurate

**Success Criteria:**
1. Exit code: 0
2. JSON meta contains: `"model_name": "base"`, `"device": "cpu"`
3. Transcription completes successfully

**Verification:**
```bash
cat whisper_json/sample.json | jq '.meta.model_name, .meta.device'
# Expected: "base", "cpu"
```

---

### Test 1.3: Transcription with Language Hint

**Objective:** Verify transcription with specified language.

**Command:**
```bash
uv run slower-whisper transcribe --language en --task transcribe
```

**Expected Behavior:**
- Forces English language detection
- Uses transcription task (not translation)

**Success Criteria:**
1. Exit code: 0
2. JSON contains: `"language": "en"`, `"meta.language_hint": "en"`
3. All segments transcribed in English

**Verification:**
```bash
cat whisper_json/sample.json | jq '.language, .meta.language_hint, .meta.task'
# Expected: "en", "en", "transcribe"
```

---

### Test 1.4: Translation Task

**Objective:** Verify translation to English from another language.

**Setup:**
- Use audio file in Spanish/French/German

**Command:**
```bash
uv run slower-whisper transcribe --language es --task translate
```

**Expected Behavior:**
- Detects Spanish audio
- Translates to English output

**Success Criteria:**
1. Exit code: 0
2. JSON contains: `"meta.task": "translate"`
3. Segments contain English text (translated)

---

### Test 1.5: Custom Advanced Options

**Objective:** Verify advanced transcription options.

**Command:**
```bash
uv run slower-whisper transcribe \
  --root test_project \
  --model medium \
  --device cuda \
  --compute-type float16 \
  --vad-min-silence-ms 1000 \
  --beam-size 8 \
  --language en
```

**Expected Behavior:**
- Uses medium model
- CUDA with float16 precision
- Longer silence threshold (1000ms)
- Larger beam size for better accuracy

**Success Criteria:**
1. Exit code: 0
2. JSON meta reflects all custom settings
3. Transcription quality is high

**Verification:**
```bash
cat whisper_json/sample.json | jq '.meta | {model_name, device, compute_type, beam_size, vad_min_silence_ms}'
```

---

### Test 1.6: Skip Existing JSON Files

**Objective:** Verify that existing transcripts are not re-transcribed.

**Setup:**
```bash
# First transcription
uv run slower-whisper transcribe
# Note the timestamp
ls -lh whisper_json/sample.json
```

**Command:**
```bash
# Second transcription (should skip)
uv run slower-whisper transcribe --skip-existing-json
```

**Expected Behavior:**
- Skips files that already have JSON output
- Prints skip message: `[skip] sample.json already exists`
- Does not modify existing JSON file

**Success Criteria:**
1. Exit code: 0
2. JSON file timestamp unchanged
3. Console output indicates files were skipped

---

### Test 1.7: Force Re-transcription

**Objective:** Verify forcing re-transcription of existing files.

**Command:**
```bash
uv run slower-whisper transcribe --no-skip-existing-json
```

**Expected Behavior:**
- Re-transcribes all files, even if JSON exists
- Updates JSON files with new timestamps

**Success Criteria:**
1. Exit code: 0
2. JSON files have updated `generated_at` timestamp
3. All files re-processed

---

### Test 1.8: Multiple Audio Files

**Objective:** Verify batch transcription of multiple files.

**Setup:**
```bash
cp sample1.wav sample2.mp3 sample3.m4a raw_audio/
```

**Command:**
```bash
uv run slower-whisper transcribe
```

**Expected Behavior:**
- Normalizes all 3 audio files
- Transcribes all 3 files
- Creates 3 JSON, 3 TXT, 3 SRT files

**Success Criteria:**
1. Exit code: 0
2. Console output: `[done] Transcribed 3 files`
3. All output files exist for all inputs

**Verification:**
```bash
ls whisper_json/ | wc -l    # Should be 3
ls transcripts/*.txt | wc -l # Should be 3
ls transcripts/*.srt | wc -l # Should be 3
```

---

## Workflow 2: Enrich Transcripts

### Test 2.1: Basic Enrichment with Defaults

**Objective:** Verify basic audio enrichment with prosody and emotion.

**Prerequisites:**
- Transcription workflow completed (JSON files exist)
- Full dependencies installed: `uv sync --extra full`

**Command:**
```bash
uv run slower-whisper enrich
```

**Expected Behavior:**
- Reads existing JSON transcripts from `whisper_json/`
- Enriches with prosody features (pitch, energy, rate, pauses)
- Enriches with dimensional emotion (valence, arousal, dominance)
- Updates JSON files with `audio_state` field
- Skips categorical emotion (disabled by default)

**Success Criteria:**
1. Exit code: 0
2. Console output: `[enriched] sample.json`
3. JSON segments contain `audio_state` field
4. `audio_state` contains: pitch, energy, rate, pauses, emotion.dimensional
5. Console output: `[done] Enriched 1 transcripts`

**Verification:**
```bash
# Check audio_state exists
cat whisper_json/sample.json | jq '.segments[0].audio_state != null'
# Should return: true

# Inspect enriched features
cat whisper_json/sample.json | jq '.segments[0].audio_state | keys'
# Expected: ["emotion", "energy", "pauses", "pitch", "rate", "rendering"]

# View enrichment
cat whisper_json/sample.json | jq '.segments[0].audio_state.rendering'
# Example: "[audio: medium pitch, moderate volume, normal speech rate]"
```

**Expected audio_state Structure:**
```json
{
  "pitch": {
    "level": "medium",
    "mean_hz": 185.3,
    "std_hz": 28.1,
    "contour": "rising"
  },
  "energy": {
    "level": "moderate",
    "db_rms": -12.4,
    "variation_coefficient": 0.22
  },
  "rate": {
    "level": "normal",
    "syllables_per_sec": 5.2,
    "words_per_sec": 3.1
  },
  "pauses": {
    "count": 1,
    "longest_sec": 0.3,
    "density": 0.25
  },
  "emotion": {
    "dimensional": {
      "valence": {
        "level": "neutral",
        "score": 0.52
      },
      "arousal": {
        "level": "medium",
        "score": 0.48
      },
      "dominance": {
        "level": "medium",
        "score": 0.55
      }
    }
  },
  "rendering": "[audio: medium pitch, moderate volume, normal speech rate]"
}
```

---

### Test 2.2: Enrichment with Categorical Emotion

**Objective:** Verify categorical emotion classification.

**Command:**
```bash
uv run slower-whisper enrich --enable-categorical-emotion
```

**Expected Behavior:**
- Enables categorical emotion recognition (slower)
- Adds `emotion.categorical` field with primary emotion and confidence

**Success Criteria:**
1. Exit code: 0
2. `audio_state.emotion.categorical` exists
3. Contains `primary` (e.g., "neutral", "happy", "sad") and `confidence` (0-1)

**Verification:**
```bash
cat whisper_json/sample.json | jq '.segments[0].audio_state.emotion.categorical'
# Expected:
# {
#   "primary": "neutral",
#   "confidence": 0.78,
#   "all_scores": { "neutral": 0.78, "happy": 0.12, ... }
# }
```

---

### Test 2.3: Prosody-Only Enrichment

**Objective:** Verify enrichment with only prosodic features.

**Command:**
```bash
uv run slower-whisper enrich --enable-prosody --no-enable-emotion
```

**Expected Behavior:**
- Extracts only prosody features (pitch, energy, rate, pauses)
- Skips emotion extraction

**Success Criteria:**
1. Exit code: 0
2. `audio_state` contains: pitch, energy, rate, pauses
3. `audio_state.emotion` does not exist

**Verification:**
```bash
cat whisper_json/sample.json | jq '.segments[0].audio_state | has("emotion")'
# Expected: false

cat whisper_json/sample.json | jq '.segments[0].audio_state | keys'
# Expected: ["energy", "pauses", "pitch", "rate", "rendering"]
```

---

### Test 2.4: Emotion-Only Enrichment

**Objective:** Verify enrichment with only emotion features.

**Command:**
```bash
uv run slower-whisper enrich --no-enable-prosody --enable-emotion
```

**Expected Behavior:**
- Extracts only dimensional emotion
- Skips prosody features

**Success Criteria:**
1. Exit code: 0
2. `audio_state` contains only: emotion
3. Prosody fields do not exist

**Verification:**
```bash
cat whisper_json/sample.json | jq '.segments[0].audio_state | keys'
# Expected: ["emotion", "rendering"]
```

---

### Test 2.5: Skip Already Enriched Files

**Objective:** Verify that enriched transcripts are not re-enriched.

**Setup:**
```bash
# First enrichment
uv run slower-whisper enrich
```

**Command:**
```bash
# Second enrichment (should skip)
uv run slower-whisper enrich --skip-existing
```

**Expected Behavior:**
- Detects existing `audio_state` field
- Skips enrichment: `[skip] sample.json already enriched`

**Success Criteria:**
1. Exit code: 0
2. Console output indicates files were skipped
3. No changes to JSON files

---

### Test 2.6: Force Re-enrichment

**Objective:** Verify forcing re-enrichment of already enriched files.

**Command:**
```bash
uv run slower-whisper enrich --no-skip-existing
```

**Expected Behavior:**
- Re-enriches all files, even if `audio_state` exists
- Updates `audio_state` with fresh analysis

**Success Criteria:**
1. Exit code: 0
2. All files re-processed
3. `audio_state` potentially updated

---

### Test 2.7: Enrichment with CUDA Device

**Objective:** Verify enrichment using GPU acceleration for emotion models.

**Prerequisites:**
- CUDA-capable GPU available

**Command:**
```bash
uv run slower-whisper enrich --device cuda --enable-emotion
```

**Expected Behavior:**
- Runs emotion models on GPU (faster)
- Prosody still uses CPU

**Success Criteria:**
1. Exit code: 0
2. Enrichment completes faster than CPU mode
3. Results are equivalent to CPU mode

---

## Workflow 3: Combined Transcribe + Enrich

### Test 3.1: Full Pipeline - Transcribe then Enrich

**Objective:** Verify complete end-to-end workflow.

**Setup:**
```bash
# Clean project
rm -rf test_project/{input_audio,whisper_json,transcripts}/*
cp sample.wav test_project/raw_audio/
```

**Commands:**
```bash
# Step 1: Transcribe
uv run slower-whisper transcribe --root test_project

# Step 2: Enrich
uv run slower-whisper enrich --root test_project --enable-categorical-emotion
```

**Expected Behavior:**
1. Transcribe creates JSON with segments
2. Enrich adds `audio_state` to all segments
3. Final JSON contains both transcription and enrichment

**Success Criteria:**
1. Both commands exit with code 0
2. Final JSON has complete structure with audio_state
3. All features present: prosody, dimensional emotion, categorical emotion

**Verification:**
```bash
# Verify complete structure
cat test_project/whisper_json/sample.json | jq '{
  schema_version,
  file,
  language,
  has_meta: (.meta != null),
  segment_count: (.segments | length),
  first_segment_has_audio_state: (.segments[0].audio_state != null),
  audio_state_features: (.segments[0].audio_state | keys)
}'
```

**Expected Final Output:**
```json
{
  "schema_version": 2,
  "file": "sample.wav",
  "language": "en",
  "has_meta": true,
  "segment_count": 3,
  "first_segment_has_audio_state": true,
  "audio_state_features": [
    "emotion",
    "energy",
    "pauses",
    "pitch",
    "rate",
    "rendering"
  ]
}
```

---

### Test 3.2: Custom Root Directory

**Objective:** Verify workflows work with custom project root.

**Setup:**
```bash
mkdir -p /tmp/custom_project/{raw_audio,input_audio,whisper_json,transcripts}
cp sample.wav /tmp/custom_project/raw_audio/
```

**Commands:**
```bash
uv run slower-whisper transcribe --root /tmp/custom_project
uv run slower-whisper enrich --root /tmp/custom_project
```

**Expected Behavior:**
- Operates on `/tmp/custom_project` instead of current directory
- Creates all outputs in custom root

**Success Criteria:**
1. Exit code: 0 for both commands
2. Files created in custom root, not current directory
3. JSON meta contains correct root path

**Verification:**
```bash
ls /tmp/custom_project/whisper_json/
cat /tmp/custom_project/whisper_json/sample.json | jq '.meta.root'
# Expected: "/tmp/custom_project"
```

---

## Error Handling Tests

### Test 4.1: Missing Raw Audio Directory

**Objective:** Verify behavior when raw_audio/ doesn't exist.

**Setup:**
```bash
mkdir empty_project
cd empty_project
```

**Command:**
```bash
uv run slower-whisper transcribe
```

**Expected Behavior:**
- Creates directory structure if missing
- Warns about no audio files found
- Exits gracefully

**Success Criteria:**
1. Exit code: 0 (or appropriate warning code)
2. Console message: "No audio files found" or similar
3. Directories created automatically

---

### Test 4.2: Corrupt Audio File

**Objective:** Verify handling of corrupt/invalid audio files.

**Setup:**
```bash
echo "not an audio file" > test_project/raw_audio/corrupt.wav
```

**Command:**
```bash
uv run slower-whisper transcribe
```

**Expected Behavior:**
- Attempts to process file
- ffmpeg fails with error
- Logs error message
- Continues with other files (if any)

**Success Criteria:**
1. Error logged for corrupt file
2. Other valid files still processed
3. Does not crash entire pipeline

**Verification:**
```bash
# Check logs for error message
# Verify valid files were still processed
```

---

### Test 4.3: Missing Input Audio for Enrichment

**Objective:** Verify behavior when audio file missing during enrichment.

**Setup:**
```bash
# Create JSON but delete audio
uv run slower-whisper transcribe
rm input_audio/sample.wav
```

**Command:**
```bash
uv run slower-whisper enrich
```

**Expected Behavior:**
- Detects missing audio file
- Skips enrichment for that file
- Logs: `[skip] Audio file not found for sample.json`

**Success Criteria:**
1. Exit code: 0
2. Skips files with missing audio
3. Processes other files normally

---

### Test 4.4: Missing JSON Directory for Enrichment

**Objective:** Verify error when whisper_json/ doesn't exist.

**Setup:**
```bash
mkdir empty_project2
cd empty_project2
mkdir input_audio
```

**Command:**
```bash
uv run slower-whisper enrich
```

**Expected Behavior:**
- Detects missing JSON directory
- Exits with error message

**Success Criteria:**
1. Exit code: non-zero (error)
2. Error message: "JSON directory does not exist: .../whisper_json"

---

### Test 4.5: Empty JSON Directory

**Objective:** Verify behavior when no JSON files exist.

**Setup:**
```bash
mkdir -p test_project/whisper_json
```

**Command:**
```bash
uv run slower-whisper enrich
```

**Expected Behavior:**
- Detects no JSON files
- Warns: `[warn] No JSON files found in whisper_json/`
- Exits gracefully

**Success Criteria:**
1. Exit code: 0
2. Warning message displayed
3. No crash

---

### Test 4.6: Permission Denied

**Objective:** Verify handling of permission errors.

**Setup:**
```bash
# Make directory read-only
chmod 444 test_project/whisper_json/
```

**Command:**
```bash
uv run slower-whisper transcribe
```

**Expected Behavior:**
- Attempts to write JSON
- Fails with permission error
- Logs error message

**Success Criteria:**
1. Error message indicates permission problem
2. Does not crash
3. Other operations continue if possible

**Cleanup:**
```bash
chmod 755 test_project/whisper_json/
```

---

### Test 4.7: Invalid Configuration Values

**Objective:** Verify validation of CLI arguments.

**Commands:**
```bash
# Invalid model name - should still work (Whisper will download)
uv run slower-whisper transcribe --model invalid-model

# Invalid device - should be caught by argparse
uv run slower-whisper enrich --device tpu

# Invalid task choice
uv run slower-whisper transcribe --task invalid-task

# Invalid beam size (non-integer)
uv run slower-whisper transcribe --beam-size abc

# Negative values
uv run slower-whisper transcribe --vad-min-silence-ms -100
```

**Expected Behavior:**
- Argparse catches invalid choices and types
- Exits with error code 2
- Shows helpful error message

**Success Criteria:**
1. Exit code: 2 (argparse error)
2. Error message indicates what's wrong
3. Suggests valid options

---

### Test 4.8: Disk Space Exhaustion

**Objective:** Verify behavior when disk is full.

**Note:** Difficult to test reliably. Manual test only if feasible.

**Expected Behavior:**
- Handles write errors gracefully
- Logs error message about disk space
- Does not corrupt existing files

---

## Help Text and Usage Tests

### Test 5.1: Global Help

**Command:**
```bash
uv run slower-whisper --help
```

**Expected Output:**
```
usage: slower-whisper [-h] {transcribe,enrich} ...

Local transcription and audio enrichment pipeline.

positional arguments:
  {transcribe,enrich}
    transcribe         Transcribe audio under a project root (Stage 1).
    enrich             Enrich existing transcripts with audio-derived features (Stage 2).

optional arguments:
  -h, --help           show this help message and exit
```

**Success Criteria:**
1. Exit code: 0
2. Shows program name and description
3. Lists available subcommands
4. Shows help option

---

### Test 5.2: Transcribe Subcommand Help

**Command:**
```bash
uv run slower-whisper transcribe --help
```

**Expected Output:**
```
usage: slower-whisper transcribe [-h] [--root ROOT] [--model MODEL]
                                 [--device DEVICE] [--compute-type COMPUTE_TYPE]
                                 [--language LANGUAGE] [--task {transcribe,translate}]
                                 [--vad-min-silence-ms VAD_MIN_SILENCE_MS]
                                 [--beam-size BEAM_SIZE]
                                 [--skip-existing-json | --no-skip-existing-json]

Transcribe audio under a project root (Stage 1).

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           Project root (contains raw_audio/, input_audio/, whisper_json/, transcripts/).
  --model MODEL         Whisper model name (default: large-v3).
  --device DEVICE       Device: cuda or cpu (default: cuda).
  --compute-type COMPUTE_TYPE
                        faster-whisper compute type (e.g. float16, int8).
  --language LANGUAGE   Force language (e.g. en). Leave empty for auto-detect.
  --task {transcribe,translate}
                        Whisper task (default: transcribe).
  --vad-min-silence-ms VAD_MIN_SILENCE_MS
                        Minimum silence duration in ms to split segments (default: 500).
  --beam-size BEAM_SIZE
                        Beam size for decoding (default: 5).
  --skip-existing-json, --no-skip-existing-json
                        Skip files with existing JSON in whisper_json/ (default: True).
```

**Success Criteria:**
1. Exit code: 0
2. Shows all transcribe options
3. Shows default values
4. Shows option types and choices

---

### Test 5.3: Enrich Subcommand Help

**Command:**
```bash
uv run slower-whisper enrich --help
```

**Expected Output:**
```
usage: slower-whisper enrich [-h] [--root ROOT] [--skip-existing | --no-skip-existing]
                             [--enable-prosody | --no-enable-prosody]
                             [--enable-emotion | --no-enable-emotion]
                             [--enable-categorical-emotion | --no-enable-categorical-emotion]
                             [--device {cpu,cuda}]

Enrich existing transcripts with audio-derived features (Stage 2).

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           Project root.
  --skip-existing, --no-skip-existing
                        Skip segments that already have audio_state (default: True).
  --enable-prosody, --no-enable-prosody
                        Enable prosody extraction (default: True).
  --enable-emotion, --no-enable-emotion
                        Enable dimensional emotion extraction (default: True).
  --enable-categorical-emotion, --no-enable-categorical-emotion
                        Enable categorical emotion (slower, default: False).
  --device {cpu,cuda}   Device to run emotion models on (default: cpu).
```

**Success Criteria:**
1. Exit code: 0
2. Shows all enrich options
3. Shows default values
4. Shows boolean toggle options

---

### Test 5.4: No Arguments Shows Error

**Command:**
```bash
uv run slower-whisper
```

**Expected Behavior:**
- Shows error: "the following arguments are required: command"
- Exits with code 2
- Suggests using `--help`

**Success Criteria:**
1. Exit code: 2
2. Error message displayed
3. Does not crash

---

### Test 5.5: Invalid Subcommand

**Command:**
```bash
uv run slower-whisper invalid-command
```

**Expected Behavior:**
- Shows error: "invalid choice: 'invalid-command'"
- Lists valid choices: transcribe, enrich
- Exits with code 2

**Success Criteria:**
1. Exit code: 2
2. Error message with valid choices
3. Helpful guidance

---

## Advanced Scenarios

### Test 6.1: Large Batch Processing

**Objective:** Verify performance with many files.

**Setup:**
```bash
# Create 50 test files
for i in {1..50}; do
  cp sample.wav test_project/raw_audio/sample_$i.wav
done
```

**Command:**
```bash
time uv run slower-whisper transcribe
```

**Expected Behavior:**
- Processes all files sequentially
- Shows progress for each file
- Completes without memory issues

**Success Criteria:**
1. Exit code: 0
2. All 50 files transcribed
3. Memory usage reasonable
4. Summary shows correct count

---

### Test 6.2: Long Audio Files

**Objective:** Verify handling of long audio files (30+ minutes).

**Setup:**
- Use a 30-60 minute audio file

**Command:**
```bash
uv run slower-whisper transcribe
```

**Expected Behavior:**
- Processes file without timeout
- Creates many segments
- Memory usage stays reasonable

**Success Criteria:**
1. Exit code: 0
2. JSON contains hundreds of segments
3. No memory errors or crashes

---

### Test 6.3: Special Characters in Filenames

**Objective:** Verify handling of filenames with spaces and special characters.

**Setup:**
```bash
cp sample.wav "test_project/raw_audio/File with spaces & symbols (2024).wav"
```

**Command:**
```bash
uv run slower-whisper transcribe
```

**Expected Behavior:**
- Handles filename correctly
- Creates output files with sanitized names

**Success Criteria:**
1. Exit code: 0
2. Output files created successfully
3. JSON references original filename

---

### Test 6.4: Mixed Audio Formats

**Objective:** Verify handling of different audio formats in same batch.

**Setup:**
```bash
cp sample.wav test_project/raw_audio/
cp sample.mp3 test_project/raw_audio/
cp sample.m4a test_project/raw_audio/
cp sample.flac test_project/raw_audio/
cp sample.ogg test_project/raw_audio/
```

**Command:**
```bash
uv run slower-whisper transcribe
```

**Expected Behavior:**
- Normalizes all formats to WAV
- Transcribes all successfully
- Each creates corresponding JSON/TXT/SRT

**Success Criteria:**
1. Exit code: 0
2. All formats processed
3. Console output: `[done] Transcribed 5 files`

---

### Test 6.5: Resume After Interruption

**Objective:** Verify --skip-existing-json allows resuming interrupted batch.

**Setup:**
```bash
# Start batch processing
cp sample{1..10}.wav test_project/raw_audio/
uv run slower-whisper transcribe

# Simulate interruption - delete some JSON files
rm test_project/whisper_json/sample{6..10}.json
```

**Command:**
```bash
# Resume - should only process missing files
uv run slower-whisper transcribe --skip-existing-json
```

**Expected Behavior:**
- Skips files 1-5 (JSON exists)
- Processes files 6-10 (JSON missing)

**Success Criteria:**
1. Exit code: 0
2. Only 5 files re-transcribed
3. Console shows skip messages for 1-5

---

### Test 6.6: CPU-Only Mode (No CUDA)

**Objective:** Verify operation on systems without GPU.

**Command:**
```bash
uv run slower-whisper transcribe --device cpu --compute-type int8
uv run slower-whisper enrich --device cpu
```

**Expected Behavior:**
- Runs on CPU successfully
- Uses quantized model for efficiency
- Slower but functional

**Success Criteria:**
1. Exit code: 0
2. Transcription completes (slower)
3. Enrichment completes on CPU

---

### Test 6.7: Partial Enrichment Features

**Objective:** Verify selective feature extraction.

**Commands:**
```bash
# Only prosody
uv run slower-whisper enrich --enable-prosody --no-enable-emotion

# Only emotion
uv run slower-whisper enrich --no-enable-prosody --enable-emotion

# Only categorical emotion
uv run slower-whisper enrich \
  --no-enable-prosody \
  --enable-emotion \
  --enable-categorical-emotion
```

**Expected Behavior:**
- Each extracts only requested features
- Skips disabled features
- JSON reflects partial enrichment

**Success Criteria:**
1. Exit code: 0
2. `audio_state` contains only requested features
3. Processing time reduced for partial enrichment

---

## Automated Test Execution

### Running Unit Tests

The project includes comprehensive unit tests in the `tests/` directory.

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/test_cli_integration.py  # CLI tests
uv run pytest tests/test_integration.py      # E2E tests
uv run pytest tests/test_audio_enrichment.py # Enrichment tests

# Run with coverage
uv run pytest --cov=transcription --cov-report=term-missing

# Run fast tests only (skip slow integration tests)
uv run pytest -m "not slow"

# Run with verbose output
uv run pytest -v
```

### Automated CLI Testing

The `tests/test_cli_integration.py` file contains automated tests for all CLI workflows:

**Key test functions:**
- `test_main_transcribe_integration()` - Tests transcribe workflow
- `test_main_enrich_integration()` - Tests enrich workflow
- `test_sequential_transcribe_then_enrich()` - Tests combined workflow
- `test_transcribe_*()` - Various transcribe option tests
- `test_enrich_*()` - Various enrich option tests
- `test_main_*_help()` - Help text tests

**Run CLI integration tests:**
```bash
uv run pytest tests/test_cli_integration.py -v
```

### Continuous Integration

For CI/CD pipelines:

```bash
#!/bin/bash
# ci-test.sh

set -e  # Exit on error

echo "Installing dependencies..."
uv sync --extra dev

echo "Running linters..."
uv run ruff check transcription/
uv run black --check transcription/

echo "Running type checks..."
uv run mypy transcription/

echo "Running unit tests..."
uv run pytest --cov=transcription --cov-report=xml

echo "All checks passed!"
```

---

## Test Checklist

Use this checklist to track test execution:

### Transcription Workflow
- [ ] Test 1.1: Basic transcription with defaults
- [ ] Test 1.2: Custom model selection
- [ ] Test 1.3: Language hint
- [ ] Test 1.4: Translation task
- [ ] Test 1.5: Advanced options
- [ ] Test 1.6: Skip existing JSON
- [ ] Test 1.7: Force re-transcription
- [ ] Test 1.8: Multiple audio files

### Enrichment Workflow
- [ ] Test 2.1: Basic enrichment
- [ ] Test 2.2: Categorical emotion
- [ ] Test 2.3: Prosody only
- [ ] Test 2.4: Emotion only
- [ ] Test 2.5: Skip enriched files
- [ ] Test 2.6: Force re-enrichment
- [ ] Test 2.7: GPU enrichment

### Combined Workflow
- [ ] Test 3.1: Full pipeline
- [ ] Test 3.2: Custom root directory

### Error Handling
- [ ] Test 4.1: Missing raw audio directory
- [ ] Test 4.2: Corrupt audio file
- [ ] Test 4.3: Missing input audio
- [ ] Test 4.4: Missing JSON directory
- [ ] Test 4.5: Empty JSON directory
- [ ] Test 4.6: Permission denied
- [ ] Test 4.7: Invalid configuration

### Help and Usage
- [ ] Test 5.1: Global help
- [ ] Test 5.2: Transcribe help
- [ ] Test 5.3: Enrich help
- [ ] Test 5.4: No arguments error
- [ ] Test 5.5: Invalid subcommand

### Advanced Scenarios
- [ ] Test 6.1: Large batch processing
- [ ] Test 6.2: Long audio files
- [ ] Test 6.3: Special characters
- [ ] Test 6.4: Mixed audio formats
- [ ] Test 6.5: Resume after interruption
- [ ] Test 6.6: CPU-only mode
- [ ] Test 6.7: Partial enrichment

### Automated Tests
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Code coverage > 80%

---

## Troubleshooting Guide

### Common Issues and Solutions

**Issue:** `ffmpeg not found`
- **Solution:** Install ffmpeg and ensure it's on PATH
- **Verify:** `ffmpeg -version`

**Issue:** `CUDA not available`
- **Solution:** Use `--device cpu` or install CUDA toolkit
- **Verify:** `nvidia-smi`

**Issue:** `Module not found: librosa`
- **Solution:** Install full dependencies: `uv sync --extra full`

**Issue:** `Permission denied writing to directory`
- **Solution:** Check directory permissions: `ls -la`
- **Fix:** `chmod 755 directory_name`

**Issue:** `JSON decode error`
- **Solution:** JSON file may be corrupted; delete and re-transcribe

**Issue:** `Model download timeout`
- **Solution:** Check internet connection; Whisper models download on first use

---

## Reporting Issues

When reporting test failures, include:

1. **Environment:**
   - OS and version
   - Python version
   - CUDA version (if applicable)
   - Package versions: `uv pip list`

2. **Test details:**
   - Test name/number
   - Command executed
   - Expected vs actual behavior

3. **Logs and outputs:**
   - Full console output
   - Error messages
   - Stack traces

4. **Reproducibility:**
   - Steps to reproduce
   - Sample files (if safe to share)
   - Configuration used

---

## Appendix: Sample Test Data

### Creating Synthetic Test Audio

For automated testing, generate synthetic audio:

```python
# generate_test_audio.py
import numpy as np
import soundfile as sf

# Generate 5 seconds of sine wave (440 Hz)
sample_rate = 16000
duration = 5
t = np.linspace(0, duration, duration * sample_rate)
audio = 0.3 * np.sin(2 * np.pi * 440 * t)

sf.write('test_audio_440hz.wav', audio, sample_rate)
print("Generated test_audio_440hz.wav")
```

### JSON Validation Schema

Expected JSON structure (minimal):

```json
{
  "schema_version": 2,
  "file": "string",
  "language": "string",
  "meta": {
    "generated_at": "ISO8601 timestamp",
    "audio_file": "string",
    "model_name": "string",
    "device": "string",
    "compute_type": "string"
  },
  "segments": [
    {
      "id": "integer",
      "start": "float",
      "end": "float",
      "text": "string",
      "speaker": "string or null",
      "tone": "string or null",
      "audio_state": "object or null"
    }
  ]
}
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-15
**Maintainer:** slower-whisper team
