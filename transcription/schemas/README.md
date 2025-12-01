# JSON Schema Specifications

This directory contains formal JSON Schema specifications for slower-whisper transcript files.

## Available Schemas

### `transcript-v2.schema.json`

**Current Version:** 2
**Status:** Active
**Description:** Complete schema for transcripts with optional audio enrichment features

This schema defines:
- Transcript metadata (file, language, ASR settings)
- Segment structure (id, start, end, text)
- Optional audio features (prosody, emotion)
- Extraction status tracking

**Use cases:**
- Validate transcript JSON files before processing
- Generate documentation for external tools
- IDE schema validation in JSON editors
- API contract validation

## Using These Schemas

### 1. Validate JSON Files (Python)

```python
import json
import jsonschema
from pathlib import Path

# Load schema
schema_path = Path(__file__).parent / "schemas" / "transcript-v2.schema.json"
with open(schema_path) as f:
    schema = json.load(f)

# Load transcript
with open("transcript.json") as f:
    transcript = json.load(f)

# Validate
try:
    jsonschema.validate(transcript, schema)
    print("✓ Valid transcript")
except jsonschema.ValidationError as e:
    print(f"✗ Validation error: {e.message}")
    print(f"  Path: {'.'.join(str(p) for p in e.path)}")
```

### 2. Validate with slower-whisper CLI (requires `jsonschema` package)

```bash
uv run slower-whisper validate transcripts/example.json
uv run slower-whisper validate transcripts/*.json --schema transcription/schemas/transcript-v2.schema.json
```

### 3. Validate with jsonschema CLI

```bash
# Install validator
pip install jsonschema

# Validate a file
jsonschema -i transcript.json transcription/schemas/transcript-v2.schema.json
```

### 4. VS Code JSON Validation

Add to your `transcript.json`:

```json
{
  "$schema": "./transcription/schemas/transcript-v2.schema.json",
  "schema_version": 2,
  "file": "audio.wav",
  ...
}
```

VS Code will now provide:
- Autocomplete for field names
- Type checking
- Inline validation errors
- Documentation on hover

### 4. Generate TypeScript Types (optional)

```bash
npm install -g json-schema-to-typescript

json-schema-to-typescript \
  transcription/schemas/transcript-v2.schema.json \
  --output transcript.types.ts
```

## Schema Versioning

### Version History

| Version | Released | Status | Notes |
|---------|----------|--------|-------|
| 2 | 2025-01 | **Active** | Added audio_state for prosody/emotion |
| 1 | 2024-XX | Legacy | Basic transcription only |

### Migration

Old v1 files are automatically migrated when loaded via `load_transcript_from_json()`.

To manually migrate:

```python
from transcription.migrations import migrate_v1_to_v2

with open("old_transcript.json") as f:
    old_data = json.load(f)

new_data = migrate_v1_to_v2(old_data)

with open("new_transcript.json", "w") as f:
    json.dump(new_data, f, indent=2)
```

## Schema Structure Overview

```
transcript-v2.schema.json
├── schema_version: 2
├── file: string
├── language: string
├── meta: object (optional)
│   ├── model_name
│   ├── device
│   ├── duration_sec
│   └── audio_enrichment
└── segments: array
    └── segment
        ├── id: int
        ├── start: number
        ├── end: number
        ├── text: string
        ├── speaker: string | null
        ├── tone: string | null
        └── audio_state: object | null
            ├── prosody
            │   ├── pitch (level, mean_hz, std_hz, variation, contour)
            │   ├── energy (level, db_rms, variation)
            │   ├── rate (level, syllables_per_sec, words_per_sec)
            │   └── pauses (count, longest_ms, density, density_per_sec)
            ├── emotion
            │   ├── valence (level, score)
            │   ├── arousal (level, score)
            │   ├── dominance (level, score)
            │   └── categorical (primary, confidence, secondary, all_scores)
            ├── rendering: string
            └── extraction_status
                ├── prosody: "success" | "failed" | "skipped"
                ├── emotion_dimensional: "success" | "failed" | "skipped"
                ├── emotion_categorical: "success" | "failed" | "skipped"
                └── errors: array of strings
```

## Validation Rules

### Required Fields (Top Level)
- `schema_version` (must be 2)
- `file` (non-empty string)
- `language` (2-letter code, e.g., "en")
- `segments` (array)

### Required Fields (Segment)
- `id` (integer >= 0)
- `start` (number >= 0)
- `end` (number > start)
- `text` (string)

### Optional Fields
- `speaker`, `tone` (segment level)
- `audio_state` (entire structure is optional)
- `meta` (all metadata)

### Validation Constraints
- Times: `start >= 0`, `end > start`
- Scores: `0.0 <= score <= 1.0` (emotion, confidence)
- Frequencies: `mean_hz >= 0`, `std_hz >= 0`
- Counts: `pause count >= 0`
- Enums: Categorical levels must match defined values

## Examples

### Minimal Valid Transcript

```json
{
  "schema_version": 2,
  "file": "audio.wav",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Hello world"
    }
  ]
}
```

### Enriched Transcript

See `transcript-v2.schema.json` examples section for complete enriched example.

## Contributing

When updating schemas:

1. **Increment version** if breaking changes
2. **Add migration utility** in `transcription/migrations.py`
3. **Update this README** with version history
4. **Add tests** in `tests/test_schema.py`
5. **Document changes** in `SCHEMA_CHANGELOG.md`

## Tools & Resources

- **JSON Schema Validator:** https://www.jsonschemavalidator.net/
- **JSON Schema Docs:** https://json-schema.org/
- **Python jsonschema:** https://python-jsonschema.readthedocs.io/
- **Pydantic Models:** See `transcription/schema.py` for Python implementation

## License

Same as parent project (slower-whisper).
