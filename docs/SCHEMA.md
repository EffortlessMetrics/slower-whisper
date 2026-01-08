# Transcript Schema (v2)

**Normative source:** [`transcription/schemas/transcript-v2.schema.json`](../transcription/schemas/transcript-v2.schema.json)

This doc provides a human-readable summary. For validation, use the JSON Schema.

---

## Stability Contract

| Tier | Fields | Commitment |
|------|--------|------------|
| **Stable** | `schema_version`, `file`, `language`, `segments[].{id,start,end,text}` | Won't change in v2.x |
| **Stable (v1.1+)** | `speakers`, `turns`, `annotations` | Shape stable; may add optional fields |
| **Stable (v1.8+)** | `segments[].words[].{word,start,end,probability}` | Shape stable |
| **Extension** | `segments[].audio_state`, `speaker_stats`, `chunks`, `meta` | Optional; may evolve |

**Breaking change policy:** Tier changes bump major version.

---

## Key Normative Names

These are the **serialized JSON keys** (not Python attribute names):

| Key | Type | Required | Notes |
|-----|------|----------|-------|
| `file` | string | ✓ | Audio filename (not `file_name`) |
| `language` | string | ✓ | ISO language code (`en`, `es`, `fr`) |
| `schema_version` | int | ✓ | Always `2` |
| `segments` | array | ✓ | Ordered by `start` |
| `speakers` | array\|null | | Global speaker table (v1.1+) |
| `turns` | array\|null | | Speaker turns (v1.1+) |
| `meta` | object\|null | | Arbitrary metadata |

### Segment shape

```json
{
  "id": 0,
  "start": 0.0,
  "end": 2.5,
  "text": "Hello world.",
  "speaker": {"id": "spk_0", "confidence": 0.95},
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.98},
    {"word": "world.", "start": 0.6, "end": 1.0, "probability": 0.92}
  ],
  "audio_state": null
}
```

### Word shape (v1.8+)

| Key | Type | Required | Notes |
|-----|------|----------|-------|
| `word` | string | ✓ | The word token |
| `start` | float | ✓ | Start time (seconds) |
| `end` | float | ✓ | End time (seconds) |
| `probability` | float | ✓ | ASR confidence (0.0–1.0) |
| `speaker` | string | | Only present if assigned |

---

## Generated Example

The snippet below is generated from test fixtures via:

```bash
python scripts/render-doc-snippets.py  # outputs docs/_snippets/
```

<details>
<summary>Minimal transcript</summary>

```json
{
  "schema_version": 2,
  "file": "example.wav",
  "language": "en",
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world."}
  ],
  "speakers": null,
  "turns": null,
  "meta": null
}
```

</details>

---

## Validation

```bash
# Validate against JSON Schema
python -m jsonschema -i output.json transcription/schemas/transcript-v2.schema.json
```

Or programmatically:

```python
from transcription.writers import validate_transcript_json
validate_transcript_json(data)  # raises on invalid
```

---

## Backward Compatibility

- v1 transcripts (missing `audio_state`, `words`) load correctly
- Deserializer accepts both `file` and `file_name` keys for REST API compatibility
- Optional fields may be `null` or absent

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) — L0-L4 layer design
- [API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md) — Python API usage
- [STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md) — Streaming protocol
