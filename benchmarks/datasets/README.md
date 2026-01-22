# Benchmark Dataset Manifests

This directory contains manifest files that describe benchmark datasets for slower-whisper evaluation.

## Directory Structure

```
datasets/
├── asr/                          # ASR (speech recognition) benchmarks
│   ├── commonvoice_en_smoke/     # Common Voice EN smoke slice (15 clips)
│   │   ├── manifest.json
│   │   └── selection.csv
│   ├── librispeech-test-clean/   # LibriSpeech test-clean subset
│   │   └── manifest.json
│   └── smoke/                    # Quick smoke tests (committed to repo)
│       └── manifest.json
├── diarization/                  # Speaker diarization benchmarks
│   ├── ami-headset/              # AMI Meeting Corpus headset recordings
│   │   └── manifest.json
│   ├── callhome-english/         # CALLHOME American English telephone speech
│   │   └── manifest.json
│   └── smoke/                    # Quick smoke tests (committed to repo)
│       └── manifest.json
└── README.md
```

## Manifest Schema

Each `manifest.json` follows a standard schema (version 1):

```json
{
  "schema_version": 1,
  "id": "dataset-name",
  "track": "asr|diarization|streaming|emotion",
  "split": "train|dev|test|smoke",
  "description": "Human-readable description",
  "source": {
    "name": "Dataset name",
    "url": "https://dataset-homepage.org/",
    "citation": "Author et al., Paper Title, Conference Year"
  },
  "license": {
    "id": "CC-BY-4.0",
    "name": "Creative Commons Attribution 4.0",
    "url": "https://creativecommons.org/licenses/by/4.0/"
  },
  "download": {
    "url": "https://download-url.org/dataset.tar.gz",
    "sha256": "abc123...",
    "size_bytes": 123456789,
    "format": "tar.gz"
  },
  "samples": [...],
  "meta": {
    "created_at": "2026-01-21T00:00:00Z",
    "total_duration_s": 3600,
    "sample_count": 100
  }
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Manifest schema version (currently 1) |
| `id` | string | Unique dataset identifier |
| `track` | string | Benchmark track: `asr`, `diarization`, `streaming`, `emotion` |
| `split` | string | Dataset split: `train`, `dev`, `test`, `smoke` |
| `source` | object | Dataset provenance information |
| `license` | object | License details |
| `meta` | object | Aggregate statistics |

### Sample Fields (ASR Track)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique sample identifier |
| `audio` | string | yes | Relative path or URL to audio file |
| `sha256` | string | yes | SHA256 hash of audio file |
| `duration_s` | float | yes | Audio duration in seconds |
| `language` | string | yes | ISO 639-1 language code |
| `reference_transcript` | string | yes | Ground truth transcript text |
| `license` | string | yes | Sample-level license identifier |
| `source` | string | yes | Sample provenance |

### Sample Fields (Diarization Track)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique sample identifier |
| `audio` | string | yes | Relative path or URL to audio file |
| `audio_sha256` | string | yes | SHA256 hash of audio file |
| `reference_rttm` | string | yes | Path to reference RTTM annotation |
| `rttm_sha256` | string | yes | SHA256 hash of RTTM file |
| `duration_s` | float | yes | Audio duration in seconds |
| `expected_speaker_count` | int | yes | Number of speakers in recording |

## Dataset Categories

### Smoke Datasets (Committed to Repo)

Small synthetic datasets for quick validation. Always available, no download required.

- `asr/smoke`: 3 TTS audio samples covering call center and meeting scenarios
- `diarization/smoke`: 3 synthetic tone patterns with known speaker boundaries

**Usage:**
```bash
slower-whisper benchmark --track asr --dataset smoke
slower-whisper benchmark --track diarization --dataset smoke
```

### Full Datasets (Download Required)

Standard benchmark datasets. Require download and setup.

#### Common Voice EN Smoke Slice (ASR)

- **Size:** ~50 MB (15 clips, ~3 minutes total)
- **Samples:** 15 fixed clips covering accent/noise variance
- **License:** CC0-1.0 (public domain audio)
- **Download:** Requires Hugging Face account with terms accepted

```bash
python benchmarks/scripts/stage_commonvoice.py
```

**Important:** Do not attempt to identify speakers or redistribute the dataset. See [docs/COMMONVOICE_SETUP.md](../../docs/COMMONVOICE_SETUP.md).

#### LibriSpeech test-clean (ASR)

- **Size:** ~350 MB compressed, 5.4 hours of audio
- **Samples:** 2,620 utterances from 40 speakers
- **License:** CC-BY-4.0
- **Download:** Automatic via `scripts/download_datasets.py`

```bash
python scripts/download_datasets.py --dataset librispeech-test-clean
```

#### AMI Meeting Corpus (Diarization)

- **Size:** ~5 GB for test set (16 meetings)
- **Samples:** 16 meetings, ~30-45 minutes each
- **License:** CC-BY-4.0 (requires citation)
- **Download:** Manual (requires license acceptance)

See [docs/AMI_SETUP.md](../../docs/AMI_SETUP.md) for setup instructions.

#### CALLHOME American English (Diarization)

- **Size:** ~500 MB for test set
- **Samples:** 20 telephone conversations, 2 speakers each
- **License:** LDC (commercial license required)
- **Download:** Manual (requires LDC membership or purchase)

CALLHOME is a standard diarization benchmark for telephone speech. It provides a different challenge from AMI due to narrowband (8kHz) audio and casual conversational speech.

## Using Datasets

### List Available Datasets

```bash
slower-whisper benchmark datasets
```

### Verify Dataset Integrity

```bash
python scripts/download_datasets.py --dataset librispeech-test-clean --verify
```

### Run Benchmark

```bash
# Smoke test (always works)
slower-whisper benchmark --track asr --dataset smoke

# Full dataset (requires download)
slower-whisper benchmark --track asr --dataset librispeech-test-clean
```

## Adding New Datasets

1. Create directory: `datasets/<track>/<dataset-name>/`
2. Create `manifest.json` following the schema above
3. For downloadable datasets:
   - Include `download.url` and `download.sha256`
   - Add support in `scripts/download_datasets.py`
4. For smoke datasets:
   - Commit audio files to `benchmarks/data/<track>/`
   - Use relative paths in manifest
5. Update this README

## Staging Policy

| Type | Location | Committed | Download |
|------|----------|-----------|----------|
| Smoke | `benchmarks/data/` | Yes | No |
| Full | `~/.cache/slower-whisper/benchmarks/` | No | Yes |

**Environment variable:** Set `SLOWER_WHISPER_BENCHMARKS` to override default cache location.

## Hash Verification

All datasets use SHA256 hashes for integrity verification:

- **Audio files:** Verify content hasn't changed
- **RTTM files:** Verify reference annotations match expected
- **Archives:** Verify download integrity before extraction

Verification runs automatically during benchmark execution. Use `--skip-verify` to disable (not recommended).

## License Compliance

Each dataset has specific license requirements:

| Dataset | License | Citation Required | Commercial Use |
|---------|---------|-------------------|----------------|
| ASR smoke | MIT | No | Yes |
| Diarization smoke | MIT | No | Yes |
| Common Voice | CC0-1.0 | No | Yes (with terms) |
| LibriSpeech | CC-BY-4.0 | Yes (for papers) | Yes |
| AMI | CC-BY-4.0 | Yes (required) | Yes |
| CALLHOME | LDC | Yes (required) | License required |

Always review the `license` field in manifest files before using datasets.

## Related Documentation

- [Benchmark baselines](../baselines/README.md)
- [AMI setup guide](../../docs/AMI_SETUP.md)
- [IEMOCAP setup guide](../../docs/IEMOCAP_SETUP.md)
- [Benchmark CLI documentation](../../docs/BENCHMARKS.md)
