# Benchmark Dataset Manifests

This directory contains manifest files that describe benchmark datasets for slower-whisper evaluation.

All manifests follow the schema defined in `../manifest_schema.json`.

## Directory Structure

```
datasets/
├── asr/                          # ASR (speech recognition) benchmarks
│   ├── commonvoice_en_smoke/     # Common Voice EN smoke slice (15 clips)
│   │   ├── manifest.json
│   │   └── selection.csv
│   ├── librispeech-test-clean/   # LibriSpeech test-clean subset
│   │   └── manifest.json
│   ├── librispeech-dev-clean/    # LibriSpeech dev-clean subset
│   │   └── manifest.json
│   ├── librispeech-test-other/   # LibriSpeech test-other subset
│   │   └── manifest.json
│   ├── librispeech-dev-other/    # LibriSpeech dev-other subset
│   │   └── manifest.json
│   └── smoke/                    # Quick smoke tests (committed to repo)
│       └── manifest.json
├── diarization/                  # Speaker diarization benchmarks
│   ├── ami-headset/              # AMI Meeting Corpus headset recordings
│   │   └── manifest.json
│   ├── callhome-english/         # CALLHOME American English telephone speech
│   │   └── manifest.json
│   ├── smoke/                    # Speech smoke tests (committed to repo)
│   │   └── manifest.json
│   └── smoke_tones/              # Legacy tone smoke tests (committed to repo)
│       └── manifest.json
└── README.md
```

## Quick Start

```bash
# List all datasets and their status
python scripts/fetch_datasets.py list

# Verify smoke datasets (always available)
python scripts/fetch_datasets.py verify --dataset asr-smoke

# Fetch all smoke datasets (no download needed)
python scripts/fetch_datasets.py fetch --smoke

# Fetch a specific full dataset
python scripts/fetch_datasets.py fetch --dataset librispeech-test-clean

# Show license info
python scripts/fetch_datasets.py license --dataset commonvoice_en_smoke

# Validate all manifests against schema
python scripts/fetch_datasets.py validate
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
- `diarization/smoke`: 3 synthetic dual-voice speech clips with RTTM references
- `diarization/smoke_tones`: 3 legacy tone fixtures for protocol/mapping checks

**Usage:**
```bash
slower-whisper benchmark --track asr --dataset smoke
slower-whisper benchmark --track diarization --dataset smoke
slower-whisper benchmark --track diarization --dataset smoke_tones
```

### Full Datasets (Download Required)

Standard benchmark datasets. Require download and setup.

#### LibriSpeech (ASR) - Recommended

LibriSpeech is the gold standard for ASR evaluation. Multiple splits available:

| Split | Size | Samples | Duration | Quality |
|-------|------|---------|----------|---------|
| test-clean | 346 MB | 2,620 | 5.4h | High |
| dev-clean | 337 MB | 2,703 | 5.4h | High |
| test-other | 328 MB | 2,939 | 5.1h | Challenging |
| dev-other | 314 MB | 2,864 | 5.3h | Challenging |

**Setup (recommended):**
```bash
# Use the setup script
python scripts/setup_benchmark_datasets.py setup librispeech-test-clean

# Or download all splits at once
python scripts/setup_benchmark_datasets.py setup --all-librispeech

# Check status
python scripts/setup_benchmark_datasets.py status
```

**Alternative (manual):**
```bash
python scripts/download_datasets.py --dataset librispeech-test-clean
```

See [docs/LIBRISPEECH_SETUP.md](../../docs/LIBRISPEECH_SETUP.md) for complete instructions.

#### Common Voice EN Smoke Slice (ASR)

- **Size:** ~50 MB (15 clips, ~3 minutes total)
- **Samples:** 15 fixed clips covering accent/noise variance
- **License:** CC0-1.0 (public domain audio)
- **Download:** Requires Hugging Face account with terms accepted

```bash
python benchmarks/scripts/stage_commonvoice.py
```

**Important:** Do not attempt to identify speakers or redistribute the dataset. See [docs/COMMONVOICE_SETUP.md](../../docs/COMMONVOICE_SETUP.md).

#### AMI Meeting Corpus (Diarization)

- **Size:** ~5 GB for test set (16 meetings)
- **Samples:** 16 meetings, ~30-45 minutes each, 4 speakers each
- **License:** CC-BY-4.0 (requires citation)
- **Download:** Manual (requires license acceptance)

**Setup:**
```bash
# Create directory structure and split files
python scripts/setup_benchmark_datasets.py setup ami-headset

# Then download audio manually from:
# https://groups.inf.ed.ac.uk/ami/download/
```

See [docs/AMI_SETUP.md](../../docs/AMI_SETUP.md) for complete setup instructions.

#### CALLHOME American English (Diarization)

- **Size:** ~500 MB for test set
- **Samples:** 20 telephone conversations, 2 speakers each
- **License:** LDC User Agreement (requires purchase or institutional membership)
- **Download:** Manual from LDC

CALLHOME provides a different diarization challenge from AMI:
- Narrowband (8kHz) telephone audio
- Casual conversational speech between family/friends
- Only 2 speakers per call (simpler than AMI)

**Setup:**
```bash
# Get setup information
python scripts/setup_benchmark_datasets.py setup callhome-english
```

See [docs/CALLHOME_SETUP.md](../../docs/CALLHOME_SETUP.md) for complete instructions including LDC access.

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
