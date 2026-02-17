# Benchmark Datasets

This document provides a comprehensive overview of benchmark datasets available for slower-whisper evaluation.

## Quick Reference

| Dataset | Track | Type | License | Download | Samples |
|---------|-------|------|---------|----------|---------|
| smoke | ASR | Smoke | MIT | No | 3 |
| diarization-smoke | Diarization | Smoke | MIT | No | 3 |
| diarization-smoke-tones | Diarization | Smoke | MIT | No | 3 |
| librispeech-test-clean | ASR | Full | CC-BY-4.0 | Yes | 2,620 |
| librispeech-dev-clean | ASR | Full | CC-BY-4.0 | Yes | 2,703 |
| librispeech-test-other | ASR | Full | CC-BY-4.0 | Yes | 2,939 |
| librispeech-dev-other | ASR | Full | CC-BY-4.0 | Yes | 2,864 |
| commonvoice_en_smoke | ASR | Full | CC0-1.0 | Yes | 15 |
| ami-headset | Diarization | Full | CC-BY-4.0 | Manual | 16 |
| callhome-english | Diarization | Full | LDC | Manual | 20 |

## Dataset Types

### Smoke Datasets

Smoke datasets are minimal test sets **committed to the repository** for quick CI validation:

- Always available (no download required)
- Synthetic or permissively licensed audio
- Fast to run (<30 seconds)
- Provide basic sanity checks

**Usage:**
```bash
slower-whisper benchmark run --track asr --dataset smoke
slower-whisper benchmark run --track diarization --dataset smoke
```

### Full Datasets

Full datasets require download and provide comprehensive evaluation:

- Standard academic benchmarks
- Larger sample counts for statistical significance
- May require license acceptance
- Used for production quality gates

## ASR Datasets

### smoke (ASR)

**Type:** Smoke (always available)

Minimal ASR test set using synthetic TTS audio covering common scenarios.

| Property | Value |
|----------|-------|
| Samples | 3 |
| Duration | 21 seconds |
| Language | English |
| Audio Format | 16kHz mono WAV |
| License | MIT |

**Samples included:**
- `call_center_narrowband` - 8kHz narrowband phone support
- `team_sync_meeting` - Clean meeting recap
- `status_update_clean` - Action-oriented status update

**Expected baseline:** WER ~5% (synthetic TTS is clean)

---

### librispeech-test-clean

**Type:** Full (download required)

LibriSpeech is the gold standard for ASR evaluation. Clean test set from LibriVox audiobooks.

| Property | Value |
|----------|-------|
| Samples | 2,620 |
| Duration | 5.4 hours |
| Speakers | 40 |
| Audio Format | 16kHz FLAC |
| License | CC-BY-4.0 |
| Download Size | 346 MB |

**Setup:**
```bash
python scripts/setup_benchmark_datasets.py setup librispeech-test-clean
```

See [docs/LIBRISPEECH_SETUP.md](../docs/LIBRISPEECH_SETUP.md) for detailed instructions.

---

### librispeech-dev-clean

**Type:** Full (download required)

LibriSpeech clean development set for hyperparameter tuning.

| Property | Value |
|----------|-------|
| Samples | 2,703 |
| Duration | 5.4 hours |
| Speakers | 40 |
| Audio Format | 16kHz FLAC |
| License | CC-BY-4.0 |
| Download Size | 337 MB |

---

### librispeech-test-other / librispeech-dev-other

**Type:** Full (download required)

LibriSpeech "other" splits contain more challenging acoustic conditions.

| Split | Samples | Duration | Speakers |
|-------|---------|----------|----------|
| test-other | 2,939 | 5.1 hours | 33 |
| dev-other | 2,864 | 5.3 hours | 33 |

Use these for robustness testing and evaluating performance on difficult audio.

---

### commonvoice_en_smoke

**Type:** Full (download required)

Common Voice English smoke slice - 15 fixed clips covering accent/feature variance.

| Property | Value |
|----------|-------|
| Samples | 15 |
| Duration | ~3 minutes |
| Accents | US, UK, India, AU/NZ, Ireland, Scotland, South Africa |
| License | CC0-1.0 |
| Download | Hugging Face (requires account) |

**Features tested:**
- Digits, apostrophes, dashes
- Quotes, commas, abbreviations
- Various accents

**Setup:**
```bash
python benchmarks/scripts/stage_commonvoice.py
```

See [docs/COMMONVOICE_SETUP.md](../docs/COMMONVOICE_SETUP.md) for setup.

## Diarization Datasets

### diarization-smoke

**Type:** Smoke (always available)

Minimal diarization test set using synthetic speech audio with known two-speaker turn patterns.

| Property | Value |
|----------|-------|
| Samples | 3 |
| Duration | 39.47 seconds |
| Speakers | 2 per sample |
| Audio Format | 16kHz mono WAV |
| Annotation | RTTM |
| License | MIT |

**Samples included:**
- `meeting_dual_voice` - Meeting recap with alternating speakers
- `support_handoff_dual_voice` - Support handoff with turn boundaries
- `planning_sync_dual_voice` - Planning sync with concise alternation

**Expected baseline:** DER <50% for smoke checks, Speaker count accuracy 100%

---

### diarization-smoke-tones

**Type:** Smoke (always available)

Legacy deterministic tone fixtures kept for protocol/mapping regression checks.

| Property | Value |
|----------|-------|
| Samples | 3 |
| Duration | 34.1 seconds |
| Speakers | 2 per sample |
| Audio Format | 16kHz mono WAV |
| Annotation | RTTM |
| License | MIT |

**Samples included:**
- `synthetic_2speaker` - Deterministic A/B tone pattern
- `overlap_tones` - Two-tone overlap stress case
- `call_mixed` - Alternating call-style tones

---

### ami-headset

**Type:** Full (manual download required)

AMI Meeting Corpus headset recordings for multi-party meeting diarization.

| Property | Value |
|----------|-------|
| Samples | 16 (test set) |
| Duration | ~9 hours |
| Speakers | 4 per meeting |
| Audio Format | 16kHz WAV |
| Annotation | RTTM |
| License | CC-BY-4.0 |

AMI meetings are project design discussions with rich annotations including:
- Speaker diarization (word-level)
- Topic segmentation
- Abstractive summaries
- Action items

**Setup:**
```bash
python scripts/setup_benchmark_datasets.py setup ami-headset
```

See [docs/AMI_SETUP.md](../docs/AMI_SETUP.md) for manual download instructions.

---

### callhome-english

**Type:** Full (LDC access required)

CALLHOME American English telephone speech corpus for 2-speaker diarization.

| Property | Value |
|----------|-------|
| Samples | 20 (test set) |
| Duration | ~2.5 hours |
| Speakers | 2 per call |
| Audio Format | 8kHz WAV |
| Annotation | RTTM |
| License | LDC User Agreement |

**Key characteristics:**
- Narrowband (8kHz) telephone audio
- Casual conversational speech
- Family/friends conversations
- Standard diarization benchmark

**Setup:**

CALLHOME requires LDC membership or purchase. See [docs/CALLHOME_SETUP.md](../docs/CALLHOME_SETUP.md).

## Manifest Infrastructure

All datasets are defined by manifest files following a standard schema.

### Manifest Location

```
benchmarks/datasets/<track>/<dataset>/manifest.json
```

### Schema

Manifests follow `benchmarks/manifest_schema.json` (JSON Schema draft 2020-12).

**Required fields:**
- `schema_version` - Always 1
- `id` - Unique dataset identifier
- `track` - Benchmark track (asr, diarization, etc.)
- `split` - Dataset split (smoke, test, dev, train)
- `source` - Dataset provenance
- `license` - License information
- `meta` - Aggregate statistics

### Validation

```bash
# Validate all manifests against schema
python scripts/fetch_datasets.py validate

# Run manifest tests
pytest tests/test_dataset_manifests.py -v
```

## Download and Verification

### Using fetch_datasets.py

```bash
# List all datasets
python scripts/fetch_datasets.py list

# Show dataset status
python scripts/fetch_datasets.py status

# Fetch all smoke datasets
python scripts/fetch_datasets.py fetch --smoke

# Fetch specific dataset
python scripts/fetch_datasets.py fetch --dataset librispeech-test-clean

# Verify dataset integrity
python scripts/fetch_datasets.py verify --dataset asr-smoke

# Show license information
python scripts/fetch_datasets.py license --dataset ami-headset
```

### Using setup_benchmark_datasets.py

```bash
# Show available datasets
python scripts/setup_benchmark_datasets.py status

# Setup LibriSpeech
python scripts/setup_benchmark_datasets.py setup librispeech-test-clean

# Setup all LibriSpeech splits
python scripts/setup_benchmark_datasets.py setup --all-librispeech

# Verify downloaded files
python scripts/setup_benchmark_datasets.py verify librispeech-test-clean
```

### Hash Verification

All datasets use SHA256 hashes for integrity verification:

- **Audio files:** Verify content hasn't changed
- **RTTM files:** Verify reference annotations
- **Archives:** Verify download integrity

## License Compliance

| Dataset | License | Citation Required | Commercial Use |
|---------|---------|-------------------|----------------|
| smoke | MIT | No | Yes |
| diarization-smoke | MIT | No | Yes |
| diarization-smoke-tones | MIT | No | Yes |
| LibriSpeech | CC-BY-4.0 | Yes (papers) | Yes |
| Common Voice | CC0-1.0 | No | Yes (with terms) |
| AMI | CC-BY-4.0 | Yes (required) | Yes |
| CALLHOME | LDC | Yes (required) | License required |

**Important:**
- Always review the license field in manifest files
- Some datasets require accepting terms before download
- Provide citations when using datasets in publications

## Adding New Datasets

1. Create manifest directory: `benchmarks/datasets/<track>/<dataset>/`
2. Create `manifest.json` following the schema
3. Validate: `python scripts/fetch_datasets.py validate`
4. For smoke datasets: commit audio to `benchmarks/data/`
5. Add iterator in `transcription/benchmarks.py` if needed
6. Register in benchmark runner
7. Create baseline: `slower-whisper benchmark save-baseline`
8. Update this document

See [docs/DATASET_MANIFEST.md](../docs/DATASET_MANIFEST.md) for detailed instructions.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLOWER_WHISPER_BENCHMARKS` | `~/.cache/slower-whisper/benchmarks` | Cache directory for downloaded datasets |

## Related Documentation

- [Benchmark CLI Reference](../docs/BENCHMARKS.md)
- [Dataset Manifest Infrastructure](../docs/DATASET_MANIFEST.md)
- [LibriSpeech Setup](../docs/LIBRISPEECH_SETUP.md)
- [AMI Setup](../docs/AMI_SETUP.md)
- [CALLHOME Setup](../docs/CALLHOME_SETUP.md)
- [IEMOCAP Setup](../docs/IEMOCAP_SETUP.md)
- [Baselines](baselines/README.md)
