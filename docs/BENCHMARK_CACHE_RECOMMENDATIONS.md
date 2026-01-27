# Benchmark Cache Strategy: Recommended Changes

**Analysis Date:** 2025-11-19
**Last Updated:** 2026-01-21
**Status:** ✅ Infrastructure is aligned; most recommendations implemented

## Overall Assessment

The benchmark dataset cache strategy is **well-architected and consistent**. All datasets properly use `CachePaths.benchmarks_root` with no hardcoded paths in dataset discovery code. Minor improvements recommended below.

For detailed analysis, see: [`docs/BENCHMARK_CACHE_STRATEGY_ANALYSIS.md`](docs/BENCHMARK_CACHE_STRATEGY_ANALYSIS.md)

---

## Recommended Changes

### 1. Update `.gitignore` (High Priority)

**Issue:** Benchmark result JSON files are not excluded, causing clutter in git status.

**Current State:**
```bash
$ git status
?? benchmarks/results/asr_diar_ami_synth.json
?? benchmarks/results/asr_diar_test.json
?? benchmarks/results/asr_diar_test_base.json
?? benchmarks/results/asr_diar_test_base_cpu.json
```

**Recommended Addition to `.gitignore`:**

```gitignore
# ===========================
# Benchmark Evaluation
# ===========================
# Benchmark result artifacts (JSON, CSV, Markdown reports)
benchmarks/results/*.json
benchmarks/results/*.csv
benchmarks/results/*.md
benchmarks/results/benchmark_*.json
benchmarks/results/ami_*.json
benchmarks/results/asr_diar_*.json

# Keep baseline results (if desired for version control)
# Uncomment to track specific baseline files:
# !benchmarks/results/baseline_v1.0.json
# !benchmarks/results/baseline_v1.1.json

# Temporary transcript cache used during evaluation
whisper_json/

# Evaluation scratch workspace
benchmarks/scratch/
benchmarks/temp/
```

**Rationale:**
- Benchmark results are transient artifacts, not source code
- They can be regenerated from benchmark datasets
- Large JSON files clutter repo history
- If you want to track specific baseline files, use explicit `!` override

---

### 2. Dataset Setup Docs (Resolved)

Setup guides are now available:

- [`docs/IEMOCAP_SETUP.md`](IEMOCAP_SETUP.md)
- [`docs/LIBRICSS_SETUP.md`](LIBRICSS_SETUP.md)

---

### 3. Update `CLAUDE.md` with Benchmark Section (Low Priority)

**Issue:** `CLAUDE.md` doesn't document the benchmark dataset infrastructure.

**Recommended Addition:**

Add after the "Testing Philosophy" section:

```markdown
## Benchmark Datasets

**Purpose:** Large evaluation datasets with reference annotations for measuring slower-whisper quality on standard tasks (diarization, summarization, emotion recognition).

**Location:** `~/.cache/slower-whisper/benchmarks/` (configurable)

**Key Distinction:**
- **`samples/`**: Small, auto-downloadable test datasets for dogfooding (e.g., synthetic audio)
- **`benchmarks/`**: Large, manually-staged evaluation corpora (e.g., AMI, IEMOCAP)

### Available Datasets

1. **AMI Meeting Corpus** (`ami/`)
   - **Tasks:** Speaker diarization, meeting summarization, action items
   - **Size:** ~100 hours of meeting recordings
   - **Setup:** See [`docs/AMI_SETUP.md`](docs/AMI_SETUP.md)
   - **License:** CC BY 4.0 (manual download required)

2. **IEMOCAP** (`iemocap/`)
   - **Tasks:** Emotion recognition (categorical and dimensional)
   - **Size:** ~12 hours of emotional speech across 5 sessions
   - **Setup:** See [`docs/IEMOCAP_SETUP.md`](docs/IEMOCAP_SETUP.md)
   - **License:** USC SAIL (academic use, manual download)

3. **LibriCSS** (`libricss/`)
   - **Tasks:** Overlapping speech, continuous diarization
   - **Size:** Derived from LibriSpeech (multi-talker scenarios)
   - **Setup:** See [`docs/LIBRICSS_SETUP.md`](docs/LIBRICSS_SETUP.md)
   - **License:** CC BY 4.0

### Usage

**Check dataset availability:**
```python
from transcription.benchmarks import list_available_benchmarks

benchmarks = list_available_benchmarks()
print(f"AMI available: {benchmarks['ami']['available']}")
print(f"AMI path: {benchmarks['ami']['path']}")
```

**Iterate over dataset:**
```python
from transcription.benchmarks import iter_ami_meetings

for sample in iter_ami_meetings(split="test", limit=5):
    print(f"{sample.id}: {sample.audio_path}")
    print(f"  Reference speakers: {sample.reference_speakers}")
    print(f"  Reference summary: {sample.reference_summary[:100]}...")
```

**Environment overrides:**
```bash
# Use custom benchmark location (e.g., mounted dataset volume)
export SLOWER_WHISPER_BENCHMARKS=/data/benchmarks
```

### Running Evaluations

```bash
# Speaker diarization evaluation (WER, DER)
uv run python benchmarks/eval_asr_diarization.py --dataset ami --n 5

# LLM-based summary evaluation (Claude-as-judge)
uv run python benchmarks/eval_summaries.py --dataset ami --n 10

# Compare two runs
uv run python benchmarks/compare_results.py \
    --before benchmarks/results/baseline.json \
    --after benchmarks/results/improved.json
```

Results saved to `benchmarks/results/*.json`.

### Important Notes

- **Manual Setup Required:** Benchmark datasets are NOT auto-downloaded
- **License Compliance:** Most require accepting terms before download
- **Size:** Each dataset is multi-GB; plan for adequate disk space
- **Cache Location:** Stored in user cache, not project directory
- **Git Exclusion:** Result artifacts in `benchmarks/results/` are gitignored
```

---

## Optional Enhancements (Future Consideration)

### Add Dataset Validation Helper

**Purpose:** Make it easier to debug dataset setup issues.

**Implementation:** Add to `transcription/benchmarks.py`:

```python
def validate_dataset_structure(dataset: str) -> dict[str, bool]:
    """Validate that a benchmark dataset has the expected structure.

    Args:
        dataset: Dataset name ("ami", "iemocap", "libricss")

    Returns:
        Dict with validation results:
        {
            "audio_dir_exists": bool,
            "annotations_dir_exists": bool,
            "has_audio_files": bool,
            "has_annotation_files": bool,
            "split_files_exist": bool,
            "num_audio_files": int,
            "num_annotation_files": int
        }

    Example:
        >>> validation = validate_dataset_structure("ami")
        >>> if not validation["audio_dir_exists"]:
        ...     print("Missing audio/ directory - see docs/AMI_SETUP.md")
    """
    root = get_benchmarks_root() / dataset

    audio_dir = root / "audio"
    annotations_dir = root / "annotations"
    splits_dir = root / "splits"

    audio_files = list(audio_dir.glob("*.wav")) if audio_dir.exists() else []
    annotation_files = list(annotations_dir.glob("*.json")) if annotations_dir.exists() else []

    return {
        "dataset": dataset,
        "root_exists": root.exists(),
        "audio_dir_exists": audio_dir.exists(),
        "annotations_dir_exists": annotations_dir.exists(),
        "splits_dir_exists": splits_dir.exists(),
        "has_audio_files": len(audio_files) > 0,
        "has_annotation_files": len(annotation_files) > 0,
        "split_files_exist": (splits_dir / "test.txt").exists() if splits_dir.exists() else False,
        "num_audio_files": len(audio_files),
        "num_annotation_files": len(annotation_files),
        "total_size_mb": sum(f.stat().st_size for f in audio_files) / (1024 * 1024) if audio_files else 0,
    }
```

**Usage in verification script:**

```python
#!/usr/bin/env python3
"""Verify benchmark dataset setup."""
from transcription.benchmarks import list_available_benchmarks, validate_dataset_structure

benchmarks = list_available_benchmarks()

print("Benchmark Dataset Status")
print("=" * 70)

for name, info in benchmarks.items():
    print(f"\n{name.upper()}")
    print(f"  Path: {info['path']}")
    print(f"  Available: {info['available']}")

    if info['available']:
        validation = validate_dataset_structure(name)
        print(f"  Audio files: {validation['num_audio_files']}")
        print(f"  Annotation files: {validation['num_annotation_files']}")
        print(f"  Total size: {validation['total_size_mb']:.1f} MB")

        if not validation['has_audio_files']:
            print(f"  ⚠️  No audio files found!")
        if not validation['split_files_exist']:
            print(f"  ⚠️  Missing split files (train/dev/test.txt)")
    else:
        print(f"  → See {info['setup_doc']} for setup instructions")
```

---

## Summary Checklist

- [x] **Add benchmark results to `.gitignore`** — Done: `benchmarks/results/` excluded
- [x] **Create `docs/IEMOCAP_SETUP.md`** — Done: Full setup guide with label mapping
- [x] **Create `docs/LIBRICSS_SETUP.md`** — Done: Full setup guide
- [ ] **Update `CLAUDE.md` with benchmark datasets section** (low priority)
- [ ] **Consider adding `validate_dataset_structure()` helper** (optional)
- [ ] **Consider adding `benchmark-datasets` CLI command** (optional)

## Files to Modify

1. `CLAUDE.md` - Add benchmark datasets section
2. `transcription/benchmarks.py` - Optionally add validation helper

---

**Conclusion:** The infrastructure is solid. These changes are polish and documentation improvements, not architectural fixes. The core `CachePaths` system and dataset iterators are production-ready and well-designed.
