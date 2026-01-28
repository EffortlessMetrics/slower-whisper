# Benchmark Dataset Cache Strategy Analysis

**Date:** 2025-11-19
**Status:** Production-Ready Infrastructure with Minor Enhancements Recommended

## Executive Summary

The benchmark dataset cache strategy in slower-whisper is **well-architected and consistent**, using a centralized `CachePaths` system with proper separation between model weights, sample datasets, and benchmark evaluation datasets. The infrastructure is production-ready with a few minor improvements recommended for better DX and maintainability.

**Overall Assessment:** ✅ **ALIGNED** - Minor enhancements recommended

---

## Current Architecture

### Cache Hierarchy

```
~/.cache/slower-whisper/                     (root)
├── hf/                                      (Hugging Face models)
├── torch/                                   (PyTorch models)
├── whisper/                                 (Whisper model weights)
├── emotion/                                 (Emotion recognition models)
├── diarization/                             (Pyannote diarization models)
├── samples/                                 (Small test datasets, auto-downloadable)
│   └── mini_diarization/                    (Synthetic/Kaggle samples)
└── benchmarks/                              (Large evaluation datasets, manually staged)
    ├── ami/                                 (AMI Meeting Corpus)
    │   ├── audio/
    │   ├── annotations/
    │   └── splits/
    ├── iemocap/                             (IEMOCAP emotion dataset)
    │   └── Session1-5/
    └── libricss/                            (LibriCSS overlapping speech)
```

### Environment Variable Control

- `SLOWER_WHISPER_CACHE_ROOT`: Override entire cache root (default: `~/.cache/slower-whisper`)
- `SLOWER_WHISPER_BENCHMARKS`: Override benchmarks location only
- `SLOWER_WHISPER_SAMPLES`: Override samples location only
- `HF_HOME`: Hugging Face cache (auto-configured if not set)
- `TORCH_HOME`: PyTorch cache (auto-configured if not set)

---

## Consistency Analysis

### ✅ What's Working Well

1. **Unified Access Pattern**
   - All benchmark iterators (`iter_ami_meetings`, `iter_iemocap_clips`) use `get_benchmarks_root()`
   - Consistent use of `CachePaths.from_env().benchmarks_root`
   - No hardcoded absolute paths in dataset discovery code

2. **Clear Separation of Concerns**
   - **Samples** (`samples/`): Small, auto-downloadable test datasets for dogfooding
   - **Benchmarks** (`benchmarks/`): Large, manually-staged evaluation datasets with reference annotations
   - Each has dedicated helper functions and documentation

3. **Documented Structure**
   - `docs/AMI_SETUP.md`: Comprehensive setup guide with expected directory layout
   - `docs/AMI_DIRECTORY_LAYOUT.md`: Visual structure reference
   - `transcription/benchmarks.py`: Inline docstrings document expected structure

4. **Proper Error Handling**
   - `FileNotFoundError` with helpful messages pointing to setup docs
   - Directory structure validation in iterators
   - Available vs. unavailable dataset detection via `list_available_benchmarks()`

5. **Environment Overrides Work Correctly**
   - `SLOWER_WHISPER_BENCHMARKS` properly overrides default location
   - Falls back to `CachePaths.benchmarks_root` if not set

### ⚠️ Inconsistencies Found

#### 1. **Hardcoded Paths in Evaluation Scripts** (Minor)

**Issue:** `benchmarks/eval_summaries.py` uses hardcoded relative paths for output:

```python
# Line 343
json_path = Path("whisper_json") / f"{sample.id}.json"

# Line 358
output_dir=Path("whisper_json")

# Line 499
results_dir = Path("benchmarks/results")
```

**Impact:** Results and transcripts written to current working directory, not cache

**Recommendation:** Add output directory configuration or use a dedicated results cache:

```python
# Option 1: Use CachePaths for results
paths = CachePaths.from_env()
results_dir = paths.benchmarks_root.parent / "eval_results"

# Option 2: Make configurable via CLI args (current pattern)
parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
parser.add_argument("--transcript-cache", type=Path, default=Path("whisper_json"))
```

**Status:** Current approach (CLI args) is acceptable for flexibility, but could benefit from documenting the convention.

#### 2. **`.gitignore` Missing Benchmark Results** (Moderate)

**Issue:** `benchmarks/results/*.json` files are NOT ignored, causing them to show up as untracked:

```bash
?? benchmarks/results/asr_diar_ami_synth.json
?? benchmarks/results/asr_diar_test.json
?? benchmarks/results/asr_diar_test_base.json
```

**Impact:** Benchmark result artifacts pollute git status, risk accidental commits

**Recommendation:** Add to `.gitignore`:

```gitignore
# Benchmark evaluation results
benchmarks/results/*.json
benchmarks/results/*.csv
benchmarks/results/*.md
benchmarks/results/benchmark_*.json
benchmarks/results/ami_*.json
benchmarks/results/asr_diar_*.json
```

#### 3. **Missing Dataset Setup Docs** (Resolved)

**Previous issue:** `transcription/benchmarks.py` referenced non-existent documentation:

```python
# Line 187 (iter_iemocap_clips docstring)
"Please see docs/IEMOCAP_SETUP.md for setup instructions."

# Line 342 (list_available_benchmarks)
"setup_doc": "docs/IEMOCAP_SETUP.md",
"setup_doc": "docs/LIBRICSS_SETUP.md",
```

**Resolution:** Both setup guides now exist and are referenced directly:
- `docs/IEMOCAP_SETUP.md`
- `docs/LIBRICSS_SETUP.md`

---

## Recommended Enhancements

### 1. **Helper Methods for Dataset Management** (Nice-to-Have)

Add to `CachePaths` or `transcription/benchmarks.py`:

```python
def validate_dataset_structure(dataset: str) -> dict[str, bool]:
    """Validate that a benchmark dataset has the expected structure.

    Returns:
        Dict with validation results:
        {
            "audio_dir_exists": bool,
            "annotations_dir_exists": bool,
            "has_audio_files": bool,
            "has_annotation_files": bool,
            "split_files_exist": bool
        }
    """
    root = get_benchmarks_root() / dataset
    return {
        "audio_dir_exists": (root / "audio").exists(),
        "annotations_dir_exists": (root / "annotations").exists(),
        "has_audio_files": len(list((root / "audio").glob("*.wav"))) > 0,
        "has_annotation_files": len(list((root / "annotations").glob("*.json"))) > 0,
        "split_files_exist": (root / "splits" / "test.txt").exists(),
    }

def get_dataset_info(dataset: str) -> dict[str, Any]:
    """Get comprehensive info about a benchmark dataset.

    Returns:
        Dict with:
        - path: Path to dataset
        - available: bool
        - num_audio_files: int
        - num_annotation_files: int
        - splits_available: list[str]
        - total_size_mb: float
    """
    # Implementation
```

**Benefit:** Better debugging, easier verification scripts

### 2. **Centralized Results Cache** (Optional)

Add `eval_results_root` to `CachePaths`:

```python
@dataclass
class CachePaths:
    # ... existing fields ...
    eval_results_root: Path  # Benchmark evaluation results (JSON, CSV, MD)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> CachePaths:
        # ...
        eval_results_root = root / "eval_results"

        return cls(
            # ...
            eval_results_root=eval_results_root,
        )
```

**Benefit:** Keeps results organized in cache, not scattered in repo

**Trade-off:** Current approach (results in repo) makes it easy to version control baselines

**Decision:** Current approach is fine; just ensure `.gitignore` excludes them

### 3. **Dataset Discovery CLI Command** (Nice-to-Have)

Add to `transcription/cli.py`:

```python
@click.command()
def benchmark_datasets():
    """List available benchmark datasets and their status."""
    from transcription.benchmarks import list_available_benchmarks

    datasets = list_available_benchmarks()

    click.echo("Benchmark Datasets:")
    click.echo("=" * 70)

    for name, info in datasets.items():
        status = "✓ Available" if info["available"] else "✗ Not found"
        click.echo(f"\n{name.upper()}: {status}")
        click.echo(f"  Path: {info['path']}")
        click.echo(f"  Description: {info['description']}")
        click.echo(f"  Tasks: {', '.join(info['tasks'])}")
        click.echo(f"  Setup: {info['setup_doc']}")
```

**Benefit:** Easy way to verify dataset availability without writing Python code

---

## `.gitignore` Recommendations

Add the following section to `.gitignore`:

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

# Keep baseline results (if desired)
# Uncomment to track specific baseline files:
# !benchmarks/results/baseline_v1.0.json

# Temporary transcript cache (used during eval)
whisper_json/

# Evaluation scratch files
benchmarks/scratch/
benchmarks/temp/
```

**Rationale:**
- Benchmark results are transient artifacts, not source code
- They can be regenerated from benchmark datasets
- Large JSON files clutter repo history
- Users may want to track specific baseline files (allow explicit override with `!`)

---

## Documentation Recommendations

### Update `CLAUDE.md`

Add a new section:

```markdown
## Benchmark Datasets

**Location:** `~/.cache/slower-whisper/benchmarks/` (configurable via `SLOWER_WHISPER_BENCHMARKS`)

**Purpose:** Large evaluation datasets with reference annotations for quality measurement.

**Available Datasets:**
- **AMI Meeting Corpus:** Speaker diarization, summarization, action items
  - Setup: `docs/AMI_SETUP.md`
  - ~100 hours of meeting recordings
  - Manually staged (requires license acceptance)

- **IEMOCAP:** Emotion recognition
  - Setup: `docs/IEMOCAP_SETUP.md`
  - ~12 hours of emotional speech
  - Manually staged (requires license)

- **LibriCSS:** Overlapping speech and diarization
  - Setup: `docs/LIBRICSS_SETUP.md`
  - Derived from LibriSpeech
  - Manually staged

**Usage:**
```python
from transcription.benchmarks import iter_ami_meetings, get_benchmarks_root

# Check if AMI is available
root = get_benchmarks_root()
ami_available = (root / "ami").exists()

# Iterate over test split
for sample in iter_ami_meetings(split="test", limit=5):
    print(f"{sample.id}: {sample.audio_path}")
```

**Environment Overrides:**
```bash
# Use custom benchmark location
export SLOWER_WHISPER_BENCHMARKS=/data/benchmarks
```

**Important:** Benchmark datasets are NOT auto-downloaded. See setup docs for staging instructions.
```

### Dataset Setup Docs

Documentation is now available:

- `docs/IEMOCAP_SETUP.md` — Emotion dataset setup (IEMOCAP)
- `docs/LIBRICSS_SETUP.md` — Overlapping speech / diarization setup (LibriCSS)

---

## Summary of Findings

### Current State: ✅ WELL-DESIGNED

| Aspect | Status | Notes |
|--------|--------|-------|
| **Cache Path Consistency** | ✅ Excellent | All iterators use `get_benchmarks_root()` |
| **Directory Structure** | ✅ Documented | `docs/AMI_SETUP.md` provides clear template |
| **Environment Overrides** | ✅ Working | `SLOWER_WHISPER_BENCHMARKS` properly respected |
| **Error Messages** | ✅ Helpful | Point to setup docs with clear instructions |
| **Separation of Concerns** | ✅ Clear | samples vs benchmarks distinction well-defined |
| **`.gitignore` Coverage** | ⚠️ Incomplete | Missing `benchmarks/results/*.json` |
| **Documentation Completeness** | ✅ Complete | AMI, IEMOCAP, and LibriCSS setup guides are documented |
| **Hardcoded Paths** | ⚠️ Minor Issue | Eval scripts use `Path("whisper_json")` |

### Recommended Actions

**High Priority:**
1. ✅ **Add `benchmarks/results/*.json` to `.gitignore`**
   - Prevents accidental commits of transient result files
   - Keeps git status clean

**Medium Priority:**
2. ✅ **Document IEMOCAP and LibriCSS setup**
   - Setup guides now exist and are referenced by error messages

**Low Priority (Nice-to-Have):**
3. 💡 **Add dataset validation helpers** to `benchmarks.py`
4. 💡 **Add `benchmark-datasets` CLI command** for easy status checking
5. 💡 **Update `CLAUDE.md`** with benchmark datasets section

### No Action Needed

- ✅ Cache path infrastructure is solid
- ✅ `CachePaths` design is extensible and well-documented
- ✅ Iterator consistency is excellent
- ✅ Hardcoded paths in eval scripts are acceptable (configurable via CLI args)

---

## Conclusion

The benchmark dataset cache strategy is **production-ready and well-aligned** across the codebase. The core infrastructure (`CachePaths`, `get_benchmarks_root()`, dataset iterators) is consistently implemented with proper separation between auto-downloadable samples and manually-staged benchmarks.

**Minor improvements** (adding `.gitignore` entries and stub docs) will enhance developer experience but do not indicate architectural issues.

**Recommended Next Step:** Apply high-priority fixes (`.gitignore` update) and consider creating minimal stub documentation for referenced but missing setup guides.
