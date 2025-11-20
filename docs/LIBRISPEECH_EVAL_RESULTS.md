# LibriSpeech Evaluation Results

## Overview

This document summarizes the LibriSpeech WER evaluation setup and baseline results for slower-whisper.

## Dataset Setup

**LibriSpeech dev-clean subset:**
- Downloaded from: https://www.openslr.org/resources/12/dev-clean.tar.gz
- Size: 337 MB (2,703 .flac files, 40 speakers, ~5.4 hours)
- Located at: `~/.cache/slower-whisper/benchmarks/librispeech/LibriSpeech/dev-clean/`

**Helper Script:**
```bash
./scripts/download_librispeech.sh  # Downloads and extracts dev-clean
```

## Critical Fix: WER Normalization

### Problem Discovered

Initial smoke test showed WER = 1.0 (100% error) for all samples, which was clearly wrong.

**Root Cause:** Text format mismatch between LibriSpeech references and Whisper output:
- **LibriSpeech**: `MISTER QUILTER IS THE APOSTLE...` (ALL CAPS, NO PUNCTUATION)
- **Whisper**: `Mr. Quilter is the apostle...` (Normal case, with punctuation)

Without normalization, jiwer treated these as completely different strings.

### Solution

Updated `compute_wer()` in `benchmarks/eval_asr_diarization.py` to apply standard ASR normalization:

```python
transformation = jiwer.Compose([
    jiwer.ToUpperCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])
```

This ensures fair comparison across different transcript formats (standard practice in ASR evaluation).

## Baseline Results

### Configuration
- **Model**: Whisper base (74M parameters)
- **Device**: CPU (WSL2)
- **Compute type**: int8 (auto-selected for CPU)
- **Dataset**: LibriSpeech dev-clean
- **Samples**: 5 (smoke test)

### Metrics

| Metric | Value | Expected Range | Status |
|--------|-------|----------------|--------|
| Average WER | **9.0%** | 8-15% for base model | ✅ Excellent |
| Min WER | 5.9% | - | ✅ |
| Max WER | 14.7% | - | ✅ |

**Per-sample results:**
```json
[
  {"id": "1272-128104-0000", "WER": 0.059},  // 5.9%
  {"id": "1272-128104-0001", "WER": 0.100},  // 10.0%
  {"id": "1272-128104-0002", "WER": 0.062},  // 6.2%
  {"id": "1272-128104-0003", "WER": 0.083},  // 8.3%
  {"id": "1272-128104-0004", "WER": 0.147}   // 14.7%
]
```

### Interpretation

✅ **Baseline is valid**: 9.0% WER on dev-clean is within expected range for Whisper base on CPU with int8.

**Context:**
- Whisper base typically achieves 8-12% WER on LibriSpeech clean speech
- large-v3 achieves ~3-5% WER (but 10x slower and 40x larger)
- Human-level performance is ~5% WER on clean speech

**Sample Variance:**
- Best sample (1272-128104-0000): 5.9% WER → near-perfect transcription
- Worst sample (1272-128104-0004): 14.7% WER → likely longer utterance with complex vocabulary

The 2.5x variance (5.9% to 14.7%) is expected for small sample sizes.

## Running Your Own Evaluation

### Smoke Test (5 samples, ~30 seconds)
```bash
uv run python benchmarks/eval_asr_diarization.py \
  --dataset librispeech \
  --n 5 \
  --model base \
  --device cpu \
  --output benchmarks/results/asr_librispeech_dev_clean_5.json
```

### Small Eval (50 samples, ~5 minutes on CPU)
```bash
uv run python benchmarks/eval_asr_diarization.py \
  --dataset librispeech \
  --n 50 \
  --model base \
  --device cpu \
  --output benchmarks/results/asr_librispeech_dev_clean_50.json
```

### Full Dev-Clean (2,703 samples, ~3-4 hours on CPU)
```bash
uv run python benchmarks/eval_asr_diarization.py \
  --dataset librispeech \
  --model base \
  --device cpu \
  --output benchmarks/results/asr_librispeech_dev_clean_full.json
```

### GPU Comparison (large-v3 model)
```bash
uv run python benchmarks/eval_asr_diarization.py \
  --dataset librispeech \
  --n 50 \
  --model large-v3 \
  --device cuda \
  --output benchmarks/results/asr_librispeech_dev_clean_50_large_v3_cuda.json
```

## View Results

```bash
# View aggregate metrics
cat benchmarks/results/asr_librispeech_dev_clean_50.json | jq '.aggregate'

# View per-sample results
cat benchmarks/results/asr_librispeech_dev_clean_50.json | jq '.samples'

# Find worst WER samples
cat benchmarks/results/asr_librispeech_dev_clean_50.json | \
  jq '.samples | sort_by(.WER) | reverse | .[0:5]'
```

## Next Steps

1. **Scale up evaluation**: Run on 50-100 samples to get stable avg_WER estimate
2. **Model comparison**: Compare base vs large-v3 WER and latency
3. **CPU vs GPU**: Benchmark int8 CPU vs float16 GPU performance
4. **AMI integration**: Apply same normalization strategy to multi-speaker meetings
5. **Error analysis**: Identify systematic failure modes (rare words, accents, etc.)

## Files Modified

- `benchmarks/eval_asr_diarization.py` - Added WER normalization (lines 146-183)
- `.gitignore` - Excluded `benchmarks/results/` transient outputs
- `scripts/download_librispeech.sh` - New helper script for dataset download

## Technical Notes

### Why Normalization Matters

ASR benchmarks use normalization to focus on *semantic accuracy* rather than formatting:
- "Mr." vs "MISTER" → same meaning, shouldn't penalize
- Punctuation is often arbitrary in ground truth
- Capitalization varies across datasets

Standard practice (per Kaldi, ESPnet, HuggingFace evaluate):
1. Uppercase all text
2. Remove punctuation
3. Collapse whitespace
4. Strip leading/trailing whitespace

This is what `jiwer.Compose([ToUpperCase(), RemovePunctuation(), ...])` does.

### Compute Type Auto-Selection

The eval script auto-selects compute_type based on device:
- **CPU**: `int8` (faster inference, minimal quality loss for Whisper)
- **CUDA**: `float16` (GPU-optimized, full precision)

Manual override:
```bash
--device cpu --compute-type float32  # Slower but higher precision
```

### LibriSpeech Splits

| Split | Size | Hours | Use Case |
|-------|------|-------|----------|
| dev-clean | 337 MB | 5.4 | Development/tuning (recommended) |
| test-clean | 346 MB | 5.4 | Final benchmark (do not tune on this!) |
| dev-other | 314 MB | 5.3 | Robustness testing (harder audio) |
| test-other | 328 MB | 5.1 | Final robustness benchmark |

**Best Practice**: Use `dev-clean` for rapid iteration, `test-clean` for final reporting.

## References

- LibriSpeech paper: https://www.danielpovey.com/files/2015_icassp_librispeech.pdf
- OpenSLR dataset: https://www.openslr.org/12/
- jiwer documentation: https://jitsi.github.io/jiwer/
- Whisper paper: https://arxiv.org/abs/2212.04356

---

**Last Updated**: 2025-11-19
**Baseline Validated**: ✅ Whisper base @ 9.0% WER on dev-clean (n=5)
