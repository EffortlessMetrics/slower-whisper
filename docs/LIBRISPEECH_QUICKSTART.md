# LibriSpeech Integration Quickstart

LibriSpeech is a corpus of 1000 hours of read English speech, derived from audiobooks. It's the gold standard for evaluating ASR accuracy (WER).

## Setup

### 1. Download LibriSpeech Subsets

Download from OpenSLR (www.openslr.org/12):

```bash
# Create benchmarks directory
mkdir -p ~/.cache/slower-whisper/benchmarks/librispeech

# Download recommended evaluation subsets (clean speech, ~350MB each)
cd ~/.cache/slower-whisper/benchmarks/librispeech

# Dev set (5.4 hours, 337 MB)
wget https://www.openslr.org/resources/12/dev-clean.tar.gz

# Test set (5.4 hours, 346 MB)
wget https://www.openslr.org/resources/12/test-clean.tar.gz

# Extract archives
tar -xzf dev-clean.tar.gz
tar -xzf test-clean.tar.gz
```

For robustness testing (noisy conditions):

```bash
# Challenging dev set (5.3 hours, 314 MB)
wget https://www.openslr.org/resources/12/dev-other.tar.gz

# Challenging test set (5.1 hours, 328 MB)
wget https://www.openslr.org/resources/12/test-other.tar.gz

tar -xzf dev-other.tar.gz
tar -xzf test-other.tar.gz
```

### 2. Verify Directory Structure

After extraction, you should have:

```
~/.cache/slower-whisper/benchmarks/librispeech/
  LibriSpeech/
    dev-clean/
      <speaker_id>/
        <chapter_id>/
          <speaker_id>-<chapter_id>.trans.txt
          <speaker_id>-<chapter_id>-<utterance_id>.flac
          ...
    test-clean/
      ...
    dev-other/
      ...
    test-other/
      ...
```

## Usage

### Python API

```python
from transcription.benchmarks import iter_librispeech

# Iterate over dev-clean samples
for sample in iter_librispeech(split="dev-clean", limit=10):
    print(f"Sample: {sample.id}")
    print(f"  Audio: {sample.audio_path}")
    print(f"  Reference: {sample.reference_transcript}")
    print(f"  Speaker: {sample.metadata['speaker_id']}")
```

### Evaluation Script

Evaluate WER on LibriSpeech:

```bash
# Quick test: 10 samples from dev-clean
uv run python benchmarks/eval_asr_diarization.py \
    --dataset librispeech \
    --n 10

# Full dev-clean evaluation (~2700 samples)
uv run python benchmarks/eval_asr_diarization.py \
    --dataset librispeech \
    --split dev-clean

# Test-clean evaluation
uv run python benchmarks/eval_asr_diarization.py \
    --dataset librispeech \
    --split test-clean

# Use different model and device
uv run python benchmarks/eval_asr_diarization.py \
    --dataset librispeech \
    --split dev-clean \
    --model base \
    --device cpu \
    --n 50
```

### Output

Results are saved to `benchmarks/results/asr_librispeech_{split}.json`:

```json
{
  "dataset": "librispeech",
  "split": "dev-clean",
  "config": {
    "model": "large-v3",
    "device": "cuda",
    "enable_diarization": false
  },
  "aggregate": {
    "num_samples": 2703,
    "avg_WER": 0.042,
    "min_WER": 0.0,
    "max_WER": 0.312
  },
  "samples": [
    {
      "id": "1272-128104-0000",
      "WER": 0.0,
      "DER": null,
      "num_segments": 1
    },
    ...
  ]
}
```

## Available Splits

| Split | Size | Duration | Quality | Purpose |
|-------|------|----------|---------|---------|
| `dev-clean` | 337 MB | 5.4 hours | Clean | Development/tuning |
| `test-clean` | 346 MB | 5.4 hours | Clean | Final evaluation |
| `dev-other` | 314 MB | 5.3 hours | Noisy | Robustness testing |
| `test-other` | 328 MB | 5.1 hours | Noisy | Final robustness eval |

For most evaluations, use `dev-clean` or `test-clean` (high-quality audio, standard WER benchmark).

## Notes

- **Single-speaker only**: LibriSpeech is read speech by individual speakers, no diarization ground truth
- **No diarization**: Evaluation script automatically disables diarization for LibriSpeech
- **FLAC audio**: LibriSpeech uses FLAC compression, which slower-whisper handles automatically
- **WER only**: Only Word Error Rate is computed (no DER, no speaker labels)
- **Fast evaluation**: Single-speaker audio is faster to process than multi-speaker conversations

## Troubleshooting

### "LibriSpeech split not found"

Ensure you extracted the archive to the correct location:

```bash
# Should exist:
ls ~/.cache/slower-whisper/benchmarks/librispeech/LibriSpeech/dev-clean/

# If not, check extraction path
tar -xzf dev-clean.tar.gz -C ~/.cache/slower-whisper/benchmarks/librispeech/
```

### Missing jiwer package

```bash
uv pip install jiwer
```

### Slow evaluation

Use a smaller model or limit sample count:

```bash
# Use base model (faster, slightly lower accuracy)
uv run python benchmarks/eval_asr_diarization.py \
    --dataset librispeech \
    --model base \
    --n 100

# Use CPU if GPU unavailable
uv run python benchmarks/eval_asr_diarization.py \
    --dataset librispeech \
    --device cpu \
    --n 50
```
