# IEMOCAP Evaluation Quick Reference

One-page reference for running IEMOCAP emotion evaluation with slower-whisper.

## Prerequisites

```bash
# 1. Install emotion dependencies
uv sync --extra emotion

# 2. Download IEMOCAP from https://sail.usc.edu/iemocap/
#    (Requires registration and signed EULA)

# 3. Stage IEMOCAP at:
mkdir -p ~/.cache/slower-whisper/benchmarks/iemocap
cp -r /path/to/IEMOCAP_full_release/Session* ~/.cache/slower-whisper/benchmarks/iemocap/
```

## Quick Commands

```bash
# Verify setup
uv run python -c "
from transcription.benchmarks import list_available_benchmarks
print(list_available_benchmarks()['iemocap'])
"

# Test on 10 clips (sanity check)
uv run python benchmarks/eval_emotion.py --limit 10 --mode categorical

# Evaluate Session 5 (standard test set) - categorical
uv run python benchmarks/eval_emotion.py --session Session5 --mode categorical

# Evaluate Session 5 - dimensional
uv run python benchmarks/eval_emotion.py --session Session5 --mode dimensional

# Full evaluation (both modes, all sessions)
uv run python benchmarks/eval_emotion.py --mode both
```

## Label Mapping (Categorical)

| IEMOCAP | Model Output | Notes |
|---------|-------------|-------|
| `neu` | neutral | Direct match |
| `ang` | angry | Direct match |
| `sad` | sad | Direct match |
| `hap` | happy | Direct match |
| `exc` | happy | Excitement → happy |
| `fru` | angry | Frustration → angry |
| `fea` | fear | Direct match |
| `sur` | surprise | Direct match |
| `dis` | disgust | Direct match |
| `oth` | (excluded) | Ambiguous |

## Scale Conversion (Dimensional)

```python
# IEMOCAP: 1-5 scale → Model: 0-1 scale
model_value = (iemocap_value - 1.0) / 4.0

# Examples:
# 1.0 (very negative) → 0.0
# 3.0 (neutral)       → 0.5
# 5.0 (very positive) → 1.0
```

## Expected Performance

| Metric | Expected Range |
|--------|----------------|
| **Categorical Accuracy** | 50-60% |
| **Weighted F1** | 0.50-0.60 |
| **Macro F1** | 0.40-0.50 |
| **Dimensional MAE** | 0.10-0.15 |
| **Dimensional Correlation** | 0.60-0.75 |

## Directory Structure

```
~/.cache/slower-whisper/benchmarks/iemocap/
├── Session1/
│   ├── sentences/
│   │   └── wav/
│   │       ├── Ses01F_impro01/
│   │       │   ├── Ses01F_impro01_F000.wav
│   │       │   └── ...
│   │       └── ...
│   └── dialog/
│       └── EmoEvaluation/
│           ├── Ses01F_impro01.txt
│           └── ...
├── Session2/
├── Session3/
├── Session4/
└── Session5/
```

## Results Location

```bash
benchmarks/results/iemocap_emotion_<timestamp>.json
```

## Full Documentation

- **Setup:** `docs/IEMOCAP_SETUP.md`
- **Label Mapping:** `docs/IEMOCAP_LABEL_MAPPING.md`
- **Integration Summary:** `IEMOCAP_INTEGRATION_SUMMARY.md`
- **Code:** `benchmarks/eval_emotion.py`
- **Iterator:** `transcription/benchmarks.py::iter_iemocap_clips()`

## Common Issues

| Issue | Solution |
|-------|----------|
| "IEMOCAP dataset not found" | Check path: `ls ~/.cache/slower-whisper/benchmarks/iemocap/` |
| "No emotion annotations found" | Verify `dialog/EmoEvaluation/*.txt` files exist |
| "Emotion dependencies not available" | Run `uv sync --extra emotion` |
| CUDA out of memory | Use `--limit` to reduce batch size or run on CPU |

## Citation

```bibtex
@article{busso2008iemocap,
  title={IEMOCAP: Interactive emotional dyadic motion capture database},
  author={Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and others},
  journal={Language resources and evaluation},
  volume={42},
  number={4},
  pages={335--359},
  year={2008}
}
```
