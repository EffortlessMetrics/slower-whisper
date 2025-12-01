# Quality Evaluation Harnesses

This document describes the **quality evaluation** harnesses for measuring slower-whisper's accuracy on benchmark datasets. For **performance benchmarking** (throughput, memory usage), see [README.md](README.md).

## Overview

**Purpose:** Systematic evaluation of:
- **Diarization quality** (DER on AMI/LibriCSS)
- **Emotion recognition** (accuracy on IEMOCAP)
- **LLM-based tasks** (summarization, action items, QA)

**Philosophy:** Use Claude as both:
1. **Task performer** (generate summaries from transcripts)
2. **Judge** (score outputs against references)
3. **Diagnostician** (categorize failure modes)

This enables rapid iteration on prompts, rendering strategies, and pipeline configurations.

## Quick Start

### 1. Set up benchmark datasets

Benchmark datasets require manual download due to licensing:

```bash
# AMI Meeting Corpus (for diarization + summarization)
# See docs/AMI_SETUP.md for detailed instructions
export SLOWER_WHISPER_BENCHMARKS="${HOME}/.cache/slower-whisper/benchmarks"
mkdir -p "${SLOWER_WHISPER_BENCHMARKS}/ami"

# IEMOCAP (for emotion recognition)
# See docs/IEMOCAP_SETUP.md
mkdir -p "${SLOWER_WHISPER_BENCHMARKS}/iemocap"
```

**Check availability:**

```bash
uv run python -c "
from transcription.benchmarks import list_available_benchmarks
import json
print(json.dumps(list_available_benchmarks(), indent=2))
"
```

### 2. Run summary evaluation

**Prerequisites:**
- AMI dataset staged (see `docs/AMI_SETUP.md`)
- `ANTHROPIC_API_KEY` set
- `uv sync --extra full` (for diarization)

**Basic usage:**

```bash
# Evaluate 10 test meetings
uv run python benchmarks/eval_summaries.py --dataset ami --split test --n 10

# Use existing transcripts (skip re-transcription)
uv run python benchmarks/eval_summaries.py --dataset ami --use-cache --n 10

# Analyze failure patterns
uv run python benchmarks/eval_summaries.py --dataset ami --n 10 --analyze-failures
```

**Compare configurations:**

```bash
# With audio cues (default)
uv run python benchmarks/eval_summaries.py --dataset ami --n 5 \
  --output results/ami_with_cues.json

# Without audio cues
uv run python benchmarks/eval_summaries.py --dataset ami --n 5 \
  --no-audio-cues --output results/ami_no_cues.json

# Compare results
jq '.aggregate_scores' results/ami_with_cues.json
jq '.aggregate_scores' results/ami_no_cues.json
```

### 3. Iterating with Claude

The evaluation harness is designed for **Claude-driven iteration**:

**Pattern:**

1. **Run evaluation** → Get results JSON with scores + failure analysis
2. **Share results with Claude** → Ask for diagnostic insights
3. **Claude suggests improvements** → Prompt changes, rendering tweaks, config adjustments
4. **Apply changes** → Update code/config
5. **Re-run evaluation** → Measure improvement
6. **Repeat**

**Example workflow:**

```bash
# Initial run
uv run python benchmarks/eval_summaries.py --dataset ami --n 10 \
  --analyze-failures --output results/baseline.json

# Review results
cat results/baseline.json

# Share with Claude (via claude.ai or API):
# "Here are my AMI summary evaluation results. What patterns do you see?
#  Failure analysis shows 60% missing_info, 30% hallucination.
#  Recommendations: [...]
#  What prompt/config changes would address this?"

# Claude responds with specific suggestions
# Apply changes to eval_summaries.py or llm_utils.py

# Re-run
uv run python benchmarks/eval_summaries.py --dataset ami --n 10 \
  --analyze-failures --output results/iteration_2.json

# Compare
jq -s '.[0].aggregate_scores, .[1].aggregate_scores' \
  results/baseline.json results/iteration_2.json
```

## Available Harnesses

### ASR WER (`eval_asr.py`)

**Status:** ✅ Tiny internal set (3 clips, CPU-friendly profiles)

**What it measures:** Word Error Rate on a small manifest you provide.

**Recent result:** On 3 short TTS clips (narrowband + meeting-style) with `call_center`/`meeting` profiles (CPU, int8), avg WER ≈ **1.8%**. See `benchmarks/ASR_REPORT.md`/`.json`.

**Workflow:**
1. Populate or extend `benchmarks/data/asr/manifest.jsonl` with `audio_path` and `reference_text` (repo includes 3 seeded clips under `benchmarks/data/asr/audio/`).
2. Pick a profile (`call_center`, `meeting`, `podcast`) or use defaults.
3. Run `PYTHONPATH=. python benchmarks/eval_asr.py --output-md benchmarks/ASR_REPORT.md --output-json benchmarks/ASR_REPORT.json`.

**Output:** Markdown + JSON with per-file WER and runtime.

**Dependencies:** `jiwer` for WER (`uv pip install jiwer`).

### Summary Evaluation (`eval_summaries.py`)

**Status:** ✅ Implemented (v1.1)

**What it measures:**
- Faithfulness (no hallucinations)
- Coverage (captures key points)
- Clarity (well-structured)

**Workflow:**
1. Load AMI meetings with reference summaries
2. Transcribe with diarization (or use cache)
3. Render transcript with `render_conversation_for_llm()`
4. Generate candidate summary (Claude)
5. Score vs reference (Claude-as-judge)
6. Categorize failures (Claude diagnostician)

**Output:** `benchmarks/results/ami_summaries_<date>.json`

**Key metrics:**
- Average faithfulness/coverage/clarity (0-10 scale)
- Per-meeting breakdown
- Failure categories (missing_info, hallucination, wrong_speaker)
- Improvement recommendations

**Usage:**

```bash
# Full evaluation with analysis
uv run python benchmarks/eval_summaries.py --dataset ami --split test \
  --n 10 --analyze-failures

# Quick smoke test (2 meetings, no analysis)
uv run python benchmarks/eval_summaries.py --dataset ami --n 2

# Use different Claude model
uv run python benchmarks/eval_summaries.py --dataset ami --n 5 \
  --model claude-opus-4-20250514
```

### Diarization Evaluation (`eval_diarization.py`)

**Status:** ✅ Implemented (tiny synthetic fixtures; pyannote-backed run pending HF_TOKEN)

**What it measures:**
- Diarization Error Rate (DER)
- Speaker count accuracy

**Recent result:** Stub backend on 3 synthetic clips: avg DER ≈ 0.67; speaker counts 3/3 (see `benchmarks/DIARIZATION_REPORT.md`).

**Workflow:**
1. Ensure dataset at `benchmarks/data/diarization` (fixtures already included).
2. Run with pyannote once HF_TOKEN is available:
   ```bash
   uv sync --extra diarization
   HF_TOKEN=... SLOWER_WHISPER_PYANNOTE_MODE=auto PYTHONPATH=. \
     python benchmarks/eval_diarization.py \
       --dataset benchmarks/data/diarization \
       --output-md benchmarks/DIARIZATION_REPORT.md \
       --output-json benchmarks/DIARIZATION_REPORT.json \
       --overwrite
   ```
3. Reports land in Markdown/JSON for DER + speaker-count accuracy.

### Emotion Evaluation (`eval_emotion.py`)

**Status:** 📋 TODO (v1.2)

**What it will measure:**
- Emotion classification accuracy
- Confusion matrix (angry/happy/sad/neutral)
- Per-emotion F1 scores

**Planned workflow:**
1. Load IEMOCAP clips with emotion labels
2. Transcribe + enrich with emotion features
3. Map `audio_state.emotion` to categorical label
4. Compute accuracy/F1 vs ground truth

## Result Format

All evaluation harnesses save results as structured JSON for easy analysis:

```json
{
  "metadata": {
    "timestamp": "2025-11-19T10:30:00Z",
    "dataset": "ami",
    "split": "test",
    "n_meetings": 10,
    "model": "claude-3-5-sonnet-20241022",
    "audio_cues_used": true
  },
  "aggregate_scores": {
    "avg_faithfulness": 7.8,
    "avg_coverage": 7.2,
    "avg_clarity": 8.5
  },
  "results": [
    {
      "meeting_id": "ES2002a",
      "reference_summary": "...",
      "candidate_summary": "...",
      "score": {
        "faithfulness": 8,
        "coverage": 7,
        "clarity": 9,
        "comments": "Good coverage but missed one action item"
      },
      "transcript_path": "whisper_json/ES2002a.json"
    }
  ],
  "failure_analysis": {
    "bad_cases": ["ES2003b", "ES2005c"],
    "categories": {
      "missing_info": 3,
      "hallucination": 1,
      "wrong_speaker": 0,
      "style_only": 2
    },
    "recommendations": [
      "Include more context from earlier turns",
      "Emphasize action items in prompt",
      "Use compact mode for longer meetings"
    ]
  }
}
```

## Using Results with Claude

### Pattern 1: Direct results sharing

Copy key sections from results JSON and paste to Claude:

```bash
# Extract aggregate scores
jq '.aggregate_scores' results/ami_summaries_20251119.json

# Extract failure analysis
jq '.failure_analysis' results/ami_summaries_20251119.json

# Show bad cases
jq '.results[] | select(.score.faithfulness < 6 or .score.coverage < 6)' \
  results/ami_summaries_20251119.json
```

### Pattern 2: Automated analysis script

Create a helper to format results for Claude:

```python
# scripts/format_results_for_claude.py
import json
import sys

results = json.load(open(sys.argv[1]))

print("=== AMI Summary Evaluation Results ===\n")
print(f"Dataset: {results['metadata']['dataset']}")
print(f"Meetings: {results['metadata']['n_meetings']}")
print(f"Model: {results['metadata']['model']}")
print(f"Audio cues: {results['metadata']['audio_cues_used']}")
print()

scores = results['aggregate_scores']
print(f"Average Scores:")
print(f"  Faithfulness: {scores['avg_faithfulness']:.1f}/10")
print(f"  Coverage: {scores['avg_coverage']:.1f}/10")
print(f"  Clarity: {scores['avg_clarity']:.1f}/10")
print()

if 'failure_analysis' in results:
    fa = results['failure_analysis']
    print(f"Failure Analysis:")
    print(f"  Bad cases: {len(fa['bad_cases'])}")
    print(f"  Categories: {fa['categories']}")
    print(f"  Recommendations:")
    for rec in fa['recommendations']:
        print(f"    - {rec}")
```

Usage:

```bash
uv run python scripts/format_results_for_claude.py \
  results/ami_summaries_20251119.json | pbcopy

# Now paste into Claude chat
```

### Pattern 3: Comparison reports

```bash
# Compare two configurations
jq -s '{
  baseline: .[0].aggregate_scores,
  improved: .[1].aggregate_scores,
  delta: {
    faithfulness: (.[1].aggregate_scores.avg_faithfulness - .[0].aggregate_scores.avg_faithfulness),
    coverage: (.[1].aggregate_scores.avg_coverage - .[0].aggregate_scores.avg_coverage),
    clarity: (.[1].aggregate_scores.avg_clarity - .[0].aggregate_scores.avg_clarity)
  }
}' results/baseline.json results/improved.json
```

## Best Practices

### Start Small

Don't run full corpus on first iteration:

```bash
# Initial smoke test: 2 meetings
uv run python benchmarks/eval_summaries.py --dataset ami --n 2

# Quick validation: 5 meetings
uv run python benchmarks/eval_summaries.py --dataset ami --n 5

# Full test set: 16 meetings (once pipeline is stable)
uv run python benchmarks/eval_summaries.py --dataset ami --split test
```

### Use Cache Aggressively

Transcription is slow; re-scoring is fast:

```bash
# First run: transcribe + score
uv run python benchmarks/eval_summaries.py --dataset ami --n 5

# After prompt changes: skip transcription
uv run python benchmarks/eval_summaries.py --dataset ami --n 5 --use-cache
```

### Track Experiments

Use descriptive output names:

```bash
uv run python benchmarks/eval_summaries.py --dataset ami --n 10 \
  --output results/ami_baseline_with_audio_cues.json

uv run python benchmarks/eval_summaries.py --dataset ami --n 10 \
  --no-audio-cues --use-cache \
  --output results/ami_no_audio_cues.json

uv run python benchmarks/eval_summaries.py --dataset ami --n 10 \
  --use-cache \
  --output results/ami_compact_mode.json
# (after changing render mode to compact in code)
```

### Version Control Results

Commit important results to track progress:

```bash
git add benchmarks/results/ami_summaries_v1.1_baseline.json
git commit -m "eval: AMI summary baseline (avg coverage 7.2/10)"
```

## Troubleshooting

**"AMI Meeting Corpus not found"**

Solution:
```bash
# Check benchmarks location
uv run python -c "
from transcription.benchmarks import get_benchmarks_root
print(get_benchmarks_root())
"

# Ensure AMI is staged there
ls ~/.cache/slower-whisper/benchmarks/ami/
```

See `docs/AMI_SETUP.md` for detailed setup.

**"ANTHROPIC_API_KEY not set"**

Solution:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
# Or add to ~/.bashrc or ~/.zshrc
```

**"No results collected"**

Possible causes:
- AMI meetings don't have reference summaries
  - Use `require_summary=True` in `iter_ami_meetings()` or stage annotations
- Transcription failures
  - Check logs for errors
  - Try single meeting: `slower-whisper transcribe --file <audio> --enable-diarization`

**Claude rate limits**

If evaluating many meetings:
- Use `--n` to limit count
- Add delays between API calls (TODO: implement in harness)
- Use tier 2+ API key for higher rate limits

## Related Documentation

- `docs/AMI_SETUP.md` - AMI corpus setup guide
- `docs/IEMOCAP_SETUP.md` - IEMOCAP setup (TODO)
- `docs/LLM_PROMPT_PATTERNS.md` - Prompt engineering guide
- [DOGFOOD_SETUP.md](../docs/DOGFOOD_SETUP.md) - Dogfooding workflow (synthetic samples)
- `transcription/benchmarks.py` - Dataset iterator API
- `transcription/llm_utils.py` - LLM rendering utilities
- `README.md` - Performance benchmarking (throughput, memory)
