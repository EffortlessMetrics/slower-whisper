# Benchmark Evaluation Quickstart

This guide shows you how to use the new **Claude-driven benchmark evaluation system** to measure and improve slower-whisper's quality on standard datasets.

## What Was Built

You now have a complete infrastructure for systematic quality evaluation that extends your existing dogfood workflow:

### Core Infrastructure

1. **Extended cache system** (`transcription/cache.py`)
   - Added `benchmarks_root` field to `CachePaths`
   - Automatic directory creation under `~/.cache/slower-whisper/benchmarks`

2. **Dataset registry** (`transcription/benchmarks.py`)
   - `EvalSample` dataclass for benchmark samples
   - `iter_ami_meetings()` - AMI corpus iterator
   - `iter_iemocap_clips()` - IEMOCAP iterator
   - `list_available_benchmarks()` - dataset status checker

3. **Summary evaluation harness** (`benchmarks/eval_summaries.py`)
   - Full AMI summarization evaluation
   - Claude-as-judge scoring (faithfulness, coverage, clarity)
   - Automatic failure analysis with recommendations
   - Configurable (with/without audio cues, caching, etc.)

4. **Documentation**
   - `docs/AMI_SETUP.md` - Detailed AMI corpus setup guide
   - `benchmarks/EVALUATION.md` - Comprehensive evaluation guide

## The Claude-Driven Evaluation Loop

This system enables rapid iteration with Claude:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Run Evaluation                                          │
│     → python benchmarks/eval_summaries.py --dataset ami     │
│     → Get results JSON with scores + failure analysis       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Share Results with Claude                               │
│     → Paste aggregate scores and failure categories         │
│     → Ask: "What patterns do you see? How to improve?"      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Claude Suggests Improvements                            │
│     → Prompt changes (emphasize action items)               │
│     → Config tweaks (use compact mode for long meetings)    │
│     → Rendering adjustments (include more context)          │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Apply Changes                                           │
│     → Update eval_summaries.py or llm_utils.py              │
│     → Adjust rendering in render_conversation_for_llm()     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Re-run & Compare                                        │
│     → python benchmarks/eval_summaries.py --use-cache       │
│     → Compare: jq '.aggregate_scores' baseline.json         │
│     → Measure improvement                                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              └──────────┐
                                         │
                                      Repeat
```

## Quick Start (5 Minutes)

### Step 1: Verify Infrastructure

```bash
# Check that everything is installed
uv run python -c "
from transcription.benchmarks import list_available_benchmarks
import json
print(json.dumps(list_available_benchmarks(), indent=2))
"
```

**Expected output:**
```json
{
  "ami": {
    "path": "/home/user/.cache/slower-whisper/benchmarks/ami",
    "available": false,
    "setup_doc": "docs/AMI_SETUP.md",
    "description": "AMI Meeting Corpus for diarization and summarization evaluation",
    "tasks": ["diarization", "summarization", "action_items"]
  },
  ...
}
```

### Step 2: Set Up AMI (10-30 minutes)

See `docs/AMI_SETUP.md` for full instructions. Quick version:

```bash
# Set benchmarks location
export SLOWER_WHISPER_BENCHMARKS="${HOME}/.cache/slower-whisper/benchmarks"

# Create AMI directory structure
mkdir -p "${SLOWER_WHISPER_BENCHMARKS}/ami/audio"
mkdir -p "${SLOWER_WHISPER_BENCHMARKS}/ami/annotations"
mkdir -p "${SLOWER_WHISPER_BENCHMARKS}/ami/splits"

# Download 2-3 sample meetings for testing
# (See AMI_SETUP.md for URLs and instructions)

# Create test split file
cat > "${SLOWER_WHISPER_BENCHMARKS}/ami/splits/test.txt" <<EOF
ES2002a
ES2002b
EOF
```

### Step 3: Set API Key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# Get key from: https://console.anthropic.com/settings/keys
```

### Step 4: Run Your First Evaluation

```bash
# Quick smoke test (2 meetings)
uv run python benchmarks/eval_summaries.py --dataset ami --n 2

# With failure analysis
uv run python benchmarks/eval_summaries.py --dataset ami --n 2 --analyze-failures

# Save results
uv run python benchmarks/eval_summaries.py --dataset ami --n 2 \
  --analyze-failures --output results/first_run.json
```

**Example output:**
```
Evaluating AMI test split (2 meetings)
Model: claude-3-5-sonnet-20241022
Audio cues: True

[1/2] Processing ES2002a...
  → Transcribing with diarization...
  ✓ Transcription complete
  → Generating summary...
  ✓ Summary generated (245 chars)
  → Scoring against reference...
  ✓ Scores: faithfulness=8, coverage=7, clarity=9

[2/2] Processing ES2002b...
  ...

============================================================
SUMMARY
============================================================
Meetings evaluated: 2
Average faithfulness: 7.5/10
Average coverage: 6.5/10
Average clarity: 8.0/10

============================================================
FAILURE ANALYSIS
============================================================
Bad cases (1): ES2002b
Failure categories:
  missing_info: 1
  hallucination: 0
Recommendations:
  1. Include more context from earlier turns
  2. Emphasize action items in prompt

Results saved to: results/first_run.json
```

### Step 5: Iterate with Claude

```bash
# Extract key insights
jq '.aggregate_scores' results/first_run.json
jq '.failure_analysis' results/first_run.json

# Share with Claude:
# "Here are my evaluation results. Coverage is 6.5/10.
#  Failure analysis shows missing_info issue.
#  Recommendation: include more context.
#  How should I adjust the rendering?"

# Claude suggests: "Try increasing context in render_conversation_for_llm()..."

# Apply changes, then re-run (using cached transcripts)
uv run python benchmarks/eval_summaries.py --dataset ami --n 2 \
  --use-cache --output results/iteration_2.json

# Compare
jq -s '{before: .[0].aggregate_scores, after: .[1].aggregate_scores}' \
  results/first_run.json results/iteration_2.json
```

## Common Workflows

### Workflow 1: Compare Audio Cues Impact

```bash
# With audio cues (default)
uv run python benchmarks/eval_summaries.py --dataset ami --n 5 \
  --output results/with_cues.json

# Without audio cues
uv run python benchmarks/eval_summaries.py --dataset ami --n 5 \
  --no-audio-cues --use-cache \
  --output results/without_cues.json

# Compare
echo "With cues:"
jq '.aggregate_scores' results/with_cues.json
echo ""
echo "Without cues:"
jq '.aggregate_scores' results/without_cues.json
```

### Workflow 2: Iterate on Prompts

1. **Baseline run:**
   ```bash
   uv run python benchmarks/eval_summaries.py --dataset ami --n 5 \
     --output results/baseline.json
   ```

2. **Modify prompt** in `benchmarks/eval_summaries.py`:
   ```python
   # Change the generate_summary() function prompt
   prompt = f"""You are analyzing a meeting transcript...

   Focus especially on:
   - Decisions made
   - Action items with owners
   - Key discussion points

   Transcript:
   {context}
   ..."""
   ```

3. **Re-run with cache:**
   ```bash
   uv run python benchmarks/eval_summaries.py --dataset ami --n 5 \
     --use-cache --output results/improved_prompt.json
   ```

4. **Compare:**
   ```bash
   jq -s '{baseline: .[0].aggregate_scores, improved: .[1].aggregate_scores}' \
     results/baseline.json results/improved_prompt.json
   ```

### Workflow 3: Full Test Set Evaluation

Once your pipeline is stable:

```bash
# Run full test set (typically 16 meetings)
uv run python benchmarks/eval_summaries.py --dataset ami --split test \
  --analyze-failures --output results/ami_test_full.json

# Review comprehensive report
cat results/ami_test_full.json | less

# Extract key metrics
jq '.aggregate_scores, .failure_analysis' results/ami_test_full.json
```

## What Each Component Does

### `transcription/cache.py`
- **What changed:** Added `samples_root` and `benchmarks_root` fields
- **Why:** Centralized cache management for all datasets
- **Impact:** Consistent paths across dogfood and benchmark workflows

### `transcription/benchmarks.py`
- **What it is:** Dataset registry and iterators
- **Key functions:**
  - `get_benchmarks_root()` - Get benchmark cache directory
  - `iter_ami_meetings()` - Iterate AMI corpus samples
  - `list_available_benchmarks()` - Check dataset availability
- **Why:** Reusable dataset loading for all evaluation harnesses

### `benchmarks/eval_summaries.py`
- **What it is:** Complete AMI summary evaluation harness
- **What it does:**
  1. Loads AMI meetings with reference summaries
  2. Transcribes with diarization (or uses cache)
  3. Renders with `render_conversation_for_llm()`
  4. Generates summaries via Claude
  5. Scores via Claude-as-judge
  6. Categorizes failures via Claude diagnostician
- **Output:** Structured JSON with scores and recommendations

## Integration with Existing Tools

### Works With Dogfood Workflow

```bash
# Dogfood workflow (synthetic samples, quick iteration)
uv run slower-whisper-dogfood --sample synthetic

# Benchmark workflow (real datasets, comprehensive evaluation)
uv run python benchmarks/eval_summaries.py --dataset ami --n 5
```

### Uses Existing LLM Utilities

The evaluation harness uses your existing `llm_utils.py`:

```python
from transcription.llm_utils import render_conversation_for_llm

# Same rendering logic used in examples/llm_integration/
context = render_conversation_for_llm(
    transcript,
    mode="turns",
    include_audio_cues=True,
    speaker_labels={"spk_0": "PM", "spk_1": "UI"}
)
```

### Compatible with Existing CLI

```bash
# Transcribe for evaluation
uv run slower-whisper transcribe --enable-diarization

# Load in evaluation harness
transcript = load_transcript("whisper_json/meeting.json")
```

## Next Steps

### Immediate (v1.1.x)

1. **Set up AMI** (see `docs/AMI_SETUP.md`)
2. **Run baseline evaluation** (5-10 meetings)
3. **Iterate with Claude** on prompts and rendering
4. **Track results** in version control

### Near-term (v1.2)

1. **Add diarization eval** (`eval_diarization.py`)
   - DER calculation
   - Speaker count accuracy
2. **Add emotion eval** (`eval_emotion.py`)
   - IEMOCAP accuracy
   - Confusion matrix
3. **Set up IEMOCAP** (see `docs/IEMOCAP_SETUP.md` when ready)

### Long-term (v1.3+)

1. **More LLM tasks** (action items, QA, conflict detection)
2. **Automated regression testing**
3. **Cost tracking** (API usage per eval)
4. **Comparison reports** (automated delta analysis)

## Documentation Index

- **This doc** - Quickstart guide
- `docs/AMI_SETUP.md` - AMI corpus setup
- `benchmarks/EVALUATION.md` - Comprehensive evaluation guide
- `docs/LLM_PROMPT_PATTERNS.md` - Prompt engineering patterns
- `transcription/llm_utils.py` - LLM rendering API reference
- `DOGFOOD_SETUP.md` - Dogfood workflow (complementary)

## Getting Help

### Check Infrastructure

```bash
# Verify benchmarks available
uv run python -c "from transcription.benchmarks import list_available_benchmarks; print(list_available_benchmarks())"

# Check API key
echo $ANTHROPIC_API_KEY

# Test eval script
uv run python benchmarks/eval_summaries.py --help
```

### Common Issues

**"AMI Meeting Corpus not found"**
- See `docs/AMI_SETUP.md`
- Check `~/.cache/slower-whisper/benchmarks/ami/`

**"ANTHROPIC_API_KEY not set"**
- Get key: https://console.anthropic.com/settings/keys
- Set: `export ANTHROPIC_API_KEY=sk-ant-...`

**"No results collected"**
- Ensure AMI meetings have reference summaries
- Check transcription logs for errors
- Try single meeting first

### Advanced Usage

See `benchmarks/EVALUATION.md` for:
- Detailed CLI options
- Result format specification
- Comparison workflows
- Troubleshooting guide

---

**You're now ready to run Claude-driven evaluation!**

Start with a small test (2-5 meetings), iterate with Claude, and scale up once your pipeline is tuned.
