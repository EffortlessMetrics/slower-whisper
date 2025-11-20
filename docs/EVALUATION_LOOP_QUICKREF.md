# Evaluation Loop Quick Reference

This is a quick reference card for the **Claude-driven evaluation loop** workflow. For detailed documentation, see [BENCHMARK_EVALUATION_QUICKSTART.md](BENCHMARK_EVALUATION_QUICKSTART.md).

---

## 🎯 The Three-Layer Approach

### Layer 0: Sanity-Check
**Goal:** Verify the harness works on a tiny sample

```bash
# Stage 2-3 AMI meetings under benchmarks_root/ami
# (see docs/AMI_SETUP.md for structure)

# Run minimal test
uv run python benchmarks/eval_summaries.py --dataset ami --n 2

# Verify: results JSON produced, no crashes, scores look sane
```

### Layer 1: Learn Something
**Goal:** Establish baseline → iterate with Claude → measure improvement

```bash
# 1. Establish baseline (10 meetings)
uv run python benchmarks/eval_summaries.py \
  --dataset ami --n 10 \
  --output benchmarks/results/ami_baseline.json \
  --analyze-failures

# 2. Review results
jq '.aggregate_scores, .failure_analysis' benchmarks/results/ami_baseline.json

# 3. Share with Claude + get suggestions
#    "Coverage is 6.8, missing_info is 60%. How to improve?"

# 4. Apply changes (e.g., prompt tweaks in ClaudeSummarizer)

# 5. Re-run with cache (fast!)
uv run python benchmarks/eval_summaries.py \
  --dataset ami --n 10 \
  --use-cache \
  --output benchmarks/results/ami_prompt_v2.json

# 6. Compare
uv run python benchmarks/compare_results.py \
  --before benchmarks/results/ami_baseline.json \
  --after benchmarks/results/ami_prompt_v2.json \
  --show-failures

# 7. Log results in EVALUATION_LOG.md
```

### Layer 2: Make It Repeatable
**Goal:** Formalize experiments, track over time

- Use EVALUATION_LOG.md template for each experiment
- Git-commit important baselines: `benchmarks/results/ami_baseline_v1.1.0.json`
- Track code changes alongside result files
- Build muscle memory for the loop

---

## 🔧 Common Commands

### Setup

```bash
# Check benchmarks directory
uv run python -c "from transcription.benchmarks import get_benchmarks_root; print(get_benchmarks_root())"

# List available datasets
uv run python -c "from transcription.benchmarks import list_available_benchmarks; import json; print(json.dumps(list_available_benchmarks(), indent=2))"

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...
export HF_TOKEN=hf_...  # For pyannote models
```

### Running Evaluations

```bash
# Basic run
uv run python benchmarks/eval_summaries.py --dataset ami --n 5

# With failure analysis (recommended)
uv run python benchmarks/eval_summaries.py \
  --dataset ami --n 10 \
  --analyze-failures

# Save to specific file
uv run python benchmarks/eval_summaries.py \
  --dataset ami --n 10 \
  --output benchmarks/results/my_experiment.json

# Use cached transcripts (re-score only, fast)
uv run python benchmarks/eval_summaries.py \
  --dataset ami --n 10 \
  --use-cache

# Disable audio cues (A/B test)
uv run python benchmarks/eval_summaries.py \
  --dataset ami --n 10 \
  --no-audio-cues \
  --use-cache \
  --output benchmarks/results/no_cues.json
```

### Comparing Results

```bash
# Basic comparison
uv run python benchmarks/compare_results.py \
  --before benchmarks/results/baseline.json \
  --after benchmarks/results/improved.json

# With failure details
uv run python benchmarks/compare_results.py \
  --before benchmarks/results/baseline.json \
  --after benchmarks/results/improved.json \
  --show-failures
```

### Inspecting Results

```bash
# Quick scores
jq '.aggregate_scores' benchmarks/results/my_experiment.json

# Failure analysis
jq '.failure_analysis' benchmarks/results/my_experiment.json

# Sample-level details
jq '.results[] | select(.meeting_id == "ES2002a")' benchmarks/results/my_experiment.json

# Find worst cases
jq '.results | sort_by(.scores.avg_score) | .[0:3] | .[] | {meeting_id, avg_score: .scores.avg_score}' \
  benchmarks/results/my_experiment.json
```

---

## 📝 Typical Iteration Cycle

### First Run (Establish Baseline)

1. Stage a small AMI subset (5-10 meetings)
2. Run evaluation with `--analyze-failures`
3. Save as `ami_baseline.json`
4. Review aggregate scores and failure categories
5. Log in EVALUATION_LOG.md

**Time:** ~10-15 minutes (includes transcription)

### Subsequent Runs (Test Changes)

1. Make a change (prompt, config, code)
2. Run with `--use-cache` (transcription already done)
3. Save as `ami_prompt_v2.json` (or similar)
4. Run `compare_results.py`
5. Log in EVALUATION_LOG.md

**Time:** ~2-5 minutes (scoring only, no transcription)

### Claude Consultation

**What to share:**
- Aggregate scores (before/after)
- Failure category breakdown
- 2-3 worst-case examples (meeting ID + reference + candidate + judge comments)
- Your hypothesis about what's wrong

**What to ask:**
- "Coverage is low but faithfulness is high. How to improve coverage without hallucinating?"
- "missing_info category is 60%. What context am I likely omitting?"
- "Compare these two summaries from the same meeting. Which better captures the key points?"

**What you get back:**
- Specific prompt tweaks
- Context construction suggestions
- Hypotheses to test

---

## 🎓 Tips and Best Practices

### Start Small
- Use 2-3 meetings for Layer 0 sanity checks
- Use 5-10 for Layer 1 baseline and iteration
- Only scale to full test set (50+) when you're confident in the harness

### Use --use-cache Aggressively
- Transcription is slow (~1-2 min/meeting with diarization)
- Scoring is fast (~5-10 sec/meeting)
- Cache lets you iterate on prompts/scoring without re-transcribing

### Git-Commit Baselines
- When you establish a good baseline, commit the result file
- Tag important milestones: `git tag ami-baseline-v1.1.0`
- Makes it easy to compare against historical performance

### Log Everything
- Use EVALUATION_LOG.md religiously
- Include: baseline, changes, results, interpretation, next steps
- Your future self will thank you

### One Change at a Time
- Don't tweak prompts AND config AND code in one iteration
- Isolate variables so you know what actually helped

### Compare Apples to Apples
- Same n_meetings for before/after comparisons
- Same dataset split (don't compare train vs test)
- Same model version (unless testing model upgrades)

### Let Claude Surprise You
- Don't just ask "how to improve coverage"
- Share failure cases and ask "what patterns do you notice?"
- Claude might spot things you wouldn't (e.g., "all failures are long meetings")

---

## 🔍 Troubleshooting

### "No such dataset: ami"
- Check `list_available_benchmarks()` output
- Verify directory structure under `benchmarks_root/ami`
- See docs/AMI_SETUP.md for expected layout

### "ANTHROPIC_API_KEY not set"
- `export ANTHROPIC_API_KEY=sk-ant-...`
- Or add to `.env` file in repo root

### Evaluation crashes mid-run
- Check individual meeting processing: add `--n 1` to isolate
- Check logs for specific error (missing audio, bad reference, etc.)
- Use `--use-cache` to resume without re-transcribing

### Scores seem wrong/random
- Verify reference summaries are actually loading (`EvalSample.reference_summary`)
- Check that Claude judge prompt is appropriate for your data
- Try `--n 2` with meetings you know well to sanity-check

### Too slow
- Use `--use-cache` after first run
- Use smaller `--n` for iteration
- Consider running on GPU-enabled machine for transcription

---

## 📚 Related Documentation

- **Full guide:** [docs/BENCHMARK_EVALUATION_QUICKSTART.md](BENCHMARK_EVALUATION_QUICKSTART.md)
- **AMI setup:** [docs/AMI_SETUP.md](AMI_SETUP.md)
- **Evaluation details:** [benchmarks/EVALUATION.md](../benchmarks/EVALUATION.md)
- **Dataset API:** [transcription/benchmarks.py](../transcription/benchmarks.py)
- **Experiment log:** [EVALUATION_LOG.md](../EVALUATION_LOG.md)

---

## 🎯 Quick Win Checklist

Ready to start? Here's the fastest path to your first iteration:

- [ ] Stage 2 AMI meetings (docs/AMI_SETUP.md)
- [ ] Run: `uv run python benchmarks/eval_summaries.py --dataset ami --n 2`
- [ ] Verify results JSON looks sane
- [ ] Expand to 5-10 meetings for baseline
- [ ] Save as `ami_baseline.json`
- [ ] Review scores with `jq '.aggregate_scores' ami_baseline.json`
- [ ] Share with Claude: "Here are my scores and failures. What do you notice?"
- [ ] Apply one suggested change
- [ ] Re-run with `--use-cache`
- [ ] Compare: `compare_results.py --before baseline --after improved`
- [ ] Log results in EVALUATION_LOG.md

**Time to first iteration:** ~30-45 minutes including AMI setup

🎉 You now have a working evaluation loop!
