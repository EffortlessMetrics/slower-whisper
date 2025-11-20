# Getting Started with Evaluation (5 Steps)

This is a **5-step quickstart** to get your first Claude-driven evaluation running in under 30 minutes.

---

## Prerequisites

- ✅ You've run `uv sync --extra full --extra diarization`
- ✅ You have `ANTHROPIC_API_KEY` set (for Claude-as-judge)
- ✅ You have `HF_TOKEN` set (for pyannote diarization models)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export HF_TOKEN=hf_...
```

---

## Step 1: Get a Tiny AMI Sample (10 minutes)

You don't need the full AMI corpus. Just grab 2-3 sample meetings to test the harness.

### Option A: Use Public AMI Samples

1. Download 2-3 meetings from the [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
   - Example: ES2002a, ES2002b (scenario meetings, ~15 minutes each)
   - Download the Mix-Headset audio (single-channel WAV)

2. Create directory structure:
   ```bash
   BENCH_ROOT=$(uv run python -c "from transcription.benchmarks import get_benchmarks_root; print(get_benchmarks_root())")
   mkdir -p "$BENCH_ROOT/ami/audio"
   mkdir -p "$BENCH_ROOT/ami/annotations"
   ```

3. Copy audio files:
   ```bash
   cp /path/to/downloaded/ES2002a.Mix-Headset.wav "$BENCH_ROOT/ami/audio/ES2002a.wav"
   cp /path/to/downloaded/ES2002b.Mix-Headset.wav "$BENCH_ROOT/ami/audio/ES2002b.wav"
   ```

4. Create minimal annotation files:
   ```bash
   # ES2002a.json
   cat > "$BENCH_ROOT/ami/annotations/ES2002a.json" << 'EOF'
   {
     "summary": "The project team held their first meeting to discuss designing a new television remote control. The project manager outlined the budget constraints and project timeline. The team agreed to focus on a simple, user-friendly design targeting a young demographic. Action items: industrial designer to research components, UI designer to create initial mockups, marketing to analyze competitor products."
   }
   EOF

   # ES2002b.json
   cat > "$BENCH_ROOT/ami/annotations/ES2002b.json" << 'EOF'
   {
     "summary": "The team presented their initial designs for the remote control. The industrial designer proposed using kinetic energy for power. The UI designer presented a minimalist button layout. The marketing expert emphasized the need for a trendy appearance to appeal to young consumers. The team discussed tradeoffs between advanced features and keeping costs low. They decided to move forward with prototyping."
   }
   EOF
   ```

### Option B: Use Synthetic/Placeholder Data (Fastest)

If you just want to test the machinery, create dummy files:

```bash
BENCH_ROOT=$(uv run python -c "from transcription.benchmarks import get_benchmarks_root; print(get_benchmarks_root())")
mkdir -p "$BENCH_ROOT/ami/audio"
mkdir -p "$BENCH_ROOT/ami/annotations"

# Copy an existing audio file you have (or use your synthetic sample)
cp /path/to/any/audio.wav "$BENCH_ROOT/ami/audio/TEST001.wav"

# Create annotation
cat > "$BENCH_ROOT/ami/annotations/TEST001.json" << 'EOF'
{
  "summary": "A brief meeting discussing project planning and next steps."
}
EOF
```

---

## Step 2: Verify Setup (2 minutes)

Run the verification script:

```bash
./verify_ami_setup.sh
```

Expected output:
```
✓ Checking directories...
  Audio files:      2
  Annotation files: 2

✓ Testing dataset iteration...
  Found 2 samples:
    ES2002a: summary=✓
    ES2002b: summary=✓

✓ AMI setup looks good!
  Ready to run: uv run python benchmarks/eval_summaries.py --dataset ami --n 2
```

If this fails, see [docs/AMI_DIRECTORY_LAYOUT.md](docs/AMI_DIRECTORY_LAYOUT.md) for troubleshooting.

---

## Step 3: Run Your First Evaluation (5 minutes)

```bash
uv run python benchmarks/eval_summaries.py \
  --dataset ami \
  --n 2 \
  --output benchmarks/results/first_run.json
```

This will:
1. Transcribe the 2 audio files (with diarization)
2. Render conversation context for LLM
3. Use Claude to generate summaries
4. Use Claude to judge quality (faithfulness, coverage, clarity)
5. Save structured results to JSON

**Time:** ~2-5 minutes (depends on audio length and GPU availability)

---

## Step 4: Review Results (3 minutes)

```bash
# Quick scores
jq '.aggregate_scores' benchmarks/results/first_run.json

# Sample output:
# {
#   "avg_faithfulness": 7.5,
#   "avg_coverage": 6.8,
#   "avg_clarity": 8.2
# }

# Per-meeting details
jq '.results[] | {id: .meeting_id, scores: .scores.avg_score, comment: .judge_outputs.overall_comment}' \
  benchmarks/results/first_run.json

# Full result for one meeting
jq '.results[] | select(.meeting_id == "ES2002a")' benchmarks/results/first_run.json
```

**Sanity checks:**
- ✅ Scores are between 0-10 (not NaN or negative)
- ✅ `generated_summary` is non-empty and looks relevant
- ✅ `judge_outputs.overall_comment` provides meaningful feedback

If anything looks wrong, check:
- Reference summaries loaded correctly (`reference_summary` field)
- Transcription succeeded (`transcript_path` exists)
- API key is valid (check for errors in output)

---

## Step 5: Iterate (10 minutes)

Now try making a change and measuring the impact:

### Example: Test Impact of Audio Cues

```bash
# Baseline (with audio cues)
uv run python benchmarks/eval_summaries.py \
  --dataset ami --n 2 \
  --output benchmarks/results/with_cues.json

# Without audio cues (use cache to skip transcription)
uv run python benchmarks/eval_summaries.py \
  --dataset ami --n 2 \
  --no-audio-cues \
  --use-cache \
  --output benchmarks/results/no_cues.json

# Compare
uv run python benchmarks/compare_results.py \
  --before benchmarks/results/no_cues.json \
  --after benchmarks/results/with_cues.json
```

Expected output:
```
======================================================================
AGGREGATE SCORES COMPARISON
======================================================================

Metric               Before      After      Delta Change
----------------------------------------------------------------------
avg_clarity            8.20       8.20      +0.00 ~
avg_coverage           6.50       7.20      +0.70 ↑ BETTER
avg_faithfulness       7.50       7.40      -0.10 ~

----------------------------------------------------------------------
Average delta: +0.20
Overall assessment: Slight improvement
```

This tells you: **audio cues improved coverage by 0.7 points with no faithfulness loss**.

---

## What You've Accomplished

🎉 Congratulations! You now have:

- ✅ Working evaluation harness
- ✅ First baseline results
- ✅ Proof that the Claude-as-judge loop works
- ✅ Evidence that a change (audio cues) made a measurable difference

---

## Next Steps

### Immediate (Next Session)

1. **Expand to 5-10 meetings** for a more robust baseline
   ```bash
   # Add more AMI samples to your benchmarks_root/ami directory
   # Then run:
   uv run python benchmarks/eval_summaries.py \
     --dataset ami --n 10 \
     --output benchmarks/results/ami_baseline_v1.json \
     --analyze-failures
   ```

2. **Share results with Claude** for analysis
   ```bash
   # Extract key findings
   jq '{scores: .aggregate_scores, failures: .failure_analysis}' \
     benchmarks/results/ami_baseline_v1.json > /tmp/summary_for_claude.json

   # Share with Claude:
   # "Here are my AMI summary evaluation results. What patterns do you see?
   #  How can I improve coverage without sacrificing faithfulness?"
   ```

3. **Apply suggested changes** and re-run
   - Tweak prompts in `benchmarks/eval_summaries.py` (ClaudeSummarizer class)
   - Or adjust context construction in `render_conversation_for_llm()`
   - Re-run with `--use-cache` to skip re-transcription

4. **Log your experiments** in `EVALUATION_LOG.md`
   - Use the template provided
   - Track what changed, what improved, what you learned

### Short-term (This Week)

- Read [docs/EVALUATION_LOOP_QUICKREF.md](docs/EVALUATION_LOOP_QUICKREF.md) for workflow tips
- Get comfortable with the iteration loop
- Try 3-5 experiments to build muscle memory
- Find a baseline you're happy with

### Medium-term (v1.1)

- Implement `benchmarks/eval_diarization.py` for DER measurement
- Implement `benchmarks/eval_emotion.py` for emotion accuracy
- Set up more complete AMI test set (50+ meetings)
- Document findings and improvements

---

## Troubleshooting

### "No samples found"
→ Run `./verify_ami_setup.sh` to diagnose
→ See [docs/AMI_DIRECTORY_LAYOUT.md](docs/AMI_DIRECTORY_LAYOUT.md)

### Evaluation crashes mid-run
→ Try `--n 1` to isolate which meeting is problematic
→ Check that audio file is valid with `ffmpeg -i <file>`
→ Check annotation JSON is valid with `jq . <file>`

### Scores seem random/unrealistic
→ Verify reference summaries are actually loaded (check `reference_summary` field)
→ Try with a meeting you know well to sanity-check Claude's output
→ Check that `ANTHROPIC_API_KEY` is valid

### Too slow
→ First run includes transcription (~1-2 min/meeting with diarization)
→ Use `--use-cache` for subsequent runs (scoring only, ~5-10 sec/meeting)
→ Use GPU if available for faster transcription

---

## Quick Reference

```bash
# Verify setup
./verify_ami_setup.sh

# Run evaluation
uv run python benchmarks/eval_summaries.py --dataset ami --n 5

# With failure analysis
uv run python benchmarks/eval_summaries.py --dataset ami --n 5 --analyze-failures

# Use cached transcripts (fast)
uv run python benchmarks/eval_summaries.py --dataset ami --n 5 --use-cache

# Compare two runs
uv run python benchmarks/compare_results.py --before baseline.json --after improved.json

# Quick scores
jq '.aggregate_scores' results.json

# Failure breakdown
jq '.failure_analysis' results.json
```

---

## Related Documentation

- **Quick reference:** [docs/EVALUATION_LOOP_QUICKREF.md](docs/EVALUATION_LOOP_QUICKREF.md)
- **Full guide:** [docs/BENCHMARK_EVALUATION_QUICKSTART.md](docs/BENCHMARK_EVALUATION_QUICKSTART.md)
- **AMI setup:** [docs/AMI_SETUP.md](docs/AMI_SETUP.md)
- **Directory layout:** [docs/AMI_DIRECTORY_LAYOUT.md](docs/AMI_DIRECTORY_LAYOUT.md)
- **Experiment log:** [EVALUATION_LOG.md](EVALUATION_LOG.md)

---

**You're ready to start!** 🚀

Begin with Step 1 and you'll have your first evaluation results in under 30 minutes.
