# Evaluation Log

This file tracks quality evaluation experiments using benchmark datasets.

Format: Each experiment is a dated entry with baseline, changes, results, and next steps.

---

## Template

```markdown
## YYYY-MM-DD – [Experiment Name]

**Baseline:** [commit SHA or previous experiment]
**Code:** [commit SHA]
**Models:**
- Summarizer: claude-3-5-sonnet-20241022
- Judge: claude-3-5-sonnet-20241022

**Config:**
- Dataset: [ami/iemocap/librispeech]
- N samples: [10]
- Mode: [turns/segments]
- Audio cues: [enabled/disabled]
- Diarization: [enabled/disabled]
- Other: [any other relevant settings]

**Changes:**
- [List what you changed from baseline]
- [Prompt tweaks, config changes, code changes]

**Results:**

| Metric        | Baseline | This Run | Δ     |
|---------------|----------|----------|-------|
| Faithfulness  | 7.2      | 7.4      | +0.2  |
| Coverage      | 6.8      | 7.6      | +0.8  |
| Clarity       | 8.4      | 8.5      | +0.1  |

**Failure Analysis:**
- Bad cases: [count] (previously: [count])
- Categories:
  - missing_info: 40% (was 60%)
  - hallucination: 20% (was 20%)
  - wrong_speaker: 10% (was 0%)
  - style_only: 30% (was 20%)

**Claude's Recommendations:**
- [Copy key recommendations from failure_analysis output]

**Interpretation:**
- [What you learned]
- [What worked, what didn't]

**Next Steps:**
- [What to try next]
```

---

## Experiments

<!-- Add your experiments below, newest first -->

### Example Entry (Delete this before your first real experiment)

## 2025-11-19 – AMI Summaries Baseline

**Baseline:** Initial implementation
**Code:** abc1234
**Models:**
- Summarizer: claude-3-5-sonnet-20241022
- Judge: claude-3-5-sonnet-20241022

**Config:**
- Dataset: ami
- N samples: 10
- Mode: turns
- Audio cues: enabled
- Diarization: enabled

**Changes:**
- Initial baseline run
- Default prompts from dogfood.py
- Standard render_conversation_for_llm() output

**Results:**

| Metric        | Baseline | This Run | Δ     |
|---------------|----------|----------|-------|
| Faithfulness  | -        | 7.2      | -     |
| Coverage      | -        | 6.8      | -     |
| Clarity       | -        | 8.4      | -     |

**Failure Analysis:**
- Bad cases: 4/10
- Categories:
  - missing_info: 60%
  - hallucination: 20%
  - style_only: 20%

**Claude's Recommendations:**
- Add explicit instruction to preserve action items
- Consider including more context from earlier turns
- Emphasize speaker roles in prompt

**Interpretation:**
- System generally faithful (no major hallucinations)
- Coverage is the main weakness (missing key details)
- Clarity is good (well-structured outputs)
- No obvious diarization failures (wrong_speaker at 0%)

**Next Steps:**
- Tweak prompt to emphasize completeness
- Add explicit "action items" instruction
- Re-run with --use-cache to isolate prompt effects
