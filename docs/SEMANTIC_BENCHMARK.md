# Semantic Benchmark Reference

**Status:** Design Draft (targeting v2.0)
**Last Updated:** 2026-01-11

> **Note:** This document describes a planned feature for v2.0. The semantic benchmark is currently in development. See [ROADMAP.md](../ROADMAP.md) for release timeline.

This document describes the semantic quality benchmark for evaluating LLM-based annotation quality in slower-whisper.

---

## Overview

The semantic benchmark measures how well slower-whisper's semantic annotator extracts structured information from conversation transcripts. It evaluates three core capabilities:

1. **Topic extraction** - Identifying discussion subjects with confidence scores
2. **Risk detection** - Flagging escalations, churn signals, compliance concerns
3. **Action extraction** - Capturing commitments with assignees and deadlines

The benchmark supports two evaluation modes:

| Mode | Purpose | Requirements | CI-Friendly |
|------|---------|--------------|-------------|
| `tags` | Deterministic F1/precision/recall against gold labels | Gold label files | Yes |
| `summary` | LLM-as-judge summary quality assessment | `ANTHROPIC_API_KEY` | No (non-deterministic) |

---

## Evaluation Modes

### Mode: `tags` (Deterministic)

Compares extracted semantic annotations against human-curated gold labels. Suitable for CI gates and regression testing.

**Metrics produced:**
- Topic F1 (micro-averaged)
- Risk precision/recall/F1 (overall + by severity)
- Action accuracy (with fuzzy matching)

**Example:**

```bash
slower-whisper benchmark run --track semantic --mode tags --dataset ami --split test
```

### Mode: `summary` (LLM-as-Judge)

Uses Claude to evaluate generated summaries against reference summaries on quality dimensions. Non-deterministic; best for qualitative assessment.

**Metrics produced:**
- Faithfulness (0-10): Factual accuracy, no hallucinations
- Coverage (0-10): Completeness of key information
- Clarity (0-10): Readability and coherence

**Example:**

```bash
ANTHROPIC_API_KEY=sk-ant-xxx slower-whisper benchmark run \
  --track semantic --mode summary --dataset ami --split test --limit 10
```

---

## Gold Label Format

Gold labels are stored as JSON files in the benchmark gold directory. Each gold label file contains human-annotated ground truth for topics, risks, and actions.

### File Location

Gold label format is defined by `benchmarks/gold/semantic/schema.json`. Sample gold files are stored at:

```
benchmarks/gold/semantic/<meeting_id>.json
```

For example: `benchmarks/gold/semantic/design_review.json`

### Schema

```json
{
  "schema_version": 1,
  "meeting_id": "design_review",
  "topics": [
    {
      "label": "pricing",
      "segment_ids": [0, 15]
    }
  ],
  "risks": [
    {
      "type": "escalation",
      "severity": "high",
      "segment_id": 3,
      "evidence": "I need to speak to a manager"
    }
  ],
  "actions": [
    {
      "text": "Send pricing proposal",
      "speaker_id": "spk_agent",
      "segment_ids": [15]
    }
  ],
  "summary": "Optional human-written summary for Track A evaluation"
}
```

### Topic Vocabulary (Controlled List)

Topics are classified into a controlled vocabulary for consistent evaluation:

| Category | Topics |
|----------|--------|
| **Business** | `pricing`, `billing`, `contract`, `features`, `subscription`, `renewal` |
| **Support** | `technical_support`, `account_access`, `setup`, `integration`, `bug_report` |
| **Relationship** | `complaint`, `feedback`, `satisfaction`, `onboarding`, `training` |
| **Operations** | `scheduling`, `delivery`, `returns`, `refund`, `warranty` |

When evaluating, model outputs are normalized to this vocabulary before comparison.

### Risk Types and Severity Mapping

| Risk Type | Description | Severity Levels |
|-----------|-------------|-----------------|
| `escalation` | Customer requests manager/supervisor | low, medium, high, critical |
| `churn_risk` | Customer mentions leaving/canceling | low, medium, high, critical |
| `compliance` | Legal threats, regulatory concerns | medium, high, critical |
| `sentiment_negative` | Strong frustration, anger | low, medium, high |
| `pricing_objection` | Budget concerns, discount demands | low, medium, high |
| `competitor_mention` | References to competing products | low, medium |
| `customer_frustration` | Repeated issues, unresolved problems | low, medium, high |
| `agent_error` | Mistakes, miscommunication | low, medium, high |

**Severity mapping for metrics:**
- `critical` = weight 4.0
- `high` = weight 2.0
- `medium` = weight 1.0
- `low` = weight 0.5

### Action Format

Actions are evaluated with fuzzy text matching (similarity threshold 0.8). Gold labels should include:

| Field | Required | Description |
|-------|----------|-------------|
| `text` | Yes | Action description text |
| `speaker_id` | No | Speaker ID who committed to the action |
| `segment_ids` | No | Segments where action was mentioned |

---

## Metric Definitions

### Topic F1 (Micro-Averaged)

Measures topic extraction accuracy using micro-averaged F1:

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Where:
- **TP (True Positive)**: Model topic matches gold topic (exact label match)
- **FP (False Positive)**: Model topic not in gold labels
- **FN (False Negative)**: Gold topic not extracted by model

**Target:** F1 > 0.8 for production readiness.

**Proxy Topic Evaluation (Tags Mode):**

In `--mode tags`, Topic F1 is computed on `predicted_topics` derived from `{risk_tags + keywords}` against gold `topics[].label`. The current `KeywordSemanticAnnotator` does not produce true topic labels; it produces keyword/risk category terms. We treat these as a proxy for topic evaluation until an LLM-based annotator is available.

Expect low scores unless your gold topic vocabulary overlaps these proxy terms. When the predicted and gold topic vocabularies are completely disjoint, a `topic_note: "vocabulary_mismatch"` field is included in the result.

### Risk Precision/Recall/F1

Risk metrics are computed at two levels:

**1. Overall (type-only matching):**
```
TP: Model risk type matches gold risk type
```

**2. By severity (type + severity matching):**
```
TP: Model risk type AND severity match gold
```

**Severity-weighted F1:**
```
Weighted_TP = sum(severity_weight * TP) for each severity level
```

### Action Accuracy

Actions are matched using fuzzy text similarity with `difflib.SequenceMatcher`:

```python
def action_match(predicted: str, gold: str, threshold: float = 0.8) -> bool:
    """Match actions using normalized text similarity."""
    pred_norm = normalize_text(predicted)  # casefold + strip punctuation + collapse whitespace
    gold_norm = normalize_text(gold)
    similarity = difflib.SequenceMatcher(None, pred_norm, gold_norm).ratio()
    return similarity >= threshold
```

**Metrics:**
- **Action Recall**: Fraction of gold actions matched by model (matched_count / gold_count)
- **Action Precision**: Fraction of model actions that match gold (matched_count / pred_count)
- **Action F1 (Accuracy)**: Harmonic mean of precision and recall

### Summary Metrics (LLM-as-Judge)

When using `--mode summary`, Claude evaluates generated summaries:

| Metric | Scale | Description |
|--------|-------|-------------|
| `faithfulness` | 0-10 | Factual accuracy; penalizes hallucinations |
| `coverage` | 0-10 | Completeness of key information from reference |
| `clarity` | 0-10 | Organization, readability, coherence |

**Scoring rubric:**
- 10 = Excellent (reference quality)
- 7-9 = Good (minor issues)
- 4-6 = Acceptable (noticeable gaps)
- 1-3 = Poor (significant problems)
- 0 = Failed (unusable output)

---

## How to Run

### Prerequisites

1. Stage benchmark dataset (see [AMI Setup](AMI_SETUP.md) or [LibriSpeech Quickstart](LIBRISPEECH_QUICKSTART.md))
2. For `summary` mode: Set `ANTHROPIC_API_KEY` environment variable

### CLI Examples

**Tags mode (CI-friendly):**

```bash
# Run semantic benchmark with deterministic metrics
slower-whisper benchmark run --track semantic --mode tags --dataset ami --split test

# Quick smoke test (10 samples)
slower-whisper benchmark run --track semantic --mode tags --dataset ami --limit 10

# Save results to JSON
slower-whisper benchmark run --track semantic --mode tags --dataset ami -o semantic_results.json
```

**Summary mode (LLM-as-judge):**

```bash
# Evaluate summary quality
ANTHROPIC_API_KEY=sk-ant-xxx slower-whisper benchmark run \
  --track semantic --mode summary --dataset ami --split test --limit 5

# Use specific Claude model
ANTHROPIC_API_KEY=sk-ant-xxx slower-whisper benchmark run \
  --track semantic --mode summary --dataset ami --model claude-3-5-sonnet-20241022
```

**Full evaluation script:**

```bash
#!/bin/bash
# scripts/run_semantic_benchmark.sh

set -e

# Run tags mode (deterministic)
slower-whisper benchmark run \
  --track semantic \
  --mode tags \
  --dataset ami \
  --split test \
  --output results/semantic_tags_$(date +%Y%m%d).json

# Run summary mode if API key available
if [ -n "$ANTHROPIC_API_KEY" ]; then
  slower-whisper benchmark run \
    --track semantic \
    --mode summary \
    --dataset ami \
    --split test \
    --limit 20 \
    --output results/semantic_summary_$(date +%Y%m%d).json
fi

echo "Benchmark complete. Results in results/"
```

### Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | `--mode summary` |
| `SLOWER_WHISPER_BENCHMARKS` | Override benchmark data directory | Optional |
| `SLOWER_WHISPER_SEMANTIC_MODEL` | Default semantic model | Optional |

---

## How to Add Gold Labels

### Labeling Workflow

1. **Select samples**: Choose representative samples from the dataset
2. **Transcribe**: Run slower-whisper transcription with diarization
3. **Label topics**: Read transcript, identify main discussion subjects
4. **Label risks**: Flag any concerning signals with severity
5. **Label actions**: Extract explicit commitments
6. **Validate**: Run validation script to check schema compliance

### Step-by-Step Example

```bash
# 1. Transcribe a sample
slower-whisper transcribe audio/support_call.wav --enable-diarization -o whisper_json/

# 2. Create gold label file (must match meeting_id used in samples)
cat > benchmarks/gold/semantic/support_call.json << 'EOF'
{
  "schema_version": 1,
  "meeting_id": "support_call",
  "topics": [
    {"label": "pricing", "segment_ids": [0, 15]}
  ],
  "risks": [
    {"type": "escalation", "severity": "high", "segment_id": 3}
  ],
  "actions": [
    {"text": "Send pricing proposal", "speaker_id": "spk_agent"}
  ]
}
EOF

# 3. Validate the gold label against schema
python -c "import json; from jsonschema import validate; \
  schema = json.load(open('benchmarks/gold/semantic/schema.json')); \
  data = json.load(open('benchmarks/gold/semantic/support_call.json')); \
  validate(data, schema); print('Valid!')"
```

### Validation Script

```bash
# Validate all gold labels against schema
for f in benchmarks/gold/semantic/*.json; do
  if [[ "$f" != *"schema.json" ]]; then
    python -c "import json; from jsonschema import validate; \
      schema = json.load(open('benchmarks/gold/semantic/schema.json')); \
      data = json.load(open('$f')); validate(data, schema); print('$f: OK')"
  fi
done
```

The validator checks:
- JSON schema compliance
- Topic labels are in controlled vocabulary
- Risk types and severities are valid
- Required fields present
- Sample ID matches filename

---

## Measurement Integrity Policy

The semantic benchmark follows strict measurement integrity rules:

### No Zeros for Unmeasured

If a metric cannot be computed (missing gold labels, API error, etc.), the result must be `null` with a reason, not `0`:

```json
{
  "metrics": {
    "topic_f1": {
      "value": null,
      "reason": "No gold labels found for sample ES2002a"
    },
    "risk_f1": {
      "value": 0.85,
      "reason": null
    }
  }
}
```

### Always Emit Null + Reason

Every metric in the result includes:
- `value`: The computed metric (or `null` if unmeasured)
- `reason`: Explanation when `value` is `null`
- `coverage`: Fraction of samples where metric was computable

```json
{
  "aggregate_metrics": {
    "topic_f1": {
      "value": 0.82,
      "reason": null,
      "coverage": 0.95,
      "samples_measured": 95,
      "samples_skipped": 5
    }
  }
}
```

### Coverage Reporting

Every benchmark run reports coverage statistics:

```json
{
  "coverage": {
    "samples_total": 100,
    "samples_with_gold_labels": 95,
    "samples_with_topics": 90,
    "samples_with_risks": 85,
    "samples_with_actions": 80,
    "gold_label_coverage": 0.95
  }
}
```

### Invalid Result Handling

If a result is later found to be invalid (e.g., gold label error discovered):

```json
{
  "valid": false,
  "invalid_reason": "Gold labels for ES2002a contained duplicates; see #157",
  "invalidated_at": "2026-01-15T10:00:00Z",
  "superseded_by": "run-20260115-semantic-001"
}
```

---

## Known Limitations (Tags Mode)

The following limitations apply to `--mode tags` evaluation:

### Proxy Topic Vocabulary

Topic F1 is computed using predicted terms from `{risk_tags + keywords}` compared against gold `topics[].label`. The `KeywordSemanticAnnotator` does not produce true topic labels—it extracts risk categories and keywords as proxies.

**Impact:** Expect low topic scores unless gold topic vocabulary overlaps annotator output terms.

### Synthetic Transcript Segments

The current tags-mode runner constructs a single-segment transcript from the sample's reference text. This means:

- All predicted annotations are associated with `segment_id=0`
- Gold labels with `segment_id > 0` will not match predicted segment IDs
- Risk metrics may be degraded when gold annotations are segment-specific

**Output note:** When synthetic segments are used, results include `tags_note: "synthetic_transcript_segments"`.

**Roadmap:** Loading real segments from AMI JSON transcripts is a follow-up improvement.

### Severity Not Measured

The `KeywordSemanticAnnotator` does not produce severity levels for detected risks. Risk matching is type-only (e.g., `escalation` matches `escalation` regardless of severity).

**Output note:** When gold risks include severity, results include `risk_note: "severity_not_measured"`.

---

## Optional Output Fields

Per-sample results may include these optional diagnostic fields:

| Field | Type | Description |
|-------|------|-------------|
| `tags_note` | string | General measurement note (e.g., `"synthetic_transcript_segments"`) |
| `topic_note` | string | Topic-specific note (e.g., `"vocabulary_mismatch"` when gold/pred vocabularies are disjoint) |
| `risk_note` | string | Risk-specific note (e.g., `"severity_not_measured"`) |
| `action_matched` | int | Count of gold actions matched by predictions |
| `action_gold` | int | Total count of gold actions |
| `tags_reason` | string | Why metrics are `null` (e.g., `"no_gold_labels"`, `"no_transcript"`) |
| `summary_reason` | string | Why summary metrics are `null` (e.g., `"missing_api_key"`) |

These fields provide transparency about measurement integrity and help diagnose low scores.

---

## Output Format

### Result JSON Schema

```json
{
  "track": "semantic",
  "mode": "tags",
  "dataset": "ami",
  "split": "test",
  "samples_evaluated": 100,
  "samples_failed": 2,
  "timestamp": "2026-01-08T12:00:00Z",
  "metrics": {
    "topic_f1": {"value": 0.82, "unit": "ratio"},
    "topic_precision": {"value": 0.85, "unit": "ratio"},
    "topic_recall": {"value": 0.79, "unit": "ratio"},
    "risk_f1": {"value": 0.78, "unit": "ratio"},
    "risk_precision": {"value": 0.80, "unit": "ratio"},
    "risk_recall": {"value": 0.76, "unit": "ratio"},
    "risk_f1_weighted": {"value": 0.81, "unit": "ratio"},
    "action_accuracy": {"value": 0.75, "unit": "ratio"},
    "action_f1": {"value": 0.72, "unit": "ratio"}
  },
  "coverage": {
    "samples_total": 100,
    "samples_with_gold_labels": 98,
    "gold_label_coverage": 0.98
  },
  "receipt": {
    "tool_version": "1.9.2",
    "semantic_model": "qwen2.5-7b",
    "semantic_backend": "local",
    "config_hash": "sha256:abc123..."
  }
}
```

### Per-Sample Results

Detailed per-sample results for debugging:

```json
{
  "sample_results": [
    {
      "sample_id": "ES2002a",
      "topic_precision": 1.0,
      "topic_recall": 0.67,
      "topic_f1": 0.80,
      "predicted_topics": ["pricing", "support"],
      "gold_topics": ["pricing", "support", "onboarding"],
      "risk_matches": [
        {"predicted": "escalation:high", "gold": "escalation:high", "match": true}
      ],
      "action_matches": [
        {"predicted": "Send pricing proposal", "gold": "Send pricing proposal", "similarity": 1.0}
      ]
    }
  ]
}
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Topic F1 | > 0.80 | Micro-averaged across all samples |
| Risk F1 | > 0.75 | Type-only matching |
| Risk F1 (weighted) | > 0.80 | Severity-weighted |
| Action Accuracy | > 0.70 | With 0.8 similarity threshold |
| Faithfulness (summary) | > 7.0 | Average across samples |
| Coverage (summary) | > 7.0 | Average across samples |
| Clarity (summary) | > 8.0 | Average across samples |

---

## Related Documentation

- [BENCHMARKS.md](BENCHMARKS.md) - Benchmark CLI overview
- [LLM_SEMANTIC_ANNOTATOR.md](LLM_SEMANTIC_ANNOTATOR.md) - Semantic annotator design
- [AMI_SETUP.md](AMI_SETUP.md) - AMI corpus setup
- [GETTING_STARTED_EVALUATION.md](GETTING_STARTED_EVALUATION.md) - Evaluation quickstart
