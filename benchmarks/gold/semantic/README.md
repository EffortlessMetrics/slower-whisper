# Semantic Benchmark Gold Labels

This directory contains gold-standard semantic annotations for evaluating the
`KeywordSemanticAnnotator` output quality. These labels are human-curated
ground truth used to measure precision, recall, and F1 scores for semantic
extraction tasks.

## Schema Overview

Each gold label file is a JSON document conforming to `schema.json` (v1).

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | integer | Always `1` for this version |
| `meeting_id` | string | Unique identifier matching the source transcript |
| `topics` | array | Topics discussed in the meeting |
| `risks` | array | Risk signals identified |
| `actions` | array | Action items extracted |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `summary` | string | Human-written summary for Track A evaluation |

## Field Specifications

### Topics

Topics identify the main subjects discussed in a meeting segment.

```json
{
  "label": "pricing",
  "segment_ids": [3, 4, 5]
}
```

- `label` (required): Topic from controlled vocabulary (see below)
- `segment_ids` (optional): Array of segment IDs where topic is discussed

### Risks

Risks capture business signals that may require attention.

```json
{
  "type": "escalation",
  "severity": "high",
  "segment_id": 3,
  "evidence": "Please bring in a supervisor"
}
```

- `type` (required): One of `escalation`, `churn`, `pricing`
- `severity` (required): One of `low`, `medium`, `high`
- `segment_id` (required): Segment ID where risk was identified
- `evidence` (optional): Quote or description supporting the label

### Actions

Actions capture commitments and next steps.

```json
{
  "text": "Follow up with customer within the hour",
  "speaker_id": "spk_lead",
  "segment_ids": [6]
}
```

- `text` (required): Description of the action item
- `speaker_id` (optional): Speaker who committed to the action
- `segment_ids` (optional): Segment IDs where action is mentioned

## Risk Type Definitions

| Type | Description | Example Keywords |
|------|-------------|------------------|
| `escalation` | Customer frustration or request for management | supervisor, manager, unacceptable, complaint |
| `churn` | Signals of potential customer loss | cancel, terminate, competitor, leaving |
| `pricing` | Cost or budget concerns | price, expensive, budget, cost |

## Severity Mapping

| Severity | Description | Criteria |
|----------|-------------|----------|
| `low` | Minor concern, easily addressed | Single mention, no emotional charge |
| `medium` | Moderate concern requiring attention | Repeated mentions, mild frustration |
| `high` | Critical concern requiring immediate action | Explicit escalation request, threat to leave |

**Severity Guidelines:**

- **Low**: Informational mentions (e.g., "What's the price?" - pricing/low)
- **Medium**: Concerns expressed (e.g., "This is more expensive than we budgeted" - pricing/medium)
- **High**: Actionable urgency (e.g., "If you can't match their price, we'll switch" - pricing/high + churn/high)

## Topic Vocabulary (Controlled List)

Use these standardized topic labels for consistency:

### Business Domains
- `pricing` - Cost, pricing discussions
- `contract` - Contract terms, agreements
- `renewal` - Subscription/contract renewal
- `onboarding` - New customer setup
- `support` - Technical or customer support
- `escalation` - Issue escalation
- `feedback` - Customer feedback
- `feature_request` - Feature requests
- `bug_report` - Bug/issue reports
- `billing` - Billing questions
- `cancellation` - Service cancellation
- `upsell` - Upselling discussions
- `cross_sell` - Cross-selling discussions
- `negotiation` - Business negotiations
- `timeline` - Schedule, deadline discussions
- `process` - Process discussions
- `project` - Project-related discussions

### Technical Domains
- `integration` - System integrations
- `api` - API-related discussions
- `security` - Security concerns
- `performance` - Performance issues
- `migration` - Data/system migration
- `training` - User training
- `documentation` - Documentation needs
- `technical` - General technical discussions
- `design` - Design discussions

### Meeting Types
- `status_update` - Status updates
- `planning` - Planning sessions
- `retrospective` - Retrospective discussions
- `demo` - Product demos
- `review` - Reviews

### General
- `introduction` - Meeting introductions
- `closing` - Meeting wrap-up
- `off_topic` - Off-topic discussion
- `other` - Uncategorized topics

## Example Gold Label File

```json
{
  "schema_version": 1,
  "meeting_id": "escalation_handoff",
  "topics": [
    {"label": "support", "segment_ids": [0, 1, 2]},
    {"label": "escalation", "segment_ids": [3, 4]},
    {"label": "closing", "segment_ids": [5, 6]}
  ],
  "risks": [
    {
      "type": "escalation",
      "severity": "medium",
      "segment_id": 2,
      "evidence": "I'll escalate now"
    },
    {
      "type": "escalation",
      "severity": "high",
      "segment_id": 3,
      "evidence": "Please bring in a supervisor"
    }
  ],
  "actions": [
    {
      "text": "Escalate ticket to engineering",
      "speaker_id": "spk_agent",
      "segment_ids": [2]
    },
    {
      "text": "Move customer to critical queue",
      "speaker_id": "spk_lead",
      "segment_ids": [4]
    },
    {
      "text": "Call customer within the hour with progress",
      "speaker_id": "spk_lead",
      "segment_ids": [6]
    }
  ],
  "summary": "Customer escalated a stalled outage ticket. Agent escalated to supervisor who committed to critical queue treatment and hourly updates."
}
```

## Validation

Run the validation script to check gold label files:

```bash
# Validate all files in benchmarks/gold/semantic/
python scripts/validate_gold_labels.py

# Verbose output
python scripts/validate_gold_labels.py --verbose

# Strict mode (treat warnings as errors)
python scripts/validate_gold_labels.py --strict

# Custom paths
python scripts/validate_gold_labels.py \
  --gold-dir path/to/labels \
  --schema path/to/schema.json
```

### Validation Checks

1. **Schema Compliance**: All required fields present, correct types
2. **Enum Validation**: Risk types and severities match allowed values
3. **Sanity Checks**: Non-empty arrays, vocabulary compliance
4. **Consistency**: No duplicate risk entries for same segment

### Exit Codes

- `0` - All files valid
- `1` - Validation errors found

## Creating New Gold Labels

1. Identify the source transcript in `benchmarks/data/speaker_analytics_samples/`
2. Create a new JSON file with matching `meeting_id`
3. Annotate topics, risks, and actions following this guide
4. Run validation to check your annotations
5. Add `summary` field if supporting Track A evaluation

## Evaluation Metrics

Gold labels are used to compute:

- **Topic F1**: Precision/recall on topic detection
- **Risk F1**: Precision/recall on risk detection (per-type and aggregate)
- **Action F1**: Precision/recall on action extraction
- **Severity Accuracy**: Exact match on severity levels

See `benchmarks/eval_semantic.py` (when implemented) for evaluation harness.
