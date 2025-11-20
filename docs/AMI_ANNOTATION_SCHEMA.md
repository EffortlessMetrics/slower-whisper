# AMI Annotation JSON Schema

This document defines the expected JSON schema for AMI Meeting Corpus annotations used by slower-whisper's evaluation infrastructure.

## Schema Overview

Annotation files are stored in `~/.cache/slower-whisper/benchmarks/ami/annotations/` with the naming convention `{meeting_id}.json` (matching the corresponding audio file in the `audio/` directory).

## Required Fields

### Top-Level Structure

```json
{
  "transcript": "Optional reference transcript text...",
  "summary": "Optional reference summary text...",
  "speakers": [
    {
      "id": "SPEAKER_00",
      "segments": [
        {"start": 0.0, "end": 3.5},
        {"start": 5.2, "end": 8.1}
      ]
    }
  ],
  "metadata": {
    "scenario": "project_meeting",
    "duration": 600.5,
    "num_speakers": 4
  }
}
```

## Field Definitions

### `transcript` (string, optional)
- **Purpose**: Reference (ground truth) transcript for WER evaluation
- **Format**: Plain text, all speaker turns concatenated
- **Used by**: `eval_asr_diarization.py` for computing Word Error Rate (WER)
- **Can be null**: Yes (if not needed for WER evaluation)

**Example:**
```json
"transcript": "So let's start the meeting. Today we'll discuss the project timeline. I think we should focus on the core features first."
```

### `summary` (string, optional)
- **Purpose**: Reference (ground truth) summary for LLM-as-judge evaluation
- **Format**: Plain text, typically 3-5 sentences or bullet points
- **Used by**: `eval_summaries.py` for Claude-as-judge scoring
- **Can be null**: Yes (if not needed for summary evaluation)

**Example:**
```json
"summary": "The team discussed the functional design of a new remote control. The industrial designer presented technical constraints, the UI designer proposed a simple button layout, and the marketing expert emphasized the need for a young, trendy appearance. They agreed to focus on TV-only functionality to keep costs down."
```

### `speakers` (list of objects, optional but recommended)
- **Purpose**: Reference (ground truth) speaker diarization for DER evaluation
- **Format**: List of speaker objects, each with an ID and list of segments
- **Used by**: `eval_asr_diarization.py` for computing Diarization Error Rate (DER)
- **Can be null**: Yes (if not needed for DER evaluation)

**Speaker object structure:**
```json
{
  "id": "SPEAKER_00",           // String: Unique speaker identifier
  "segments": [                  // List: All segments spoken by this speaker
    {
      "start": 0.0,              // Float: Start time in seconds
      "end": 3.5                 // Float: End time in seconds
    }
  ]
}
```

**Notes:**
- Speaker IDs should be consistent within a meeting (e.g., "SPEAKER_00", "SPEAKER_01")
- Segments for a single speaker can be non-contiguous (speaker speaks multiple times)
- Segments should not overlap for the same speaker
- Gaps between segments represent pauses or other speakers talking

**Full example:**
```json
"speakers": [
  {
    "id": "SPEAKER_00",
    "segments": [
      {"start": 0.0, "end": 3.0},
      {"start": 6.2, "end": 9.2}
    ]
  },
  {
    "id": "SPEAKER_01",
    "segments": [
      {"start": 3.2, "end": 6.2},
      {"start": 9.4, "end": 12.4}
    ]
  }
]
```

### `metadata` (object, optional)
- **Purpose**: Additional context about the meeting (not used for metrics)
- **Format**: Arbitrary key-value pairs
- **Used by**: Informational only, passed to evaluation results
- **Can be null**: Yes

**Common metadata fields:**
```json
"metadata": {
  "scenario": "project_meeting",           // Meeting type
  "duration": 600.5,                       // Duration in seconds
  "num_speakers": 4,                       // Expected number of speakers
  "roles": {                               // Speaker role mappings (AMI-specific)
    "PM": "Project Manager",
    "UI": "UI Designer",
    "ME": "Marketing Expert",
    "ID": "Industrial Designer"
  }
}
```

## Complete Example

Here's a complete annotation file for a sample AMI meeting:

```json
{
  "transcript": "Good morning everyone. Let's start by reviewing the agenda. I think we should begin with the budget discussion. That sounds good to me. I agree, the budget is critical.",
  "summary": "The team held their weekly project meeting to review the agenda and prioritize the budget discussion. All participants agreed that budget planning should be the first topic addressed.",
  "speakers": [
    {
      "id": "SPEAKER_00",
      "segments": [
        {"start": 0.0, "end": 4.2},
        {"start": 8.5, "end": 12.1}
      ]
    },
    {
      "id": "SPEAKER_01",
      "segments": [
        {"start": 4.5, "end": 7.8}
      ]
    },
    {
      "id": "SPEAKER_02",
      "segments": [
        {"start": 12.3, "end": 15.0}
      ]
    }
  ],
  "metadata": {
    "scenario": "weekly_standup",
    "duration": 15.0,
    "num_speakers": 3
  }
}
```

## Usage in Evaluation

### WER Evaluation (`eval_asr_diarization.py`)
Requires: `transcript` field

The reference transcript is compared against slower-whisper's ASR output to compute Word Error Rate.

### DER Evaluation (`eval_asr_diarization.py`)
Requires: `speakers` field

The reference speaker segments are compared against slower-whisper's diarization output using pyannote.metrics to compute Diarization Error Rate.

### Summary Evaluation (`eval_summaries.py`)
Requires: `summary` field

The reference summary is used by Claude-as-judge to score the LLM-generated summary from slower-whisper's transcript.

## Minimal Valid Files

### Diarization-only evaluation
```json
{
  "speakers": [
    {
      "id": "SPEAKER_00",
      "segments": [{"start": 0.0, "end": 10.0}]
    }
  ]
}
```

### Summary-only evaluation
```json
{
  "summary": "Meeting summary goes here..."
}
```

### Full evaluation
```json
{
  "transcript": "Full transcript text...",
  "summary": "Meeting summary...",
  "speakers": [
    {
      "id": "SPEAKER_00",
      "segments": [{"start": 0.0, "end": 5.0}]
    }
  ]
}
```

## Common Mistakes

### ❌ Using `reference_summary` instead of `summary`
```json
{
  "reference_summary": "This won't work!"  // Wrong field name
}
```

**Fix:** Use `summary` as the field name.

### ❌ Using `reference_transcript` instead of `transcript`
```json
{
  "reference_transcript": "This won't work!"  // Wrong field name
}
```

**Fix:** Use `transcript` as the field name.

### ❌ Missing `segments` array in speakers
```json
{
  "speakers": [
    {
      "id": "SPEAKER_00"  // Missing segments!
    }
  ]
}
```

**Fix:** Always include the `segments` array, even if empty.

### ❌ Overlapping segments for same speaker
```json
{
  "speakers": [
    {
      "id": "SPEAKER_00",
      "segments": [
        {"start": 0.0, "end": 5.0},
        {"start": 3.0, "end": 7.0}  // Overlaps with previous!
      ]
    }
  ]
}
```

**Fix:** Ensure segments for the same speaker don't overlap.

## See Also

- [AMI_SETUP.md](AMI_SETUP.md) - Complete AMI corpus setup instructions
- [AMI_DIRECTORY_LAYOUT.md](AMI_DIRECTORY_LAYOUT.md) - Directory structure verification
- [BENCHMARK_EVALUATION_QUICKSTART.md](BENCHMARK_EVALUATION_QUICKSTART.md) - Running evaluations
