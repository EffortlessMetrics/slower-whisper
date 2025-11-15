# LLM Integration Examples

This directory contains practical examples for integrating enriched transcripts with Large Language Models (LLMs). The audio enrichment system produces transcripts with prosodic and emotional annotations, enabling LLMs to understand not just *what* was said, but *how* it was said.

## Overview

The enriched transcript format includes:
- **Text**: Transcribed speech from Whisper/Faster-Whisper
- **Audio Annotations**: Human-readable descriptions like `[audio: high pitch, loud volume, excited tone]`
- **Prosody Features**: Detailed pitch, energy, speech rate, and pause data
- **Emotion Features**: Valence, arousal, and categorical emotion classifications

This enables sophisticated LLM analysis that would be impossible with text-only transcripts.

## Scripts

### 1. `prompt_builder.py` - Build LLM-Ready Prompts

Convert enriched JSON transcripts into formatted prompts for LLM consumption.

**Features:**
- Multiple formatting styles (inline, separate, minimal)
- Speaker-aware formatting for dialogues
- Audio summary statistics
- Context injection for RAG applications
- Configurable timestamp and annotation display

**Usage:**

```bash
# Show demonstration examples
python prompt_builder.py --demo

# Generate basic prompt from transcript
python prompt_builder.py enriched_transcript.json

# Generate speaker-aware prompt
python prompt_builder.py enriched_transcript.json --style speaker

# Generate analysis prompt with task
python prompt_builder.py enriched_transcript.json --style analysis --task "Find moments of escalation"
```

**Python API:**

```python
from prompt_builder import TranscriptPromptBuilder, PromptConfig

# Basic usage
builder = TranscriptPromptBuilder("enriched_transcript.json")
prompt = builder.build_basic_prompt()

# Analysis prompt with audio summary
prompt = builder.build_analysis_prompt(
    task="Identify moments where the speaker's emotional state shifts",
    include_audio_summary=True
)

# Speaker-aware formatting
prompt = builder.build_speaker_aware_prompt()

# Custom configuration
config = PromptConfig(
    include_timestamps=True,
    include_audio_annotations=True,
    format_style="inline",
    speaker_aware=True
)
prompt = builder.build_basic_prompt(config)
```

**Output Example:**

```
TASK
Identify moments where the speaker's emotional state shifts significantly.

AUDIO CHARACTERISTICS SUMMARY
Total segments: 145
Segments with audio features: 145
Pitch distribution: medium (68/145), high (42/145), low (35/145)
Emotion distribution: neutral (85/145), joy (24/145), concern (14/145)

TRANSCRIPT
File: meeting.wav
Language: en

[0.0s - 2.5s] Hello everyone, thanks for joining today. [audio: neutral]
[2.5s - 5.8s] I'm really excited about our progress! [audio: high pitch, loud volume, excited tone]
[5.8s - 9.2s] However, we need to address some concerns. [audio: low pitch, slow speech, concerned tone]
```

---

### 2. `analysis_queries.py` - Pre-Built Analysis Prompts

Ready-to-use query templates for common analysis tasks.

**Available Queries:**
- `find_escalation_moments()` - Detect rising intensity/conflict
- `find_uncertain_statements()` - Identify hesitation and uncertainty
- `detect_sarcasm_contradiction()` - Find mismatches between words and tone
- `summarize_emotional_arc()` - Track emotional journey
- `find_high_confidence_statements()` - Locate assertive moments
- `identify_questions_by_tone()` - Detect questions using intonation
- `detect_agreement_disagreement()` - Analyze consensus patterns
- `find_key_moments()` - Identify most significant moments
- `analyze_speaker_states()` - Profile each speaker's emotional patterns

**Usage:**

```bash
# Show demonstration examples
python analysis_queries.py --demo

# Generate specific query
python analysis_queries.py enriched_transcript.json --query escalation
python analysis_queries.py enriched_transcript.json --query uncertain
python analysis_queries.py enriched_transcript.json --query sarcasm
python analysis_queries.py enriched_transcript.json --query emotional_arc

# Custom query
python analysis_queries.py enriched_transcript.json --custom "Find moments where the speaker contradicts themselves"
```

**Python API:**

```python
from analysis_queries import QueryTemplates

templates = QueryTemplates("enriched_transcript.json")

# Pre-built queries
escalation_prompt = templates.find_escalation_moments()
uncertainty_prompt = templates.find_uncertain_statements()
sarcasm_prompt = templates.detect_sarcasm_contradiction()
emotional_arc_prompt = templates.summarize_emotional_arc()

# Custom query
custom_prompt = templates.custom_query(
    "Identify moments where speakers talk over each other",
    include_summary=True
)
```

**Example Query Output:**

See the demonstration mode for detailed examples of expected LLM responses to each query type.

---

### 3. `llm_integration_demo.py` - Full LLM Integration Workflow

Complete end-to-end demonstration of integrating enriched transcripts with LLM APIs.

**Supported Providers:**
- **OpenAI** (GPT-4, GPT-3.5, etc.)
- **Anthropic** (Claude Sonnet, Opus, etc.)
- **Local/Custom** (Any OpenAI-compatible API)

**Features:**
- API key management from environment variables
- Error handling and retry logic
- Token usage tracking
- Response parsing
- Batch processing support
- Streaming response examples

**Usage:**

```bash
# Show API integration examples
python llm_integration_demo.py --examples

# Set API key (choose one)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run full workflow (requires API key)
python llm_integration_demo.py enriched_transcript.json --provider openai

# Dry run (show prompts without API calls)
python llm_integration_demo.py enriched_transcript.json --dry-run

# Use specific model
python llm_integration_demo.py enriched_transcript.json --provider anthropic --model claude-3-5-sonnet-20241022

# Local model
python llm_integration_demo.py enriched_transcript.json --provider local --model llama3
```

**Python API:**

```python
from llm_integration_demo import LLMIntegration, LLMConfig
from prompt_builder import TranscriptPromptBuilder

# Configure LLM
config = LLMConfig(
    provider="openai",
    model="gpt-4",
    temperature=0.7,
    max_tokens=2000
)

llm = LLMIntegration(config)

# Build prompt
builder = TranscriptPromptBuilder("enriched_transcript.json")
prompt = builder.build_analysis_prompt("Find moments of escalation")

# Query LLM
result = llm.query(
    prompt=prompt,
    system_message="You are an expert at analyzing audio-enriched transcripts."
)

print(result["response"])
print(f"Tokens used: {result['usage']['total_tokens']}")
```

**Installation:**

```bash
# For OpenAI
pip install openai

# For Anthropic
pip install anthropic

# Both are optional - only install what you need
```

---

### 4. `comparison_demo.py` - With vs Without Audio Enrichment

Demonstrates the difference in analysis quality between enriched and text-only transcripts.

**Features:**
- Side-by-side prompt comparisons
- Concrete scenarios showing improvement
- Quantitative benefit analysis
- Real-world use case examples

**Usage:**

```bash
# Show demonstration scenarios
python comparison_demo.py --demo

# Compare prompt sizes
python comparison_demo.py enriched_transcript.json

# Compare specific task
python comparison_demo.py enriched_transcript.json --task "Detect sarcasm in this conversation"
```

**Demonstration Scenarios:**

1. **Sarcasm Detection** - How tone reveals sarcasm that words hide
2. **Emotional Shift Tracking** - Precise measurement of emotional changes
3. **Speaker Mental State** - Assessing stress, confidence, cognitive load
4. **Question Detection** - Finding questions without question marks using intonation

**Key Findings:**

Audio enrichment provides:
- **Higher confidence** in tone/emotion classification (30-50% → 85-95%)
- **Quantitative metrics** for emotional intensity (not just present/absent)
- **Multiple corroborating signals** reducing false positives
- **Detection of pragmatic meaning** beyond literal text

---

## Use Cases

### 1. Customer Service Analysis

**Goal:** Identify frustrated customers and escalation moments

```python
from analysis_queries import QueryTemplates

templates = QueryTemplates("customer_call.json")

# Find escalation
escalation_prompt = templates.find_escalation_moments()

# Find uncertainty (may indicate confusion)
uncertainty_prompt = templates.find_uncertain_statements()

# Detect sarcasm (hidden frustration)
sarcasm_prompt = templates.detect_sarcasm_contradiction()
```

**Benefits:**
- Detect frustration before explicit complaints
- Identify where agents need support
- Measure emotional de-escalation effectiveness

### 2. Meeting Intelligence

**Goal:** Summarize meetings with emotional context

```python
from prompt_builder import TranscriptPromptBuilder

builder = TranscriptPromptBuilder("meeting.json")

# Speaker-aware transcript
prompt = builder.build_speaker_aware_prompt()

# Add task
full_prompt = f"""{prompt}

TASK: Summarize key decisions and action items.
For each decision, note:
- Who proposed it
- Level of agreement (check audio tone)
- Any hesitation or concerns (check pauses, uncertain tone)
"""
```

**Benefits:**
- Detect reluctant agreement vs. enthusiastic buy-in
- Identify unresolved concerns (uncertain tone on "agreements")
- Find moments needing follow-up

### 3. Interview Analysis

**Goal:** Assess candidate confidence and communication style

```python
from analysis_queries import QueryTemplates

templates = QueryTemplates("interview.json")

# Analyze confidence
confidence_prompt = templates.find_high_confidence_statements()

# Analyze uncertainty
uncertainty_prompt = templates.find_uncertain_statements()

# Analyze overall state
speaker_state_prompt = templates.analyze_speaker_states()
```

**Benefits:**
- Objective measurement of communication confidence
- Detect areas of genuine expertise vs. uncertainty
- Reduce interviewer bias through quantitative metrics

### 4. Podcast/Content Analysis

**Goal:** Find engaging moments for clips and highlights

```python
from analysis_queries import QueryTemplates

templates = QueryTemplates("podcast.json")

# Find key moments
key_moments_prompt = templates.find_key_moments()

# Find emotional peaks
emotional_arc_prompt = templates.summarize_emotional_arc()
```

**Benefits:**
- Automatically identify clip-worthy moments
- Find emotional peaks (excitement, surprise)
- Understand audience engagement trajectory

### 5. Therapy/Counseling Session Analysis

**Goal:** Track patient emotional state and progress

```python
from analysis_queries import QueryTemplates

templates = QueryTemplates("session.json")

# Track emotional journey
emotional_arc_prompt = templates.summarize_emotional_arc()

# Identify breakthroughs (shifts in tone)
# Measure confidence/uncertainty patterns
```

**Benefits:**
- Objective emotional state tracking
- Identify breakthrough moments
- Monitor treatment effectiveness over time

---

## Enriched JSON Format

The enriched transcript JSON structure:

```json
{
  "schema_version": 2,
  "file": "audio.wav",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "I'm really excited about this!",
      "speaker": "Alice",
      "audio_state": {
        "rendering": "[audio: high pitch, loud volume, fast speech, excited tone]",
        "prosody": {
          "pitch": {
            "level": "high",
            "mean_hz": 245.3,
            "std_hz": 32.1,
            "contour": "rising"
          },
          "energy": {
            "level": "loud",
            "db_rms": -12.1
          },
          "rate": {
            "level": "fast",
            "syllables_per_sec": 5.2
          },
          "pauses": {
            "count": 0,
            "density": 0.0
          }
        },
        "emotion": {
          "valence": {
            "level": "positive",
            "score": 0.78
          },
          "arousal": {
            "level": "high",
            "score": 0.82
          },
          "categorical": {
            "primary": "joy",
            "confidence": 0.85
          }
        }
      }
    }
  ],
  "meta": {
    "audio_enrichment": {
      "enriched_at": "2025-11-15T12:00:00Z",
      "features_enabled": {
        "prosody": true,
        "emotion_dimensional": true,
        "emotion_categorical": true
      }
    }
  }
}
```

**Key Fields for LLM Prompts:**
- `segments[].text` - The transcribed speech
- `segments[].audio_state.rendering` - Human-readable audio annotation
- `segments[].audio_state.prosody` - Detailed prosodic features
- `segments[].audio_state.emotion` - Emotional analysis

---

## Best Practices

### 1. Prompt Design

**Include audio summary:**
```python
# Good
prompt = builder.build_analysis_prompt(
    task="Find escalation",
    include_audio_summary=True  # Provides context
)

# Less effective
prompt = builder.build_basic_prompt()  # Missing overall context
```

**Use specific tasks:**
```python
# Good
task = "Identify moments where pitch and volume increase together, suggesting escalation"

# Less effective
task = "Analyze this transcript"  # Too vague
```

### 2. Audio Feature Interpretation

**In your system prompts, teach the LLM:**

```
Audio annotations follow this pattern: [audio: features]

Prosodic features:
- Pitch: low/medium/high (relative to speaker baseline)
- Volume: quiet/normal/loud
- Speech rate: slow/normal/fast
- Pauses: minimal/moderate/frequent

Emotional tones:
- Joy, sadness, anger, fear, surprise, neutral
- Excited, calm, agitated, uncertain, confident

Interpretation guidelines:
- High pitch + loud + fast = excitement or agitation
- Low pitch + quiet + slow = sadness or uncertainty
- Frequent pauses + slow = cognitive load or uncertainty
- Rising pitch contour = question (even without "?")
```

### 3. Handling Missing Features

Some segments may lack audio features. Handle gracefully:

```python
config = PromptConfig(
    include_audio_annotations=True,
    format_style="inline"  # Only shows annotations when present
)
```

The LLM should be instructed:
```
Segments without [audio: ...] annotations have neutral or unanalyzed audio.
Base analysis on text content for these segments.
```

### 4. Token Budget Management

**For long transcripts:**

```python
# Option 1: Limit segments
config = PromptConfig(max_segments=50)
prompt = builder.build_basic_prompt(config)

# Option 2: Focus on relevant segments
relevant_ids = [23, 45, 67, 89]  # From prior analysis
prompt = builder.build_comparison_prompt(
    segment_ids=relevant_ids,
    context="Compare these segments"
)

# Option 3: Summarize first, then deep-dive
summary_prompt = templates.summarize_emotional_arc()
# Get LLM response identifying key moments
# Then analyze those moments in detail
```

### 5. Combining with RAG

**For long conversations, use semantic search + audio filtering:**

```python
# 1. Semantic search finds topically relevant segments
relevant_segments = semantic_search(query="budget discussion")

# 2. Filter by audio features
high_emotion_segments = [
    s for s in relevant_segments
    if s.audio_state and s.audio_state.get('emotion', {}).get('arousal', {}).get('level') == 'high'
]

# 3. Build focused prompt
prompt = builder.build_context_injection_prompt(
    user_query="What concerns were raised about the budget?",
    relevant_segment_ids=[s.id for s in high_emotion_segments]
)
```

---

## Advanced Examples

### Multi-Pass Analysis

```python
from llm_integration_demo import LLMIntegration, LLMConfig
from analysis_queries import QueryTemplates

config = LLMConfig(provider="openai", model="gpt-4")
llm = LLMIntegration(config)
templates = QueryTemplates("transcript.json")

# Pass 1: Get emotional arc
arc_result = llm.query(templates.summarize_emotional_arc())

# Pass 2: Deep-dive on identified peaks
peak_segments = parse_peak_segments(arc_result["response"])
detail_prompt = builder.build_comparison_prompt(
    segment_ids=peak_segments,
    context="Analyze these emotional peaks in detail"
)
detail_result = llm.query(detail_prompt)
```

### Batch Processing

```python
from pathlib import Path
import json

results = []
for transcript_file in Path("transcripts/").glob("*.json"):
    templates = QueryTemplates(transcript_file)
    prompt = templates.summarize_emotional_arc()

    result = llm.query(prompt)
    results.append({
        "file": transcript_file.name,
        "analysis": result["response"],
        "tokens": result["usage"]["total_tokens"]
    })

# Save batch results
with open("batch_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Custom Query Templates

```python
class CustomQueryTemplates(QueryTemplates):
    def find_product_mentions_with_sentiment(self):
        task = """Identify all mentions of product names and features.

        For each mention, determine:
        1. What product/feature is mentioned
        2. Sentiment (positive, negative, neutral)
        3. Audio evidence (tone, pitch, energy)
        4. Context (complaint, praise, question)

        Use the audio tone to distinguish between:
        - Genuine praise (positive words + positive tone)
        - Sarcastic criticism (positive words + negative/sarcastic tone)
        - Neutral information (neutral words + neutral tone)
        """

        return self.builder.build_analysis_prompt(task, include_audio_summary=True)
```

---

## Troubleshooting

### Issue: API Key Not Found

**Solution:**
```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"

# Verify
echo $OPENAI_API_KEY
```

### Issue: Import Errors

**Solution:**
```bash
# Ensure you're in the project root or examples directory
cd /path/to/slower-whisper
python examples/llm_integration/prompt_builder.py --demo

# Or add to Python path
export PYTHONPATH=/path/to/slower-whisper:$PYTHONPATH
```

### Issue: Missing Audio Features

Some segments may not have audio_state. This is normal for:
- Very short segments (< 0.1s)
- Silence or background noise
- Extraction failures

**Handle in prompts:**
```python
config = PromptConfig(
    include_audio_annotations=True,
    format_style="inline"  # Only shows when present
)
```

### Issue: Prompts Too Long

**Solutions:**
1. Limit segments: `config = PromptConfig(max_segments=100)`
2. Focus on key moments: Use `find_key_moments()` first
3. Use RAG approach: Semantic search + audio filtering
4. Multi-pass: Summary first, then details

---

## Requirements

**Required:**
- Python 3.8+
- Project dependencies (transcription module)

**Optional (for LLM integration):**
- `openai` - For OpenAI API
- `anthropic` - For Anthropic API

**Install:**
```bash
# For OpenAI
pip install openai

# For Anthropic
pip install anthropic

# Both
pip install openai anthropic
```

---

## Examples Output Directory Structure

```
examples/llm_integration/
├── README.md                      (this file)
├── prompt_builder.py              (Build LLM prompts)
├── analysis_queries.py            (Pre-built queries)
├── llm_integration_demo.py        (Full API integration)
└── comparison_demo.py             (With vs without enrichment)
```

---

## Next Steps

1. **Try the demos:**
   ```bash
   python prompt_builder.py --demo
   python analysis_queries.py --demo
   python comparison_demo.py --demo
   python llm_integration_demo.py --examples
   ```

2. **Process your own transcript:**
   ```bash
   # First, create an enriched transcript
   python audio_enrich.py your_audio.wav

   # Then use with LLM integration
   python llm_integration_demo.py output/your_audio.json --dry-run
   ```

3. **Experiment with queries:**
   ```bash
   python analysis_queries.py output/your_audio.json --query emotional_arc
   ```

4. **Integrate with your LLM:**
   ```bash
   export OPENAI_API_KEY="your-key"
   python llm_integration_demo.py output/your_audio.json --provider openai
   ```

---

## Contributing

To add new query templates:

1. Extend `QueryTemplates` in `analysis_queries.py`
2. Follow the existing pattern for task descriptions
3. Add demonstration examples
4. Update this README

---

## References

- Main project: `../../README.md`
- Audio enrichment: `../../transcription/audio_enrichment.py`
- Other examples: `../README_EXAMPLES.md`
- Data models: `../../transcription/models.py`

---

## License

Same as main project.
