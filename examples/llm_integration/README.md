# LLM Integration Examples

This directory contains example scripts demonstrating how to integrate slower-whisper transcripts with Large Language Models for conversation analysis.

## Overview

slower-whisper produces rich, structured JSON transcripts with:
- Speaker diarization (who said what)
- Turn structure (contiguous segments by speaker)
- Audio enrichment (prosody, emotion, tone)

These examples show how to:
1. Load transcripts using the public API
2. Render them in LLM-friendly text formats
3. Send to Claude/GPT for downstream tasks

## Examples

### `summarize_with_diarization.py`

**What it does:**
- Loads a transcript JSON file
- Infers speaker roles (Agent/Customer) from talk time
- Renders conversation with turn structure + audio cues
- Sends to Claude for quality analysis and coaching feedback

**Prerequisites:**
```bash
# Install Anthropic SDK
uv pip install anthropic

# Set API key
export ANTHROPIC_API_KEY="your-key-here"
```

**Usage:**
```bash
# Basic usage (with audio cues)
python examples/llm_integration/summarize_with_diarization.py \
    whisper_json/support_call_001.json

# Without audio cues
python examples/llm_integration/summarize_with_diarization.py \
    whisper_json/support_call_001.json \
    --no-audio-cues
```

**Output:**
```
ANALYSIS
================================================================================
1. Summary
The customer called about login issues. The agent helped reset credentials
and verified the issue was resolved.

2. Quality Score: 8/10
Good empathy and clear instructions. Minor room for improvement in proactive
follow-up.

3. Coaching Feedback
- [00:02:15] Acknowledge frustration earlier when customer's tone escalates
- [00:04:30] Offer proactive next steps before customer asks
...
```

## How It Works

### Step 1: Transcribe Audio

```bash
# Generate transcript with diarization and audio enrichment
slower-whisper transcribe \
    --enable-diarization \
    --min-speakers 2 \
    --max-speakers 2 \
    --enable-prosody \
    --enable-emotion
```

This produces `whisper_json/your_file.json` with rich structured data.

### Step 2: Load and Render

```python
from transcription import load_transcript, render_conversation_for_llm

# Load transcript
transcript = load_transcript("whisper_json/support_call_001.json")

# Render for LLM
context = render_conversation_for_llm(
    transcript,
    mode="turns",              # Use turn structure
    include_audio_cues=True,   # Include [audio: ...] cues
    include_timestamps=True,   # Add [HH:MM:SS] prefixes
    speaker_labels={           # Map spk_0 → Agent
        "spk_0": "Agent",
        "spk_1": "Customer"
    }
)
```

**Output:**
```
Conversation: support_call_001.wav (en)
Duration: 00:05:23 | Speakers: 2 | Turns: 12

[00:00:00] [Agent | calm tone, moderate pitch] Hello, thank you for calling...
[00:00:04] [Customer | frustrated tone, high pitch, fast rate] I can't log in...
[00:00:12] [Agent | empathetic tone, slow rate] I understand that's frustrating...
```

### Step 3: Analyze with LLM

```python
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=2048,
    messages=[{
        "role": "user",
        "content": f"""{context}

Please summarize this call and provide coaching feedback for the agent.
"""
    }]
)

print(response.content[0].text)
```

## More Patterns

See `docs/LLM_PROMPT_PATTERNS.md` for additional use cases:
- Meeting summaries and action items
- Sales call scoring
- Sentiment analysis over time
- Conflict detection
- Intent classification
