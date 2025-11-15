# LLM Integration Examples - File Index

This directory contains practical examples for integrating audio-enriched transcripts with Large Language Models.

## Quick Start

```bash
# 1. See demonstrations
python quick_start.py

# 2. Try the demos
python prompt_builder.py --demo
python analysis_queries.py --demo
python comparison_demo.py --demo
python llm_integration_demo.py --examples

# 3. Use with your transcript
python prompt_builder.py your_enriched_transcript.json
python analysis_queries.py your_enriched_transcript.json --query escalation
```

## Files

### Core Modules

| File | Purpose | Key Features |
|------|---------|--------------|
| `prompt_builder.py` | Build LLM-ready prompts | Multiple formats, speaker-aware, audio summaries |
| `analysis_queries.py` | Pre-built analysis queries | 9+ query templates for common tasks |
| `llm_integration_demo.py` | Full API integration | OpenAI, Anthropic, local model support |
| `comparison_demo.py` | Benefits demonstration | Side-by-side with/without audio comparison |

### Documentation

| File | Content |
|------|---------|
| `README.md` | Complete documentation with use cases and best practices |
| `EXAMPLES_OUTPUT.md` | Concrete example outputs for each script |
| `INDEX.md` | This file - quick reference |
| `quick_start.py` | Interactive walkthrough guide |

### Supporting Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package initialization |

## Usage Patterns

### Pattern 1: Build Simple Prompts

```python
from prompt_builder import TranscriptPromptBuilder

builder = TranscriptPromptBuilder("transcript.json")
prompt = builder.build_basic_prompt()
```

### Pattern 2: Use Query Templates

```python
from analysis_queries import QueryTemplates

templates = QueryTemplates("transcript.json")
prompt = templates.find_escalation_moments()
```

### Pattern 3: Full LLM Integration

```python
from llm_integration_demo import LLMIntegration, LLMConfig
from analysis_queries import QueryTemplates

config = LLMConfig(provider="openai", model="gpt-4")
llm = LLMIntegration(config)

templates = QueryTemplates("transcript.json")
prompt = templates.summarize_emotional_arc()

result = llm.query(prompt)
print(result["response"])
```

## Feature Matrix

### prompt_builder.py

- ✓ Basic inline formatting
- ✓ Speaker-aware formatting
- ✓ Analysis prompts with tasks
- ✓ Comparison prompts
- ✓ Context injection for RAG
- ✓ Audio summary statistics
- ✓ Configurable timestamp/annotation display

### analysis_queries.py

Pre-built queries:
- ✓ Escalation detection
- ✓ Uncertainty identification
- ✓ Sarcasm detection
- ✓ Emotional arc summary
- ✓ High confidence statements
- ✓ Question detection (by tone)
- ✓ Agreement/disagreement analysis
- ✓ Key moments identification
- ✓ Speaker state profiling

### llm_integration_demo.py

- ✓ OpenAI integration
- ✓ Anthropic Claude integration
- ✓ Local model support
- ✓ API key management
- ✓ Error handling
- ✓ Token usage tracking
- ✓ Batch processing examples
- ✓ Streaming support examples

### comparison_demo.py

Demonstration scenarios:
- ✓ Sarcasm detection improvement
- ✓ Emotional shift tracking
- ✓ Speaker mental state assessment
- ✓ Question detection via prosody
- ✓ Side-by-side comparisons
- ✓ Quantitative benefit analysis

## Command Reference

### prompt_builder.py

```bash
# Show demonstrations
python prompt_builder.py --demo

# Generate basic prompt
python prompt_builder.py transcript.json

# Speaker-aware format
python prompt_builder.py transcript.json --style speaker

# Analysis format
python prompt_builder.py transcript.json --style analysis --task "Find escalation"
```

### analysis_queries.py

```bash
# Show demonstrations
python analysis_queries.py --demo

# Specific queries
python analysis_queries.py transcript.json --query escalation
python analysis_queries.py transcript.json --query uncertain
python analysis_queries.py transcript.json --query sarcasm
python analysis_queries.py transcript.json --query emotional_arc
python analysis_queries.py transcript.json --query confident
python analysis_queries.py transcript.json --query questions
python analysis_queries.py transcript.json --query agreement
python analysis_queries.py transcript.json --query key_moments
python analysis_queries.py transcript.json --query speaker_states

# Custom query
python analysis_queries.py transcript.json --custom "Your custom task"
```

### llm_integration_demo.py

```bash
# Show API examples
python llm_integration_demo.py --examples

# Dry run (no API calls)
python llm_integration_demo.py transcript.json --dry-run

# OpenAI integration
export OPENAI_API_KEY="sk-..."
python llm_integration_demo.py transcript.json --provider openai

# Anthropic integration
export ANTHROPIC_API_KEY="sk-ant-..."
python llm_integration_demo.py transcript.json --provider anthropic --model claude-3-5-sonnet-20241022

# Local model
python llm_integration_demo.py transcript.json --provider local --model llama3
```

### comparison_demo.py

```bash
# Show demonstration scenarios
python comparison_demo.py --demo

# Compare prompt sizes
python comparison_demo.py transcript.json

# Compare specific task
python comparison_demo.py transcript.json --task "Detect sarcasm"
```

### quick_start.py

```bash
# Interactive walkthrough
python quick_start.py
```

## Dependencies

### Required
- Python 3.8+
- Project modules (transcription package)

### Optional (for LLM integration)
- `openai` - OpenAI API integration
- `anthropic` - Anthropic Claude integration

```bash
pip install openai anthropic
```

## Common Workflows

### Workflow 1: Analyze Single Transcript

```bash
# 1. Enrich transcript (if not already enriched)
python audio_enrich.py audio.wav

# 2. Generate analysis prompt
python analysis_queries.py output/audio.json --query emotional_arc > prompt.txt

# 3. Send to LLM (manual or via integration)
# Manual: Copy prompt.txt to ChatGPT/Claude
# Automated: Use llm_integration_demo.py
```

### Workflow 2: Batch Analysis

```python
from pathlib import Path
from analysis_queries import QueryTemplates
from llm_integration_demo import LLMIntegration, LLMConfig

config = LLMConfig(provider="openai", model="gpt-4")
llm = LLMIntegration(config)

for transcript in Path("transcripts/").glob("*.json"):
    templates = QueryTemplates(transcript)
    prompt = templates.find_key_moments()
    result = llm.query(prompt)
    # Save result...
```

### Workflow 3: Custom Analysis

```python
from prompt_builder import TranscriptPromptBuilder

builder = TranscriptPromptBuilder("transcript.json")

custom_task = """
Analyze this sales call and identify:
1. Customer objections (look for uncertain or concerned tones)
2. Successful persuasion moments (tone shifts from negative to positive)
3. Overall customer satisfaction (emotional arc)
"""

prompt = builder.build_analysis_prompt(
    task=custom_task,
    include_audio_summary=True
)
```

## Integration Examples

### With RAG Systems

```python
# Semantic search for relevant segments
relevant_segments = semantic_search("budget concerns")

# Filter by audio features
high_concern = [s for s in relevant_segments
                if s.audio_state.get('emotion', {}).get('categorical', {}).get('primary') == 'concern']

# Build focused prompt
prompt = builder.build_context_injection_prompt(
    user_query="What budget concerns were raised?",
    relevant_segment_ids=[s.id for s in high_concern]
)
```

### With Streaming LLMs

```python
import openai

client = openai.OpenAI()
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Troubleshooting

### Issue: Module not found
```bash
# Ensure you're in project root
cd /path/to/slower-whisper
python examples/llm_integration/script.py
```

### Issue: API key not found
```bash
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

### Issue: No audio features
Check that transcript is enriched:
```bash
python audio_enrich.py audio.wav  # Creates enriched JSON
```

## Performance Tips

1. **Token Management**: Use `max_segments` to limit prompt size
2. **Batch Processing**: Group similar queries to reduce API calls
3. **Caching**: Save LLM responses to avoid re-analyzing
4. **Multi-pass**: Summary first, then detailed analysis on key segments

## Learning Path

1. **Start here**: `python quick_start.py`
2. **Try demos**: Run all `--demo` modes
3. **Read examples**: See `EXAMPLES_OUTPUT.md`
4. **Build prompts**: Experiment with `prompt_builder.py`
5. **Use templates**: Try pre-built queries
6. **Integrate**: Connect to your LLM of choice
7. **Customize**: Create your own query templates

## Related Documentation

- Main project: `../../README.md`
- Audio enrichment: `../../transcription/audio_enrichment.py`
- Other examples: `../README_EXAMPLES.md`
- Complete workflow: `../complete_workflow.py`

## Version

Current version: 1.0.0

Last updated: 2025-11-15
