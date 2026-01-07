# LLM Prompt Patterns for slower-whisper

**Status:** v1.2 - Production-ready patterns for conversation analysis (includes speaker analytics)

**Target audience:** Application developers building LLM-powered conversation intelligence

---

## Overview

This guide provides **reference prompts** for using slower-whisper's JSON output with text-only LLMs (like Claude, GPT-4, etc.).

slower-whisper's JSON format is designed to be **LLM-native**:

- Text-based renderings of audio features (`audio_state.rendering`, `segment.speaker`)
- Structured turn-level and speaker-level aggregates
- Timestamped segments for precise retrieval and citation

This guide shows how to:

1. **Transform JSON into text context** for LLM prompts
2. **Structure prompts** for common conversation analysis tasks
3. **Handle multi-speaker scenarios** with diarization data

**Quick Start:** See [`examples/llm_integration/`](../examples/llm_integration/) for working code that demonstrates:
- Loading transcripts with the public API
- Rendering for LLM consumption
- Sending to Claude/GPT for analysis
- Speaker role inference and labeling
- For a tiny bake-off and expected lift (modest, clearest on attribution tasks), see [`benchmarks/SPEAKER_ANALYTICS_MVP.md`](../benchmarks/SPEAKER_ANALYTICS_MVP.md).

---

## Quick Reference: What's in the JSON

**Segment-level** (array of segments):
```json
{
  "id": 0,
  "start": 0.0,
  "end": 4.2,
  "text": "Hello, how can I help you today?",
  "speaker": {"id": "spk_0", "confidence": 0.87},
  "audio_state": {
    "rendering": "[audio: warm tone, moderate pitch, calm pace]",
    "prosody": { /* pitch, energy, rate details */ },
    "emotion": { /* valence, arousal details */ }
  }
}
```

**Turn-level** (grouped by speaker):
```json
{
  "id": "turn_0",
  "speaker_id": "spk_0",
  "start": 0.0,
  "end": 8.5,
  "text": "Hello, how can I help you today? Let me pull up your account.",
  "segment_ids": [0, 1]
}
```

**Speaker-level** (aggregates):
```json
{
  "id": "spk_0",
  "label": null,
  "total_speech_time": 45.3,
  "num_segments": 12
}
```

---

## Pattern 1: Conversation Summary with Speaker Awareness

**Use case:** Summarize a customer support call, sales call, or meeting with attention to who said what.

### Example Input (JSON snippet)

```json
{
  "file": "support_call_001.wav",
  "language": "en",
  "speakers": [
    {"id": "spk_0", "total_speech_time": 45.3, "num_segments": 12},
    {"id": "spk_1", "total_speech_time": 32.1, "num_segments": 9}
  ],
  "turns": [
    {
      "id": "turn_0",
      "speaker_id": "spk_0",
      "text": "Hello, thank you for calling support. How can I help you today?",
      "start": 0.0,
      "end": 3.5
    },
    {
      "id": "turn_1",
      "speaker_id": "spk_1",
      "text": "Hi, I'm having trouble logging into my account. It keeps saying my password is invalid.",
      "start": 4.0,
      "end": 8.2
    },
    {
      "id": "turn_2",
      "speaker_id": "spk_0",
      "text": "I understand. Let me look that up for you. Can you confirm your email address?",
      "start": 8.5,
      "end": 12.1
    }
  ]
}
```

### Rendering with Public API (Python)

**slower-whisper v1.1+ provides a built-in rendering API:**

```python
from transcription import load_transcript, render_conversation_for_llm

# Load transcript from JSON
transcript = load_transcript("whisper_json/support_call_001.json")

# Render for LLM consumption
context = render_conversation_for_llm(
    transcript,
    mode="turns",              # Use turn structure (recommended)
    include_audio_cues=False,  # Set True to include prosody/emotion cues
    include_timestamps=False,  # Set True for [HH:MM:SS] prefixes
    include_metadata=True      # Include conversation header with stats
)
```

**Output example:**
```
Conversation: support_call_001.wav (en)
Duration: 00:05:23 | Speakers: 2 | Turns: 12

[spk_0] Hello, thank you for calling support. How can I help you today?
[spk_1] Hi, I'm having trouble logging into my account...
[spk_0] I can help with that. Let me pull up your account details...
```

**Available rendering functions:**
- `render_conversation_for_llm(transcript, ...)` - Full conversation with metadata
- `render_conversation_compact(transcript, max_tokens=None)` - Minimal format for token budgets
- `render_segment(segment, ...)` - Single segment rendering

See API documentation: `python -c "from transcription import render_conversation_for_llm; help(render_conversation_for_llm)"`

### Example Prompt

````markdown
<transcript>
# Conversation: support_call_001.wav
Language: en

## Participants
- spk_0: 45.3s speech time, 12 segments
- spk_1: 32.1s speech time, 9 segments

## Transcript

spk_0 [0.0s - 3.5s]:
Hello, thank you for calling support. How can I help you today?

spk_1 [4.0s - 8.2s]:
Hi, I'm having trouble logging into my account. It keeps saying my password is invalid.

spk_0 [8.5s - 12.1s]:
I understand. Let me look that up for you. Can you confirm your email address?

[... rest of transcript ...]
</transcript>

Analyze this customer support conversation and provide:

1. **Summary**: 2-3 sentences capturing the issue and resolution
2. **Key Moments**: Timestamped highlights (reference the [Xs] timestamps)
3. **Speaker Roles**: Identify which speaker is the agent vs. customer based on behavior
4. **Outcome**: Was the issue resolved? If so, how?

Format your response as structured sections for easy parsing.
````

### Expected LLM Output

```markdown
## Summary
Customer (spk_1) contacted support unable to log into their account due to an invalid password error. Agent (spk_0) verified the account, discovered a locked status due to multiple failed attempts, and successfully reset access.

## Key Moments
- [4.0s] Customer reports login issue with invalid password message
- [12.1s] Agent begins account lookup and verification
- [45.2s] Agent identifies account lock due to failed login attempts
- [58.3s] Agent confirms successful password reset and account unlock

## Speaker Roles
- **spk_0 = Agent**: Professional greeting, verification questions, technical diagnosis
- **spk_1 = Customer**: Reports issue, provides requested information, confirms resolution

## Outcome
✅ Resolved - Account unlocked and new password set. Customer confirmed successful login at call end.
```

---

## Pattern 2: Coaching Feedback with Audio Cues

**Use case:** Analyze sales or support calls for coaching, with attention to *how* things were said (tone, pace, energy).

### Example Input (with audio_state)

```json
{
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "Hello, thank you for calling. How can I help you?",
      "speaker": {"id": "spk_0", "confidence": 0.92},
      "audio_state": {
        "rendering": "[audio: warm tone, moderate pitch, calm pace]"
      }
    },
    {
      "id": 5,
      "start": 45.2,
      "end": 48.7,
      "text": "I understand your frustration. Let me see what I can do to fix this right away.",
      "speaker": {"id": "spk_0", "confidence": 0.88},
      "audio_state": {
        "rendering": "[audio: empathetic tone, slower pace, reassuring]"
      }
    }
  ]
}
```

### Rendering with Audio Cues (Built-in API)

```python
from transcription import load_transcript, render_conversation_for_llm

# Load transcript with audio enrichment
transcript = load_transcript("whisper_json/call_with_enrichment.json")

# Render with audio cues inline
context = render_conversation_for_llm(
    transcript,
    mode="segments",           # Segment-by-segment for fine-grained audio cues
    include_audio_cues=True,   # Include [audio: ...] renderings
    include_timestamps=True,   # Include [HH:MM:SS] timestamps
    include_metadata=False      # Skip header for compact output
)
```

**Output:**
```
[00:00:00] [spk_0 | warm tone, moderate pitch, calm pace] Hello, thank you for calling...
[00:00:04] [spk_1 | elevated pitch, fast speech, tense] I'm really frustrated...
[00:00:09] [spk_0 | empathetic tone, slower pace, reassuring] I understand your frustration...
```

### Example Prompt

````markdown
<call_transcript>
spk_0 [0.0s]: Hello, thank you for calling. How can I help you?
  [audio: warm tone, moderate pitch, calm pace]

spk_1 [3.8s]: I'm really frustrated. I've been on hold for 20 minutes and this is the third time I've called about the same issue.
  [audio: elevated pitch, fast speech, tense]

spk_0 [8.5s]: I understand your frustration. Let me see what I can do to fix this right away.
  [audio: empathetic tone, slower pace, reassuring]

[... rest of transcript ...]
</call_transcript>

You are a customer support coach. Analyze this call transcript and provide feedback for the agent (spk_0):

1. **Strengths**: What did the agent do well? Reference specific moments with timestamps.
2. **Opportunities**: Where could the agent improve their approach or communication style?
3. **Audio/Tone Analysis**: How effectively did the agent use tone, pace, and delivery to manage the interaction?
4. **Overall Assessment**: Rate the call (1-5) and provide 2-3 actionable next steps for improvement.

Focus on **behavioral specifics** tied to actual transcript moments.
````

### Expected LLM Output

```markdown
## Strengths
- **[0.0s] Strong opening**: Warm, professional greeting with clear offer to help
- **[8.5s] Empathy under pressure**: When customer expressed frustration, agent acknowledged feelings before pivoting to solution ("I understand your frustration")
- **Tone calibration**: Agent consistently matched slower pace and reassuring tone when customer showed stress (e.g., [8.5s], [45.2s])

## Opportunities
- **[3.8s response] Acknowledge wait time**: Customer mentioned 20-minute hold, but agent didn't explicitly apologize for the wait before moving to solution
- **[28.3s] Technical jargon**: Used "backend validation error" without checking if customer understood—could simplify language
- **Proactive closure**: Could have offered follow-up or direct contact info to prevent "third call" scenario in future

## Audio/Tone Analysis
✅ **Excellent**: Agent's audio cues show strong emotional intelligence:
- Maintained calm, moderate pace even when customer was fast/tense
- Used empathetic tone markers at key frustration points
- Avoided defensive or rushed speech patterns

This is a model example of tone management under pressure.

## Overall Assessment
**Rating: 4/5** - Strong call with minor process gaps

**Action Steps**:
1. Add explicit acknowledgment of wait times in opening (e.g., "I apologize for the hold—let's get this resolved now")
2. Practice translating technical errors into customer-friendly language
3. Close with proactive offer: "Here's my direct line if this happens again"
```

---

## Pattern 3: Action Item Extraction with Ownership

**Use case:** Extract tasks, commitments, and next steps from meetings or calls, with clear speaker attribution.

### Example Prompt

````markdown
<meeting_transcript>
spk_0 [0.0s - 15.2s]:
Alright team, let's review the launch timeline. We need to finalize the landing page copy by Friday.

spk_1 [15.5s - 22.3s]:
I can have the first draft to you by Wednesday. That gives us two days for revisions.

spk_0 [22.8s - 28.1s]:
Perfect. And can someone take ownership of the email campaign setup in Mailchimp?

spk_2 [28.5s - 35.7s]:
I'll handle that. I need the final copy from spk_1 first, but I can start the template design this week.

[... rest of transcript ...]
</meeting_transcript>

Extract all action items from this meeting. For each action item, provide:

1. **Task**: What needs to be done
2. **Owner**: Which speaker committed to it (use speaker ID)
3. **Deadline**: If mentioned
4. **Dependencies**: Any blockers or prerequisites
5. **Timestamp**: Where in the call this was discussed

Output as a structured table or JSON for easy parsing.
````

### Expected LLM Output

```markdown
## Action Items

| Task | Owner | Deadline | Dependencies | Timestamp |
|------|-------|----------|--------------|-----------|
| Draft landing page copy | spk_1 | Wednesday | None | [15.5s] |
| Finalize landing page copy (review + revisions) | spk_0 | Friday | Draft from spk_1 | [0.0s] |
| Set up email campaign in Mailchimp | spk_2 | (not specified) | Final copy from spk_1 | [28.5s] |
| Design email template | spk_2 | This week | None | [28.5s] |

**Critical Path**:
1. spk_1 delivers draft (Wed) → 2. spk_0 finalizes (Fri) → 3. spk_2 sets up campaign

**Risks**:
- Email campaign has no explicit deadline—recommend clarifying in follow-up
```

---

## Pattern 4: Sentiment Shift Detection

**Use case:** Identify emotional turning points in conversations (customer going from frustrated to satisfied, sales prospect showing interest, etc.).

### Example Prompt

````markdown
<conversation_with_audio>
spk_1 [0.0s]: This is ridiculous. I've been trying to get this resolved for a week.
  [audio: high pitch, fast speech, elevated energy]

spk_0 [3.2s]: I hear you, and I'm going to make this right. Let me escalate this to our senior team right now.
  [audio: calm, steady pace, reassuring tone]

spk_1 [8.5s]: Okay... I appreciate that. I just really need this fixed today.
  [audio: moderate pitch, slower pace, tentative]

[... middle of conversation ...]

spk_1 [62.3s]: Oh wow, that actually worked! Thank you so much, I really appreciate your help.
  [audio: lower pitch, relaxed pace, positive energy]
</conversation_with_audio>

Identify **sentiment shifts** in this conversation:

1. **Initial state**: How did each speaker start the conversation? (tone, emotion)
2. **Turning points**: When did sentiment change? What triggered it?
3. **Final state**: How did the conversation end?
4. **Key moves**: What specific actions or phrases by spk_0 contributed to positive shift?

Reference timestamps and audio cues in your analysis.
````

### Expected LLM Output

```markdown
## Sentiment Trajectory

### Initial State (0.0s - 8.5s)
- **spk_1 (Customer)**: Frustrated, angry
  - Audio cues: high pitch, fast speech, elevated energy
  - Language: "ridiculous", "trying for a week" (escalation indicators)
- **spk_0 (Agent)**: Calm, solution-focused
  - Audio cues: steady pace, reassuring tone
  - Language: "I hear you", "make this right" (validation + commitment)

### Turning Point 1: [3.2s] - De-escalation begins
**Trigger**: Agent's immediate acknowledgment + escalation commitment
**Evidence**: spk_1 shifts from anger to tentative cooperation
- Audio changes: pitch moderates, pace slows, "Okay... I appreciate that"

### Middle Phase (8.5s - 62.3s)
Customer remains cautiously engaged while agent works on solution

### Turning Point 2: [62.3s] - Resolution + Positive shift
**Trigger**: Technical solution actually works
**Evidence**: spk_1 shifts to gratitude and relief
- Audio changes: lower pitch, relaxed pace, positive energy
- Language: "Oh wow", "thank you so much" (genuine satisfaction)

## Key Moves by Agent
1. **[3.2s] Validation before solution**: "I hear you" acknowledged emotion before action
2. **[3.2s] Immediate escalation**: Demonstrated urgency and commitment
3. **[throughout] Tone consistency**: Maintained calm, reassuring delivery even when customer was agitated

## Outcome
✅ Successfully de-escalated and resolved
- Customer sentiment: Angry → Tentative → Satisfied
- Relationship: Potentially preserved through effective handling
```

---

## Pattern 5: Conversation Metrics Dashboard

**Use case:** Generate quantitative and qualitative metrics for conversation quality, balance, and engagement.

### Example Prompt

````markdown
<conversation_metadata>
File: team_standup_2024_01_15.wav
Duration: 482 seconds (8m 2s)
Speakers: 4

Speaker Stats:
- spk_0: 145.3s (30.1%), 18 turns, avg 8.1s per turn
- spk_1: 98.2s (20.4%), 12 turns, avg 8.2s per turn
- spk_2: 167.4s (34.7%), 22 turns, avg 7.6s per turn
- spk_3: 71.1s (14.8%), 9 turns, avg 7.9s per turn

Turn Pattern:
spk_0 → spk_1 → spk_0 → spk_2 → spk_0 → spk_3 → spk_0 → spk_1 → ...
(spk_0 appears to be moderating/facilitating)
</conversation_metadata>

Generate a **conversation quality dashboard** with:

1. **Participation Balance**: Is speaking time evenly distributed? Any concerns?
2. **Engagement Pattern**: Are people interrupting, back-and-forth, or taking turns?
3. **Facilitator Effectiveness**: If there's a moderator (spk_0), are they enabling others to speak?
4. **Recommendations**: Specific, actionable improvements for next meeting

Focus on **inclusive, productive conversation dynamics**.
````

### Expected LLM Output

```markdown
## Conversation Quality Dashboard

### Participation Balance
📊 **Moderately Imbalanced**

| Speaker | Talk Time | % of Total | Assessment |
|---------|-----------|------------|------------|
| spk_2   | 167.4s    | 34.7%      | ⚠️ Dominant speaker |
| spk_0   | 145.3s    | 30.1%      | ✅ Appropriate for facilitator |
| spk_1   | 98.2s     | 20.4%      | ✅ Reasonable contribution |
| spk_3   | 71.1s     | 14.8%      | ⚠️ Underrepresented |

**Concern**: spk_3 had only 9 turns (50% fewer than average). Check if this was due to:
- Topic relevance (not their area)
- Disengagement
- Not being given space to contribute

### Engagement Pattern
✅ **Structured Turn-Taking**
- Facilitator (spk_0) consistently returned as "hub" after each speaker
- No evidence of interruptions or talking over
- Average turn length: 7.6-8.2s (consistent, suggests prepared updates)

**Pattern**: This looks like a **standup meeting** with structured check-ins.

### Facilitator Effectiveness
✅ **Good structure, opportunity to improve inclusion**

**Strengths**:
- Maintained consistent rhythm (spk_0 as hub between speakers)
- Gave each participant multiple turns (not just one-and-done)
- Equal avg turn length suggests no one was rushed

**Opportunities**:
- spk_3's lower participation (14.8%) could indicate facilitator didn't probe for input
- Consider explicit "Anyone else?" or "spk_3, anything to add?" prompts

### Recommendations
1. **For next meeting**: Explicitly invite spk_3 to contribute ("spk_3, let's hear your update next")
2. **Monitor**: Track spk_2's dominance over time—is this pattern or one-off?
3. **Format tweak**: Consider round-robin with explicit time boxes (2 min each) to ensure balance
4. **Check-in**: Ask spk_3 directly if they felt they had space to contribute

**Overall Score: 7/10** - Well-structured, but minor inclusion gap
```

---

## Pattern 6: Speaker Analytics with Turn Metadata

**Use case:** Prepend per-speaker analytics and turn-level metadata so the LLM can reason about interruptions, disfluency, and question frequency.

### Example Input (with analytics)

```json
{
  "speaker_stats": [
    {
      "speaker_id": "spk_0",
      "total_talk_time": 45.3,
      "num_turns": 12,
      "avg_turn_duration": 3.8,
      "interruptions_initiated": 2,
      "interruptions_received": 1,
      "question_turns": 4
    },
    {
      "speaker_id": "spk_1",
      "total_talk_time": 32.1,
      "num_turns": 10,
      "avg_turn_duration": 3.2,
      "interruptions_initiated": 1,
      "interruptions_received": 3,
      "question_turns": 1
    }
  ],
  "turns": [
    {
      "id": "turn_3",
      "speaker_id": "spk_1",
      "text": "Uh, I think we should push this to tomorrow?",
      "start": 12.0,
      "end": 14.5,
      "metadata": {
        "question_count": 1,
        "interruption_started_here": true,
        "avg_pause_ms": 60.0,
        "disfluency_ratio": 0.42
      }
    }
  ]
}
```

### Example Prompt

````markdown
<conversation_analytics>
Speaker stats summary:
- spk_0: 45.3s talk time across 12 turns; 2 interruptions started, 1 received; 4 question turns
- spk_1: 32.1s talk time across 10 turns; 1 interruption started, 3 received; 1 question turn

Focus:
- Which speaker is most interrupted?
- Which turns show high disfluency (metadata.disfluency_ratio > 0.35)?
</conversation_analytics>

<conversation_turns>
[spk_1 | interruption_started_here=true | disfluency=0.42] Uh, I think we should push this to tomorrow?
[spk_0 | interruption_started_here=false | disfluency=0.08] Let's keep today and remove scope instead.
</conversation_turns>

Provide:
1. Who dominates the conversation vs. who is interrupted most (cite speaker_stats).
2. List turns with interruptions (`interruption_started_here=true`) and high disfluency (`disfluency_ratio>0.35`) with timestamps.
3. Coaching tips to reduce interruptions and disfluency.
````

### Expected LLM Output

```markdown
## Participation + Interruptions
- spk_0 leads (45.3s, 12 turns); spk_1 is interrupted most (3 received vs 1 initiated).

## High-Disfluency / Interruption Turns
- [12.0s-14.5s] spk_1: disfluency=0.42, interruption_started_here=true — struggles to finish thought.

## Coaching
- Invite spk_1 to finish before responding; pause 0.5s after they start speaking.
- Ask spk_1 to restate key points without fillers; offer guided questions to reduce disfluency.
```

---

## Best Practices

### 1. Choose the Right Rendering Level

**Segment-level**: Use for detailed analysis (e.g., coaching, sentiment shifts)
```python
render_with_audio_cues(transcript)  # Every segment + audio states
```

**Turn-level**: Use for summaries, action items, meeting notes
```python
render_conversation_for_llm(transcript, include_audio_cues=False)  # Cleaner, grouped by speaker
```

**Speaker-level**: Use for participation metrics, balance analysis
```python
render_speaker_stats(transcript)  # Just aggregates
```

### 2. Always Include Timestamps

LLMs are better at "citing their work" when you give them timestamps:
```
[45.2s] Agent identifies account lock  ← LLM can reference this
```

### 3. Structure Your Requests

LLMs perform better with clear, numbered requests:
```markdown
Analyze this conversation and provide:
1. Summary (2-3 sentences)
2. Key moments (with timestamps)
3. Action items (table format)
```

### 4. Use Context Windows Wisely

For long conversations (>30 min), consider:
- **Chunking**: Send turns 0-20, 21-40, etc. with separate prompts
- **Hierarchical summarization**: Summarize chunks first, then summarize summaries
- **Key moments only**: Filter to turns with audio state changes or keywords

### 5. Validate LLM Outputs

LLMs can hallucinate. For production systems:
- **Cross-reference timestamps**: Check that [45.2s] actually exists in transcript
- **Speaker ID validation**: Ensure spk_0, spk_1 are real speaker IDs
- **Numeric sanity checks**: If LLM says "3 action items", count them

---

## Advanced Patterns

### Multi-Modal Prompting (Audio + Text)

If your LLM supports audio input (e.g., GPT-4V with audio, Claude with audio), you can:

1. Send the **raw audio file** for acoustic analysis
2. Send the **JSON transcript** for structured reasoning
3. Ask the LLM to **cross-validate**: "Does your audio analysis match the `audio_state` features in the JSON?"

This is useful for debugging slower-whisper's acoustic feature extraction or for high-stakes analysis.

### Chained Analysis (Multi-Step Reasoning)

For complex tasks (e.g., sales coaching), break into stages:

1. **Pass 1**: Extract facts (who said what, when)
2. **Pass 2**: Analyze sentiment and tone shifts
3. **Pass 3**: Generate coaching feedback based on passes 1+2

Each pass produces JSON output that feeds the next stage.

### Real-Time Streaming

For live transcription + analysis:

1. slower-whisper runs in chunks (e.g., 30s segments)
2. Send each chunk's JSON to LLM as it's produced
3. LLM maintains running summary/dashboard
4. At end of call, final pass for comprehensive analysis

---

## Example: Complete End-to-End Workflow

**Working example:** See [`examples/llm_integration/summarize_with_diarization.py`](../examples/llm_integration/summarize_with_diarization.py) for a complete, runnable script.

**Using slower-whisper v1.1+ with built-in rendering API:**

```python
from transcription import (
    load_transcript,
    render_conversation_for_llm,
)
import anthropic
import os

# 1. Assume transcripts were generated via CLI:
#    $ slower-whisper transcribe --enable-diarization --min-speakers 2 --max-speakers 2 \
#                                  --enable-prosody --enable-emotion

# 2. Load transcript from JSON
transcript = load_transcript("whisper_json/support_call_001.json")

# 3. Render for LLM using built-in API
context = render_conversation_for_llm(
    transcript,
    mode="turns",              # Turn-level structure
    include_audio_cues=True,   # Include prosody/emotion cues
    include_timestamps=True,   # Add [HH:MM:SS] timestamps
    include_metadata=True      # Include conversation header
)

# 4. Send to Claude for analysis
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=2048,
    messages=[{
        "role": "user",
        "content": f"""
<call_transcript>
{context}
</call_transcript>

You are a customer support quality analyst. Analyze this call and provide:

1. **Summary** (2-3 sentences)
2. **Quality Score** (1-10 with justification)
3. **Coaching Feedback** (2-3 actionable improvements for the agent)
4. **Customer Sentiment** (start, middle, end - note any shifts indicated by audio cues)

Use timestamps to reference specific moments.
"""
    }]
)

print(response.content[0].text)
```

**Key advantages of the built-in API:**
- No need to write custom rendering functions
- Consistent formatting across all your applications
- Supports both turn-level and segment-level rendering
- Handles audio cue aggregation automatically
- Type-safe with proper Transcript objects

---

## Pattern 7: Streaming Semantics for Real-Time Analysis

**Use case:** Process live conversation transcription in real-time, applying semantic tagging to finalized segments as they arrive, and using those tags to route each turn to the appropriate LLM analysis prompt (e.g., escalation detection, pricing discussion, churn risk).

### Architecture Overview

```
┌─────────────────┐
│   ASR Stream    │ (e.g., faster-whisper in chunked mode)
│  (text chunks)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│   StreamingSession          │ Aggregates chunks into segments
│   (state machine)           │ based on speaker continuity
└────────┬────────────────────┘
         │
    ┌────┴────┬─────────────────┐
    │          │                 │
    ▼          ▼                 ▼
PARTIAL     FINAL_SEGMENT    FINAL_SEGMENT
(UI update)  (emit event)     (emit event)
             │                 │
             ▼                 ▼
        ┌──────────────────────────────┐
        │ KeywordSemanticAnnotator     │ Attach semantic tags:
        │ (rule-based tagging)         │ - escalation_tags
        └─────┬────────────────────────┘ - churn_tags
              │ (tags: {"risk_level",  │ - pricing_tags
              │          "keywords",   │ - action_items
              │          ...})         │
              ▼                         │
        ┌──────────────────────────────┐
        │ Prompt Router                │ Select LLM prompt template
        │ (tag → prompt mapping)       │ based on detected tags
        └─────┬────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ LLM Analysis                 │ Real-time feedback:
        │ (via selected prompt)        │ - Escalation alert
        │ (streaming or batch)         │ - Action items
        └──────────────────────────────┘ - Customer sentiment
```

**Key principle**: Semantic tagging runs only on **finalized segments**, not partial/interim chunks, so tags remain stable and reliable for routing decisions.

### Example Input (Streaming ASR Chunks)

ASR produces chunks with timing and speaker info:

```json
{
  "chunks": [
    {
      "start": 0.0,
      "end": 2.1,
      "text": "Hi, I need to cancel my account.",
      "speaker_id": "spk_1"
    },
    {
      "start": 2.5,
      "end": 5.3,
      "text": "I'm unhappy with the pricing and would like to switch to a competitor.",
      "speaker_id": "spk_1"
    },
    {
      "start": 5.8,
      "end": 8.2,
      "text": "Let me escalate this to my supervisor right away.",
      "speaker_id": "spk_0"
    },
    {
      "start": 8.5,
      "end": 12.3,
      "text": "I can offer you a 30% discount if you stay with us.",
      "speaker_id": "spk_0"
    }
  ]
}
```

### Pattern Implementation

Here's a complete, runnable streaming analysis engine:

```python
"""
Real-time streaming semantics for conversation intelligence.

This pattern demonstrates:
1. Ingesting ASR chunks in order
2. Emitting finalized segments when speaker changes or gap occurs
3. Running semantic tagging on finalized segments
4. Routing to context-appropriate LLM prompts based on detected tags
5. Streaming analysis results back to the caller
"""

from typing import Callable, Iterator
from dataclasses import dataclass
import json

from transcription import (
    Segment,
    Transcript,
    StreamConfig,
    StreamingSession,
    StreamChunk,
    StreamEventType,
    KeywordSemanticAnnotator,
)
import anthropic


@dataclass
class StreamingAnalysisResult:
    """Result of analyzing a single turn."""

    segment_id: int
    start: float
    end: float
    text: str
    speaker_id: str | None
    semantic_tags: dict
    llm_analysis: str


class StreamingSemanticAnalyzer:
    """
    Orchestrates streaming transcription → semantic tagging → LLM analysis.

    Example usage:
        analyzer = StreamingSemanticAnalyzer()
        for result in analyzer.analyze_stream(asr_chunks):
            print(f"[{result.start}s] Tags: {result.semantic_tags}")
            print(f"Analysis: {result.llm_analysis[:100]}...")
    """

    def __init__(self, enable_llm: bool = True):
        self.session = StreamingSession(StreamConfig(max_gap_sec=1.0))
        self.annotator = KeywordSemanticAnnotator()
        self.final_segments: list[Segment] = []
        self.enable_llm = enable_llm
        self.llm_client = anthropic.Anthropic() if enable_llm else None

    def analyze_stream(
        self, chunks: list[dict], on_partial: Callable | None = None
    ) -> Iterator[StreamingAnalysisResult]:
        """
        Process ASR chunks and yield analysis results for finalized segments.

        Args:
            chunks: List of ASR chunks with {"start", "end", "text", "speaker_id"}
            on_partial: Optional callback for partial segment updates (UI hints)

        Yields:
            StreamingAnalysisResult for each finalized segment
        """
        for chunk in chunks:
            # Feed chunk into streaming state machine
            stream_chunk: StreamChunk = {
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
                "speaker_id": chunk.get("speaker_id"),
            }

            events = self.session.ingest_chunk(stream_chunk)

            for event in events:
                if event.type == StreamEventType.PARTIAL_SEGMENT:
                    # Partial updates (good for live UI)
                    if on_partial:
                        on_partial(
                            segment_id=len(self.final_segments),
                            text=event.segment.text,
                            speaker=event.segment.speaker_id,
                        )

                elif event.type == StreamEventType.FINAL_SEGMENT:
                    # Finalized segment: run analysis
                    stream_seg = event.segment

                    # Hydrate into Transcript schema
                    final_segments_list = list(self.final_segments)
                    final_segments_list.append(
                        Segment(
                            id=len(self.final_segments),
                            start=stream_seg.start,
                            end=stream_seg.end,
                            text=stream_seg.text,
                            speaker={"id": stream_seg.speaker_id}
                            if stream_seg.speaker_id
                            else None,
                        )
                    )

                    # Run semantic annotation
                    transcript = Transcript(
                        file_name="live.wav",
                        language="en",
                        segments=final_segments_list,
                    )
                    annotated = self.annotator.annotate(transcript)

                    # Extract semantic tags for this turn
                    semantic_tags = annotated.annotations.get("semantic", {})

                    # Route to appropriate LLM prompt
                    llm_analysis = ""
                    if self.enable_llm:
                        llm_analysis = self._get_llm_analysis(
                            text=stream_seg.text,
                            speaker_id=stream_seg.speaker_id,
                            semantic_tags=semantic_tags,
                        )

                    result = StreamingAnalysisResult(
                        segment_id=len(self.final_segments),
                        start=stream_seg.start,
                        end=stream_seg.end,
                        text=stream_seg.text,
                        speaker_id=stream_seg.speaker_id,
                        semantic_tags=semantic_tags,
                        llm_analysis=llm_analysis,
                    )

                    self.final_segments.append(
                        Segment(
                            id=result.segment_id,
                            start=stream_seg.start,
                            end=stream_seg.end,
                            text=stream_seg.text,
                            speaker={"id": stream_seg.speaker_id}
                            if stream_seg.speaker_id
                            else None,
                        )
                    )

                    yield result

        # Finalize any remaining partial segment
        final_events = self.session.end_of_stream()
        for event in final_events:
            if event.type == StreamEventType.FINAL_SEGMENT:
                stream_seg = event.segment

                final_segments_list = list(self.final_segments)
                final_segments_list.append(
                    Segment(
                        id=len(self.final_segments),
                        start=stream_seg.start,
                        end=stream_seg.end,
                        text=stream_seg.text,
                        speaker={"id": stream_seg.speaker_id}
                        if stream_seg.speaker_id
                        else None,
                    )
                )

                transcript = Transcript(
                    file_name="live.wav",
                    language="en",
                    segments=final_segments_list,
                )
                annotated = self.annotator.annotate(transcript)
                semantic_tags = annotated.annotations.get("semantic", {})

                llm_analysis = ""
                if self.enable_llm:
                    llm_analysis = self._get_llm_analysis(
                        text=stream_seg.text,
                        speaker_id=stream_seg.speaker_id,
                        semantic_tags=semantic_tags,
                    )

                result = StreamingAnalysisResult(
                    segment_id=len(self.final_segments),
                    start=stream_seg.start,
                    end=stream_seg.end,
                    text=stream_seg.text,
                    speaker_id=stream_seg.speaker_id,
                    semantic_tags=semantic_tags,
                    llm_analysis=llm_analysis,
                )

                self.final_segments.append(
                    Segment(
                        id=result.segment_id,
                        start=stream_seg.start,
                        end=stream_seg.end,
                        text=stream_seg.text,
                        speaker={"id": stream_seg.speaker_id}
                        if stream_seg.speaker_id
                        else None,
                    )
                )

                yield result

    def _get_llm_analysis(
        self, text: str, speaker_id: str | None, semantic_tags: dict
    ) -> str:
        """
        Route to context-appropriate LLM prompt based on semantic tags.

        Example routes:
        - escalation_tags detected → escalation response template
        - churn_tags detected → retention template
        - pricing_tags detected → pricing negotiation template
        """
        if not self.enable_llm or not self.llm_client:
            return ""

        # Example: detect risk level and select prompt
        risk_tags = semantic_tags.get("escalation_tags", set())
        churn_tags = semantic_tags.get("churn_tags", set())
        pricing_tags = semantic_tags.get("pricing_tags", set())

        # Route based on detected tags
        if risk_tags and "escalation" in risk_tags:
            prompt = self._escalation_response_prompt(text, speaker_id)
        elif churn_tags:
            prompt = self._retention_prompt(text, speaker_id)
        elif pricing_tags:
            prompt = self._pricing_response_prompt(text, speaker_id)
        else:
            prompt = self._general_response_prompt(text, speaker_id)

        # Call LLM synchronously (streaming can use server-sent events in production)
        response = self.llm_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def _escalation_response_prompt(self, text: str, speaker_id: str | None) -> str:
        return f"""
Customer message (escalation detected): "{text}"

Provide a brief (1-2 sentences) de-escalation response that:
1. Acknowledges the concern
2. Shows immediate action

Keep response concise for real-time delivery.
"""

    def _retention_prompt(self, text: str, speaker_id: str | None) -> str:
        return f"""
Customer message (churn risk detected): "{text}"

Provide a brief (1-2 sentences) retention counter-offer that:
1. Acknowledges their intent to leave
2. Offers a specific value retention tactic

Keep response concise for real-time delivery.
"""

    def _pricing_response_prompt(self, text: str, speaker_id: str | None) -> str:
        return f"""
Customer message (pricing discussion): "{text}"

Provide a brief (1-2 sentences) pricing response that:
1. Validates their budget concern
2. Offers a negotiation path

Keep response concise for real-time delivery.
"""

    def _general_response_prompt(self, text: str, speaker_id: str | None) -> str:
        return f"""
Customer message: "{text}"

Provide a brief (1-2 sentences) acknowledgment response.
Keep response concise for real-time delivery.
"""
```

### Example Output

Running the streaming analyzer on the input chunks above produces real-time results:

```
Analyzing live conversation stream...

[0.0-2.1s] spk_1:
  Text: "Hi, I need to cancel my account."
  Tags: {'churn_tags': {'cancel'}}
  Routing: → Retention prompt
  Analysis: We'd love to keep you on board! Can I explore what's driving
    this decision? I may have some flexibility on pricing or features.

[2.5-5.3s] spk_1:
  Text: "I'm unhappy with the pricing and would like to switch to a competitor."
  Tags: {'churn_tags': {'switch', 'competitor'}, 'pricing_tags': {'pricing'}}
  Routing: → Retention + Pricing prompt (combined)
  Analysis: I completely understand—pricing is crucial. Before you go, let me
    check if our enterprise plan (20% savings) might work better for your use case.

[5.8-8.2s] spk_0:
  Text: "Let me escalate this to my supervisor right away."
  Tags: {'escalation_tags': {'escalation', 'supervisor'}}
  Routing: → Escalation response prompt
  Analysis: Excellent—escalating shows commitment. Here's our direct line
    for priority support: [contact info]. You'll hear from us within 30 min.

[8.5-12.3s] spk_0:
  Text: "I can offer you a 30% discount if you stay with us."
  Tags: {'pricing_tags': {'discount'}, 'action_items': {'offer'}}
  Routing: → Pricing response prompt
  Analysis: Strong retention move! Confirm the discount tier and lock in
    a 12-month agreement for maximum impact.

Stream complete. Summary:
- 4 turns analyzed
- Churn risk: HIGH (2 churn_tags segments)
- Escalation: 1 escalation event (handled)
- Recommended follow-up: Confirm pricing agreement and send written confirmation
```

### Best Practices for Streaming Semantics

1. **Segment finalization only**: Only run semantic tagging and LLM analysis on `FINAL_SEGMENT` events, not partials. This ensures tags are stable and production-ready.

2. **Speaker-aware chunking**: Configure `StreamConfig.max_gap_sec` based on your use case:
   - Call centers (high interruption): `0.5s` (finalize quickly)
   - Meetings (natural pauses): `1.5s` (group long pauses as same speaker)
   - One-on-one interviews: `2.0s` (allow for natural thinking time)

3. **Tag-driven routing**: Map semantic tags → prompt templates at initialization time to avoid LLM latency:
   ```python
   # Pre-load prompt templates (fast lookup)
   self.prompt_routes = {
       'escalation': self.escalation_template,
       'churn': self.retention_template,
       'pricing': self.pricing_template,
   }

   # Route in O(1) time
   for tag in semantic_tags:
       if tag in self.prompt_routes:
           prompt = self.prompt_routes[tag]
   ```

4. **Batch vs. streaming LLM calls**: For production:
   - **Streaming**: Use server-sent events (SSE) or WebSocket for real-time feedback to agents
   - **Batching**: Group 3-5 turns before LLM call to reduce API overhead and latency
   - Example with batching:
     ```python
     batch_size = 3
     batch_results = []
     for result in analyzer.analyze_stream(chunks):
         batch_results.append(result)
         if len(batch_results) >= batch_size:
             llm_analysis = self._batch_analyze(batch_results)
             for r in batch_results:
                 yield r._replace(llm_analysis=llm_analysis)
             batch_results = []
     ```

5. **Partial events for UI**: Use `on_partial` callback to update agent dashboards in real-time without waiting for finalization:
   ```python
   def update_ui(segment_id, text, speaker):
       # Push live transcription to frontend
       emit('transcription_update', {'segment_id': segment_id, 'text': text})

   for result in analyzer.analyze_stream(chunks, on_partial=update_ui):
       emit('analysis_ready', result.semantic_tags)
   ```

6. **Error handling**: Wrap LLM calls in try/except to maintain streaming even if analysis fails:
   ```python
   try:
       llm_analysis = self._get_llm_analysis(...)
   except anthropic.APIError as e:
       logger.error(f"LLM analysis failed: {e}")
       llm_analysis = "[Analysis unavailable]"
       yield result._replace(llm_analysis=llm_analysis)
   ```

### When to Use This Pattern

**Use Pattern 7 (Streaming Semantics) when:**

- Processing live conversation with real-time agent feedback needed
- Routing decisions must happen mid-call (escalation detection, topic routing)
- Latency-sensitive: Agent needs LLM suggestions within 2-5 seconds
- Volume-high: Need to batch LLM calls to manage API costs
- Multi-prompt: Different conversation types need different analysis templates

**Use Pattern 1-6 instead when:**

- Post-call analysis (recording already complete)
- Batch processing (night job over call archives)
- Latency not critical (e.g., coaching report generation)
- Single, unified prompt for all conversations

---

## Schema Reference

For full JSON schema details, see:
- [`transcription/schemas/transcript-v2.schema.json`](../transcription/schemas/transcript-v2.schema.json)
- [`docs/SPEAKER_DIARIZATION.md`](SPEAKER_DIARIZATION.md) - Diarization features
- [`docs/AUDIO_ENRICHMENT.md`](AUDIO_ENRICHMENT.md) - Prosody and emotion features

---

## Feedback and Improvements

These patterns are **living examples**. If you develop new prompt patterns or find improvements:

1. Open an issue with your pattern and use case
2. Contribute examples via PR to this doc
3. Share results (LLM performance, task accuracy, etc.)

The goal is to build a shared library of proven prompts for conversation intelligence.
