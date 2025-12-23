# Turn, Speaker Stats, and Enrichment Module Integration Analysis

## Executive Summary

The integration between `turns.py`, `turns_enrich.py`, and `speaker_stats.py` forms a **three-stage enrichment pipeline** that converts diarization output into conversation-level analytics. While the architecture is modular and intentional, there are several inconsistencies in turn handling and opportunities for better typing.

---

## 1. Data Flow: Diarization → Turns → Speaker Stats → JSON

### Stage 1: Diarization (diarization.py)
- **Input**: Audio file + existing transcript with segments
- **Output**: Speaker-labeled segments + turns array

```
Audio → Diarizer.run() → list[SpeakerTurn]
         ↓
      assign_speakers() → transcript.segments[].speaker = {"id", "confidence"}
                       → transcript.speakers = [...]
         ↓
      build_turns() → transcript.turns = [dict, dict, ...]
```

**Key Function**: `assign_speakers(transcript, speaker_turns)` (diarization.py:349-456)
- Mutates `transcript.segments[].speaker` (dict with "id" and "confidence")
- Creates `transcript.speakers` array (list of speaker metadata)
- Returns transcript for chaining

### Stage 2: Turn Building (turns.py)
- **Input**: Transcript with speaker-labeled segments
- **Output**: Conversational turns grouped by speaker

```
transcript.segments (with speaker labels)
         ↓
    build_turns() → transcript.turns = [dict, dict, ...]
```

**Key Function**: `build_turns(transcript, pause_threshold=None)` (turns.py:45-125)
- Returns **list of dicts** (not dataclass instances)
- Each turn dict: `{id, speaker_id, start, end, segment_ids, text}`
- Line 103: `turns: list[Turn | dict[str, Any]] = []` — **type annotation shows mixed union**
- Line 112, 122: `turns.append(_finalize_turn(...))` — returns dict, not Turn
- **INCONSISTENCY 1**: Type annotation says `Turn | dict` but code only produces dicts

**Key Helper**: `_finalize_turn(turn_id, segments, speaker_id)` (turns.py:128-152)
- Returns **plain dict**, not Turn dataclass
- Manually constructs: `{"id", "speaker_id", "start", "end", "segment_ids", "text"}`

### Stage 3A: Turn Enrichment (turns_enrich.py)
- **Input**: Transcript with turns array (dicts)
- **Output**: Enriched turns with metadata

```
transcript.turns (list of dicts)
    ↓
enrich_turns_metadata(transcript)
    ↓
For each turn:
  1. turn_to_dict(turn, copy=True) → dict
  2. Extract segments by segment_ids
  3. Compute metadata:
     - question_count
     - interruption_started_here
     - avg_pause_ms
     - disfluency_ratio
  4. Attach metadata dict to turn
    ↓
transcript.turns = enriched_turns (now with "metadata" key)
```

**Key Function**: `enrich_turns_metadata(transcript)` (turns_enrich.py:50-117)
- Line 66: `turn_dict = turn_to_dict(turn, copy=True)` — **converts to dict**
- Line 93: `prev_dict = turn_to_dict(prev)` — **converts to dict**
- Line 107: `new_turn = dict(turn_dict)` — **creates copy**
- Line 108: Manually adds `metadata` key with dict
- **INCONSISTENCY 2**: Mutates `transcript.turns` by replacing dicts with enhanced dicts

### Stage 3B: Speaker Statistics (speaker_stats.py)
- **Input**: Transcript with turns (now with metadata) and segments
- **Output**: Per-speaker aggregates

```
transcript.turns (with metadata) + transcript.segments
    ↓
compute_speaker_stats(transcript)
    ↓
For each speaker:
  1. Collect segment durations
  2. Collect prosody values (pitch, energy)
  3. Collect sentiment/emotion values
  4. Group turns by speaker
  5. Count interruptions from turn.metadata
    ↓
Create SpeakerStats dataclass → convert to dict via .to_dict()
    ↓
transcript.speaker_stats = [dict, dict, ...]
```

**Key Function**: `compute_speaker_stats(transcript)` (speaker_stats.py:85-160)
- Line 99, 108, 111: `t_dict = turn_to_dict(t)` — **converts to dict**
- Line 100: `sid = t_dict.get("speaker_id") or "spk_0"` — reads from dict
- Line 109: `meta = t_dict.get("metadata") or {}` — reads metadata from dict
- Line 156: `stats_out.append(s.to_dict())` — converts SpeakerStats to dict
- Line 159: `transcript.speaker_stats = stats_out` — assigns list of dicts

### Stage 4: JSON Serialization (writers.py)
- **Input**: Transcript with turns + speaker_stats
- **Output**: JSON file

```
transcript
    ↓
write_json(transcript, out_path)
    ↓
Line 94: data["turns"] = [_to_dict(t) for t in transcript.turns]
Line 96: data["speaker_stats"] = [_to_dict(s) for s in transcript.speaker_stats or []]
    ↓
json.dumps(data) → output file
```

**Key Function**: `_to_dict(obj)` (writers.py:12-37)
- Handles dict passthrough (already dicts)
- Calls `.to_dict()` if available
- Falls back to `asdict()` for dataclasses
- **Lenient**: Returns unchanged object if conversion fails

---

## 2. Turn Conversion Helpers

### Primary Helper: `turn_to_dict()` (turn_helpers.py:11-43)

**Location**: Dedicated module `transcription/turn_helpers.py`

**Signature**:
```python
def turn_to_dict(t: Any, *, copy: bool = False) -> dict[str, Any]
```

**Behavior**:
- Dict → returns as-is or shallow copy based on `copy` param
- Object with `.to_dict()` → calls method
- Dataclass instance → converts via `asdict()`
- Other → raises `TypeError`

**Used in**:
- `turns_enrich.py` (lines 66, 93) — 2 calls
- `speaker_stats.py` (lines 99, 108, 111) — 3 calls
- `tests/test_turn_helpers.py` — comprehensive tests

**Contrast**:
- `_to_dict()` in writers.py (lenient, returns original on failure)
- `_as_dict()` in llm_utils.py (strict, raises on failure)

### Alternative Helper: `_normalize_turns()` (chunking.py:51-81)

**Location**: Embedded in `transcription/chunking.py`

**Behavior**:
- Handles both dataclass instances and dicts
- Extracts fields manually: `.id`, `.start`, `.end`, `.text`, `.segment_ids`, `.speaker_id`
- Returns normalized list of dicts
- **Does NOT use `turn_to_dict()`**

**Why Separate?**
- Chunk building needs to normalize turns to extract specific fields for chunking logic
- Doesn't need full `.to_dict()` conversion; just field extraction

---

## 3. Where turn_to_dict is Called

### Current Usage (53 references across 5 files):

**transcription/turns_enrich.py**:
- Line 12: Import
- Line 66: `turn_dict = turn_to_dict(turn, copy=True)` — initial copy
- Line 93: `prev_dict = turn_to_dict(prev)` — previous turn lookup

**transcription/speaker_stats.py**:
- Line 19: Import
- Line 99: `t_dict = turn_to_dict(t)` — speaker stats grouping
- Line 108: `t_dict = turn_to_dict(t)` — interruption counting
- Line 111: `prev = turn_to_dict(transcript.turns[idx - 1])` — previous turn

**tests/test_turn_helpers.py**:
- Lines 22-237: Comprehensive test suite (11 test functions)
- Tests dict passthrough, copying, dataclass conversion, error handling

**transcription/turn_helpers.py**:
- Line 11: Definition

**transcription/chunking.py**:
- Line 51: Defines **separate** `_normalize_turns()` function (does NOT use turn_to_dict)

### Notable Places That DON'T Use turn_to_dict:

**transcription/exporters.py** (lines 44-60):
```python
for idx, turn in enumerate(transcript.turns):
    if isinstance(turn, dict):
        turn_dict = dict(turn)  # manual copy
    elif hasattr(turn, "to_dict"):
        turn_dict = turn.to_dict()  # manual call
    else:
        turn_dict = {"id": str(idx)}  # fallback
```
- **Inconsistency**: Reimplements turn-to-dict logic locally

**transcription/llm_utils.py** (lines 317, 408):
```python
turn_dict = _as_dict(turn_item)  # uses _as_dict(), not turn_to_dict()
```
- Uses strict `_as_dict()` from llm_utils module instead

**transcription/writers.py** (line 94):
```python
data["turns"] = [_to_dict(t) for t in transcript.turns]
```
- Uses lenient `_to_dict()` from writers module

---

## 4. Inconsistencies in Turn Handling

### Inconsistency 1: Type Annotation vs. Reality
**Location**: turns.py:103-104

```python
turns: list[Turn | dict[str, Any]] = []  # Type says Turn | dict
# ...
turns.append(_finalize_turn(...))  # But _finalize_turn returns dict
```

**Problem**:
- Type annotation promises `Turn | dict` but code only produces dicts
- `_finalize_turn()` is hardcoded to return dict (lines 145-152)
- Never creates `Turn` dataclass instances

**Impact**: Type checkers will accept Turn instances, but at runtime only dicts exist.

---

### Inconsistency 2: Multiple Turn-to-Dict Implementations

**Location**: Three different converter functions

| Function | Module | Type | Behavior |
|----------|--------|------|----------|
| `turn_to_dict()` | turn_helpers.py | Strict | Raises TypeError on failure |
| `_normalize_turns()` | chunking.py | Hybrid | Manual field extraction, no to_dict() call |
| `_to_dict()` | writers.py | Lenient | Returns original object on failure |
| `_as_dict()` | llm_utils.py | Strict | Raises TypeError on failure |
| exporters.py inline | exporters.py | Manual | Reimplements logic |

**Problem**:
- Four different ways to convert turns in the codebase
- `turn_to_dict()` not used in exporters.py or llm_utils.py
- `_normalize_turns()` doesn't use `turn_to_dict()`

**Result**:
- Code duplication
- Inconsistent error handling
- Hard to maintain

---

### Inconsistency 3: Mutation of Turns During Enrichment
**Location**: turns_enrich.py:116

```python
for idx, turn in enumerate(transcript.turns or []):
    turn_dict = turn_to_dict(turn, copy=True)
    # ... compute metadata ...
    new_turn = dict(turn_dict)
    new_turn["metadata"] = {...}
    enriched_turns.append(new_turn)

transcript.turns = enriched_turns  # type: ignore[assignment]
```

**Problem**:
- `build_turns()` produces dicts
- `enrich_turns_metadata()` reads those dicts and produces NEW dicts with metadata
- Original turns are discarded and replaced
- Type annotation uses `# type: ignore[assignment]` (suppresses type error)

**Why It Happens**:
- `Turn` dataclass (models.py:68-107) has metadata field
- But turns from `build_turns()` are plain dicts without metadata
- Enrichment can't just call a method; must reconstruct

---

### Inconsistency 4: Turn Field Extraction Patterns

**turns.py** (_finalize_turn):
```python
segment_ids = [seg.id for seg in segments]
text = " ".join(texts)
return {
    "id": f"turn_{turn_id}",
    "speaker_id": speaker_id,
    # ...
}
```

**chunking.py** (_normalize_turns):
```python
if hasattr(turn, "id"):
    turn_id = turn.id or f"turn_{idx}"
    text = str(turn.text or "")
    segment_ids = list(turn.segment_ids or [])
elif isinstance(turn, dict):
    turn_id = str(turn.get("id") or f"turn_{idx}")
    text = str(turn.get("text", ""))
    segment_ids = list(turn.get("segment_ids", []) or [])
```

**Problem**:
- chunking.py duplicates field extraction logic
- Handles both dataclass and dict, but doesn't use turn_to_dict()
- Fragile: has to check for both attribute and dict access patterns

---

### Inconsistency 5: Speaker Stats Depends on Turn Metadata
**Location**: speaker_stats.py:109

```python
meta = t_dict.get("metadata") or {}
if idx > 0 and meta.get("interruption_started_here"):
```

**Dependency Chain**:
1. `build_turns()` creates basic turns
2. `enrich_turns_metadata()` adds metadata (optional)
3. `compute_speaker_stats()` REQUIRES metadata to exist

**Problem**:
- If turn enrichment is skipped, speaker stats incomplete
- No clear validation that metadata exists before computing stats
- api.py has to explicitly call both in correct order (lines 105-118)

---

## 5. Integration Points and Data Flow Diagram

```
diarization.py
  ├─ Diarizer.run(audio) → list[SpeakerTurn]
  └─ assign_speakers(transcript, speaker_turns)
       └─ transcript.segments[].speaker = {"id", "confidence"}
       └─ transcript.speakers = [...]
            ↓
turns.py
  └─ build_turns(transcript)
       └─ _finalize_turn() → dict (not Turn dataclass!)
       └─ transcript.turns = [dict, ...]
            ↓
turns_enrich.py
  └─ enrich_turns_metadata(transcript)
       ├─ turn_to_dict(turn, copy=True) [2 calls]
       └─ transcript.turns = [dict with metadata, ...]
            ↓
speaker_stats.py
  └─ compute_speaker_stats(transcript)
       ├─ turn_to_dict(t) [3 calls]
       ├─ Group by speaker
       ├─ Count interruptions
       └─ transcript.speaker_stats = [dict, ...]
            ↓
writers.py
  └─ write_json(transcript, path)
       ├─ _to_dict(turn) for each turn
       ├─ _to_dict(stat) for each stat
       └─ json.dumps()
```

---

## 6. Typing Issues and Opportunities

### Issue 1: Union Type Not Realized
**Location**: models.py:299, turns.py:103

```python
# models.py
turns: list[Turn | dict[str, Any]] | None = None

# turns.py
turns: list[Turn | dict[str, Any]] = []
```

**Reality**: Code only creates `dict`, never `Turn` instances

**Solution**:
- Change to `list[dict[str, Any]]` in turns.py
- Or start producing `Turn` instances in `_finalize_turn()`

---

### Issue 2: Type Ignore on Assignment
**Location**: turns_enrich.py:116

```python
transcript.turns = enriched_turns  # type: ignore[assignment]
```

**Problem**: Type checker sees mutation as invalid

**Root Cause**:
- Input is `list[Turn | dict]`
- Output is `list[dict]`
- Even though both are assignable to same type

**Solution**:
- Use consistent type throughout
- Or use proper type narrowing

---

### Issue 3: Missing Type Annotations in Helpers
**Location**: chunking.py:51, exporters.py:40

```python
def _collect_segments(transcript: Transcript, unit: str = "segments") -> list[dict[str, Any]]:
```

**Problem**: Parameter `unit` is a string, not a Literal type

**Solution**:
```python
from typing import Literal
def _collect_segments(transcript: Transcript, unit: Literal["segments", "turns"] = "segments") -> list[dict[str, Any]]:
```

---

### Issue 4: turn_to_dict() Not Used Everywhere
**Location**: exporters.py, chunking.py, llm_utils.py

**Problem**: Each module reimplements turn conversion logic

**Solution**:
- Make `turn_to_dict()` the canonical function
- Update all modules to use it
- Add to public API exports

---

## 7. Recommendations

### Short Term (v1.1)

1. **Resolve Type Union Inconsistency**
   - Change turns.py line 103 to `list[dict[str, Any]]`
   - Remove `# type: ignore` from turns_enrich.py:116
   - Update models.py Transcript.turns to be more specific

2. **Centralize Turn Conversion**
   - Export `turn_to_dict()` from turn_helpers.py in `__init__.py`
   - Update exporters.py to use `turn_to_dict()`
   - Update llm_utils.py to use `turn_to_dict()` with try/except

3. **Add Type Narrowing**
   - Use `Literal` for `unit` parameters in exporters.py, chunking.py
   - Add type guards for optional fields (metadata, audio_state)

### Medium Term (v1.2)

1. **Create Turn Dataclass Instances**
   - Option A: Have `_finalize_turn()` return `Turn` instances
   - Option B: Keep turns as dicts but properly type them
   - Decision: Need to evaluate downstream impact

2. **Consolidate Metadata Flow**
   - Make turn metadata required (not optional) in v1.2
   - Or create separate "enriched_turn" type
   - Document dependency order clearly

3. **Add Validation Layer**
   - Validate turns have required fields before speaker_stats
   - Validate speaker_stats only created when turns available
   - Add schema validation for turn structure

### Long Term (v1.3)

1. **TypedDict for Turn Structure**
   - Define: `class TurnDict(TypedDict, total=False): id, speaker_id, ...`
   - Use for all turn handling
   - Replaces `dict[str, Any]`

2. **Better Error Messages**
   - When turn metadata missing, raise clear error
   - When turn structure invalid, suggest fixes
   - Add debugging helpers

---

## 8. Summary of Current State

**What Works**:
- Clear modular pipeline stages
- Proper turn building from diarization
- Metadata enrichment working
- Speaker stats computation working
- JSON serialization correct

**What Needs Improvement**:
- Type inconsistencies between promises and reality
- Multiple competing conversion functions
- Missing centralization of turn handling
- Unclear dependencies between stages
- Some type checker suppression needed

**Risk Areas**:
- If Turn dataclass becomes required in future, major refactor needed
- Inconsistent handling could cause bugs if new consumers added
- No validation that enrichment stages completed in order
