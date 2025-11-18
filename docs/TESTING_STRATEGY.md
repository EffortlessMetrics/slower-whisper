# Testing Strategy

**Last Updated:** 2025-11-17
**Status:** Active

This document defines "good enough" quality thresholds and testing strategies for each layer of slower-whisper.

---

## Philosophy

**Testing at slower-whisper follows three principles:**

1. **Stability over perfection** — We don't need SOTA accuracy; we need consistent, reliable behavior
2. **BDD scenarios are contracts** — Breaking tests = breaking user expectations
3. **Minimal but sufficient** — Test what matters, not everything possible

---

## Layer 1: ASR (Whisper)

**Responsibility:** Transcribe audio to text with timestamps.

### What We Test

**Correctness:**
- ✅ Text accuracy (Word Error Rate - WER)
- ✅ Timestamp accuracy (segment start/end reasonable)
- ✅ Language detection
- ✅ No crashes on edge cases (silence, noise, very short/long files)

**Performance:**
- ✅ Processing speed (realtime factor)
- ✅ Memory usage (doesn't OOM on long files)

### Quality Thresholds

| Metric | Threshold | Dataset |
|--------|-----------|---------|
| WER | < 15% | LibriSpeech test-clean |
| Timestamp drift | < 500ms | Synthetic aligned data |
| Language detection accuracy | > 95% | Multi-language test set |
| Crash rate | 0% | Edge case battery (100 files) |

**Edge case battery:**
- Silent audio (0s, 5s, 60s)
- Pure noise
- Very short (<1s)
- Very long (>2 hours)
- Multiple languages in one file
- Music + speech
- Heavily accented speech

### Test Implementation

**Unit tests:**
```bash
uv run pytest tests/test_asr_engine.py -v
```

**BDD scenarios:**
```gherkin
Scenario: Transcribe clean English audio
  Given a clear English audio file "test_clean.wav"
  When I transcribe with model "base"
  Then the transcript contains expected text
  And timestamps are within 500ms of reference
```

**Regression tests:**
- Frozen test set (20 files across languages)
- Run on every release
- WER must not degrade >2% from baseline

---

## Layer 2: Speaker Diarization

**Responsibility:** Determine *who* spoke *when*.

### What We Test

**Correctness:**
- ✅ Speaker count (detect correct number of speakers)
- ✅ Diarization Error Rate (DER)
- ✅ Speaker consistency (same person gets same ID across segments)
- ✅ Confidence scores are calibrated

**Robustness:**
- ✅ Single-speaker edge case (1 speaker detected, not 2+)
- ✅ Overlapping speech (marked as overlap, not attributed to one speaker only)
- ✅ Silence segments (no speaker assigned)

### Quality Thresholds

| Metric | Threshold | Dataset |
|--------|-----------|---------|
| DER (Diarization Error Rate) | < 0.25 | AMI Meeting Corpus (10 files) |
| Speaker count accuracy | > 90% | Synthetic test set (2-4 speakers known) |
| Confidence calibration | ECE < 0.15 | AMI subset with manual labels |

**DER components:**
- False Alarm (speech detected when none)
- Missed Speech (speech not detected)
- Speaker Error (wrong speaker assigned)

### Test Implementation

**Synthetic tests:**
```python
# tests/test_diarization.py
def test_two_speaker_synthetic():
    """Two distinct voices → exactly 2 speakers detected."""
    audio = generate_two_speaker_audio()  # TTS with different voices
    transcript = transcribe_and_diarize(audio)
    assert len(transcript.speakers) == 2
    assert all(s.cluster_confidence > 0.7 for s in transcript.speakers)
```

**BDD scenarios:**
```gherkin
Scenario: Two-speaker conversation
  Given a 2-speaker audio file "interview_2spk.wav"
  When I transcribe with diarization enabled
  Then exactly 2 speakers are detected
  And each segment has a speaker ID
  And speaker confidence > 0.7 for 80%+ of segments

Scenario: Single-speaker monologue
  Given a single-speaker audio file "lecture.wav"
  When I transcribe with diarization enabled
  Then exactly 1 speaker is detected
```

**Real-world benchmark:**
```bash
# benchmarks/eval_diarization.py
uv run python benchmarks/eval_diarization.py \
  --dataset ami_subset \
  --metric DER \
  --threshold 0.25
```

---

## Layer 2: Prosody Extraction

**Responsibility:** Extract pitch, energy, speaking rate, pauses from audio.

### What We Test

**Correctness:**
- ✅ Pitch (rising/falling/flat contours detected correctly)
- ✅ Energy (loud vs soft segments distinguished)
- ✅ Speaking rate (fast vs slow speech distinguished)
- ✅ Pauses (silent regions detected accurately)

**Robustness:**
- ✅ Noisy audio (graceful degradation, not crash)
- ✅ Music/non-speech (detected as unreliable)
- ✅ Very short segments (<0.5s)

### Quality Thresholds

| Feature | Correctness Test | Threshold |
|---------|-----------------|-----------|
| Pitch contour | Synthetic rising/falling tones | 100% correct classification |
| Energy levels | Synthetic loud (+6dB) vs soft (-6dB) | Levels differ by ≥10 dB |
| Speaking rate | Synthetic fast (180 wpm) vs slow (80 wpm) | Differ by ≥50 wpm |
| Pauses | Synthetic 500ms silences | Detected with ±100ms accuracy |

### Test Implementation

**Synthetic tests:**
```python
# tests/test_prosody.py
def test_rising_pitch_contour():
    """Synthetic tone rising from 200Hz to 400Hz → contour='rising'."""
    audio = generate_rising_tone(start_hz=200, end_hz=400, duration=2.0)
    prosody = extract_prosody(audio)
    assert prosody["pitch"]["contour"] == "rising"

def test_loud_vs_soft():
    """Loud segment (+6dB) vs soft (-6dB) → energy levels differ."""
    loud_audio = generate_noise(amplitude=0.5)
    soft_audio = generate_noise(amplitude=0.1)

    prosody_loud = extract_prosody(loud_audio)
    prosody_soft = extract_prosody(soft_audio)

    assert prosody_loud["energy"]["db_rms"] - prosody_soft["energy"]["db_rms"] > 10
```

**BDD scenarios:**
```gherkin
Scenario: High-energy angry speech
  Given a high-energy angry sample "angry_mono.wav"
  When I extract prosody features
  Then energy level is "loud"
  And pitch level is "high"
  And speech rate level is "fast"
```

**Per-Speaker Baseline Test:**
```gherkin
Scenario: Per-speaker prosody baselines
  Given a 2-speaker conversation
  And Speaker A has low baseline pitch (120 Hz median)
  And Speaker B has high baseline pitch (220 Hz median)
  When I enrich the transcript with prosody
  Then Speaker A at 160 Hz is labeled "high pitch"
  And Speaker B at 200 Hz is labeled "low pitch"
```

---

## Layer 2: Emotion Recognition

**Responsibility:** Classify emotional state from audio.

### What We Test

**Correctness:**
- ✅ Dimensional emotion (valence, arousal) aligned with ground truth
- ✅ Categorical emotion (happy/sad/angry) accuracy on labeled data
- ✅ Confidence scores calibrated

**Robustness:**
- ✅ Minimum segment length (< 0.5s handled gracefully)
- ✅ Non-speech audio (low confidence or "neutral" label)

### Quality Thresholds

| Metric | Threshold | Dataset |
|--------|-----------|---------|
| 4-class emotion accuracy | > 60% | IEMOCAP (happy/sad/angry/neutral) |
| Valence correlation | r > 0.6 | IEMOCAP dimensional labels |
| Arousal correlation | r > 0.6 | IEMOCAP dimensional labels |
| Confidence calibration (ECE) | < 0.20 | IEMOCAP subset |

**Note:** We don't aim for SOTA (>80% accuracy). We aim for **stable, consistent** emotion estimates that are **better than text-only inference**.

### Test Implementation

**Labeled dataset tests:**
```bash
# benchmarks/eval_emotion.py
uv run python benchmarks/eval_emotion.py \
  --dataset iemocap_subset \
  --metric accuracy \
  --threshold 0.60
```

**BDD scenarios:**
```gherkin
Scenario: Clearly angry segment
  Given an obviously angry audio sample "angry_speech.wav"
  When I extract emotion features
  Then valence is negative (< 0)
  And arousal is high (> 0.6)
  And categorical label is "angry" OR "frustrated"

Scenario: Calm neutral segment
  Given a calm neutral sample "neutral_speech.wav"
  When I extract emotion features
  Then valence is near zero (-0.3 to +0.3)
  And arousal is low (< 0.4)
```

---

## Layer 2: Turn Structure

**Responsibility:** Group segments by speaker into turns.

### What We Test

**Correctness:**
- ✅ Turn count matches speaker changes
- ✅ Turn boundaries align with speaker transitions
- ✅ No overlapping turns (unless overlap explicitly detected)

**Metadata:**
- ✅ Question detection (turns ending with `?` flagged)
- ✅ Interruption detection (overlapping turn starts)

### Quality Thresholds

| Test | Expected Behavior |
|------|-------------------|
| A-B-A-B conversation | 4 turns detected |
| Single speaker (AAAA) | 1 turn detected |
| Overlapping speech (A+B) | Overlap flag set on both turns |

### Test Implementation

**Synthetic tests:**
```python
# tests/test_turns.py
def test_turn_grouping():
    """A-B-A-B speaker pattern → 4 turns."""
    transcript = create_synthetic_transcript(speakers=["A", "B", "A", "B"])
    turns = build_turns(transcript)
    assert len(turns) == 4
    assert turns[0].speaker_id == "A"
    assert turns[1].speaker_id == "B"
```

**BDD scenarios:**
```gherkin
Scenario: Speaker change creates new turn
  Given segments [spk_A, spk_A, spk_B, spk_B, spk_A]
  When I build turns
  Then 3 turns are created
  And turn 0 has speaker spk_A with 2 segments
  And turn 1 has speaker spk_B with 2 segments
  And turn 2 has speaker spk_A with 1 segment
```

---

## Integration: Evaluation Harness (v1.2+)

**Purpose:** Measure **downstream utility** of enrichment for LLM tasks.

### What We Test

**Tasks:**
1. Speaker-aware summarization
2. Action item extraction with responsibility (who must do what)
3. Conflict/objection detection

### Quality Thresholds

| Condition | Task | Metric | Threshold |
|-----------|------|--------|-----------|
| Text only | Summarization | F1 on action items | Baseline |
| Text + speaker | Summarization | F1 on action items | +10% vs baseline |
| Text + speaker + stats | Summarization | F1 on action items | +15% vs baseline |

**Evaluation method:**

**Option A: LLM-as-judge (MVP for v1.2)**
- 20-50 labeled segments
- Two summaries generated (text-only vs enriched)
- GPT-4o-mini judges which is better
- Success: Enriched wins >60% of pairwise comparisons

**Option B: Human evaluation (future)**
- 3-5 human raters
- Score summaries on 1-5 scale for:
  - Accuracy
  - Completeness
  - Speaker attribution correctness

### Test Implementation

```bash
# benchmarks/eval_speaker_utility.py
uv run python benchmarks/eval_speaker_utility.py \
  --dataset ami_subset \
  --task summarization \
  --judge gpt-4o-mini \
  --output results/speaker_utility_v1.2.json
```

**Success criteria:**
- Enriched condition wins >60% (LLM-as-judge)
- At least 3 clear examples where speaker info helps

---

## BDD Scenarios as Behavioral Contracts

**All BDD scenarios are guaranteed behaviors.**

Locations:
- **Library BDD**: `tests/features/` (transcription, enrichment)
- **API BDD**: `features/` (REST API black-box tests)

**Running BDD tests:**
```bash
# Library BDD scenarios
uv run pytest tests/steps/ -v

# API BDD scenarios (smoke tests = hard gate)
uv run pytest features/ -v -m "api and smoke"

# API functional tests (recommended, not required)
uv run pytest features/ -v -m "api and functional"
```

**Contract enforcement:**
- All BDD scenarios must pass before merging
- Breaking a scenario = breaking the contract = requires versioning discussion
- See `docs/BDD_IAC_LOCKDOWN.md` for versioning policy

---

## Continuous Integration

**CI pipeline runs:**

1. ✅ Code quality (ruff format, ruff check)
2. ✅ Type checking (mypy)
3. ✅ Unit tests (pytest -m "not slow")
4. ✅ Integration tests (pytest tests/test_*integration*)
5. ✅ Library BDD scenarios (pytest tests/steps/)
6. ✅ API BDD scenarios - smoke (pytest features/ -m "api and smoke")
7. ✅ API BDD scenarios - functional (pytest features/ -m "api and functional", continue-on-error)
8. ✅ Docker smoke tests (build + `--help` check)
9. ✅ K8s validation (kubectl apply --dry-run)

**All must pass before merge** (except functional API tests, which are recommended but not required).

---

## Performance Benchmarks

**Track processing speed over time:**

```bash
# Benchmark full pipeline on standard test file
uv run python benchmarks/benchmark_pipeline.py \
  --file test_audio_10min.wav \
  --output results/benchmark_$(date +%Y%m%d).json
```

**Metrics tracked:**
- Realtime factor (processing time / audio duration)
- Memory peak (MB)
- GPU utilization (%)

**Regression thresholds:**
- Realtime factor must not degrade >10% between versions
- Memory must not increase >20% between versions

---

## Release Checklist

Before releasing any version:

- [ ] All unit tests pass
- [ ] All BDD scenarios pass (library + API smoke)
- [ ] Benchmarks run without degradation
- [ ] DER < 0.25 on AMI subset (if diarization changed)
- [ ] Emotion accuracy > 60% on IEMOCAP subset (if emotion changed)
- [ ] Prosody synthetic tests pass 100%
- [ ] Docker smoke tests pass
- [ ] K8s manifests validate
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

---

## Test Data Sources

**Synthetic:**
- Generated via `tests/fixtures/generate_synthetic.py`
- Controlled signals (rising tones, loud/soft, fast/slow speech)
- TTS with different voices (2-4 speakers)

**Public datasets:**
- **LibriSpeech**: ASR accuracy (test-clean)
- **AMI Meeting Corpus**: Diarization, turn structure
- **IEMOCAP**: Emotion recognition
- **Common Voice**: Multi-language, accents

**Internal test set:**
- 100 edge-case files (silence, noise, very short/long, etc.)
- Frozen regression set (20 files, representative)

---

## When "Good Enough" is Enough

**We optimize for:**
- ✅ Stability (same input → same output across versions)
- ✅ Consistency (similar quality across languages, speakers, domains)
- ✅ Robustness (no crashes on edge cases)

**We explicitly do NOT optimize for:**
- ❌ SOTA accuracy on academic benchmarks
- ❌ Beating commercial APIs on WER
- ❌ Perfect emotion classification

**Rationale:**

slower-whisper is **infrastructure**, not a research project. Users care about:
1. Does it work reliably?
2. Can I build on it without it breaking?
3. Is it better than text-only?

If the answer to all three is "yes," we've succeeded.

---

## Questions or Contributions?

- **Testing issues**: Open GitHub issue with `testing` label
- **New test ideas**: Open discussion or PR
- **Benchmark datasets**: Suggest in GitHub discussions

---

**Document History:**
- 2025-11-17: Initial testing strategy (v1.0.0 baseline)
