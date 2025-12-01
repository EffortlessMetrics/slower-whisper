# Dogfood Notes – v1.1.0

Notes from testing slower-whisper v1.1.0 with sample datasets before public release.

---

## Test Session Template

Copy this template for each sample you test:

```markdown
### Sample: [filename.wav]
**Date:** YYYY-MM-DD
**Source:** [Dataset name / source]

#### Setup
- Model: [large-v3 / medium / etc.]
- Device: [cuda / cpu]
- Min speakers: [N]
- Max speakers: [N]

#### Diarization Quality
- [ ] Speaker count correct? (expected: X, got: Y)
- [ ] Segments mostly correct? (spot-check 10 random)
- [ ] Turn structure sensible? (A-B-A-B pattern)
- [ ] No phantom speakers? (no unexpected spk_N)

#### LLM Rendering (`summarize_with_diarization.py`)
- [ ] Speaker labels clear? (human names vs spk_0/spk_1)
- [ ] Summary accurate? (doesn't misattribute dialogue)
- [ ] Audio cues helpful? (or too verbose?)
- [ ] Timestamps useful? (for temporal context)

#### Issues Found
- (List specific problems with timestamps, examples)

#### Notes
- (General observations, edge cases, suggestions)

#### Stats Output
```
[Paste output from: uv run python scripts/diarization_stats.py whisper_json/file.json]
```
```

---

## Session 1: Mini Speaker Diarization Test

### Sample: mini_diarization_test.wav
**Date:** [TBD - run your first test]
**Source:** [Kaggle Mini Speaker Diarization](https://www.kaggle.com/datasets/wiradkp/mini-speech-diarization)

#### Setup
- Model: large-v3
- Device: cuda (or cpu if no GPU)
- Min speakers: 2
- Max speakers: 2

#### Diarization Quality
- [ ] Speaker count correct? (expected: 2, got: ___)
- [ ] Segments mostly correct? (spot-check 10 random)
- [ ] Turn structure sensible? (A-B-A-B pattern)
- [ ] No phantom speakers? (no unexpected spk_N)

#### LLM Rendering (`summarize_with_diarization.py`)
- [ ] Speaker labels clear? (Student/Professor vs spk_0/spk_1)
- [ ] Summary accurate? (doesn't misattribute dialogue)
- [ ] Audio cues helpful? (or too verbose?)
- [ ] Timestamps useful? (for temporal context)

#### Issues Found
- (List specific problems after running)

#### Notes
- (Observations after running)

#### Stats Output
```
[Run: uv run python scripts/diarization_stats.py whisper_json/mini_diarization_test.json]
```

---

## Summary: Issues to Address

After testing 1-2 samples, triage findings here:

### Quick Wins (1.1.1 candidates)
- [ ] Doc clarifications (e.g., "For noisy audio, use --min-speakers=1")
- [ ] Default tweaks (e.g., audio_cues=False in compact rendering)
- [ ] Error message improvements (e.g., clearer HF_TOKEN instructions)

### Real Features (1.2+ roadmap)
- [ ] Speaker overlap detection
- [ ] Per-speaker prosody baselines
- [ ] AMI corpus benchmarking
- [ ] Turn-level analytics (interruptions, questions)

### Blockers (Must fix before public release)
- [ ] (List any critical bugs or UX issues)

---

## Decision: What to Release

After dogfooding:

- [ ] **Ship 1.1.0 as-is** - No major issues, ready to publish GitHub Release
- [ ] **Polish → 1.1.1** - Quick wins worth fixing first, then publish 1.1.1
- [ ] **Defer** - Systemic issues found, need deeper work before public release

**Rationale:** (Fill in after testing)
