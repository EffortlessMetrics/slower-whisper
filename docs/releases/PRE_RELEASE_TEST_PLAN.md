# Pre-Release Test Plan: v1.1.0

This is the focused, **manual** test plan for v1.1.0 before tagging the release. All automated tests already pass; this covers the integration scenarios that require real audio, API keys, and end-to-end validation.

---

## Test Environment Setup

### Required Dependencies

```bash
# In repo root
uv sync --extra full --extra diarization
```

### Required Credentials

```bash
# For diarization (pyannote.audio)
export HF_TOKEN=hf_...

# For LLM integration example
export ANTHROPIC_API_KEY=sk-ant-...
```

### Required System Tools

```bash
# Verify ffmpeg is installed
ffmpeg -version

# Verify CUDA (optional, for GPU acceleration)
nvidia-smi  # Should show GPU if available
```

---

## Test Scenarios

### Scenario 1: Basic Transcription (5-Minute Quickstart Validation)

**Goal**: Verify the exact commands from README work for a new user.

```bash
# 1. Prepare test audio
mkdir -p raw_audio
cp /path/to/test_audio.wav raw_audio/

# 2. Run transcription (from README quickstart)
uv run slower-whisper transcribe

# 3. Verify outputs
ls whisper_json/    # Should have test_audio.json
ls transcripts/     # Should have test_audio.txt and test_audio.srt
```

**Expected Results**:
- `whisper_json/test_audio.json` contains structured transcript with segments
- `transcripts/test_audio.txt` is readable plain text
- `transcripts/test_audio.srt` is valid subtitle format
- No crashes, clear progress output

**Pass Criteria**:
- [ ] All outputs generated without errors
- [ ] JSON validates against schema v2 (has `schema_version: 2`)
- [ ] Text is accurate transcription of audio
- [ ] Time elapsed is reasonable for audio length

---

### Scenario 2: Speaker Diarization End-to-End

**Goal**: Verify diarization works with 2-speaker audio and populates schema correctly.

```bash
# 1. Use or create 2-speaker audio
cp /path/to/two_speaker_call.wav raw_audio/

# 2. Run transcription with diarization
uv run slower-whisper transcribe \
  --enable-diarization \
  --min-speakers 2 \
  --max-speakers 2

# 3. Inspect output
cat whisper_json/two_speaker_call.json | jq .speakers
cat whisper_json/two_speaker_call.json | jq .turns
```

**Expected Results**:
- `speakers` array has 2 entries: `{"id": "spk_0", ...}`, `{"id": "spk_1", ...}`
- Each speaker has `first_seen`, `last_seen`, `total_speech_time`, `num_segments`
- `turns` array groups contiguous segments by speaker
- Each turn has `speaker_id`, `start`, `end`, `segment_ids`, `text`
- `meta.diarization.status` is `"success"`
- `meta.diarization.backend` is `"pyannote.audio"`

**Pass Criteria**:
- [ ] `speakers` count matches expected (2 for test file)
- [ ] All segments have `speaker` field populated (not `null`)
- [ ] Turns correctly group segments by speaker (no broken turns)
- [ ] Speaker IDs are canonical (`spk_0`, `spk_1`, not raw pyannote UUIDs)
- [ ] No diarization errors in `meta.diarization`

---

### Scenario 3: LLM Integration Example (End-to-End)

**Goal**: Run the `summarize_with_diarization.py` example and verify it produces useful LLM output.

**Prerequisites**:
- Scenario 2 completed (have a transcript with diarization)
- `ANTHROPIC_API_KEY` set

```bash
# Run the example
python examples/llm_integration/summarize_with_diarization.py \
  whisper_json/two_speaker_call.json
```

**Expected Results**:
1. **No crashes** - Script runs to completion
2. **Speaker labels inferred** - Console shows "Inferred speaker roles: Agent (spk_0), Customer (spk_1)" or similar
3. **Rendered conversation** - Output includes formatted transcript with speaker labels:
   ```
   [Agent] Hello, how can I help you today?
   [Customer | high pitch, fast speech] I can't log into my account!
   ```
4. **LLM response** - Claude returns:
   - A summary of the conversation
   - A quality/score evaluation
   - Coaching feedback with specific timestamps
   - Action items extracted from conversation

**Pass Criteria**:
- [ ] Script completes without Python exceptions
- [ ] Console output shows speaker role inference (not raw `spk_0`/`spk_1`)
- [ ] Rendered conversation uses mapped labels (`Agent`, `Customer`)
- [ ] LLM response is coherent and references speakers correctly
- [ ] LLM response includes timestamps (validates temporal context)
- [ ] Audio cues appear in rendered text if transcript was enriched

---

### Scenario 4: Audio Enrichment Integration

**Goal**: Verify enrichment populates `audio_state` and renders correctly for LLMs.

```bash
# 1. Enrich an existing transcript
uv run slower-whisper enrich

# 2. Verify audio_state populated
cat whisper_json/test_audio.json | jq '.segments[0].audio_state'

# 3. Check rendering includes audio cues
python -c "
from transcription import load_transcript, render_conversation_for_llm

t = load_transcript('whisper_json/test_audio.json')
print(render_conversation_for_llm(t, mode='segments', include_audio_cues=True))
"
```

**Expected Results**:
- Segments have `audio_state` with `prosody`, `emotion`, `rendering`
- Rendering includes text like `"[high pitch, loud volume, fast speech]"`
- `extraction_status` shows which features succeeded

**Pass Criteria**:
- [ ] `audio_state` is not `null` for majority of segments
- [ ] At least one of `prosody` or `emotion` populated per segment
- [ ] `rendering` field contains human-readable audio cues
- [ ] LLM rendering includes audio cues when `include_audio_cues=True`

---

### Scenario 5: Error Handling (Graceful Degradation)

**Goal**: Verify friendly error messages for common user mistakes.

#### 5.1 Missing HF_TOKEN for Diarization

```bash
# Unset token
unset HF_TOKEN

# Try diarization
uv run slower-whisper transcribe --enable-diarization
```

**Expected**: Clear error message pointing to `docs/SPEAKER_DIARIZATION.md` for setup.

**Pass Criteria**:
- [ ] Error message mentions `HF_TOKEN`
- [ ] Error message references setup docs
- [ ] No Python traceback (user-friendly error)

#### 5.2 Missing ffmpeg

```bash
# Temporarily rename ffmpeg (if safe to test)
# OR: Test by reading error handling code

# Try transcription
uv run slower-whisper transcribe
```

**Expected**: Clear error message with ffmpeg install instructions.

**Pass Criteria**:
- [ ] Error message mentions `ffmpeg not found`
- [ ] Error message includes install instructions or link

#### 5.3 Diarization Fails Gracefully

```bash
# Use audio file with no speech or single speaker
# Try diarization with min-speakers 2
uv run slower-whisper transcribe \
  --enable-diarization \
  --min-speakers 2 \
  --max-speakers 2
```

**Expected**: Transcription succeeds, diarization fails gracefully with `meta.diarization.status = "failed"`.

**Pass Criteria**:
- [ ] Transcript generated (transcription not blocked by diarization failure)
- [ ] `meta.diarization.status` is `"failed"`
- [ ] Error logged but pipeline continues

---

### Scenario 6: Fresh Install Test (Simulates New User)

**Goal**: Verify install process works from scratch.

```bash
# In a temporary directory
cd /tmp
git clone https://github.com/EffortlessMetrics/slower-whisper.git
cd slower-whisper

# Base install
uv sync

# Verify version
uv run slower-whisper --version

# Quick transcription test
mkdir -p raw_audio
cp /path/to/test.wav raw_audio/
uv run slower-whisper transcribe
```

**Pass Criteria**:
- [ ] Clone succeeds with correct URL
- [ ] `uv sync` completes without errors
- [ ] `--version` shows `1.1.0`
- [ ] Basic transcription works
- [ ] Optional: Test `uv sync --extra diarization` also works

---

## Success Criteria Summary

**Minimum for Release**:
- [x] Scenario 1: Basic transcription works (5-min quickstart)
- [x] Scenario 2: Diarization populates schema correctly
- [x] Scenario 3: LLM example produces useful output
- [ ] Scenario 5: Error messages are friendly (code review OK if manual test infeasible)

**Nice to Have** (if time permits):
- [ ] Scenario 4: Enrichment integration verified
- [ ] Scenario 6: Fresh install smoke test

**Blockers** (must fix before release):
- Any crash in Scenarios 1-3
- Missing/broken schema fields in Scenario 2
- Unhelpful error messages in Scenario 5

---

## Post-Test Actions

### If All Tests Pass

1. Update CHANGELOG date:
   ```bash
   # Change "TBD" to today's date
   sed -i 's/## \[1.1.0\] - TBD/## [1.1.0] - 2025-11-18/' CHANGELOG.md
   ```

2. Commit release:
   ```bash
   git add .
   git commit -m "Release v1.1.0: Speaker diarization and LLM integration

   - Add experimental speaker diarization with pyannote.audio
   - Add first-class LLM integration API (render_conversation_for_llm)
   - Add working example: summarize_with_diarization.py
   - Add comprehensive docs: SPEAKER_DIARIZATION.md, LLM_PROMPT_PATTERNS.md
   - Polish README with 5-minute quickstart and LLM cross-links
   - Bump version to 1.1.0 in pyproject.toml and __init__.py

   See CHANGELOG.md for full details."
   ```

3. Tag release:
   ```bash
   git tag -a v1.1.0 -m "Release v1.1.0: Speaker diarization and LLM integration"
   ```

4. Push:
   ```bash
   git push origin main
   git push origin v1.1.0
   ```

5. Create GitHub Release (template in `RELEASE_CHECKLIST.md`)

6. (Optional) Publish to PyPI:
   ```bash
   uv build
   twine check dist/*
   twine upload dist/*
   ```

### If Tests Fail

1. Document failure in GitHub issue
2. Fix bug(s)
3. Re-run affected scenarios
4. Do NOT tag/release until all critical scenarios pass

---

## Quick Test Commands

For rapid iteration during final testing:

```bash
# Run fast automated tests
uv run pytest -m "not slow and not requires_gpu and not requires_diarization and not api" -q

# Test CLI version
uv run slower-whisper --version

# Test basic transcription (Scenario 1)
uv run slower-whisper transcribe

# Test diarization (Scenario 2)
uv run slower-whisper transcribe --enable-diarization --min-speakers 2 --max-speakers 2

# Test LLM example (Scenario 3)
python examples/llm_integration/summarize_with_diarization.py whisper_json/test.json
```

---

## Notes for Tester

- **Test audio recommendations**:
  - For basic transcription: Any clear speech audio (podcast, meeting recording)
  - For diarization: 2-speaker conversation (support call, interview, dialogue)
  - Duration: 1-3 minutes ideal for quick testing

- **Expected timing** (on modern hardware):
  - Scenario 1 (basic transcription): ~30 seconds for 1-minute audio
  - Scenario 2 (with diarization): ~1-2 minutes for 1-minute audio (pyannote is slower)
  - Scenario 3 (LLM example): Depends on transcript length + Claude API latency

- **Debugging tips**:
  - If diarization fails, check `meta.diarization.error_type` in JSON
  - If LLM example fails, check `ANTHROPIC_API_KEY` is valid
  - If transcription is slow, check if GPU is detected: `torch.cuda.is_available()`

- **GPU vs CPU**:
  - GPU recommended for diarization (pyannote is slow on CPU)
  - GPU recommended for large Whisper models (large-v3)
  - CPU mode works but is 5-10x slower

---

## Appendix: Example Test Audio Sources

**For basic transcription** (Scenario 1):
- [LibriVox public domain audiobooks](https://librivox.org/)
- Your own voice recordings (quick and easy)
- Meeting recordings (if available)

**For diarization** (Scenario 2):
- [CallHome corpus](https://catalog.ldc.upenn.edu/LDC97S42) (academic, may require license)
- [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) (academic, free for research)
- Your own 2-person call recordings
- Synthetic: Record yourself having a "conversation" with role-play

**Quick synthetic 2-speaker test**:
```bash
# Use text-to-speech to create synthetic 2-speaker audio
# (requires additional tools like espeak or macOS 'say' command)

# Example on macOS:
say -v "Alex" "Hello, this is the first speaker" -o speaker1.aiff
say -v "Samantha" "And this is the second speaker responding" -o speaker2.aiff

# Merge with ffmpeg
ffmpeg -i speaker1.aiff -i speaker2.aiff -filter_complex \
  "[0:a][1:a]concat=n=2:v=0:a=1[out]" -map "[out]" test_2speakers.wav
```
