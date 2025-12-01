# v1.1.0 – Speaker Diarization & LLM Integration

This release adds **experimental speaker diarization** and **first-class LLM integration** to slower-whisper, enabling speaker-aware conversation analysis with LLMs.

---

## 🎯 What's New

### Speaker Diarization (Experimental)

Turn audio into speaker-labeled conversations:

```bash
# Install diarization dependencies
uv sync --extra diarization

# Set HuggingFace token (required for pyannote.audio models)
export HF_TOKEN=hf_...

# Transcribe with speaker detection
uv run slower-whisper transcribe \
  --enable-diarization \
  --min-speakers 2 \
  --max-speakers 2
```

**What you get:**
- Segments labeled with speaker IDs (`spk_0`, `spk_1`, etc.)
- Global `speakers[]` table with per-speaker stats
- `turns[]` structure grouping contiguous speaker segments
- Graceful error handling with detailed diagnostics

See [`docs/SPEAKER_DIARIZATION.md`](docs/SPEAKER_DIARIZATION.md) for setup guide.

---

### LLM Integration API

Convert transcripts to LLM-ready text with speaker awareness:

```python
from transcription.api import load_transcript
from transcription.llm_utils import render_conversation_for_llm

# Load transcript with diarization
transcript = load_transcript("whisper_json/meeting.json")

# Render for LLM
conversation = render_conversation_for_llm(
    transcript,
    mode="turns",  # Group by speaker turns
    speaker_labels={"spk_0": "Agent", "spk_1": "Customer"},
    include_audio_cues=True,  # Include prosody/emotion
    include_timestamps=False
)

# Send to LLM
print(conversation)
# Output:
# Conversation (2 speakers, 3 turns, 45.2s)
#
# [Agent | calm tone, low pitch] Hello, how can I help you today?
#
# [Customer | high pitch, fast speech] I'm having trouble with my account...
```

**API Functions:**
- `render_conversation_for_llm()` – Full-featured conversation rendering
- `render_conversation_compact()` – Token-efficient format
- `render_segment()` – Individual segment rendering

See [`docs/LLM_PROMPT_PATTERNS.md`](docs/LLM_PROMPT_PATTERNS.md) for prompt engineering patterns.

---

### Working Example

```bash
# 1. Clone and install
git clone https://github.com/EffortlessMetrics/slower-whisper.git
cd slower-whisper
uv sync --extra diarization

# 2. Transcribe with diarization
export HF_TOKEN=hf_...
mkdir -p raw_audio
cp /path/to/conversation.wav raw_audio/
uv run slower-whisper transcribe --enable-diarization

# 3. Analyze with LLM
export ANTHROPIC_API_KEY=sk-ant-...
python examples/llm_integration/summarize_with_diarization.py \
  whisper_json/conversation.json
```

See [`examples/llm_integration/README.md`](examples/llm_integration/README.md) for complete guide.

---

## 📋 Full Changelog

### Added

**Speaker Diarization (Experimental)**
- v1.1 speaker diarization (L2 enrichment layer) using pyannote.audio
  - Optional `diarization` extra: `uv sync --extra diarization`
  - Populates `segment.speaker` with speaker IDs (`"spk_0"`, `"spk_1"`, etc.)
  - Builds global `speakers[]` table and `turns[]` structure in schema v2
  - Normalized canonical speaker IDs backend-agnostic
  - Overlap-based segment-to-speaker mapping with configurable threshold
- Diarization metadata in `meta.diarization`:
  - `status: "success" | "failed"` - Overall diarization result
  - `requested: bool` - Whether diarization was enabled
  - `backend: "pyannote.audio"` - Backend used for diarization
  - `error_type: "auth" | "missing_dependency" | "file_not_found" | "unknown"` - Error category for debugging
- Turn structure - Contiguous segments grouped by speaker:
  - Each turn includes `speaker_id`, `start`, `end`, `segment_ids`, `text`
  - Foundation for turn-level analysis (interruptions, questions, backchanneling)
- Speaker table - Per-speaker aggregates:
  - `id`, `first_seen`, `last_seen`, `total_speech_time`, `num_segments`
  - Foundation for speaker-relative prosody baselines (v1.2)

**LLM Integration API**
- `render_conversation_for_llm()` - Convert transcripts to LLM-ready text:
  - Modes: `"turns"` (speaker turns) or `"segments"` (individual segments)
  - Optional speaker labels: `speaker_labels={"spk_0": "Agent", "spk_1": "Customer"}`
  - Audio cue inclusion: `include_audio_cues=True` for prosody/emotion annotations
  - Timestamp prefixes: `include_timestamps=True` for temporal context
  - Metadata header with conversation stats
- `render_conversation_compact()` - Token-efficient rendering for constrained contexts:
  - Simple `Speaker: text` format without cues
  - Automatic truncation with `max_tokens` parameter
  - Preserves speaker labels and turn structure
- `render_segment()` - Render individual segments with speaker/audio cues:
  - Format: `"[Agent | calm tone, low pitch] Hello, how can I help you?"`
  - Configurable cue inclusion and timestamp formatting
- Speaker label mapping - Map raw IDs to human-readable labels across all rendering functions
- Graceful degradation - Handles missing speakers, turns, or audio state without errors

**Examples and Documentation**
- Working example scripts in `examples/llm_integration/`:
  - `summarize_with_diarization.py` - Complete end-to-end QA scoring with Claude
  - Speaker role inference (heuristic talk time-based)
  - Demonstrates rendering + LLM API integration
- Comprehensive test coverage (21 new tests):
  - Unit tests for all rendering functions
  - Speaker label mapping validation
  - Edge case handling (empty transcripts, missing speakers, no turns)
  - Graceful degradation verification
- Documentation:
  - `docs/SPEAKER_DIARIZATION.md` - Complete diarization implementation guide
  - `docs/LLM_PROMPT_PATTERNS.md` - Reference prompts and rendering strategies
  - `docs/TESTING_STRATEGY.md` - Updated with synthetic fixtures methodology
  - `examples/llm_integration/README.md` - LLM integration guide with alternative providers
  - Updated `README.md` with 5-minute quickstart and LLM integration section

### Improved
- CLI help text now references docs for diarization setup (`docs/SPEAKER_DIARIZATION.md`)
- `--version` flag added to main CLI
- README structure - Added 5-minute quickstart and LLM cross-links
- `docs/INDEX.md` - Enhanced LLM integration flow with new examples

### Fixed
- Speaker type consistency - All speaker fields use string IDs throughout schema
- Linting - Fixed module-level import order issues with proper ruff configuration

### Changed
- Schema v2 remains backward compatible:
  - `speakers` and `turns` are optional, default to `null`
  - Existing v1 transcripts still load correctly
  - Diarization disabled by default (`--enable-diarization` required)

---

## 📦 Installation

**Basic (transcription only):**
```bash
uv sync
```

**With diarization:**
```bash
uv sync --extra diarization
```

**Full installation (diarization + enrichment + dev tools):**
```bash
uv sync --extra dev
```

**Alternative (pip):**
```bash
pip install -e ".[diarization]"
```

---

## 📚 Documentation

- **[5-Minute Quickstart](README.md#quick-start-5-minutes)** - Get started immediately
- **[Speaker Diarization Guide](docs/SPEAKER_DIARIZATION.md)** - Setup and troubleshooting
- **[LLM Integration Guide](examples/llm_integration/README.md)** - Prompt patterns and examples
- **[LLM Prompt Patterns](docs/LLM_PROMPT_PATTERNS.md)** - Reference prompts for common tasks
- **[Documentation Index](docs/INDEX.md)** - Complete documentation map

---

## 🧪 Testing

This release includes:
- **268 passing tests** (21 new for diarization + LLM integration)
- **100% coverage** for new modules (`diarization.py`, `llm_utils.py`, `cache.py`)
- **Synthetic fixtures** for repeatable diarization testing
- **Integration tests** for end-to-end workflows

Run tests:
```bash
uv run pytest
```

---

## 🔒 Privacy & Local-First

All processing runs **100% locally**:
- ✅ Whisper transcription (faster-whisper)
- ✅ Speaker diarization (pyannote.audio)
- ✅ Audio enrichment (librosa, praat-parselmouth)
- ✅ No data leaves your machine

Only external dependencies:
- Model weights downloaded from HuggingFace (one-time, cached locally)
- LLM API calls (optional, user-controlled)

---

## 🚀 What's Next

See [`ROADMAP.md`](ROADMAP.md) for upcoming features:
- **v1.2**: Speaker-relative prosody baselines
- **v1.3**: Multi-language diarization, conversation analytics
- **v2.0**: Real-time streaming, production API server

---

## 🙏 Acknowledgments

Built on the shoulders of giants:
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Efficient Whisper implementation
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - State-of-the-art speaker diarization
- [librosa](https://librosa.org/) - Audio analysis library
- [Anthropic Claude](https://www.anthropic.com/claude) - LLM integration example

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details
