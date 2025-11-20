# Local Evaluation Workflow

This document describes the **pure-local evaluation workflow** for `slower-whisper`.

The core idea:

1. The **local pipeline** (Whisper + diarization + enrichment) runs entirely on your machine (ffmpeg, faster-whisper, pyannote.audio, etc.).
2. The **evaluation harness** computes metrics (e.g., WER, DER) against ground-truth annotations.
3. An LLM (e.g., Claude via "Code" or ChatGPT) is used **only as an analyst**: it reads small JSON + text snippets, explains failure modes, and suggests changes. It does **not** run ASR, diarization, or process audio.

All benchmark data lives outside git, under:

```bash
~/.cache/slower-whisper/benchmarks/
```

All evaluation results are written to:

```bash
benchmarks/results/
```

Note: `benchmarks/results/` is git-ignored by default.

---

## 1. Prerequisites

### 1.1 System tools

```bash
# ffmpeg must be installed
ffmpeg -version

# uv must be installed
uv --version
```

On Ubuntu/WSL:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg jq
```

### 1.2 Python environment

From the repo root:

```bash
cd /path/to/slower-whisper

# Install base + enrichment + diarization + dev deps
uv sync --extra full --extra diarization --extra dev

# Quick sanity checks
uv run python -c "import jiwer; print('jiwer OK')"
uv run python -c "import benchmarks.eval_asr_diarization as m; print('eval_asr_diarization OK')"
```

### 1.3 Environment variables

For ASR-only evaluation (LibriSpeech), no extra env is needed.

For diarization (AMI, etc.), you must set a Hugging Face token so `pyannote.audio` can download its models:

```bash
# Obtain from https://huggingface.co/settings/tokens
export HF_TOKEN=hf_xxx
```

---

## 2. Dataset Layout & Cache Strategy

All benchmarks are expected under `CachePaths.benchmarks_root`, typically:

```text
~/.cache/slower-whisper/benchmarks/
  librispeech/
    LibriSpeech/
      dev-clean/
      test-clean/
      ...
  ami/
    audio/
      ES2002a.wav
      ES2002b.wav
      ...
    annotations/
      ES2002a.json
      ES2002b.json
      ...
  iemocap/          # optional, for emotion eval
  ...
```

Nothing under `benchmarks/` is committed to git. The repo only contains:

* Dataset iterators (`transcription/benchmarks.py`)
* Evaluation scripts (`benchmarks/eval_*.py`)
* Documentation and helper scripts (`scripts/*.sh`)

---

## 3. LibriSpeech: ASR / WER Evaluation

### 3.1 Download dev-clean

From the repo root:

```bash
# One-time helper script
./scripts/download_librispeech.sh  # defaults to dev-clean

# Verify structure
LIB_ROOT="$HOME/.cache/slower-whisper/benchmarks/librispeech"
ls "$LIB_ROOT/LibriSpeech/dev-clean" | head
```

You should see speaker ID directories (e.g. `1272`, `1462`, …).

### 3.2 Inner-ring smoke test (n=5–10)

```bash
cd /path/to/slower-whisper

uv run python benchmarks/eval_asr_diarization.py \
  --dataset librispeech \
  --n 10 \
  --model base \
  --device cpu \
  --output benchmarks/results/asr_librispeech_dev_clean_10.json
```

Inspect results:

```bash
cat benchmarks/results/asr_librispeech_dev_clean_10.json | jq '.aggregate'
```

Typical baseline for `base` on dev-clean (CPU, `int8`):

* `avg_WER` ≈ 0.08–0.15
* `min_WER` near 0.0 (many perfect utterances)
* `max_WER` can be high for very short utterances or weird proper nouns

### 3.3 Outer-ring baseline (n=50–200)

```bash
uv run python benchmarks/eval_asr_diarization.py \
  --dataset librispeech \
  --n 50 \
  --model base \
  --device cpu \
  --output benchmarks/results/asr_librispeech_dev_clean_50.json

cat benchmarks/results/asr_librispeech_dev_clean_50.json | jq '.aggregate'
```

For deeper baselines:

```bash
uv run python benchmarks/eval_asr_diarization.py \
  --dataset librispeech \
  --n 200 \
  --model base \
  --device cpu \
  --output benchmarks/results/asr_librispeech_dev_clean_200.json
```

### 3.4 Interpreting WER

The eval harness uses a **standard ASR normalization pipeline** before computing WER with `jiwer`:

* Convert to uppercase
* Remove punctuation
* Collapse extra whitespace

This aligns LibriSpeech's all-caps/no-punctuation transcripts with Whisper's natural-case, punctuated output.

Expected patterns:

* Very short utterances (2–4 words) → small absolute errors produce large WER (e.g. 1 wrong word out of 2 = 50%).
* Proper nouns (names like "KALIKO", "BRION'S") are frequent sources of error.
* Abbreviations and titles (`M A`) may be normalized differently.

---

## 4. AMI: WER + DER (Diarization) Evaluation

### 4.1 Stage AMI audio & annotations

Target layout:

```text
~/.cache/slower-whisper/benchmarks/ami/
  audio/
    ES2002a.wav         # 16 kHz mono, normalized via ffmpeg
    ES2002b.wav
    ...
  annotations/
    ES2002a.json
    ES2002b.json
    ...
```

Each annotation JSON should include at least:

```jsonc
{
  "meeting_id": "ES2002a",
  "reference_transcript": "full meeting transcript text...",
  "speakers": [
    {
      "id": "A",
      "segments": [
        { "start": 0.0, "end": 3.5 },
        { "start": 10.0, "end": 15.2 }
      ]
    },
    {
      "id": "B",
      "segments": [
        { "start": 3.5, "end": 10.0 }
      ]
    }
  ]
}
```

For initial smoke tests, you can reuse the synthetic fixtures you already built (`TEST001`, `TEST002`) as `audio/TEST001.wav` + `annotations/TEST001.json`, etc.

You can verify your AMI staging via:

```bash
./verify_ami_setup.sh
```

(or whatever helper script you've added).

### 4.2 Run AMI smoke test (n=2–3)

```bash
cd /path/to/slower-whisper

export HF_TOKEN=hf_xxx  # required for pyannote.audio

uv run python benchmarks/eval_asr_diarization.py \
  --dataset ami \
  --n 2 \
  --model base \
  --device cpu \
  --output benchmarks/results/asr_diar_ami_small.json
```

Inspect:

```bash
cat benchmarks/results/asr_diar_ami_small.json | jq '.aggregate, .samples'
```

You should see, per sample:

* `WER`: numeric (0–1)
* `DER`: numeric for samples with `speakers[]` in annotations

### 4.3 Interpreting DER

The harness constructs `pyannote.core.Annotation` objects from:

* **Reference**: `speakers[*].segments` in annotations JSON
* **Hypothesis**: `transcript.segments[*].speaker.id` and span (`start`, `end`)

Notes:

* Unlabeled segments (`speaker = null`) are treated as missed speech and increase DER.
* Overlap / cross-talk is not fully modeled yet (v1.1 focuses on max-overlap assignment).
* Tie-breaking for equal overlaps is deterministic (first speaker ID in lexical order).

Common DER failure modes:

* **Boundary drift**: segments slightly shifted; mostly OK for higher-level tasks.
* **Speaker swaps**: segments assigned to the wrong `spk_N` (these matter more).
* **Excessive splitting/merging**: too many or too few turns, impacting DER.

---

## 5. Using an LLM as Analyst (Optional but Recommended)

Once you have JSON results, you can use an LLM (via "Code" or similar) as an analysis aid:

### 5.1 Example: analyze LibriSpeech WER outliers

1. Generate worst-case samples:

   ```bash
   cat benchmarks/results/asr_librispeech_dev_clean_50.json \
     | jq '.samples | sort_by(.WER) | reverse | .[0:5]'
   ```

2. For a specific sample (e.g. `1272-135031-0011`), inspect reference text:

   ```bash
   REF_TXT=$(grep '^1272-135031-0011 ' \
     "$HOME/.cache/slower-whisper/benchmarks/librispeech/LibriSpeech/dev-clean/1272/135031/"*.txt)
   echo "$REF_TXT"
   ```

3. Transcribe that sample locally (if needed) and show both reference/hypothesis to the LLM, then ask:

   * What kind of error is this? (proper noun, short-utterance, etc.)
   * Is this a real quality problem or an expected edge case?

### 5.2 Example: analyze AMI DER outliers

1. Generate worst DER samples:

   ```bash
   cat benchmarks/results/asr_diar_ami_small.json \
     | jq '.samples | sort_by(.DER) | reverse | .[0:3]'
   ```

2. For a bad sample ID (e.g. `ES2002a`):

   * Open `~/.cache/slower-whisper/benchmarks/ami/annotations/ES2002a.json` to see `speakers[*].segments`.
   * Open the corresponding transcript JSON (`whisper_json/ES2002a.json`) and extract only the misassigned segments.

3. Ask the LLM:

   * Classify errors as boundary drift / wrong speaker / unlabeled.
   * Suggest adjustments to:

     * `min_speakers`, `max_speakers`, `overlap_threshold`.
     * The `assign_speakers()` logic (e.g., ignoring ultra-short segments).

The critical rule: **never send raw audio**, and avoid dumping entire meeting transcripts into the LLM. Let the code slice and summarize; keep the LLM in a pure "read JSON and reason about it" role.

---

## 6. Summary

* The evaluation harness (`benchmarks/eval_asr_diarization.py`) is now **production-ready** for:

  * ASR WER on LibriSpeech (with proper normalization).
  * WER + DER on AMI (once annotations are in place).
* LibriSpeech `dev-clean` baseline with `base+cpu` shows:

  * `avg_WER ≈ 10.9%` over 50 samples (in line with expectations).
* Next steps:

  1. Maintain inner-ring runs (n=5–10) for quick checks.
  2. Use outer-ring runs (n=50–200) for stable baselines.
  3. Grow AMI annotations incrementally and start analyzing DER.
  4. Only then consider model/hardware changes (e.g., `large-v3` + GPU) based on real baselines.
