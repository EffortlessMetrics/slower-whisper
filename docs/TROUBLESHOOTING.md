# Troubleshooting Guide

Common issues and solutions for slower-whisper transcription and audio enrichment.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Stage 1: Transcription Issues](#stage-1-transcription-issues)
3. [Stage 2: Audio Enrichment Issues](#stage-2-audio-enrichment-issues)
4. [GPU and CUDA Issues](#gpu-and-cuda-issues)
5. [Model Download Issues](#model-download-issues)
6. [Performance Issues](#performance-issues)
7. [Output and Format Issues](#output-and-format-issues)

---

## Installation Issues

### "Command 'ffmpeg' not found"

**Problem:** ffmpeg is not installed or not on PATH

**Solution:**

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows (with Chocolatey):**
```powershell
choco install ffmpeg -y
```

**Verify installation:**
```bash
ffmpeg -version
```

---

### "No module named 'faster_whisper'"

**Problem:** Base dependencies not installed

**Solution:**
```bash
# Install base dependencies
pip install -r requirements-base.txt

# Or if using pyproject.toml
pip install .
```

**Verify:**
```bash
python -c "import faster_whisper; print('OK')"
```

---

### "No module named 'transformers'" or "No module named 'librosa'"

**Problem:** Audio enrichment dependencies not installed

**Solution:**
```bash
# Install full dependencies
pip install -r requirements-enrich.txt

# Or if using pyproject.toml
pip install .[full]
```

**Verify:**
```bash
python -c "import librosa, transformers, torch; print('OK')"
```

---

### Import errors with numpy 2.x

**Problem:** `AttributeError: module 'numpy' has no attribute 'X'` (usually from an old audio dependency pinned to numpy 1.x)

**Solution:** Reinstall with the current audio stack (numpy 2.x compatible)
```bash
# uv (preferred)
uv sync --extra enrich-basic

# or pip
pip install -U "numpy>=2.3.5,<3" "librosa>=0.11.0" "soundfile>=0.13.1"
```

---

## Stage 1: Transcription Issues

### "No audio files found in raw_audio/"

**Problem:** No files to process or wrong directory

**Solution:**
```bash
# Check directory exists
ls raw_audio/

# Verify audio files present
ls raw_audio/*.{mp3,wav,m4a}

# If using custom root, specify it
slower-whisper transcribe --root /path/to/project
```

---

### "ffmpeg failed to normalize audio"

**Problem:** Input audio file is corrupted or unsupported format

**Solution:**
```bash
# Test audio file manually
ffmpeg -i raw_audio/yourfile.mp3 -t 5 test.wav

# If that fails, try re-downloading or converting the file
# Check audio file integrity
file raw_audio/yourfile.mp3
```

**Supported formats:** mp3, m4a, wav, flac, ogg, aac, wma

---

### Transcription is inaccurate

**Problem:** Wrong language detection, incorrect model, or poor audio quality

**Solutions:**

**1. Force language:**
```bash
slower-whisper transcribe --language en
```

**2. Use larger model:**
```bash
slower-whisper transcribe --model large-v3
```

**3. Check audio quality:**
- Remove background noise
- Ensure speech is clear
- Increase recording volume
- Use higher bitrate audio

**4. Adjust VAD settings:**
```bash
# More aggressive VAD (shorter silences)
slower-whisper transcribe --vad-min-silence-ms 300

# Less aggressive (longer silences)
slower-whisper transcribe --vad-min-silence-ms 800
```

---

### "Transcription too slow"

**Problem:** Using CPU or large model

**Solutions:**

**1. Use GPU:**
```bash
slower-whisper transcribe --device cuda
```

**2. Use smaller model:**
```bash
slower-whisper transcribe --model base
```

**3. Use int8 quantization:**
```bash
slower-whisper transcribe --compute-type int8_float16
```

---

## Stage 2: Audio Enrichment Issues

### "Audio segment too short for emotion extraction"

**Problem:** Warning message for segments < 0.5 seconds

**This is expected behavior:**
- Emotion models work best on segments > 0.5 seconds
- Prosody extraction still works
- Enrichment continues (emotion may be None for short segments)

**Solutions:**
- Ignore warning (not an error)
- Adjust VAD settings in Stage 1 to create longer segments
- Disable emotion if not needed: `slower-whisper enrich --no-enable-emotion`

---

### "Pitch extraction returns None"

**Problem:** Parselmouth cannot extract pitch from segment

**Possible causes:**
- Audio is silence
- Audio is non-speech (music, noise)
- Audio quality too poor
- Segment too short (< 0.1 seconds)

**Solutions:**

**1. Check audio:**
```bash
# Play segment audio to verify it's speech
ffplay input_audio/yourfile.wav -ss 10 -t 2
```

**2. Verify parselmouth installed:**
```bash
pip install praat-parselmouth
```

**3. Check for silence:**
- Prosody module detects silence and may return None
- This is expected for silent segments

---

### "Emotion scores don't match text sentiment"

**Problem:** Audio emotion is "frustrated" but text is neutral

**This is expected:**
- Audio features capture *how* something is said, not *what* is said
- A person can say "I'm fine" (neutral text) with frustrated tone (negative audio)
- Combining audio and text gives complete picture

**Example:**
```
Text: "I'm fine."
Audio: high arousal, negative valence
Interpretation: Sarcastic or frustrated
```

---

### Enrichment is very slow

**Problem:** Emotion extraction on CPU

**Solutions:**

**1. Use GPU:**
```bash
slower-whisper enrich --device cuda
```

**2. Disable emotion (prosody-only is 10x faster):**
```bash
slower-whisper enrich --no-enable-emotion
```

**3. Process single file to test:**
```bash
slower-whisper enrich --file whisper_json/test.json
```

**4. Skip existing:**
```bash
slower-whisper enrich --skip-existing
```

---

## GPU and CUDA Issues

### "CUDA out of memory"

**Problem:** GPU VRAM insufficient for model

**Solutions:**

**Stage 1 (Transcription):**
```bash
# Use smaller model
slower-whisper transcribe --model medium

# Use CPU
slower-whisper transcribe --device cpu

# Use quantized weights
slower-whisper transcribe --compute-type int8
```

**Stage 2 (Enrichment):**
```bash
# Use CPU
slower-whisper enrich --device cpu

# Skip emotion (saves 2-3 GB VRAM)
slower-whisper enrich --no-enable-emotion
```

---

### "CUDA available: False" (but you have NVIDIA GPU)

**Problem:** PyTorch not installed with CUDA support or driver issues

**Solutions:**

**1. Verify NVIDIA driver:**
```bash
nvidia-smi
```

Should show GPU and CUDA version. If not, install/update NVIDIA drivers.

**2. Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your CUDA version (check `nvidia-smi`).

**3. Verify CUDA availability:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

### "RuntimeError: CUDA error: device-side assert triggered"

**Problem:** CUDA error in model inference

**Solutions:**

**1. Fallback to CPU:**
```bash
slower-whisper enrich --device cpu
```

**2. Update CUDA drivers**

**3. Reinstall PyTorch**

**4. Check GPU health:**
```bash
nvidia-smi dmon
```

---

## Model Download Issues

### Model download fails

**Problem:** `OSError: Can't load model weights`

**Solutions:**

**1. Check internet connection:**
- Whisper models: ~3 GB (download on first use)
- Emotion models: ~2.5 GB total (download on first use)

**2. Behind proxy:**
```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
slower-whisper transcribe
```

**3. Custom cache directory:**
```bash
export HF_HOME=/path/to/cache
slower-whisper enrich
```

**4. Manual download:**
```python
# Download Whisper model manually
from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cpu", download_root="/path/to/cache")
```

---

### Model download is very slow

**Problem:** Large models downloading over slow connection

**Solutions:**

**1. Use smaller model:**
```bash
slower-whisper transcribe --model base  # ~150 MB instead of ~3 GB
```

**2. Download overnight (first-time only)**

**3. Use different mirror:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
slower-whisper transcribe
```

---

## Performance Issues

### Transcription is very slow on CPU

**Expected:**
- CPU is 5-10x slower than GPU
- Large models are slowest
- This is normal behavior

**Solutions:**

**1. Use GPU** (recommended)

**2. Use smaller model:**
```bash
slower-whisper transcribe --model tiny  # Fastest
```

**3. Reduce audio quality:**
- Already normalized to 16 kHz (optimal)

**4. Process in batches:**
- Pipeline processes all files automatically
- Just wait for completion

---

### Enrichment takes too long

**Problem:** Full enrichment (prosody + dimensional + categorical emotion) is slow

**Solutions:**

**Timing expectations:**
- Prosody only: ~0.1x real-time (very fast)
- + Dimensional emotion: ~0.5-1x real-time
- + Categorical emotion: ~1-2x real-time

**1. Disable categorical emotion (default):**
```bash
slower-whisper enrich  # Dimensional only
```

**2. Prosody only (fastest):**
```bash
slower-whisper enrich --no-enable-emotion
```

**3. Use GPU:**
```bash
slower-whisper enrich --device cuda
```

---

### High memory usage

**Problem:** Python process using too much RAM

**Solutions:**

**1. Process fewer files at once:**
```bash
# Process single file
slower-whisper enrich --file whisper_json/file1.json
```

**2. Close other applications**

**3. Use smaller Whisper model:**
```bash
slower-whisper transcribe --model base
```

---

## Output and Format Issues

### JSON is malformed

**Problem:** `JSONDecodeError` when loading transcript

**Solutions:**

**1. Validate JSON:**
```bash
python -m json.tool whisper_json/file.json
```

**2. Re-transcribe file:**
```bash
rm whisper_json/file.json
slower-whisper transcribe
```

**3. Check disk space:**
```bash
df -h
```

---

### SRT subtitles have wrong timestamps

**Problem:** Subtitles out of sync with video

**Solutions:**

**1. Check VAD settings:**
```bash
# Longer minimum silence (fewer, longer segments)
slower-whisper transcribe --vad-min-silence-ms 800
```

**2. Verify audio normalization:**
- Input audio should be 16 kHz mono WAV
- Check `input_audio/` directory

**3. Manual adjustment:**
- Use subtitle editing software (Subtitle Edit, Aegisub)
- Shift all timestamps by offset

---

### Missing audio_state in enriched JSON

**Problem:** Enrichment ran but `audio_state` is None

**Solutions:**

**1. Check extraction status:**
```python
import json
with open('whisper_json/file.json') as f:
    data = json.load(f)
    print(data['segments'][0].get('audio_state', {}).get('extraction_status'))
```

**2. Check logs:**
- Look for error messages during enrichment
- Check if features were disabled

**3. Re-run enrichment with overwrite:**
```bash
slower-whisper enrich --file whisper_json/file.json
# (overwrites by default unless --skip-existing used)
```

---

### Unicode errors in output

**Problem:** `UnicodeEncodeError` or garbled text

**Solutions:**

**1. Use UTF-8 encoding:**
```python
# When reading JSON
with open('whisper_json/file.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
```

**2. Set system locale:**
```bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

**3. Specify language:**
```bash
slower-whisper transcribe --language en
```

---

## Error Code Reference

Quick reference for common error messages and their solutions.

### Runtime Errors

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `RuntimeError: No audio found in file` | Empty or corrupted audio file | Check file with `ffprobe input.wav` |
| `RuntimeError: Failed to load audio` | Unsupported format or codec | Convert with `ffmpeg -i input -ar 16000 output.wav` |
| `RuntimeError: Model not found` | Model name misspelled or not downloaded | Use valid name: tiny, base, small, medium, large-v3 |
| `RuntimeError: CUDA out of memory` | Insufficient GPU VRAM | Use `--device cpu` or smaller model |
| `RuntimeError: CTranslate2 error` | CUDA/driver mismatch | Reinstall ctranslate2 or check driver version |

### Import Errors

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `ModuleNotFoundError: No module named 'faster_whisper'` | Base deps not installed | `pip install .` or `uv sync` |
| `ModuleNotFoundError: No module named 'librosa'` | Enrichment deps not installed | `pip install .[full]` or `uv sync --extra full` |
| `ModuleNotFoundError: No module named 'pyannote'` | Diarization deps not installed | `pip install .[diarization]` |
| `ImportError: cannot import name 'WhisperModel'` | Wrong package version | `pip install --upgrade faster-whisper` |

### Configuration Errors

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `ValueError: Invalid compute_type for device` | Incompatible compute type | Use `int8` for CPU, `float16` for CUDA |
| `ValueError: model must be one of` | Invalid model name | Check valid models in CONFIGURATION.md |
| `FileNotFoundError: raw_audio directory not found` | Wrong --root path | Verify path exists: `ls /path/to/root/raw_audio/` |
| `PermissionError: [Errno 13]` | No write permission to output dir | `chmod 755 whisper_json/` or run with appropriate permissions |

### Network Errors

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `OSError: Can't load model weights` | Network issue during download | Check internet, retry, or use `HF_HOME` for manual cache |
| `ConnectionError: Unable to reach Hugging Face` | Firewall/proxy blocking | Set `HTTP_PROXY`/`HTTPS_PROXY` env vars |
| `ReadTimeoutError` | Slow connection during model download | Retry or download manually |

### Diarization Errors

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `ValueError: No speakers detected` | Audio too short or silent | Provide audio with clear speech > 1 second |
| `RuntimeError: pyannote model not found` | Missing HF token or model access | Accept terms at huggingface.co/pyannote/speaker-diarization |
| `ValueError: min_speakers > max_speakers` | Invalid speaker config | Ensure `--min-speakers <= --max-speakers` |

### Enrichment Errors

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `ValueError: Segment too short for analysis` | Segment < 0.1s | Adjust VAD settings for longer segments |
| `RuntimeError: Parselmouth extraction failed` | Invalid audio segment | Check audio isn't silence/noise |
| `RuntimeError: Emotion model failed` | Model loading issue | Reinstall transformers: `pip install -U transformers` |

---

## Still Having Issues?

### Collect diagnostic information

```bash
# System info
python --version
pip list | grep -E "(faster-whisper|torch|librosa|transformers|parselmouth)"
ffmpeg -version
nvidia-smi  # If using GPU

# Test basic functionality
python -c "import faster_whisper; print('Base install: OK')"
python -c "import librosa, torch, transformers; print('Enrichment install: OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Check logs

```bash
# Run with verbose output
slower-whisper transcribe 2>&1 | tee transcribe.log
slower-whisper enrich 2>&1 | tee enrich.log
```

### Report an issue

When reporting issues, please include:
1. Command you ran
2. Full error message and traceback
3. System information (OS, Python version, GPU)
4. Diagnostic output from above
5. Sample audio file (if possible)

**GitHub Issues:** https://github.com/EffortlessMetrics/slower-whisper/issues

---

## Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| ffmpeg not found | Install ffmpeg, add to PATH |
| CUDA out of memory | Use `--device cpu` or smaller model |
| Slow transcription | Use GPU, smaller model, or `--compute-type int8` |
| Model won't download | Check internet, try `export HF_HOME=/tmp/cache` |
| Inaccurate transcription | Use `--language en`, larger model |
| Enrichment too slow | Use `--no-enable-emotion` or `--device cuda` |
| Missing audio_state | Re-run enrichment, check logs |
| Import errors | Install dependencies: `pip install -r requirements-enrich.txt` |

---

**See Also:**
- [Quickstart Guide](QUICKSTART.md) - Getting started tutorial
- [Installation Guide](INSTALLATION.md) - Detailed installation
- [Audio Enrichment Guide](AUDIO_ENRICHMENT.md) - Stage 2 documentation
- [GitHub Issues](https://github.com/EffortlessMetrics/slower-whisper/issues) - Report problems
