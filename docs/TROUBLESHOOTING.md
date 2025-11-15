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

**Problem:** `AttributeError: module 'numpy' has no attribute 'X'`

**Solution:** Downgrade to numpy 1.x (required by some dependencies)
```bash
pip install "numpy>=1.24.0,<2.0"
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
python transcribe_pipeline.py --root /path/to/project
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
python transcribe_pipeline.py --language en
```

**2. Use larger model:**
```bash
python transcribe_pipeline.py --model large-v3
```

**3. Check audio quality:**
- Remove background noise
- Ensure speech is clear
- Increase recording volume
- Use higher bitrate audio

**4. Adjust VAD settings:**
```bash
# More aggressive VAD (shorter silences)
python transcribe_pipeline.py --vad-min-silence-ms 300

# Less aggressive (longer silences)
python transcribe_pipeline.py --vad-min-silence-ms 800
```

---

### "Transcription too slow"

**Problem:** Using CPU or large model

**Solutions:**

**1. Use GPU:**
```bash
python transcribe_pipeline.py --device cuda
```

**2. Use smaller model:**
```bash
python transcribe_pipeline.py --model base
```

**3. Use int8 quantization:**
```bash
python transcribe_pipeline.py --compute-type int8_float16
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
- Disable emotion if not needed: `python audio_enrich.py --no-enable-emotion`

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
python audio_enrich.py --device cuda
```

**2. Disable emotion (prosody-only is 10x faster):**
```bash
python audio_enrich.py --no-enable-emotion
```

**3. Process single file to test:**
```bash
python audio_enrich.py --file whisper_json/test.json
```

**4. Skip existing:**
```bash
python audio_enrich.py --skip-existing
```

---

## GPU and CUDA Issues

### "CUDA out of memory"

**Problem:** GPU VRAM insufficient for model

**Solutions:**

**Stage 1 (Transcription):**
```bash
# Use smaller model
python transcribe_pipeline.py --model medium

# Use CPU
python transcribe_pipeline.py --device cpu

# Use quantized weights
python transcribe_pipeline.py --compute-type int8
```

**Stage 2 (Enrichment):**
```bash
# Use CPU
python audio_enrich.py --device cpu

# Skip emotion (saves 2-3 GB VRAM)
python audio_enrich.py --no-enable-emotion
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
python audio_enrich.py --device cpu
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
python transcribe_pipeline.py
```

**3. Custom cache directory:**
```bash
export HF_HOME=/path/to/cache
python audio_enrich.py
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
python transcribe_pipeline.py --model base  # ~150 MB instead of ~3 GB
```

**2. Download overnight (first-time only)**

**3. Use different mirror:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python transcribe_pipeline.py
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
python transcribe_pipeline.py --model tiny  # Fastest
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
python audio_enrich.py  # Dimensional only
```

**2. Prosody only (fastest):**
```bash
python audio_enrich.py --no-enable-emotion
```

**3. Use GPU:**
```bash
python audio_enrich.py --device cuda
```

---

### High memory usage

**Problem:** Python process using too much RAM

**Solutions:**

**1. Process fewer files at once:**
```bash
# Process single file
python audio_enrich.py --file whisper_json/file1.json
```

**2. Close other applications**

**3. Use smaller Whisper model:**
```bash
python transcribe_pipeline.py --model base
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
python transcribe_pipeline.py
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
python transcribe_pipeline.py --vad-min-silence-ms 800
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
python audio_enrich.py --file whisper_json/file.json
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
python transcribe_pipeline.py --language en
```

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
python transcribe_pipeline.py 2>&1 | tee transcribe.log
python audio_enrich.py 2>&1 | tee enrich.log
```

### Report an issue

When reporting issues, please include:
1. Command you ran
2. Full error message and traceback
3. System information (OS, Python version, GPU)
4. Diagnostic output from above
5. Sample audio file (if possible)

**GitHub Issues:** https://github.com/yourusername/slower-whisper/issues

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
- [GitHub Issues](https://github.com/yourusername/slower-whisper/issues) - Report problems
