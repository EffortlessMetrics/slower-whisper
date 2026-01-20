# GPU Setup Guide

This guide covers GPU configuration for faster transcription with slower-whisper.

## Quick Start

### Local (with CUDA)

```bash
# Auto-detect CUDA (recommended)
slower-whisper transcribe --device auto

# Explicitly use CUDA
slower-whisper transcribe --device cuda
```

Output shows device selection:
```
[Device] CUDA | compute_type=float16 | model=large-v3
```

### Local (CPU fallback)

If CUDA is unavailable, the banner shows why:
```
[Device] CPU (fallback from auto) | compute_type=int8 | model=large-v3
         └─ Reason: No CUDA devices found by CTranslate2
```

### Docker GPU

```bash
# Build GPU image
docker build -f config/Dockerfile.gpu -t slower-whisper:gpu .

# Run with GPU
docker run --rm --gpus all \
  -v $(pwd)/raw_audio:/app/raw_audio \
  -v $(pwd)/whisper_json:/app/whisper_json \
  slower-whisper:gpu \
  transcribe --device auto
```

## Model Cache Location

Models are downloaded on first use and cached:

| Component | Default Location |
|-----------|------------------|
| Whisper models | `~/.cache/slower-whisper/whisper/` |
| HuggingFace models | `~/.cache/slower-whisper/huggingface/` |
| Torch models | `~/.cache/slower-whisper/torch/` |

Override with environment variable:
```bash
export SLOWER_WHISPER_CACHE_ROOT=/path/to/cache
```

View cache status:
```bash
slower-whisper cache --show
```

## Troubleshooting

### "CPU fallback" when CUDA should work

**Check 1: CTranslate2 CUDA support**

slower-whisper uses CTranslate2 (via faster-whisper), not PyTorch, for ASR.
Verify CTranslate2 sees your GPU:

```python
import ctranslate2
print(f"CUDA devices: {ctranslate2.get_cuda_device_count()}")
```

If this returns 0 but `nvidia-smi` works, reinstall ctranslate2 with CUDA:
```bash
pip install --force-reinstall ctranslate2
```

**Check 2: NVIDIA driver version**

CTranslate2 requires NVIDIA driver >= 525 for CUDA 12.x:
```bash
nvidia-smi  # Check "Driver Version" in header
```

**Check 3: Correct CUDA version**

Verify torch and ctranslate2 were built for your CUDA version:
```bash
python -c "import torch; print(torch.version.cuda)"
```

### Docker: "could not select device driver"

Install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
```

Test:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Performance tips

1. **Use float16 on CUDA** (default): Best speed/quality balance
2. **Use int8 on CPU** (default): Fastest CPU inference
3. **large-v3 model**: Best accuracy, requires ~3GB VRAM
4. **base model**: Fastest, ~150MB, good for testing

```bash
# Fast testing
slower-whisper transcribe --model base --device auto

# Production quality
slower-whisper transcribe --model large-v3 --device auto
```

## VRAM Requirements by Model

| Model | Model Size | Min VRAM (float16) | Recommended VRAM | Speed (RTF) |
|-------|------------|-------------------|------------------|-------------|
| tiny | 39M | ~1 GB | 2 GB | ~32x |
| base | 74M | ~1 GB | 2 GB | ~16x |
| small | 244M | ~2 GB | 4 GB | ~6x |
| medium | 769M | ~5 GB | 8 GB | ~2x |
| large-v2 | 1.5B | ~10 GB | 12 GB | ~1x |
| large-v3 | 1.5B | ~10 GB | 12 GB | ~1x |

**RTF** = Real-Time Factor (e.g., 16x means 16 minutes of audio processed per minute)

**With diarization enabled**, add ~2-3 GB additional VRAM for the pyannote model.

**With emotion enrichment**, add ~1.5 GB for the dimensional model, ~1.5 GB for categorical.

## CUDA Compatibility Matrix

| NVIDIA Driver | CUDA Version | CTranslate2 Support | PyTorch Support |
|---------------|--------------|---------------------|-----------------|
| 525+ | CUDA 12.x | Yes | Yes (2.0+) |
| 515+ | CUDA 11.8 | Yes | Yes (1.13+) |
| 470+ | CUDA 11.4 | Yes | Yes (1.10+) |
| < 470 | CUDA 11.x | Partial | Limited |

Check your driver version:
```bash
nvidia-smi  # Look for "Driver Version" in header
```

Check CUDA version seen by Python:
```bash
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import ctranslate2; print(f'CTranslate2 CUDA devices: {ctranslate2.get_cuda_device_count()}')"
```

## Compute Types

| Compute Type | Device | Speed | Quality | VRAM |
|--------------|--------|-------|---------|------|
| float16 | CUDA | Fast | Best | Higher |
| int8 | CPU/CUDA | Fastest | Good | Lower |
| float32 | CPU | Slow | Best | N/A |

Auto-selection defaults:
- CUDA → float16
- CPU → int8

Override:
```bash
slower-whisper transcribe --device cuda --compute-type int8_float16
```
