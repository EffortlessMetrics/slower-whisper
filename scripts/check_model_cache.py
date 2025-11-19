#!/usr/bin/env python3
"""
Check what models are already cached and what will be downloaded.

Usage:
    uv run python scripts/check_model_cache.py
"""

import os
from pathlib import Path


def get_hf_cache_dir():
    """Get HuggingFace cache directory."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def check_whisper_models():
    """Check which Whisper models are cached."""
    cache_dir = get_hf_cache_dir() / "hub"

    # faster-whisper models are stored as model--Systran--faster-whisper-*
    whisper_models = {
        "tiny": "models--Systran--faster-whisper-tiny",
        "base": "models--Systran--faster-whisper-base",
        "small": "models--Systran--faster-whisper-small",
        "medium": "models--Systran--faster-whisper-medium",
        "large-v2": "models--Systran--faster-whisper-large-v2",
        "large-v3": "models--Systran--faster-whisper-large-v3",
    }

    print("=== Whisper Models ===")
    for name, cache_name in whisper_models.items():
        model_path = cache_dir / cache_name
        if model_path.exists():
            size_mb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / (
                1024 * 1024
            )
            print(f"✓ {name:12s} - {size_mb:6.1f} MB cached at {model_path}")
        else:
            print(f"✗ {name:12s} - NOT cached (will download on first use)")
    print()


def check_pyannote_models():
    """Check which pyannote models are cached."""
    cache_dir = get_hf_cache_dir() / "hub"

    # pyannote models
    pyannote_models = {
        "segmentation-3.0": "models--pyannote--segmentation-3.0",
        "speaker-diarization-3.1": "models--pyannote--speaker-diarization-3.1",
        "wespeaker-voxceleb-resnet34-LM": "models--pyannote--wespeaker-voxceleb-resnet34-LM",
    }

    print("=== Pyannote Models (for diarization) ===")
    for name, cache_name in pyannote_models.items():
        model_path = cache_dir / cache_name
        if model_path.exists():
            size_mb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / (
                1024 * 1024
            )
            print(f"✓ {name:40s} - {size_mb:6.1f} MB cached at {model_path}")
        else:
            print(f"✗ {name:40s} - NOT cached (will download on first use)")
    print()


def check_emotion_models():
    """Check which emotion recognition models are cached."""
    cache_dir = get_hf_cache_dir() / "hub"

    # emotion models
    emotion_models = {
        "wav2vec2-large-robust-12-ft-emotion-msp-dim": "models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim",
        "wav2vec2-lg-xlsr-en-speech-emotion-recognition": "models--ehcalabres--wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    }

    print("=== Emotion Recognition Models (for enrichment) ===")
    for name, cache_name in emotion_models.items():
        model_path = cache_dir / cache_name
        if model_path.exists():
            size_mb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / (
                1024 * 1024
            )
            print(f"✓ {name:50s} - {size_mb:6.1f} MB")
        else:
            print(f"✗ {name:50s} - NOT cached")
    print()


def estimate_download_sizes():
    """Estimate what will be downloaded for a typical dogfood run."""
    print("=== Estimated Downloads for Dogfooding ===")
    print("\nFor basic transcription + diarization (minimal setup):")
    print("  - Whisper large-v3:              ~3.1 GB (one-time)")
    print("  - pyannote segmentation-3.0:     ~40 MB")
    print("  - pyannote diarization-3.1:      ~17 MB (pipeline config)")
    print("  - pyannote wespeaker embedding:  ~80 MB")
    print("  Total: ~3.2 GB (cached forever after first download)")
    print()
    print("For enrichment (optional, can skip for now):")
    print("  - wav2vec2 emotion models:       ~1.2 GB each")
    print()
    print("All models cached at:", get_hf_cache_dir())


def main():
    print("Checking model cache status...\n")

    check_whisper_models()
    check_pyannote_models()
    check_emotion_models()
    estimate_download_sizes()

    print("\nNotes:")
    print("- Models are cached permanently after first download")
    print("- Cache location: $HF_HOME or ~/.cache/huggingface/")
    print("- Use 'slower-whisper cache --show' for human-readable cache info")
    print("- Use 'slower-whisper cache --clear' to remove cached models")


if __name__ == "__main__":
    main()
