#!/usr/bin/env python3
"""
Quick Start Guide for Prosody Feature Extraction

This script shows the minimal code needed to extract prosody features.
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transcription.prosody import extract_prosody


def quickstart_example():
    """Minimal example of prosody extraction."""

    # 1. Generate or load your audio
    #    In production, you'd load from a file:
    #    import soundfile as sf
    #    audio, sr = sf.read("your_audio.wav")

    # For demo, generate 2 seconds of synthetic audio at 16kHz
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    frequency = 200  # Hz
    audio = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    # 2. Your transcribed text
    text = "This is a sample sentence for prosody analysis."

    # 3. Extract prosody features (that's it!)
    features = extract_prosody(audio, sr, text)

    # 4. Access the results
    print("Prosody Features:")
    print("-" * 50)
    print(f"Pitch Level:      {features['pitch']['level']}")
    print(f"Pitch Mean:       {features['pitch']['mean_hz']:.1f} Hz")
    print(f"Pitch Contour:    {features['pitch']['contour']}")
    print(f"Energy Level:     {features['energy']['level']}")
    print(f"Energy dB:        {features['energy']['db_rms']:.1f} dB")
    print(f"Rate Level:       {features['rate']['level']}")
    print(f"Rate:             {features['rate']['syllables_per_sec']:.1f} syl/sec")
    print(f"Pause Count:      {features['pauses']['count']}")
    print(f"Pause Density:    {features['pauses']['density']}")

    return features


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Prosody Extraction - Quick Start")
    print("=" * 50 + "\n")

    try:
        features = quickstart_example()
        print("\n" + "=" * 50)
        print("Success! Features extracted.")
        print("=" * 50)
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install praat-parselmouth librosa numpy")
    except Exception as e:
        print(f"\nError: {e}")
