#!/usr/bin/env python3
"""
Demonstration of prosody feature extraction.

This script shows how to use the prosody module to extract and analyze
prosodic features from audio segments.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transcription.prosody import compute_speaker_baseline, extract_prosody


def generate_sample_audio(sr=16000, duration=2.0, frequency=200, amplitude=0.3):
    """
    Generate sample audio with specified characteristics.

    Args:
        sr: Sample rate in Hz
        duration: Duration in seconds
        frequency: Fundamental frequency in Hz
        amplitude: Amplitude (0.0 to 1.0)

    Returns:
        numpy array of audio samples
    """
    t = np.linspace(0, duration, int(sr * duration))

    # Generate fundamental frequency with slight variation
    f0_variation = np.sin(2 * np.pi * 0.5 * t) * 20  # ±20 Hz variation
    frequency_array = frequency + f0_variation

    # Generate signal with harmonics for more natural sound
    signal = np.zeros_like(t)
    for harmonic in range(1, 4):
        signal += (amplitude / harmonic) * np.sin(2 * np.pi * frequency_array * harmonic * t)

    # Add some noise for realism
    noise = np.random.randn(len(t)) * 0.02
    signal += noise

    return signal.astype(np.float32)


def demo_basic_extraction():
    """Demonstrate basic prosody extraction."""
    print("=" * 70)
    print("DEMO 1: Basic Prosody Extraction")
    print("=" * 70)

    # Generate sample audio
    sr = 16000
    audio = generate_sample_audio(sr=sr, duration=2.0, frequency=180)
    text = "This is a sample sentence for prosody analysis."

    # Extract prosody features
    features = extract_prosody(audio, sr, text)

    # Display results
    print("\nExtracted Prosody Features:")
    print(json.dumps(features, indent=2))

    print("\nInterpretation:")
    print(
        f"  Pitch: {features['pitch']['level']} "
        f"({features['pitch']['mean_hz']:.1f} Hz) - "
        f"{features['pitch']['contour']} contour"
    )
    print(f"  Energy: {features['energy']['level']} " f"({features['energy']['db_rms']:.1f} dB)")
    print(
        f"  Speech Rate: {features['rate']['level']} "
        f"({features['rate']['syllables_per_sec']:.1f} syllables/sec)"
    )
    print(
        f"  Pauses: {features['pauses']['count']} pauses, "
        f"{features['pauses']['density']} density"
    )


def demo_baseline_normalization():
    """Demonstrate speaker baseline normalization."""
    print("\n" + "=" * 70)
    print("DEMO 2: Speaker Baseline Normalization")
    print("=" * 70)

    sr = 16000

    # Generate multiple segments for baseline computation
    print("\nGenerating speaker baseline from 5 segments...")
    baseline_segments = []
    for i in range(5):
        audio = generate_sample_audio(
            sr=sr,
            duration=1.5,
            frequency=np.random.uniform(170, 190),  # Vary pitch slightly
            amplitude=np.random.uniform(0.25, 0.35),
        )
        baseline_segments.append(
            {"audio": audio, "sr": sr, "text": f"This is baseline segment number {i + 1}."}
        )

    # Compute baseline
    baseline = compute_speaker_baseline(baseline_segments)
    print("\nComputed Speaker Baseline:")
    print(json.dumps(baseline, indent=2))

    # Extract features for a new segment WITHOUT baseline
    print("\n" + "-" * 70)
    print("Test Segment 1: Normal pitch (180 Hz)")
    print("-" * 70)
    audio1 = generate_sample_audio(sr=sr, frequency=180)
    text1 = "This segment has normal pitch for this speaker."
    features1 = extract_prosody(audio1, sr, text1)
    print(f"Without baseline: Pitch level = {features1['pitch']['level']}")

    features1_norm = extract_prosody(audio1, sr, text1, speaker_baseline=baseline)
    print(f"With baseline:    Pitch level = {features1_norm['pitch']['level']}")

    # Extract features for a high-pitch segment
    print("\n" + "-" * 70)
    print("Test Segment 2: High pitch (240 Hz)")
    print("-" * 70)
    audio2 = generate_sample_audio(sr=sr, frequency=240)
    text2 = "This segment has unusually high pitch for this speaker."
    features2 = extract_prosody(audio2, sr, text2)
    print(f"Without baseline: Pitch level = {features2['pitch']['level']}")

    features2_norm = extract_prosody(audio2, sr, text2, speaker_baseline=baseline)
    print(f"With baseline:    Pitch level = {features2_norm['pitch']['level']}")
    print("\nNotice how baseline normalization provides speaker-relative categories!")


def demo_speech_variations():
    """Demonstrate prosody extraction for different speech styles."""
    print("\n" + "=" * 70)
    print("DEMO 3: Different Speech Styles")
    print("=" * 70)

    sr = 16000

    # Scenario 1: Calm, slow speech
    print("\n" + "-" * 70)
    print("Scenario 1: Calm, slow speech")
    print("-" * 70)
    audio_calm = generate_sample_audio(sr=sr, duration=3.0, frequency=160, amplitude=0.2)
    text_calm = "This is calm slow speech."
    features_calm = extract_prosody(audio_calm, sr, text_calm)
    print(
        f"Pitch: {features_calm['pitch']['level']} " f"({features_calm['pitch']['mean_hz']:.1f} Hz)"
    )
    print(
        f"Energy: {features_calm['energy']['level']} "
        f"({features_calm['energy']['db_rms']:.1f} dB)"
    )
    print(
        f"Rate: {features_calm['rate']['level']} "
        f"({features_calm['rate']['syllables_per_sec']:.1f} syl/sec)"
    )

    # Scenario 2: Excited, fast speech
    print("\n" + "-" * 70)
    print("Scenario 2: Excited, fast speech")
    print("-" * 70)
    audio_excited = generate_sample_audio(sr=sr, duration=1.0, frequency=250, amplitude=0.5)
    text_excited = "This is excited fast speech wow!"
    features_excited = extract_prosody(audio_excited, sr, text_excited)
    print(
        f"Pitch: {features_excited['pitch']['level']} "
        f"({features_excited['pitch']['mean_hz']:.1f} Hz)"
    )
    print(
        f"Energy: {features_excited['energy']['level']} "
        f"({features_excited['energy']['db_rms']:.1f} dB)"
    )
    print(
        f"Rate: {features_excited['rate']['level']} "
        f"({features_excited['rate']['syllables_per_sec']:.1f} syl/sec)"
    )

    # Scenario 3: Question (rising intonation)
    print("\n" + "-" * 70)
    print("Scenario 3: Question with rising intonation")
    print("-" * 70)
    # Generate audio with rising pitch
    t = np.linspace(0, 1.5, int(sr * 1.5))
    frequency = 180 + 40 * t  # Rise from 180 to 220 Hz
    audio_question = np.zeros_like(t)
    for i, f in enumerate(frequency):
        audio_question[i] = 0.3 * np.sin(2 * np.pi * f * t[i])
    text_question = "Are you asking a question here?"
    features_question = extract_prosody(audio_question.astype(np.float32), sr, text_question)
    print(
        f"Pitch: {features_question['pitch']['level']} - "
        f"Contour: {features_question['pitch']['contour']}"
    )
    print(f"Energy: {features_question['energy']['level']}")


def demo_json_schema():
    """Demonstrate JSON schema compliance."""
    print("\n" + "=" * 70)
    print("DEMO 4: JSON Schema Output")
    print("=" * 70)

    sr = 16000
    audio = generate_sample_audio(sr=sr, duration=2.0)
    text = "This demonstrates the JSON schema output format."

    features = extract_prosody(audio, sr, text)

    print("\nJSON Schema-Compliant Output:")
    print(json.dumps(features, indent=2))

    print("\nThis output can be directly serialized to JSON and stored")
    print("alongside transcription data for further analysis or display.")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║        Prosody Feature Extraction - Demonstration Suite           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    try:
        demo_basic_extraction()
        demo_baseline_normalization()
        demo_speech_variations()
        demo_json_schema()

        print("\n" + "=" * 70)
        print("All demonstrations completed successfully!")
        print("=" * 70)
        print("\nNext Steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run tests: python -m pytest tests/test_prosody.py")
        print("  3. Integrate with your transcription pipeline")
        print("  4. Customize thresholds and parameters as needed")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("\nNote: Some features require optional dependencies:")
        print("  - praat-parselmouth: For pitch extraction")
        print("  - librosa: For energy and advanced audio processing")
        print("\nInstall with: pip install praat-parselmouth librosa")


if __name__ == "__main__":
    main()
