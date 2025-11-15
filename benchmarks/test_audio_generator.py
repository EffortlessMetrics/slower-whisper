"""
Test audio generator for benchmarking.

This module generates synthetic audio files with speech-like characteristics
for benchmarking the audio enrichment pipeline without requiring real audio files.
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def generate_sine_wave(
    frequency: float, duration: float, sample_rate: int, amplitude: float = 0.5
) -> np.ndarray:
    """
    Generate a sine wave.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0.0 to 1.0)

    Returns:
        Audio samples as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)


def generate_speech_like_signal(
    duration: float,
    sample_rate: int = 16000,
    base_pitch: float = 150.0,
    pitch_variation: float = 50.0,
    syllables_per_second: float = 5.0,
    add_pauses: bool = True,
) -> np.ndarray:
    """
    Generate synthetic audio with speech-like characteristics.

    This creates audio that mimics human speech patterns with:
    - Varying pitch (fundamental frequency)
    - Syllabic rhythm
    - Energy variations
    - Optional pauses

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        base_pitch: Base pitch in Hz (typical: 100-250 Hz)
        pitch_variation: Pitch variation range in Hz
        syllables_per_second: Speech rate (typical: 4-7 syllables/sec)
        add_pauses: Whether to add pauses between phrases

    Returns:
        Audio samples as numpy array (mono, float32)
    """
    num_samples = int(duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    # Time array
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Generate pitch contour (varying F0)
    # Use low-frequency modulation to simulate natural pitch variation
    pitch_contour = base_pitch + pitch_variation * np.sin(2 * np.pi * 0.3 * t)

    # Generate syllabic envelope
    # Create amplitude modulation at syllable rate
    syllable_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * syllables_per_second * t)

    # Generate formant-like structure
    # Human speech has multiple formants (resonant frequencies)
    # We'll simulate simplified formants
    formant1 = generate_sine_wave(800, duration, sample_rate, amplitude=0.3)  # F1
    formant2 = generate_sine_wave(1200, duration, sample_rate, amplitude=0.2)  # F2
    formant3 = generate_sine_wave(2500, duration, sample_rate, amplitude=0.1)  # F3

    # Generate fundamental frequency (pitch)
    # Use pitch contour to modulate
    phase = np.cumsum(pitch_contour / sample_rate)
    fundamental = 0.4 * np.sin(2 * np.pi * phase)

    # Add harmonics for richer sound
    harmonic2 = 0.2 * np.sin(4 * np.pi * phase)
    harmonic3 = 0.1 * np.sin(6 * np.pi * phase)

    # Combine components
    audio = fundamental + harmonic2 + harmonic3 + formant1 + formant2 + formant3

    # Apply syllabic envelope
    audio *= syllable_envelope

    # Add pauses if requested
    if add_pauses:
        # Add a pause every 3-5 seconds
        pause_interval = np.random.uniform(3.0, 5.0)
        pause_duration = 0.3  # 300ms pauses

        current_time = pause_interval
        while current_time < duration:
            pause_start = int(current_time * sample_rate)
            pause_end = int((current_time + pause_duration) * sample_rate)
            pause_end = min(pause_end, num_samples)

            # Fade out before pause
            fade_samples = int(0.05 * sample_rate)  # 50ms fade
            if pause_start > fade_samples:
                fade_start = pause_start - fade_samples
                fade = np.linspace(1, 0, fade_samples)
                audio[fade_start:pause_start] *= fade

            # Silence during pause
            audio[pause_start:pause_end] = 0

            # Fade in after pause
            if pause_end + fade_samples < num_samples:
                fade = np.linspace(0, 1, fade_samples)
                audio[pause_end : pause_end + fade_samples] *= fade

            current_time += pause_interval + pause_duration

    # Add some noise for realism (very low level)
    noise = np.random.normal(0, 0.01, num_samples)
    audio += noise

    # Normalize to prevent clipping
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.8  # Leave some headroom

    return audio.astype(np.float32)


def generate_varying_audio(
    duration: float, sample_rate: int = 16000, num_variations: int = 5
) -> np.ndarray:
    """
    Generate audio with varying characteristics over time.

    This creates segments with different pitch, energy, and rate characteristics
    to test the feature extraction robustness.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        num_variations: Number of different variation segments

    Returns:
        Audio samples as numpy array
    """
    segment_duration = duration / num_variations
    audio_segments = []

    for _ in range(num_variations):
        # Vary parameters for each segment
        base_pitch = np.random.uniform(100, 200)
        pitch_variation = np.random.uniform(20, 60)
        syllables_per_sec = np.random.uniform(3.5, 6.5)

        segment = generate_speech_like_signal(
            duration=segment_duration,
            sample_rate=sample_rate,
            base_pitch=base_pitch,
            pitch_variation=pitch_variation,
            syllables_per_second=syllables_per_sec,
            add_pauses=True,
        )

        audio_segments.append(segment)

    # Concatenate segments with cross-fades
    fade_samples = int(0.1 * sample_rate)  # 100ms cross-fade

    result = audio_segments[0]
    for segment in audio_segments[1:]:
        # Cross-fade
        fade_out = np.linspace(1, 0, fade_samples)
        fade_in = np.linspace(0, 1, fade_samples)

        result[-fade_samples:] *= fade_out
        segment[:fade_samples] *= fade_in

        # Overlap and add
        result[-fade_samples:] += segment[:fade_samples]

        # Concatenate rest
        result = np.concatenate([result, segment[fade_samples:]])

    return result.astype(np.float32)


def generate_test_audio_file(
    output_path: Path,
    duration_seconds: float,
    sample_rate: int = 16000,
    add_speech_characteristics: bool = True,
) -> Path:
    """
    Generate a test audio file for benchmarking.

    Args:
        output_path: Path where to save the audio file
        duration_seconds: Duration of the audio file
        sample_rate: Sample rate in Hz (default: 16000)
        add_speech_characteristics: If True, generate speech-like audio;
                                   if False, generate simple tone

    Returns:
        Path to the generated file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    logger.info(f"Generating test audio: {duration_seconds}s at {sample_rate}Hz")

    if add_speech_characteristics:
        # Generate speech-like audio with variations
        if duration_seconds > 30:
            # Use varying characteristics for longer audio
            num_variations = max(5, int(duration_seconds / 60))
            audio = generate_varying_audio(
                duration=duration_seconds, sample_rate=sample_rate, num_variations=num_variations
            )
        else:
            # Simple speech-like audio for short durations
            audio = generate_speech_like_signal(
                duration=duration_seconds,
                sample_rate=sample_rate,
                base_pitch=150.0,
                pitch_variation=50.0,
                syllables_per_second=5.0,
                add_pauses=True,
            )
    else:
        # Simple sine wave for basic testing
        audio = generate_sine_wave(
            frequency=440.0, duration=duration_seconds, sample_rate=sample_rate, amplitude=0.5
        )

    # Save to file
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Generated: {output_path} ({file_size_mb:.2f} MB)")

    return output_path


def generate_benchmark_test_suite(output_dir: Path):
    """
    Generate a standard suite of test audio files for benchmarking.

    Creates test files for common durations:
    - 10 seconds
    - 1 minute (60s)
    - 5 minutes (300s)
    - 30 minutes (1800s)

    Args:
        output_dir: Directory where to save test files

    Returns:
        Dictionary mapping duration to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    durations = [10, 60, 300, 1800]
    files = {}

    logger.info(f"Generating benchmark test suite in: {output_dir}")

    for duration in durations:
        file_path = output_dir / f"test_audio_{duration}s.wav"

        if file_path.exists():
            logger.info(f"Skipping existing file: {file_path}")
        else:
            generate_test_audio_file(
                output_path=file_path,
                duration_seconds=duration,
                sample_rate=16000,
                add_speech_characteristics=True,
            )

        files[duration] = file_path

    logger.info(f"Test suite generation complete: {len(files)} files")
    return files


# CLI for standalone usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate test audio files for benchmarking")
    parser.add_argument("--duration", type=float, required=True, help="Duration in seconds")
    parser.add_argument("--output", type=Path, required=True, help="Output file path")
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Generate simple sine wave instead of speech-like audio",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    generate_test_audio_file(
        output_path=args.output,
        duration_seconds=args.duration,
        sample_rate=args.sample_rate,
        add_speech_characteristics=not args.simple,
    )
