"""
Generate synthetic 2-speaker audio for diarization testing.

This fixture creates a deterministic audio file with two distinct speakers
alternating in a known pattern. It's used to validate diarization mapping
logic without requiring a pretrained model.

Pattern:
    Speaker A (120 Hz tone): 0.0-3.0s, 6.2-9.2s
    Speaker B (220 Hz tone): 3.2-6.2s, 9.4-12.4s

Expected speaker turns after diarization:
    [
        SpeakerTurn(start=0.0, end=3.0, speaker_id="SPEAKER_00"),
        SpeakerTurn(start=3.2, end=6.2, speaker_id="SPEAKER_01"),
        SpeakerTurn(start=6.2, end=9.2, speaker_id="SPEAKER_00"),
        SpeakerTurn(start=9.4, end=12.4, speaker_id="SPEAKER_01"),
    ]

Usage:
    uv run python tests/fixtures/generate_synthetic_2speaker.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Optional dependency: only needed for fixture generation, not tests
try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

SR = 16_000  # 16 kHz mono (Whisper standard)


def tone(f_hz: float, duration: float) -> np.ndarray:
    """
    Generate a pure sine wave tone.

    Args:
        f_hz: Frequency in Hz.
        duration: Duration in seconds.

    Returns:
        Audio samples as float32 array.
    """
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return 0.1 * np.sin(2 * np.pi * f_hz * t).astype(np.float32)


def silence(duration: float) -> np.ndarray:
    """
    Generate silence (zeros).

    Args:
        duration: Duration in seconds.

    Returns:
        Zero-filled float32 array.
    """
    return np.zeros(int(SR * duration), dtype=np.float32)


def main(out_path: Path) -> None:
    """
    Generate synthetic_2speaker.wav fixture.

    Pattern:
        Speaker A: 3s, gap 0.2s, Speaker B: 3s, gap 0.2s,
        Speaker A: 3s, gap 0.2s, Speaker B: 3s

    Total duration: ~12.6s
    """
    if not SOUNDFILE_AVAILABLE:
        raise ImportError(
            "soundfile is required to generate fixtures. Install with: uv sync --extra enrich-basic"
        )

    # Speaker A: 120 Hz tone (low pitch)
    a1 = tone(120.0, 3.0)  # 0.0 - 3.0s
    a2 = tone(120.0, 3.0)  # 6.2 - 9.2s

    # Speaker B: 220 Hz tone (higher pitch)
    b1 = tone(220.0, 3.0)  # 3.2 - 6.2s
    b2 = tone(220.0, 3.0)  # 9.4 - 12.4s

    gap = silence(0.2)  # 200ms gaps between speakers

    # Concatenate in A-B-A-B pattern
    audio = np.concatenate([a1, gap, b1, gap, a2, gap, b2], axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, SR)

    print(f"Generated {out_path} ({len(audio) / SR:.2f}s)")
    print("Expected speaker turns:")
    print("  A: 0.0-3.0s, 6.2-9.2s")
    print("  B: 3.2-6.2s, 9.4-12.4s")


if __name__ == "__main__":
    out = Path(__file__).parent / "synthetic_2speaker.wav"
    main(out)
