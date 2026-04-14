import numpy as np

def _detect_speech_frames(audio: np.ndarray, frame_size_ms: int = 30) -> list[bool]:
    frame_size = int(16000 * frame_size_ms / 1000)
    num_frames = len(audio) // frame_size

    if num_frames == 0:
        return []

    truncated = audio[: num_frames * frame_size]
    frames = truncated.reshape(num_frames, frame_size)
    energies = np.sqrt(np.mean(frames**2, axis=1))

    return [bool(x) for x in (energies > 0.01)]
