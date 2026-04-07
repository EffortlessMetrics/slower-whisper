import numpy as np

def detect_speech_frames_orig(audio: np.ndarray, sample_rate, threshold, frame_size_ms: int = 30):
    frame_size = int(sample_rate * frame_size_ms / 1000)
    num_frames = len(audio) // frame_size

    def calc_energy(frame):
        if len(frame) == 0: return 0.0
        return float(np.sqrt(np.mean(frame**2)))

    speech_frames = []
    for i in range(num_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        energy = calc_energy(frame)
        speech_frames.append(energy > threshold)

    return speech_frames

def detect_speech_frames_new(audio: np.ndarray, sample_rate, threshold, frame_size_ms: int = 30):
    frame_size = int(sample_rate * frame_size_ms / 1000)
    num_frames = len(audio) // frame_size

    if num_frames == 0:
        return []

    # Reshape audio into frames
    frames = audio[:num_frames * frame_size].reshape(num_frames, frame_size)

    # Calculate energy across frames using vectorized operations
    energies = np.sqrt(np.mean(frames**2, axis=1))

    # Compare with threshold
    is_speech = energies > threshold

    return [bool(x) for x in is_speech]

# Test
np.random.seed(42)
audio = np.random.randn(16000 * 5).astype(np.float32) # 5 seconds
orig = detect_speech_frames_orig(audio, 16000, 0.5)
new = detect_speech_frames_new(audio, 16000, 0.5)

print(f"Orig length: {len(orig)}, New length: {len(new)}")
print(f"Equal? {orig == new}")

# Benchmark
import time
t0 = time.time()
for _ in range(100):
    detect_speech_frames_orig(audio, 16000, 0.5)
t1 = time.time()
print(f"Orig: {t1 - t0:.4f}s")

t0 = time.time()
for _ in range(100):
    detect_speech_frames_new(audio, 16000, 0.5)
t1 = time.time()
print(f"New: {t1 - t0:.4f}s")
