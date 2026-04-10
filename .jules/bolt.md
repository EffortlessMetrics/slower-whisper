## 2024-10-27 - Vectorizing frame-wise audio processing loops
**Learning:** Using Python `for` loops to iterate over and slice numpy arrays drops out of C-level vectorization, creating a massive bottleneck in hot paths like VAD energy calculations.
**Action:** Use `np.reshape` to convert the 1D audio array into a 2D `(num_frames, frame_size)` array, then use `axis=1` operations (e.g., `np.mean(frames**2, axis=1)`) to maintain C-level performance. Ensure boolean numpy arrays are cast properly using list comprehensions (`[bool(x) for x in array]`) to satisfy strict typing.
