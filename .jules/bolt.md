## 2024-05-24 - [Vectorizing Audio Slicing]
**Learning:** Python `for` loops slicing `numpy` arrays (e.g. `audio[i*size:(i+1)*size]`) in hot audio processing paths cause significant performance bottlenecks.
**Action:** Replace `for` loop slicing with vectorized operations using `np.reshape()` and `axis=1` operations (e.g. `np.mean(frames**2, axis=1)`) to achieve >30x C-level speedups. Make sure to cast NumPy booleans to Python `bool` if `list[bool]` typing is strictly required.
