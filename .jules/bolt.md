## 2024-05-23 - Pipeline I/O Redundancy
**Learning:** The pipeline loop was redundantly loading JSON files multiple times when multiple features (chunking, diarization) were enabled. Refactoring to load once and pass the object in memory eliminates this overhead.
**Action:** Always check if an object loaded from disk can be reused across multiple steps in a processing loop instead of reloading it for each step.
