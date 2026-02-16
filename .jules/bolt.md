## 2024-05-23 - Thread Pool Overhead Optimization
**Learning:** Checking task preconditions (like file timestamps) *before* submitting to `ThreadPoolExecutor` significantly reduces overhead for up-to-date states.
**Action:** Extract precondition checks to helper functions and run them in the main thread loop. Measured ~74% speedup for 5000 no-op files (1.15s -> 0.30s).
