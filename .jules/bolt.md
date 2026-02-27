# BOLT'S JOURNAL - CRITICAL LEARNINGS ONLY

## 2024-05-23 - [Start]
**Learning:** Initialized Bolt's journal.
**Action:** Always check for this file before starting.

## 2024-05-23 - [ThreadPoolExecutor Overhead]
**Learning:** `normalize_all` submits ALL files to the ThreadPoolExecutor before checking if they are up-to-date inside the worker function. This incurs significant overhead when most files are already normalized.
**Action:** Filter files by timestamp *before* submitting to the executor. This avoids thread creation and context switching for files that don't need processing.
