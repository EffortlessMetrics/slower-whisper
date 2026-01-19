## 2025-11-17 - [FFmpeg Option Injection]
**Vulnerability:** Filenames starting with `-` can be interpreted as flags by ffmpeg even when using list arguments in subprocess.
**Learning:** `subprocess.run(["ffmpeg", "-i", "-filename"])` treats `-filename` as a flag.
**Prevention:** Validate paths do not start with `-` or ensure they are prefixed (e.g., `./-filename`).
