# Sentinel Journal

## 2025-05-27 - Path Safety Verification Pattern
**Vulnerability:** Potential for path traversal or shell injection if filename sanitization logic is duplicated or bypassed in service endpoints calling `subprocess.run` (e.g., `ffprobe`).
**Learning:** Reliance on ad-hoc filename construction in API endpoints creates maintenance risks. Even internally generated paths should be explicitly validated before being passed to shell commands.
**Prevention:**
1.  Centralized `get_safe_audio_extension` helper to enforce allowed extensions.
2.  Explicit `validate_path_safety` call immediately before any `subprocess.run` invocation involving file paths, serving as a final gatekeeper.
