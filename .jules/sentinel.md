## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2025-02-14 - Prevent Command Option Injection in subprocess.run

**Vulnerability:** Even when using `subprocess.run` with a list of arguments (which avoids standard shell injection), passing an unsanitized user-provided path as an argument to an external binary (like `ffprobe`) can still lead to "Option Injection". If the user-provided filename starts with a dash (e.g., `-unsafe_flag`), the executable may interpret it as a command-line option rather than a positional file path argument.
**Learning:** `subprocess.run(["cmd", user_path])` is not completely safe if `user_path` can start with a `-`. This specific codebase has an existing defense-in-depth utility, `transcription.audio_io._validate_path_safety`, which explicitly blocks leading dashes and shell metacharacters.
**Prevention:** Always use `_validate_path_safety` before passing paths to `subprocess.run`, or ensure arguments are explicitly separated from positional arguments using `--` (e.g., `["cmd", "--", user_path]`) when the target CLI supports it.
