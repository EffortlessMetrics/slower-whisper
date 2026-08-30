## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2024-01-28 - Unvalidated Path Input in ffmpeg
**Vulnerability:** Unvalidated user input (audio file path) was passed directly to subprocess `ffprobe`, allowing potential option injection (e.g. `-filename`).
**Learning:** Subprocess arguments must be validated, particularly checking for leading hyphens to avoid unintended flags.
**Prevention:** Use `_validate_path_safety` before passing paths to subprocesses.
