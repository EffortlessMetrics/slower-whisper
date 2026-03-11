## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-01-29 - Option Injection in Subprocess
**Vulnerability:** Option injection risk despite using list-based `subprocess.run` arguments when user-provided path starts with a hyphen (e.g., `-i`).
**Learning:** List arguments prevent shell injection but do not inherently stop command option injection. Tools like ffmpeg/ffprobe may incorrectly parse paths starting with `-` as CLI flags.
**Prevention:** Always sanitize/validate paths using `_validate_path_safety` before passing them to external command executions.
