## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-02-09 - Path Validation for Subprocess Option Injection
**Vulnerability:** Option injection vulnerability via user-controlled file paths passed to `subprocess.run` (e.g., `ffprobe`).
**Learning:** Even when using `subprocess.run` with a list of arguments (which prevents shell injection), an attacker can supply a filename starting with `-` to inject options into the command.
**Prevention:** Use `_validate_path_safety` before executing external tools on user-provided paths, and catch `ValueError` to raise appropriate HTTP errors in endpoints.
