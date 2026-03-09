## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-01-29 - Subprocess Option Injection
**Vulnerability:** Option injection via filenames starting with `-` bypassing list-based `subprocess.run` protection in FastAPI upload validation.
**Learning:** Using list-based arguments in `subprocess.run` prevents shell injection (e.g., `; rm -rf /`), but it does not prevent option injection where a program (like `ffprobe`) interprets an argument starting with `-` as a command-line flag instead of a file path.
**Prevention:** Always validate user-provided paths using `_validate_path_safety` before passing them to external tools, and handle potential errors securely by catching `ValueError` and raising generic `HTTPException(400)` to prevent unhandled server errors.
