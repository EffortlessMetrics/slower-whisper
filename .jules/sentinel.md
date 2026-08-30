## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-01-30 - Fallback Validation Security
**Vulnerability:** Weak Python-based fallback validation for audio files allowed invalid files (e.g., text files masked as MP3) to pass when `ffprobe` was missing or failed.
**Learning:** Fallback mechanisms often receive less scrutiny than primary paths. When a robust tool (`ffprobe`) fails, falling back to a weak implementation (simple extension/header check that logs warnings but doesn't block) creates a bypass.
**Prevention:** Ensure fallback validation logic is as strict as the primary method. If a check fails (like an invalid header), it must raise an exception, not just log a warning.
