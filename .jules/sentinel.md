## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-02-18 - Fallback Validation Security
**Vulnerability:** File upload validation relied on `ffprobe`, with a Python fallback that only checked file extensions for certain formats (OGG, FLAC, WMA). This allowed malicious files to bypass validation if `ffprobe` was missing or failed.
**Learning:** Fallback mechanisms must provide equivalent security guarantees to the primary mechanism. A "soft fail" to a weaker check creates a bypass vector.
**Prevention:** Implement rigorous content-based validation (e.g., magic bytes) in all fallback paths, ensuring no file type is accepted based on extension alone.
