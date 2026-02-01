## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-01-29 - FastAPI Memory Exhaustion via UploadFile.read()
**Vulnerability:** Unbounded memory consumption when reading uploaded files using `await UploadFile.read()` in FastAPI endpoints.
**Learning:** `UploadFile.read()` loads the entire file into RAM. Even if `validate_file_size` is called afterwards, the damage is already done (OOM risk).
**Prevention:** Always use streaming writes (e.g., `save_upload_file_streaming`) to save uploads to disk with a chunk-based size limit before processing.
