## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-02-05 - DoS Protection & Path Safety
**Vulnerability:** Unbounded file read in `enrich_audio` endpoint allowed reading potentially large transcript files entirely into memory, leading to DoS via memory exhaustion. Also identified potential path injection risk in validation utilities.
**Learning:** `UploadFile.read()` without size limit is dangerous for user inputs. Always use `save_upload_file_streaming` or bounded reads. Defense-in-depth requires validating paths even when they come from "trusted" sources like tempfile, as utility functions might be reused.
**Prevention:** Enforced `save_upload_file_streaming` for all file uploads. Exposed and utilized `validate_path_safety` in validation modules.
