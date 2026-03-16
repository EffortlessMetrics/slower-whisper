## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-02-23 - Subprocess Path Injection
**Vulnerability:** Command and Option injection when calling `ffprobe` in `validate_audio_format`.
**Learning:** While using an argument list in `subprocess.run` mitigates shell injection, it does not prevent option injection where a user provides a path like `-filename.mp3`, tricking `ffprobe` into treating it as a command line flag.
**Prevention:** Always use `_validate_path_safety(audio_path)` from `transcription.audio_io` to validate inputs before using them in external processes, specifically catching `ValueError` to raise safe, generic HTTP errors.