## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-03-05 - Missing Path Safety Validation in Subprocess Command
**Vulnerability:** Path inputs provided directly from service endpoints to the ffmpeg validation subprocess were not validated against path safety checks. This exposed the potential for argument option injection or unsafe filesystem interaction.
**Learning:** While list-based `subprocess.run` calls mitigate direct shell injection, options manipulation (e.g. paths starting with `-`) remain a viable attack vector in utilities like `ffmpeg` or `ffprobe`.
**Prevention:** Ensure `_validate_path_safety(audio_path)` (or an equivalent sanitization logic) is strictly applied across all execution paths prior to subprocess calls, converting subsequent validation failures into safe, handled exceptions (e.g. `HTTPException 400`).
