## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-06-26 - CORS Configuration
**Vulnerability:** Missing Cross-Origin Resource Sharing (CORS) policy leaves APIs open to potential abuse from any web origin.
**Learning:** Default API configuration exposes REST and WebSocket endpoints to all domains if not explicitly secured.
**Prevention:** Use `CORSMiddleware` with `allow_origins` bound to an environment variable (`SLOWER_WHISPER_ALLOWED_ORIGINS`) with an empty default to enforce a strict same-origin policy securely.
