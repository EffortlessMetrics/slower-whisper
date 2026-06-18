## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-01-29 - CORS Configuration Missing
**Vulnerability:** The FastAPI service lacked Cross-Origin Resource Sharing (CORS) middleware configuration, preventing secure integration with frontend clients and potentially exposing endpoints if not properly restricted.
**Learning:** By default, FastAPI doesn't add CORS headers, which either breaks valid cross-origin requests (like WebSockets) or requires explicit configuration. A default-deny approach via `allow_origins=[]` unless explicitly configured via `SLOWER_WHISPER_ALLOWED_ORIGINS` is safer.
**Prevention:** Always configure `CORSMiddleware` when exposing REST APIs or WebSockets, using environment-driven allowlists instead of wildcard `["*"]`.
