## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-01-28 - FastAPI CORS Configuration
**Vulnerability:** Missing Cross-Origin Resource Sharing (CORS) protection in FastAPI service.
**Learning:** Default FastAPI has no CORS restrictions, allowing unauthorized cross-origin requests by default if exposed to a browser context.
**Prevention:** Always add `CORSMiddleware` with explicit, strict allowed origins (e.g., specific localhost ports during development) rather than relying on defaults or using wildcard `*` alongside credentials.
