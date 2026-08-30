## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-05-06 - Gitleaks Action Node Deprecation
**Vulnerability:** Gitleaks action v2 throws Node 20 deprecation warnings and missing license errors if not configured properly, and downgrading introduces severe security risks.
**Learning:** Never downgrade security actions like `gitleaks-action` to bypass checks or deprecation warnings. Instead, fix the underlying license issue by passing the expected GitHub Secret token to the action environment block.
**Prevention:** Fix gitleaks license errors by injecting the `GITLEAKS_LICENSE` environment variable via `${{ secrets.GITLEAKS_LICENSE }}`.
