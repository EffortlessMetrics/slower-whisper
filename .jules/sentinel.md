## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-03-31 - [SQL Injection in StoreQuery]
**Vulnerability:** Dynamic column names in ORDER BY clauses without allowlisting.
**Learning:** SQLite standard placeholders (?) cannot parameterize column names in ORDER BY. This leads to SQL injection.
**Prevention:** Always validate user-provided column names against a strict allowlist of known safe columns before concatenating them into SQL strings.
