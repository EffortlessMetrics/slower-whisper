## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-01-29 - SQLite Injection in ORDER BY Clause
**Vulnerability:** SQL injection vulnerability via string concatenation in StoreQuery's order_by field.
**Learning:** Standard SQL parameters (?) cannot be used for column names or identifiers like ORDER BY clauses.
**Prevention:** Always validate user-provided identifiers (column names) against a strict allowlist before concatenating them into SQL strings.
