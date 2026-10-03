## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-05-12 - SQL Injection in StoreQuery order_by
**Vulnerability:** SQL injection via the `order_by` field in `StoreQuery` due to direct string interpolation into the `ORDER BY` clause.
**Learning:** Dynamic `ORDER BY` clauses cannot use parameterized queries (prepared statements).
**Prevention:** Always implement a strict, hardcoded exact-match string allowlist for all permitted sorting columns and aliases.
