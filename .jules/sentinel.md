## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-06-08 - SQL Injection in StoreQuery ORDER BY
**Vulnerability:** Direct string interpolation of the `order_by` field in `StoreQuery` into a SQLite SQL string, allowing arbitrary SQL expressions to be evaluated during sort.
**Learning:** Validating dynamic query properties against a strict allowlist is required because parameterized queries (?) do not work for identifiers like column names.
**Prevention:** Always validate dynamic column names or table names against an explicit allowlist before interpolating them into SQL strings.
