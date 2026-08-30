## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2024-04-06 - SQL Injection in StoreQuery.order_by
**Vulnerability:** The `order_by` parameter in `StoreQuery` was directly interpolated into the SQL query without validation.
**Learning:** Standard SQLite parameterization (`?`) does not support dynamic column names in `ORDER BY` clauses.
**Prevention:** Always use a strict allowlist to validate user-provided column names before string interpolation in queries.
