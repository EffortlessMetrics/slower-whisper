## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-06-15 - Fix SQL Injection in ORDER BY Clause
**Vulnerability:** SQL Injection via string interpolation in the SQLite `ORDER BY` clause of `StoreQuery`.
**Learning:** Parameterized queries (`?`) cannot be used for column names or `ORDER BY` fields. Directly interpolating user-controlled input like `query.order_by` allows attackers to inject arbitrary subqueries (e.g., Boolean-based blind SQLi).
**Prevention:** Always validate column names against a strict allowlist of known, safe columns before interpolating them into a SQL query.
