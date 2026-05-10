## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2024-05-10 - SQL Injection in ORDER BY Clause
**Vulnerability:** Dynamic string interpolation in SQL `ORDER BY` clauses allows SQL injection.
**Learning:** Parameterized queries (`?`) cannot be used to parameterize column names in `ORDER BY` clauses. Direct string interpolation was used, leaving the code vulnerable to injection.
**Prevention:** Implement a strict, hardcoded exact-match string allowlist for all permitted sorting columns and aliases before interpolating them into the SQL query string.
