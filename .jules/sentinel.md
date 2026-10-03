## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-05-11 - SQL Injection in Dynamic ORDER BY
**Vulnerability:** SQL Injection via dynamically constructed `ORDER BY` clause in SQLite search query.
**Learning:** `ORDER BY` clauses cannot be parameterized in SQLite using standard prepared statements (`?`).
**Prevention:** Always use a hardcoded string exact-match allowlist for sorting column names and map them safely to table aliases before constructing the SQL query.
