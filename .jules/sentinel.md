## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-03-19 - SQL Injection in FTS5 Order By
**Vulnerability:** SQL injection vector in sqlite FTS5 StoreQuery via unparameterized order_by string.
**Learning:** Database drivers (like sqlite3) cannot parameterize column names or direction keywords (ASC/DESC) in ORDER BY clauses. Simply concatenating user input is inherently insecure.
**Prevention:** Always use a strict allowlist to validate user-provided column strings before concatenating them into dynamic SQL strings.
