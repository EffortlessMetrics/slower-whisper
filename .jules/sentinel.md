## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-03-28 - Prevent SQL Injection via ORDER BY Clause
**Vulnerability:** Dynamic column names in ORDER BY clauses were vulnerable to SQL injection because they cannot be parameterized with ? placeholders.
**Learning:** SQLite cannot parameterize column or table names. If user input specifies an ordering column, it must be validated against a strict allowlist before string interpolation.
**Prevention:** Use an allowlist to validate user-supplied column names before concatenating them into the SQL query string.
