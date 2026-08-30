## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-01-29 - SQLite ORDER BY Parameterization & SQL Injection
**Vulnerability:** Dynamic column names in ORDER BY clauses in SQLite cannot be parameterized using standard ? placeholders, leading to potential SQL injection if user input is concatenated into the SQL string.
**Learning:** Relying on ORM parameterization or ? placeholders for column names is insufficient for ORDER BY clauses and leaves the query vulnerable to option injection.
**Prevention:** To prevent SQL injection attacks in ORDER BY clauses, always validate user-provided column names against a strict allowlist of known safe columns before concatenating them into SQL strings.
