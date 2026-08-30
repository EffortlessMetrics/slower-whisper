## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-06-06 - SQL Injection in SQLite ORDER BY Clauses
**Vulnerability:** SQL Injection via un-sanitized string concatenation in `ORDER BY` clauses when passing user input to `sqlite3`.
**Learning:** SQLite cannot parameterize column names in `ORDER BY` clauses (e.g., `ORDER BY ?` does not work as intended for column names), which frequently leads to dangerous string concatenation patterns.
**Prevention:** Always implement a strict exact-match string allowlist for column names when dynamic sorting is required. Never use prepared statement parameterization for column identifiers.
