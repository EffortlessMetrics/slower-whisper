## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-01-29 - SQL Injection in ORDER BY clause
**Vulnerability:** SQL injection via string concatenation in the `ORDER BY` clause of the search method (`sql_parts.append(f"ORDER BY {order_col} {order_dir}")`).
**Learning:** Even if `WHERE` clauses correctly use parameterized queries (`?`), `ORDER BY` and `GROUP BY` column names cannot be parameterized in SQLite. Allowing user input here directly leads to SQL injection.
**Prevention:** Always validate column names against a strict allowlist of allowed columns before concatenating them into the query.
