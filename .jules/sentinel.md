## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-01-29 - SQLite SQL Injection in ORDER BY
**Vulnerability:** SQL injection vulnerability in `ConversationStore.search()` where the `order_by` field was directly interpolated into the query via an f-string (`f"ORDER BY {order_col} {order_dir}"`).
**Learning:** Standard SQLite parameterization (`?`) does not support dynamic column names in `ORDER BY` clauses, making string interpolation a common but dangerous pattern if user input is not validated.
**Prevention:** Always use a strict allowlist (whitelist) of permitted column names to validate user-provided sorting or grouping keys before using them in SQL string interpolation.
