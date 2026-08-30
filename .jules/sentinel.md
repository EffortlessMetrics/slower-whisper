## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2025-04-04 - SQLite ORDER BY Parameterization SQL Injection
**Vulnerability:** SQL injection via unparameterized `order_by` parameter appended directly to `ORDER BY` clause using string concatenation (`f"ORDER BY {order_col}"`) in `ConversationStore.search()`.
**Learning:** Standard SQLite parameterization (using `?` placeholders) does not work for structural parts of SQL queries, such as table names or column names in `ORDER BY` or `GROUP BY` clauses. Direct string concatenation of these variables allows arbitrary SQL execution if the input is untrusted.
**Prevention:** Always validate dynamically provided column names against a strict allowlist before using them in `ORDER BY` or other structural SQL clauses. Do not rely on parameterization for column names.
