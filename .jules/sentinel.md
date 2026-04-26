## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-01-28 - SQL Injection in ORDER BY
**Vulnerability:** SQL injection vulnerability via string formatting in the `ORDER BY` clause of `ConversationStore.search()`. The `order_col` parameter was concatenated directly into the query string: `f"ORDER BY {order_col} {order_dir}"`.
**Learning:** Using raw string manipulation or string interpolation for `ORDER BY` columns in SQLite leads to potential SQL injection using CASE expressions or inline comments `/*.id`, even if you try to split or sanitize the string naively.
**Prevention:** Always use strict explicit allowlisting (e.g. `if order_col not in allowed_columns: raise QueryError()`) to validate `ORDER BY` column parameters before appending them to the query string, and ensure exact string matching rather than substring searching.
