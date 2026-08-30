## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2025-05-02 - SQLite FTS5 ORDER BY SQL Injection
**Vulnerability:** The `order_by` property of `StoreQuery` was directly concatenated into the `ConversationStore.search()` SQLite query: `f"ORDER BY {order_col} {order_dir}"`.
**Learning:** In standard SQL implementations, including SQLite, `ORDER BY` clauses cannot be parameterized safely. Using dynamic field names directly from user/API input introduces a vulnerability where attackers can inject arbitrary subqueries or manipulate sort logic.
**Prevention:** Always use a strict, hardcoded allowlist to validate user-supplied column names before using them in dynamic `ORDER BY` or `GROUP BY` clauses.
