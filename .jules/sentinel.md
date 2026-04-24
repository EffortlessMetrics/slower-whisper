## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2024-05-28 - SQL Injection in Dynamic ORDER BY
**Vulnerability:** The `order_by` parameter in `StoreQuery` was interpolated directly into a SQLite FTS5 search query (`f"ORDER BY {order_col} {order_dir}"`) without validation.
**Learning:** `ORDER BY` clauses typically cannot be parameterized natively by the DB engine, making dynamic sorting inputs a common vector for SQL injection.
**Prevention:** Always use strict exact string match allowlists for sort columns. Ensure the allowlist explicitly includes valid table aliases (e.g., `s.start_time`) to match how the query engine references them, rather than attempting to strip aliases via string manipulation which can be bypassed.
