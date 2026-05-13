## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-05-13 - SQL Injection in StoreQuery ORDER BY
**Vulnerability:** The `StoreQuery` object in `transcription/store/store.py` allows arbitrary strings in the `order_by` field, which are directly concatenated into the SQL query's `ORDER BY` clause, allowing SQL injection.
**Learning:** Even if primary query parameters are parameterized, dynamic `ORDER BY` clauses are vulnerable if they blindly concatenate user input, as prepared statements cannot parameterize column names.
**Prevention:** Implement a strict, hardcoded string allowlist for all permitted sorting columns (including aliases) and validate `order_col` against it before execution.
