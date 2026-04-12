## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-01-28 - SQL Injection in StoreQuery ORDER BY
**Vulnerability:** The SQLite `ORDER BY` clause in `transcription/store/store.py` directly interpolated the user-controlled `query.order_by` field, allowing potential SQL injection.
**Learning:** Standard SQLite parameterization (`?`) does not support dynamic column names in `ORDER BY` clauses. Direct interpolation must never be used on unsanitized input.
**Prevention:** Always use a strict allowlist (e.g., `ALLOWED_ORDER_COLS`) to validate user-provided column names before string interpolation in SQL queries.
