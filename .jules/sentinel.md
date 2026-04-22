## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-01-28 - SQL Injection in SQLite Dynamic ORDER BY
**Vulnerability:** The `StoreQuery`'s `order_by` field was directly interpolated into the SQL `ORDER BY` clause in `transcription/store/store.py` without validation. Since parameterized queries (`?`) cannot be used for column identifiers, this exposed the database to SQL injection attacks.
**Learning:** Even internal APIs that seem to only take simple strings can be vulnerable if those strings are used for dynamic table/column references or `ORDER BY`/`GROUP BY` clauses, as standard parameterization doesn't protect these areas.
**Prevention:** Always use strict exact-match allowlists (including any necessary table aliases) to validate column names before using them in dynamic string interpolation for `ORDER BY` clauses. Raise a custom exception (e.g., `QueryError`) if validation fails to prevent execution and avoid leaking internal error states.
