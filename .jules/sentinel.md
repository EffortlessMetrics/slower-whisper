## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-02-12 - SQLite ORDER BY SQL Injection
**Vulnerability:** SQL injection in `transcription/store/store.py` via unvalidated `order_by` string passed into `ORDER BY {order_col}`.
**Learning:** SQLite cannot parameterize column or table names. Using f-strings to format user-provided column names into `ORDER BY` clauses creates a critical blind SQL injection vulnerability (even if it's returning empty results, boolean inferences can be made).
**Prevention:** Always validate `ORDER BY` columns against a strict exact-match allowlist of known schema columns before formatting them into SQL strings.
