## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-01-29 - SQL Injection via un-parameterized ORDER BY
**Vulnerability:** SQL injection vulnerability in `transcription/store/store.py` where the `order_by` field from user-controlled `StoreQuery` was directly concatenated into the SQL string.
**Learning:** Prepared statements (parameterized queries) cannot parameterize column names or `ASC`/`DESC` keywords. Using string interpolation for these fields is a critical risk if input isn't strictly validated.
**Prevention:** Always implement a strict, hardcoded exact-match string allowlist for all permitted sorting columns and directions before concatenating them into a SQL query.
