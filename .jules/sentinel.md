## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-04-19 - SQL Injection in SQLiteConversationStore
**Vulnerability:** SQL injection vulnerability in `transcription/store/store.py` where user-provided `order_by` string is directly interpolated into SQL query.
**Learning:** Even internal API query models (`StoreQuery`) can be attack vectors if they pass unsanitized strings directly into database operations. Specifically, SQLite `ORDER BY` allows arbitrary SQL execution if the column name is directly injected.
**Prevention:** Use an explicit dictionary-based allowlist to map user-facing column names to safe SQL table aliases instead of direct string interpolation.
