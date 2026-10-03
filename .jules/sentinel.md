## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-06-09 - SQL Injection in Conversation Store
**Vulnerability:** SQL injection vulnerability in `transcription/store/store.py` where `order_by` from user input is directly concatenated into the SQL query without validation.
**Learning:** SQLite queries constructed dynamically using string formatting are vulnerable to injection if the formatted inputs (like column names) are not validated against an allowlist, as parameterized queries cannot be used for column names or identifiers.
**Prevention:** Always validate identifiers (like `order_by` columns) against a strict allowlist of known safe strings before appending them to a SQL query string.
