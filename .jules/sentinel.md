## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-01-29 - SQLite SQL Injection via ORDER BY
**Vulnerability:** SQL injection vulnerability in `transcription/store/store.py` where a user-provided `order_by` field is directly concatenated into an `ORDER BY` clause.
**Learning:** `ORDER BY` cannot be safely parameterized using standard SQLite parameterization (`?`).
**Prevention:** Implement a strict allowlist of column names and validate against it before string interpolation. Default to a safe column if validation fails.
