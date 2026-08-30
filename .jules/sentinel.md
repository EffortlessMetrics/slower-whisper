## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2024-05-24 - Prevent SQL Injection in Dynamic ORDER BY
**Vulnerability:** Dynamic ORDER BY clause in ConversationStore search allowed SQL injection via string formatting.
**Learning:** User input should never be interpolated into an ORDER BY clause. Prepared statements cannot parameterize column names in ORDER BY.
**Prevention:** Always use a strict, hardcoded string exact-match allowlist for sorting columns and aliases.
