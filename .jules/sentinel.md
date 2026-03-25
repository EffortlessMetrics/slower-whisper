## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-03-25 - SQL Injection in StoreQuery ORDER BY
**Vulnerability:** SQL Injection via unvalidated dynamic column names in `ORDER BY` clauses in `ConversationStore.search`.
**Learning:** Dynamic column names in `ORDER BY` clauses cannot be parameterized using standard `?` placeholders in SQLite, leading to direct string concatenation vulnerabilities.
**Prevention:** Always validate user-provided column names (like `query.order_by`) against a strict hardcoded allowlist of known safe columns before concatenating them into SQL queries.
