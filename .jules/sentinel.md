## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-05-07 - Store SQL Injection Prevented
**Vulnerability:** SQL injection in `ConversationStore.search()` via unsanitized `order_by` string parameter.
**Learning:** Default parameterization does not work for column names in ORDER BY clauses.
**Prevention:** Implement strict exact-match string allowlists for order column parameters to ensure only valid identifiers are appended to SQL queries.
