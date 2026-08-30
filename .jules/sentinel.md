## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-01-29 - SQL Injection in ORDER BY
**Vulnerability:** SQL injection vulnerability in `ConversationStore.search()` where the `ORDER BY` clause was constructed using unvalidated string interpolation of the user-provided `query.order_by` value.
**Learning:** SQL parameterization (using `?` or `%s`) only works for data values, not for identifiers like column names, table names, or SQL keywords. When dynamic sorting is required, standard parameterization is ineffective and unvalidated string formatting creates critical SQL injection risks.
**Prevention:** Never use direct string interpolation for dynamic SQL columns. Always use a strict, hardcoded exact-match string allowlist to validate user-provided sort options before formatting them into queries.
