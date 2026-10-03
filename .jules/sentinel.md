## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-06-10 - SQL Injection in StoreQuery order_by
**Vulnerability:** SQL injection via unvalidated `order_by` property of `StoreQuery` in `SQLiteConversationStore.search`.
**Learning:** The `ORDER BY` clause cannot be parameterized in sqlite3 like normal values. Unvalidated values mapped directly from user input or queries into the SQL string lead to SQL injection.
**Prevention:** Strictly validate `ORDER BY` columns against a hardcoded allowlist of legitimate column names before string formatting them into queries.
