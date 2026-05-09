## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-05-09 - SQL Injection in dynamic ORDER BY
**Vulnerability:** SQL injection vulnerability in `ConversationStore.search()` where the `order_by` field from user input is directly concatenated into the SQL query's `ORDER BY` clause.
**Learning:** Dynamic `ORDER BY` clauses cannot be parameterized using standard prepared statements, making them a common vector for blind SQL injection and schema leakage if not properly validated.
**Prevention:** Implement a strict, hardcoded string allowlist for all permitted sorting columns and aliases to ensure untrusted input is never executed as SQL.
