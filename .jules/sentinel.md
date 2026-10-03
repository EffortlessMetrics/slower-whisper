## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-02-05 - SQL Injection in Dynamic ORDER BY Clauses
**Vulnerability:** SQL injection vulnerability in the `SQLiteConversationStore.search()` method due to unsanitized interpolation of the `order_by` parameter.
**Learning:** Prepared statements (parameterized queries) cannot parameterize column names or `ORDER BY` directions. If `order_by` is concatenated into a SQL string directly from user input, attackers can inject subqueries or case statements to exfiltrate data, bypass checks, or cause denial of service.
**Prevention:** Implement a strict, hardcoded exact-match string allowlist for all permitted sorting columns and aliases before concatenating them into the SQL query string. Raise an error if the input does not match the allowlist.
