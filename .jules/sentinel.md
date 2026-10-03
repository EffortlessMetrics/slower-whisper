## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-05-14 - Fix SQL injection in StoreQuery order_by
**Vulnerability:** The `StoreQuery` `order_by` parameter was dynamically inserted into a SQL `ORDER BY` clause using f-strings, allowing an attacker to pass arbitrary SQL or perform blind SQL injection attacks.
**Learning:** SQL databases cannot parameterize column names in `ORDER BY` clauses. Any user-supplied sorting field must be strictly validated before being added to a query string.
**Prevention:** Implement a strict, hardcoded exact-match string allowlist for all permitted sorting columns and aliases whenever an `ORDER BY` clause depends on dynamic input.
