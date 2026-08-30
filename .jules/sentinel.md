## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2026-01-29 - SQL Injection in ORDER BY Clause
**Vulnerability:** SQL injection vulnerability in `ConversationStore.search()` via unsanitized `order_by` input.
**Learning:** Prepared statements (parameterized queries) cannot parameterize column names in `ORDER BY` clauses. Direct string interpolation creates SQL injection risks.
**Prevention:** Implement a strict, hardcoded exact-match string allowlist for all permitted sorting columns and aliases when dynamic sorting is required.
