## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2025-10-25 - SQL Injection in ORDER BY Clause
**Vulnerability:** SQL Injection in `ConversationStore.search()` via `StoreQuery.order_by` where unsanitized user input was interpolated directly into the `ORDER BY` clause string.
**Learning:** Parameterized queries (`?`) cannot be used for column names or table names in SQL. Consequently, any dynamic `ORDER BY` column must be strictly validated against an allowlist before string interpolation.
**Prevention:** Use an exact string match allowlist for all permissible column names (including table aliases) when constructing `ORDER BY` clauses dynamically.
