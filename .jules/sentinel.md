## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2024-04-29 - SQL Injection in Store Search
**Vulnerability:** ORDER BY clause in SQLite search query was directly formatted with user-provided `order_by` strings, allowing SQL injection.
**Learning:** `ORDER BY` cannot be parameterized in standard SQL, so developers often mistakenly fall back to string interpolation. Even for internal APIs or "memory" stores, this can be exploited if the query reaches the store.
**Prevention:** Always use a strict hardcoded allowlist dictionary (exact match) for column names in `ORDER BY` clauses instead of string manipulation or direct formatting.
