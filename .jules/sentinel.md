## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-01-28 - StoreQuery SQL Injection
**Vulnerability:** SQL Injection in SQLiteConversationStore search logic via unvalidated `order_by` parameter.
**Learning:** Even internal query objects (`StoreQuery.order_by`) can lead to SQL injection if their values are directly formatted into SQL strings (`f"ORDER BY {order_col} {order_dir}"`).
**Prevention:** Always validate columns against an explicit allowlist before using them in dynamic SQL constructs like `ORDER BY`, which cannot be parameterized.
