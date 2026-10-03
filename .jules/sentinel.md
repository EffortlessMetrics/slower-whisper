## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-06-11 - SQL Injection in ORDER BY Clause
**Vulnerability:** Unescaped string interpolation for `order_by` in `StoreQuery.search` (`sql_parts.append(f"ORDER BY {order_col} {order_dir}")`). Allowed arbitrary query modification, potentially exposing hidden columns or performing timing attacks.
**Learning:** Even internal queries that don't directly handle web input can be vulnerable to SQL injection if input from a typed data model (like `StoreQuery`) allows arbitrary strings that are then directly concatenated into a SQL statement.
**Prevention:** For SQL components that cannot be parameterized (like table/column names in `ORDER BY`), use a strict allowlist of predefined valid names. Reject unrecognized inputs immediately.
