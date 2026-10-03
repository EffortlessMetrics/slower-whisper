## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2025-02-12 - SQL Injection in StoreQuery order_by
**Vulnerability:** SQL Injection in `ConversationStore.search()` where the `query.order_by` property is directly interpolated into a query string using `f"ORDER BY {order_col} {order_dir}"`.
**Learning:** Always validate and allowlist dynamically provided order by columns before string interpolation in SQL queries, as user-provided ordering directives cannot easily be parameterized.
**Prevention:** Strictly restrict acceptable `order_by` columns to a pre-defined set of valid keys (e.g. `start_time`, `end_time`, `speaker_id`).
