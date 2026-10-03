## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2026-06-14 - Fix SQL injection in search order_by clause
**Vulnerability:** SQL injection vulnerability in `ConversationStore.search()` where the `query.order_by` field was directly interpolated into the query string (`f"ORDER BY {order_col} {order_dir}"`).
**Learning:** Dynamic column names in `ORDER BY` clauses cannot be parameterized using standard SQLite placeholders (`?`). Direct string concatenation enables attackers to execute arbitrary SQL or extract data via error-based injection.
**Prevention:** Strictly enforce explicit allowlists for dynamic sorting columns to validate user input before constructing SQL queries.
