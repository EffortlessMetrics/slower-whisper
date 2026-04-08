## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.
## 2024-05-24 - SQL Injection in SQLite ORDER BY
**Vulnerability:** SQL injection vulnerability via unvalidated `order_by` parameter in `StoreQuery` interpolation (`f"ORDER BY {order_col} {order_dir}"`).
**Learning:** SQLite parameterization (`?`) does not support dynamic column names in `ORDER BY` clauses. Direct string interpolation allows arbitrary SQL execution.
**Prevention:** Always use a strict allowlist (e.g., `ALLOWED_ORDER_COLS`) to validate user-provided column names before string interpolation in queries.
## 2024-05-24 - Package Distribution Security configuration
**Vulnerability:** CI wheel build failures due to missing package directories. While not a direct exploit vulnerability, failing securely includes ensuring build pipelines distribute full integrity artifacts.
**Learning:** `pyproject.toml` explicit package definitions and Docker `COPY` commands must precisely mirror application subpackage structure to avoid broken distributions.
**Prevention:** If missing module errors happen in wheel build or docker build, explicitly ensure all nested packages (like `transcription.store` and `slower_whisper`) are included in `[tool.setuptools] packages = [...]` in `pyproject.toml` and manually copied in Dockerfiles if not relying on `COPY . .`.
