## 2026-01-28 - FastAPI Security Headers & CSP
**Vulnerability:** Missing security headers (X-Content-Type-Options, X-Frame-Options, CSP) in FastAPI service.
**Learning:** Default strict CSP (`default-src 'self'`) breaks FastAPI's auto-generated docs (Swagger UI/Redoc) which rely on `cdn.jsdelivr.net` and `unsafe-inline` styles/scripts.
**Prevention:** Use a middleware to add security headers, but ensure CSP explicitly allows `cdn.jsdelivr.net` and `fastapi.tiangolo.com` if API docs are enabled.

## 2024-05-24 - SQL Injection in ORDER BY Clause
**Vulnerability:** SQL injection vulnerability due to direct string interpolation of user-provided column names in SQLite `ORDER BY` clauses within `transcription/store/store.py`.
**Learning:** Standard database parameterization (`?` or `%s`) cannot be used for structural elements like table names or column names in `ORDER BY` clauses. Direct interpolation allows attackers to execute arbitrary SQL commands if the column name is derived from user input.
**Prevention:** Always validate user-provided column names against a strict allowlist (whitelist) of safe, known column names before performing string interpolation in an `ORDER BY` clause.
