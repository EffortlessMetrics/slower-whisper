## 2024-05-24 - Strict Allowlists for ORDER BY Clauses
**Vulnerability:** SQL Injection in `ORDER BY` interpolation in SQLite store search function.
**Learning:** SQLite cannot parameterize column names or `ORDER BY` values (using `?`). Direct string interpolation (e.g., `f"ORDER BY {col}"`) allows attackers to inject arbitrary SQL statements, bypass security, or perform destructive operations.
**Prevention:** Use a strict dictionary-based allowlist that maps safe, expected string values to fully qualified, safe database column aliases (e.g., `{"start_time": "s.start_time"}`). Raise a clear `QueryError` if the requested value is not explicitly present in the allowlist.
