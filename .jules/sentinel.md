## 2025-02-21 - [Bypass in Allowlist Validation for SQL Injection]
**Vulnerability:** A bypass in the SQL injection prevention logic for `ORDER BY` clauses.
**Learning:** The previous implementation used `order_col.split('.')[-1]` to strip table aliases before validating against the allowlist. An attacker could bypass this by appending a comment containing a valid column name, e.g., `(CASE WHEN 1=1 THEN s.id ELSE s.start_time END) /*.id`. The split logic would extract `id`, validate it as safe, and inject the entire malicious string into the query.
**Prevention:** Always use strict, exact string matching for allowlists. If table aliases are expected, include them explicitly in the allowlist rather than relying on string manipulation to sanitize the input.
