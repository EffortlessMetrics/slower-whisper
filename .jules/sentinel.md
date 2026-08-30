## 2024-06-21 - SQL Injection in StoreQuery ORDER BY
**Vulnerability:** The `ConversationStore.search()` method interpolates the `query.order_by` field directly into a SQL query string (`ORDER BY {order_col} {order_dir}`) without sanitizing it or ensuring it refers to an allowed column name.
**Learning:** `order_by` fields supplied by the query model are vulnerable to SQL injection because standard parameterized queries (using `?`) can only be used for literal values, not column names or identifiers.
**Prevention:** Use an explicit allowlist of valid column names to map the `order_by` field before interpolating it into the SQL query string.

## 2024-06-21 - SQL Injection in StoreQuery ORDER BY
**Vulnerability:** The `ConversationStore.search()` method interpolates the `query.order_by` field directly into a SQL query string (`ORDER BY {order_col} {order_dir}`) without sanitizing it or ensuring it refers to an allowed column name.
**Learning:** `order_by` fields supplied by the query model are vulnerable to SQL injection because standard parameterized queries (using `?`) can only be used for literal values, not column names or identifiers.
**Prevention:** Use an explicit allowlist of valid column names to map the `order_by` field before interpolating it into the SQL query string.
