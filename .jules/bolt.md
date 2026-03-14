## 2024-05-24 - [Keyword Annotator Fast Path]
**Learning:** Pre-computing lowercase keywords and using `in` fast-path string checks prior to regex searches provides a massive speedup for rule-based semantic annotation in Python.
**Action:** Always consider string `in` checks to filter inputs before applying more expensive `re.search` operations.
