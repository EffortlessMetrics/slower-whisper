## 2024-03-11 - [Optimize regex by string inclusion check]
**Learning:** For regex-based keyword annotators, performance can be significantly improved by pre-computing lowercase keywords and utilizing fast-path string inclusion checks (`keyword in text_lower`) prior to expensive regex evaluation. This is safe when the literal string is strictly required by the regex pattern.
**Action:** When evaluating regex patterns over large texts inside a loop, always consider if there is a fast-path string inclusion check that can quickly skip non-matching texts without executing the regular expression.
