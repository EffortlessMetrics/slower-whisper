## 2024-03-09 - [Avoid redundant method calls with Walrus Operator]
**Learning:** Calling the same parsing method (`_count_questions(text)`) twice in an `if` condition and body is a redundant computation.
**Action:** Use Python's assignment expression (walrus operator `:=`) to compute the value once and bind it to a variable for the condition check and subsequent use.
