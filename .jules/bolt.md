## 2025-05-27 - Optimized Segment Construction
**Learning:** Creating intermediate lists of tuples for validation before object instantiation creates unnecessary memory pressure and CPU overhead in tight loops.
**Action:** Prefer single-pass object instantiation with inline validation. If sorting is conditionally required, sort the objects directly rather than the intermediate data structures.
