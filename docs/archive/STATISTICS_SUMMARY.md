# Project Statistics - Quick Summary

## Code Metrics

```
Total Lines of Code:        17,079
├─ Core Source:              3,656 (transcription module)
├─ Tests:                    3,409 (142 test functions)
├─ Examples:                 7,999 (20 example scripts)
└─ Benchmarks:               1,339 (4 benchmark scripts)

Source Code Composition:
├─ Code:                     2,045 lines (55.9%)
├─ Documentation:              787 lines (21.5%)
├─ Comments:                   183 lines (5.0%)
└─ Blank:                      641 lines (17.5%)
```

## Structure

```
Modules:                     17 files
├─ Core:                      5 modules (457 lines, 6 functions, 8 classes)
├─ Enrichment:                5 modules (2,074 lines, 20 functions, 2 classes)
├─ Interfaces:                4 modules (925 lines, 17 functions, 1 class)
└─ Output:                    2 modules (148 lines, 7 functions)

Functions:                   51 (avg 48 lines per function)
Classes:                     11
Test Functions:             142
```

## Quality

```
Documentation Coverage:
├─ Functions:                96.1% (49/51 with docstrings)
├─ Classes:                 100.0% (11/11 with docstrings)
└─ Total Docstrings:         78

Test Coverage:
├─ Module Coverage:         100% (16/16 modules tested)
├─ Estimated Line Coverage:  75-85%
└─ Test-to-Source Ratio:     0.93:1

Documentation Files:         38 markdown files
```

## Dependencies

```
Runtime Dependencies:         7 packages
├─ Core:                      1 (faster-whisper)
├─ Basic Enrichment:          3 (soundfile, numpy, librosa)
├─ Prosody:                  +1 (praat-parselmouth)
└─ Emotion:                  +2 (torch, transformers)

Development Dependencies:    27 packages
├─ Testing:                   5 (pytest suite)
├─ Code Quality:              5 (black, ruff, mypy, etc.)
├─ Documentation:             3 (sphinx suite)
└─ Security/Profiling:        6
```

## Files

```
Total Project Files:        119 (excluding venv)
├─ Python (.py):             54
├─ Documentation (.md):      38
├─ Data/Config (.txt/.json): 24
└─ Other (.toml/.yml/.sh):    3
```

## Development

```
Git Statistics:
├─ Total Commits:            12
├─ Contributors:              2
├─ Active Period:             1 day
└─ Latest Commit:        2025-11-15

Code Quality:
├─ Linters:              flake8, ruff
├─ Formatters:           black, isort
├─ Type Checking:        mypy
├─ Line Length:          100 chars
└─ Python Version:       3.10+
```

## Highlights

- **100% Module Test Coverage** - All 16 core modules have tests
- **96%+ Documentation** - Nearly all functions and classes documented
- **2.2:1 Example Ratio** - 20 comprehensive example scripts
- **Professional Setup** - Complete linting, testing, and type checking
- **Modular Design** - Clear separation into Core, Enrichment, and Interface layers
- **Minimal Core** - Only 1 core dependency (faster-whisper)
- **Flexible Extras** - Optional dependencies for advanced features

---

For detailed statistics, see [PROJECT_STATISTICS.md](PROJECT_STATISTICS.md)
