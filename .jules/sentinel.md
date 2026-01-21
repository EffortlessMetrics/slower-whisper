# Sentinel Journal 🛡️

This journal tracks CRITICAL security learnings, patterns, and decisions.

## Entries

## 2024-05-21 - [Argument Injection in Dogfood CLI]
**Vulnerability:** `transcription/dogfood.py` passed user-controlled filenames to `subprocess.run` without using `--` separator, allowing filenames starting with `-` to be interpreted as flags by the called python script.
**Learning:** `argparse` based scripts are vulnerable to argument injection if positional arguments are not separated by `--` when invoked via `subprocess`. Validating paths (like `exists()`) is not enough to prevent this.
**Prevention:** Always use `--` separator when passing untrusted arguments to CLI tools that support it, or strictly validate filenames to not start with `-`.
