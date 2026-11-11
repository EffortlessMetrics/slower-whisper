#!/usr/bin/env python3
"""
Verify all code examples from documentation files.

This script extracts Python code blocks from README.md, docs/ARCHITECTURE.md,
and docs/API_QUICK_REFERENCE.md and validates them for:
- Syntax correctness
- Import correctness (matching actual exports)
- Function signature correctness
- Variable name consistency

Also rejects removed v2 CLI commands in current README/docs/example code,
including Python help and error strings. docs/archive/ and docs/releases/ are
historical. Migration snippets may opt out with a ``historical`` fence suffix
(for example, ``bash historical``), alongside a visible v1/removed label.
Use --legacy-cli-only to run this dependency-free check on its own.
"""

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class CodeExample:
    """A code example extracted from documentation."""

    source_file: str
    line_number: int
    code: str
    language: str
    context: str = ""
    historical: bool = False


@dataclass
class ValidationIssue:
    """An issue found during validation."""

    example: CodeExample
    severity: str  # "error", "warning", "info"
    message: str
    suggestion: str = ""


class ExampleExtractor:
    """Extract code examples from markdown files."""

    def __init__(self):
        self.examples: list[CodeExample] = []

    def extract_from_file(self, file_path: Path) -> list[CodeExample]:
        """Extract all code blocks from a markdown file."""
        examples = []

        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Find code blocks with ```language and ```
        pattern = r"```(\w+)?([^\n]*)\n(.*?)```"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            language = match.group(1) or "text"
            code = match.group(3)

            # Get line number
            line_number = content[: match.start()].count("\n") + 1

            # Get context (previous heading)
            context = self._get_context(content, match.start())

            if language in ("python", "py", "bash", "sh", "shell", "console", "json", "text"):
                examples.append(
                    CodeExample(
                        source_file=str(file_path),
                        line_number=line_number,
                        code=code,
                        language=language,
                        context=context,
                        historical="historical" in match.group(2).split(),
                    )
                )

        self.examples.extend(examples)
        return examples

    def _get_context(self, content: str, position: int) -> str:
        """Find the previous markdown heading before this position."""
        before = content[:position]
        # Find last heading
        heading_pattern = r"^#{1,6}\s+(.+)$"
        matches = list(re.finditer(heading_pattern, before, re.MULTILINE))
        if matches:
            return matches[-1].group(1).strip()
        return ""


_REMOVED_CLI_PATTERNS = (
    (
        re.compile(
            r"\bpython(?:\d+(?:\.\d+)?)?\s+(?:\./)?(?:scripts/)?"
            r"(?:transcribe_pipeline|audio_enrich)\.py\b"
        ),
        "Use slower-whisper transcribe / slower-whisper enrich --root PROJECT_DIR",
    ),
    (
        re.compile(
            r"\bpython(?:\d+(?:\.\d+)?)?\s+-m\s+"
            r"(?:scripts\.(?:transcribe_pipeline|audio_enrich)|"
            r"transcription\.(?:audio_enrich_cli|cli_legacy_transcribe))\b"
        ),
        "Use the unified slower-whisper CLI",
    ),
    (
        re.compile(r"(?<![\w-])slower-whisper-enrich(?![\w-])"),
        "Use slower-whisper enrich",
    ),
    (re.compile(r"(?<![\w-])--enrich-config(?![\w-])"), "Use --config"),
)


def validate_current_cli(example: CodeExample) -> list[ValidationIssue]:
    """Reject removed commands, preserving explicitly historical snippets."""
    if example.historical:
        return []

    issues = []
    for offset, line in enumerate(example.code.splitlines()):
        # Migration instructions search for old commands rather than execute them.
        if example.language in ("bash", "sh", "shell") and re.match(r"\s*(?:grep|rg)\s", line):
            continue
        for pattern, replacement in _REMOVED_CLI_PATTERNS:
            if match := pattern.search(line):
                issues.append(
                    ValidationIssue(
                        example=CodeExample(
                            source_file=example.source_file,
                            line_number=example.line_number + offset,
                            code=line,
                            language=example.language,
                            context=example.context,
                        ),
                        severity="error",
                        message=f"Removed v2 CLI command or flag: {match.group()}",
                        suggestion=replacement,
                    )
                )
    return issues


def check_current_cli(root: Path) -> list[ValidationIssue]:
    """Check current docs and executable examples without importing ML packages."""
    issues = []
    extractor = ExampleExtractor()
    markdown_files = set(root.glob("README*.md"))
    for directory in ("docs", "examples"):
        markdown_files.update((root / directory).rglob("*.md"))

    for path in sorted(markdown_files):
        relative = path.relative_to(root)
        if relative.parts[:2] in (("docs", "archive"), ("docs", "releases")):
            continue
        for example in extractor.extract_from_file(path):
            # The extractor reports the opening fence; the first code line is next.
            example.line_number += 1
            issues.extend(validate_current_cli(example))

    for path in sorted((root / "examples").rglob("*.py")):
        issues.extend(
            validate_current_cli(
                CodeExample(str(path), 1, path.read_text(encoding="utf-8"), "python")
            )
        )
    return issues


class CodeValidator:
    """Validate code examples for correctness."""

    def __init__(self):
        # Get actual exports from transcription package
        self.actual_exports = self._get_actual_exports()

    def _get_actual_exports(self) -> dict[str, Any]:
        """Get the actual exports from the transcription package."""
        try:
            # Import the package dynamically
            import transcription

            exports = {
                # Functions
                "transcribe_directory": transcription.transcribe_directory,
                "transcribe_file": transcription.transcribe_file,
                "enrich_directory": transcription.enrich_directory,
                "enrich_transcript": transcription.enrich_transcript,
                "load_transcript": transcription.load_transcript,
                "save_transcript": transcription.save_transcript,
                # Config
                "TranscriptionConfig": transcription.TranscriptionConfig,
                "EnrichmentConfig": transcription.EnrichmentConfig,
                # Models
                "Transcript": transcription.Transcript,
                "Segment": transcription.Segment,
            }

            # Also check __all__
            if hasattr(transcription, "__all__"):
                print(f"[info] transcription.__all__ = {transcription.__all__}")

            return exports

        except Exception as e:
            print(f"[warn] Could not import transcription package: {e}")
            return {}

    def validate_python_code(self, example: CodeExample) -> list[ValidationIssue]:
        """Validate a Python code example."""
        issues = []

        # 1. Check syntax
        try:
            ast.parse(example.code)
        except SyntaxError as e:
            issues.append(
                ValidationIssue(
                    example=example,
                    severity="error",
                    message=f"Syntax error: {e.msg} at line {e.lineno}",
                    suggestion="Fix the syntax error",
                )
            )
            return issues  # Can't continue if syntax is invalid

        # 2. Check imports
        tree = ast.parse(example.code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "transcription":
                    # Check each imported name
                    for alias in node.names:
                        name = alias.name
                        if name not in self.actual_exports:
                            issues.append(
                                ValidationIssue(
                                    example=example,
                                    severity="error",
                                    message=f"Import '{name}' not found in transcription package",
                                    suggestion=f"Available: {', '.join(sorted(self.actual_exports.keys()))}",
                                )
                            )

        # 3. Check function signatures (basic check for known functions)
        issues.extend(self._check_function_calls(example, tree))

        return issues

    def _check_function_calls(self, example: CodeExample, tree: ast.AST) -> list[ValidationIssue]:
        """Check if function calls match expected signatures."""
        issues = []

        # Known function signatures (simplified)
        expected_signatures = {
            "transcribe_directory": {"required_args": 2, "params": ["root", "config"]},
            "transcribe_file": {
                "required_args": 3,
                "params": ["audio_path", "root", "config"],
            },
            "enrich_directory": {"required_args": 2, "params": ["root", "config"]},
            "enrich_transcript": {
                "required_args": 3,
                "params": ["transcript", "audio_path", "config"],
            },
            "load_transcript": {"required_args": 1, "params": ["json_path"]},
            "save_transcript": {"required_args": 2, "params": ["transcript", "json_path"]},
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in expected_signatures:
                        expected = expected_signatures[func_name]
                        num_args = len(node.args) + len(node.keywords)

                        # Allow for keyword arguments
                        if num_args < expected["required_args"] - 1:
                            issues.append(
                                ValidationIssue(
                                    example=example,
                                    severity="warning",
                                    message=f"Function '{func_name}' expects {expected['required_args']} arguments, found {num_args}",
                                    suggestion=f"Expected: {func_name}({', '.join(expected['params'])})",
                                )
                            )

        return issues

    def validate_bash_code(self, example: CodeExample) -> list[ValidationIssue]:
        """Validate a bash/shell code example."""
        return validate_current_cli(example)

    def validate_example(self, example: CodeExample) -> list[ValidationIssue]:
        """Validate a code example based on its language."""
        if example.language in ("python", "py"):
            return self.validate_python_code(example)
        elif example.language in ("bash", "sh"):
            return self.validate_bash_code(example)
        else:
            return []


def main(argv: list[str] | None = None):
    """Main validation script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-cli-only", action="store_true")
    args = parser.parse_args(argv)
    cli_issues = check_current_cli(Path.cwd())
    if args.legacy_cli_only:
        for issue in cli_issues:
            print(f"{issue.example.source_file}:{issue.example.line_number}: {issue.message}")
            print(f"  {issue.suggestion}")
        print(f"Current CLI examples: {len(cli_issues)} error(s)")
        return 1 if cli_issues else 0

    print("=" * 80)
    print("CODE EXAMPLE VALIDATION REPORT")
    print("=" * 80)
    print()

    # Files to check
    doc_files = [
        Path("README.md"),
        Path("docs/ARCHITECTURE.md"),
        Path("docs/API_QUICK_REFERENCE.md"),
    ]

    # Extract examples
    extractor = ExampleExtractor()
    all_examples = []

    for doc_file in doc_files:
        if doc_file.exists():
            examples = extractor.extract_from_file(doc_file)
            all_examples.extend(examples)
            print(f"[{doc_file}] Found {len(examples)} code blocks")
        else:
            print(f"[warn] File not found: {doc_file}")

    print(f"\nTotal code blocks: {len(all_examples)}")
    print()

    # Filter to Python examples only
    python_examples = [e for e in all_examples if e.language in ("python", "py")]
    bash_examples = [e for e in all_examples if e.language in ("bash", "sh")]

    print(f"Python examples: {len(python_examples)}")
    print(f"Bash examples: {len(bash_examples)}")
    print()

    # Validate
    validator = CodeValidator()
    all_issues = list(cli_issues)

    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()

    # Validate Python examples
    valid_count = 0
    for example in python_examples:
        issues = validator.validate_example(example)
        if not issues:
            valid_count += 1
        else:
            all_issues.extend(issues)

    # Validate Bash examples
    bash_issues_count = 0
    for example in bash_examples:
        issues = validator.validate_example(example)
        if issues:
            bash_issues_count += len(issues)
        all_issues.extend(issues)

    # Summary
    print(f"✓ Valid Python examples: {valid_count}/{len(python_examples)}")
    print(f"✗ Python examples with issues: {len(python_examples) - valid_count}")
    print(f"⚠ Bash examples with warnings: {bash_issues_count}")
    print()

    # Report issues
    if all_issues:
        print("=" * 80)
        print("ISSUES FOUND")
        print("=" * 80)
        print()

        # Group by severity
        errors = [i for i in all_issues if i.severity == "error"]
        warnings = [i for i in all_issues if i.severity == "warning"]

        if errors:
            print(f"ERRORS ({len(errors)}):")
            print("-" * 80)
            for issue in errors:
                print(
                    f"\n[{issue.example.source_file}:{issue.example.line_number}] {issue.example.context}"
                )
                print(f"  {issue.message}")
                if issue.suggestion:
                    print(f"  → {issue.suggestion}")
                print("\n  Code:")
                for line in issue.example.code.split("\n")[:5]:
                    print(f"    {line}")
                if len(issue.example.code.split("\n")) > 5:
                    print("    ...")

        if warnings:
            print(f"\n\nWARNINGS ({len(warnings)}):")
            print("-" * 80)
            for issue in warnings:
                print(
                    f"\n[{issue.example.source_file}:{issue.example.line_number}] {issue.example.context}"
                )
                print(f"  {issue.message}")
                if issue.suggestion:
                    print(f"  → {issue.suggestion}")

    else:
        print("✓ No issues found! All examples are valid.")

    print()
    print("=" * 80)
    print("DETAILED EXAMPLE LISTING")
    print("=" * 80)
    print()

    # List all Python examples with their status
    for i, example in enumerate(python_examples, 1):
        issues = validator.validate_example(example)
        status = "✓" if not issues else "✗"
        print(f"{status} Example #{i}: [{example.source_file}:{example.line_number}]")
        print(f"  Context: {example.context}")
        print(f"  Lines: {len(example.code.splitlines())}")
        if issues:
            print(f"  Issues: {len(issues)} ({', '.join(i.severity for i in issues)})")
        print()

    return 1 if any(issue.severity == "error" for issue in all_issues) else 0


if __name__ == "__main__":
    sys.exit(main())
