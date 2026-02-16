# Support

## Before Opening an Issue

1. Check the [documentation index](https://github.com/EffortlessMetrics/slower-whisper/blob/main/docs/INDEX.md).
2. Run the local quick gate: `./scripts/ci-local.sh fast`.
3. Capture receipts (commands, logs, and versions) so maintainers can reproduce quickly.

## Where To Ask For Help

- Bug report: [open a bug issue](https://github.com/EffortlessMetrics/slower-whisper/issues/new?template=bug_report.yml)
- Feature request: [open a feature request](https://github.com/EffortlessMetrics/slower-whisper/issues/new?template=feature_request.yml)
- Security concern: [private vulnerability report](https://github.com/EffortlessMetrics/slower-whisper/security/advisories/new)
- Documentation improvements: open an issue and use the `documentation` label

## What To Include

- `slower-whisper --version` output
- Python version and install method (`nix`, `uv`, `pip`, or `docker`)
- Device context (`--device auto|cpu|cuda`, GPU model if applicable)
- Minimal reproduction commands and expected vs actual behavior
- Logs or tracebacks

## Response Expectations

Maintainers triage issues in batches. Security reports are handled privately first, then disclosed in coordinated release notes.
