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

## Common Issues

- **GPU / CUDA problems** — See [GPU_SETUP.md](https://github.com/EffortlessMetrics/slower-whisper/blob/main/docs/GPU_SETUP.md) for device selection, compute types, and driver requirements
- **Configuration questions** — See [CONFIGURATION.md](https://github.com/EffortlessMetrics/slower-whisper/blob/main/docs/CONFIGURATION.md) for config file precedence, environment variables, and CLI flags
- **General troubleshooting** — See [TROUBLESHOOTING.md](https://github.com/EffortlessMetrics/slower-whisper/blob/main/docs/TROUBLESHOOTING.md) for common errors and solutions

## Contributing

Want to contribute? See [CONTRIBUTING.md](https://github.com/EffortlessMetrics/slower-whisper/blob/main/CONTRIBUTING.md) for development setup, PR process, and coding standards.

## Response Expectations

Maintainers triage issues in batches — expect initial response within a few business days for bugs and feature requests. Security reports are handled privately first, then disclosed in coordinated release notes.
