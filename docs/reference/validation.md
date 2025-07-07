# Validation Script Guide

This document explains how to use the comprehensive validation script to ensure code quality in the Mistral NER project.

## Overview

The validation script (`scripts/validate.py`) provides a unified interface for running all quality checks:
- **Ruff**: Code linting and formatting
- **MyPy**: Static type checking
- **Tests**: Unit tests with coverage reporting

## Quick Start

```bash
# Run all checks
python scripts/validate.py

# Run with verbose output
python scripts/validate.py --verbose

# Auto-fix issues where possible
python scripts/validate.py --fix
```

## Selective Execution

Run specific checks only:

```bash
# Run only ruff checks
python scripts/validate.py --ruff-only

# Run only mypy checks  
python scripts/validate.py --mypy-only

# Run only tests
python scripts/validate.py --tests-only
```

## File and Directory Targeting

Target specific files or directories:

```bash
# Check specific files
python scripts/validate.py --files src/config.py src/model.py

# Check specific directories
python scripts/validate.py --dirs src/ tests/

# Combine with selective execution
python scripts/validate.py --ruff-only --files src/config.py
```

## Output Formats

Generate different report formats:

```bash
# Generate JSON report
python scripts/validate.py --json --output validation_report.json

# Generate HTML report
python scripts/validate.py --html --output validation_report.html

# Quiet output (minimal)
python scripts/validate.py --quiet
```

## Integration with Development Workflow

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
python scripts/validate.py --quiet
if [ $? -ne 0 ]; then
    echo "Validation failed. Please fix issues before committing."
    exit 1
fi
```

### VS Code Integration

Add to `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Validate All",
            "type": "shell",
            "command": "python",
            "args": ["scripts/validate.py", "--verbose"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Quick Lint",
            "type": "shell", 
            "command": "python",
            "args": ["scripts/validate.py", "--ruff-only", "--mypy-only"],
            "group": "test"
        }
    ]
}
```

## CI/CD Integration

The validation script is integrated into our CI pipeline (`.github/workflows/ci.yml`):

- **Main validation job**: Runs comprehensive validation on all PRs and pushes
- **Individual check jobs**: Available for manual triggering via workflow_dispatch
- **Coverage reporting**: Automatically uploads to Codecov
- **Artifact preservation**: Saves validation reports for debugging

## Configuration

Optional configuration can be provided via `configs/validation_config.yaml`:

```yaml
# Timeout settings (seconds)
timeouts:
  ruff: 300
  mypy: 300
  tests: 600

# Output preferences
output:
  verbose: false
  format: "console"  # console, json, html

# Tool-specific settings
tools:
  ruff:
    auto_fix: false
    parallel: true
  mypy:
    cache_dir: ".mypy_cache"
  tests:
    coverage_threshold: 85
    parallel: true
```

## Requirements

- **Coverage**: Minimum 85% coverage required for all files
- **Ruff**: All linting and formatting rules must pass
- **MyPy**: No type errors allowed
- **Tests**: All tests must pass

## Troubleshooting

### Common Issues

1. **Low coverage**: Add more unit tests or integration tests
2. **Ruff failures**: Run with `--fix` to auto-resolve most issues
3. **MyPy errors**: Add proper type hints or ignore specific lines with `# type: ignore[error-code]`

### Debug Mode

For detailed debugging information:

```bash
# Show all command output
python scripts/validate.py --verbose

# Generate detailed reports
python scripts/validate.py --html --output debug_report.html
```

### Performance Issues

If validation is slow:

```bash
# Run checks separately for faster feedback
python scripts/validate.py --ruff-only  # Fast
python scripts/validate.py --mypy-only  # Medium  
python scripts/validate.py --tests-only # Slowest
```

## Exit Codes

- `0`: All checks passed
- `1`: One or more checks failed
- `130`: Interrupted by user (Ctrl+C)

## Rich Output

The script uses the Rich library for beautiful console output with:
- Progress bars during execution
- Colored status indicators
- Detailed results tables
- Error highlighting

If Rich is not available, the script falls back to plain text output.

## Examples

```bash
# Complete validation with auto-fix
python scripts/validate.py --fix --verbose

# Quick pre-commit check
python scripts/validate.py --quiet

# Generate comprehensive report  
python scripts/validate.py --html --output reports/validation_$(date +%Y%m%d_%H%M%S).html

# Check only modified files (with git)
git diff --name-only | grep '\.py$' | xargs python scripts/validate.py --files
```