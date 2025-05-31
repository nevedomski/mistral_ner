#!/usr/bin/env python3
"""
Comprehensive validation script for Mistral NER project.

Runs all quality checks and generates detailed reports:
- Ruff linting and formatting
- MyPy type checking
- Unit tests with coverage
- Beautiful console output with progress indicators
- Multiple output formats (console, JSON, HTML)
- Selective execution and file targeting
"""

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Protocol

# Rich for beautiful console output
try:
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not available. Install with: pip install rich")


class ConsoleProtocol(Protocol):
    """Protocol for console objects."""

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print method."""
        ...


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    duration: float
    output: str
    error: str
    details: dict[str, Any]
    exit_code: int


@dataclass
class ValidationReport:
    """Complete validation report."""

    timestamp: str
    git_commit: str
    git_branch: str
    python_version: str
    environment: str
    total_duration: float
    overall_status: str
    checks: list[CheckResult]
    summary: dict[str, Any]


class FileSelector:
    """Handle file and directory pattern matching."""

    def __init__(self, files: list[str] | None = None, dirs: list[str] | None = None):
        self.files = files or []
        self.dirs = dirs or []

    def get_target_args(self, tool: str) -> list[str]:
        """Get appropriate target arguments for a tool."""
        if not self.files and not self.dirs:
            return []

        targets = []
        for pattern in self.files:
            targets.append(pattern)
        for pattern in self.dirs:
            targets.append(pattern)

        return targets


class BaseRunner:
    """Abstract base class for tool runners."""

    def __init__(self, console: ConsoleProtocol | None = None):
        self.console = console

    def run(self, target_args: list[str] | None = None, fix: bool = False) -> CheckResult:
        """Run the check and return results."""
        raise NotImplementedError


class RuffRunner(BaseRunner):
    """Runner for Ruff linting and formatting."""

    def run(self, target_args: list[str] | None = None, fix: bool = False) -> CheckResult:
        """Run ruff checks."""
        start_time = time.time()
        target_args = target_args or ["."]

        # Run ruff check
        check_cmd = [sys.executable, "-m", "ruff", "check"] + (["--fix"] if fix else []) + target_args
        format_cmd = [sys.executable, "-m", "ruff", "format"] + ([] if fix else ["--check"]) + target_args

        if self.console and RICH_AVAILABLE:
            self.console.print(f"[blue]Running:[/blue] {' '.join(check_cmd)}")

        try:
            # Run check
            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=300)

            # Run format
            format_result = subprocess.run(format_cmd, capture_output=True, text=True, timeout=300)

            # Combine results
            passed = check_result.returncode == 0 and format_result.returncode == 0
            output = f"Check output:\n{check_result.stdout}\n\nFormat output:\n{format_result.stdout}"
            error = f"Check errors:\n{check_result.stderr}\n\nFormat errors:\n{format_result.stderr}"

            # Parse details from output
            details = self._parse_ruff_output(check_result.stdout, format_result.stdout)

            return CheckResult(
                name="Ruff",
                passed=passed,
                duration=time.time() - start_time,
                output=output,
                error=error,
                details=details,
                exit_code=max(check_result.returncode, format_result.returncode),
            )

        except subprocess.TimeoutExpired:
            return CheckResult(
                name="Ruff",
                passed=False,
                duration=time.time() - start_time,
                output="",
                error="Timeout after 300 seconds",
                details={"error": "timeout"},
                exit_code=124,
            )
        except Exception as e:
            return CheckResult(
                name="Ruff",
                passed=False,
                duration=time.time() - start_time,
                output="",
                error=str(e),
                details={"error": str(e)},
                exit_code=1,
            )

    def _parse_ruff_output(self, check_output: str, format_output: str) -> dict[str, Any]:
        """Parse ruff output for detailed information."""
        details: dict[str, Any] = {"check_issues": 0, "format_issues": 0, "files_checked": 0, "rules_violated": []}

        # Count issues from check output
        if check_output:
            lines = check_output.split("\n")
            for line in lines:
                if re.match(r".*:\d+:\d+:", line):
                    details["check_issues"] += 1
                    # Extract rule code
                    match = re.search(r"\[([A-Z0-9]+)\]", line)
                    if match:
                        rule = match.group(1)
                        if rule not in details["rules_violated"]:
                            details["rules_violated"].append(rule)

        # Count format issues
        if "would reformat" in format_output:
            details["format_issues"] = len(re.findall(r"would reformat", format_output))

        return details


class MypyRunner(BaseRunner):
    """Runner for MyPy type checking."""

    def run(self, target_args: list[str] | None = None, fix: bool = False) -> CheckResult:
        """Run mypy checks."""
        start_time = time.time()

        # MyPy targets are configured in pyproject.toml, but we can override
        cmd = [sys.executable, "-m", "mypy"]
        if target_args:
            cmd.extend(target_args)

        if self.console and RICH_AVAILABLE:
            self.console.print(f"[blue]Running:[/blue] {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            passed = result.returncode == 0
            details = self._parse_mypy_output(result.stdout)

            return CheckResult(
                name="MyPy",
                passed=passed,
                duration=time.time() - start_time,
                output=result.stdout,
                error=result.stderr,
                details=details,
                exit_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return CheckResult(
                name="MyPy",
                passed=False,
                duration=time.time() - start_time,
                output="",
                error="Timeout after 300 seconds",
                details={"error": "timeout"},
                exit_code=124,
            )
        except Exception as e:
            return CheckResult(
                name="MyPy",
                passed=False,
                duration=time.time() - start_time,
                output="",
                error=str(e),
                details={"error": str(e)},
                exit_code=1,
            )

    def _parse_mypy_output(self, output: str) -> dict[str, Any]:
        """Parse mypy output for detailed information."""
        details: dict[str, Any] = {"errors": 0, "files_processed": 0, "error_codes": []}

        if output:
            lines = output.split("\n")
            for line in lines:
                if re.match(r".*:\d+:", line):
                    details["errors"] += 1
                    # Extract error code if present
                    match = re.search(r"\[([a-z-]+)\]", line)
                    if match:
                        code = match.group(1)
                        if code not in details["error_codes"]:
                            details["error_codes"].append(code)

            # Count success message
            if "Success: no issues found" in output:
                details["files_processed"] = len(re.findall(r"files processed", output))

        return details


class TestRunner(BaseRunner):
    """Runner for pytest with coverage."""

    def run(self, target_args: list[str] | None = None, fix: bool = False) -> CheckResult:
        """Run pytest with coverage."""
        start_time = time.time()

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--cov=src",
            "--cov=api",
            "--cov=scripts",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "-v",
        ]

        if target_args:
            cmd.extend(target_args)
        else:
            cmd.append("tests/")

        if self.console and RICH_AVAILABLE:
            self.console.print(f"[blue]Running:[/blue] {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            passed = result.returncode == 0
            details = self._parse_pytest_output(result.stdout)

            return CheckResult(
                name="Tests",
                passed=passed,
                duration=time.time() - start_time,
                output=result.stdout,
                error=result.stderr,
                details=details,
                exit_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return CheckResult(
                name="Tests",
                passed=False,
                duration=time.time() - start_time,
                output="",
                error="Timeout after 600 seconds",
                details={"error": "timeout"},
                exit_code=124,
            )
        except Exception as e:
            return CheckResult(
                name="Tests",
                passed=False,
                duration=time.time() - start_time,
                output="",
                error=str(e),
                details={"error": str(e)},
                exit_code=1,
            )

    def _parse_pytest_output(self, output: str) -> dict[str, Any]:
        """Parse pytest output for detailed information."""
        details: dict[str, Any] = {"tests_run": 0, "passed": 0, "failed": 0, "skipped": 0, "coverage_total": 0.0, "coverage_files": {}}

        if output:
            # Parse test results
            result_match = re.search(r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?", output)
            if result_match:
                details["passed"] = int(result_match.group(1))
                details["failed"] = int(result_match.group(2) or 0)
                details["skipped"] = int(result_match.group(3) or 0)
                details["tests_run"] = details["passed"] + details["failed"] + details["skipped"]

            # Parse coverage
            coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
            if coverage_match:
                details["coverage_total"] = float(coverage_match.group(1))

        # Try to read coverage.json for more detailed coverage info
        try:
            with open("coverage.json") as f:
                coverage_data = json.load(f)
                if "files" in coverage_data:
                    for file_path, file_data in coverage_data["files"].items():
                        if "summary" in file_data:
                            percentage = file_data["summary"]["percent_covered"]
                            details["coverage_files"][file_path] = percentage
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        return details


class ValidationScript:
    """Main validation orchestrator."""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.runners = {
            "ruff": RuffRunner(self.console),
            "mypy": MypyRunner(self.console),
            "tests": TestRunner(self.console),
        }

    def run_validation(
        self,
        checks: list[str],
        target_args: list[str] | None = None,
        fix: bool = False,
        output_format: str = "console",
        output_file: str | None = None,
        verbose: bool = False,
    ) -> ValidationReport:
        """Run validation checks and generate report."""

        if self.console and RICH_AVAILABLE:
            self.console.print(
                Panel.fit(
                    f"[bold blue]Mistral NER Validation[/bold blue]\nRunning checks: {', '.join(checks)}", style="blue"
                )
            )

        # Get environment info
        env_info = self._get_environment_info()

        # Run checks
        start_time = time.time()
        results = []

        if RICH_AVAILABLE and self.console:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Running validation...", total=len(checks))

                for check in checks:
                    if check in self.runners:
                        progress.update(task, description=f"Running {check}...")
                        result = self.runners[check].run(target_args, fix)
                        results.append(result)
                        progress.advance(task)
                    else:
                        self.console.print(f"[red]Unknown check: {check}[/red]")
        else:
            for check in checks:
                if check in self.runners:
                    print(f"Running {check}...")
                    result = self.runners[check].run(target_args, fix)
                    results.append(result)
                else:
                    print(f"Unknown check: {check}")

        # Create report
        total_duration = time.time() - start_time
        overall_status = "PASSED" if all(r.passed for r in results) else "FAILED"

        report = ValidationReport(
            timestamp=env_info["timestamp"],
            git_commit=env_info["git_commit"],
            git_branch=env_info["git_branch"],
            python_version=env_info["python_version"],
            environment=env_info["environment"],
            total_duration=total_duration,
            overall_status=overall_status,
            checks=results,
            summary=self._generate_summary(results),
        )

        # Output report
        self._output_report(report, output_format, output_file, verbose)

        return report

    def _get_environment_info(self) -> dict[str, str]:
        """Get environment information."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version.split()[0],
            "environment": "development",  # Could be enhanced
            "git_commit": "unknown",
            "git_branch": "unknown",
        }

        # Try to get git info
        try:
            commit_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
            if commit_result.returncode == 0:
                info["git_commit"] = commit_result.stdout.strip()[:8]
        except (subprocess.SubprocessError, OSError):
            pass

        try:
            branch_result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
            if branch_result.returncode == 0:
                info["git_branch"] = branch_result.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass

        return info

    def _generate_summary(self, results: list[CheckResult]) -> dict[str, Any]:
        """Generate summary statistics."""
        summary: dict[str, Any] = {
            "total_checks": len(results),
            "passed_checks": sum(1 for r in results if r.passed),
            "failed_checks": sum(1 for r in results if not r.passed),
            "total_duration": sum(r.duration for r in results),
            "check_details": {},
        }

        for result in results:
            summary["check_details"][result.name] = {
                "passed": result.passed,
                "duration": result.duration,
                "exit_code": result.exit_code,
            }

        return summary

    def _output_report(self, report: ValidationReport, output_format: str, output_file: str | None, verbose: bool):
        """Output the report in the specified format."""

        if output_format == "console":
            self._output_console_report(report, verbose)
        elif output_format == "json":
            self._output_json_report(report, output_file)
        elif output_format == "html":
            self._output_html_report(report, output_file)

    def _output_console_report(self, report: ValidationReport, verbose: bool):
        """Output beautiful console report."""
        if not self.console or not RICH_AVAILABLE:
            # Fallback to plain text
            self._output_plain_report(report, verbose)
            return

        # Header
        status_color = "green" if report.overall_status == "PASSED" else "red"
        self.console.print(
            Panel.fit(
                f"[bold {status_color}]{report.overall_status}[/bold {status_color}]\n"
                f"Duration: {report.total_duration:.2f}s | "
                f"Branch: {report.git_branch} | "
                f"Commit: {report.git_commit}",
                style=status_color,
            )
        )

        # Results table
        table = Table(title="Validation Results")
        table.add_column("Check", style="cyan", width=12)
        table.add_column("Status", width=8)
        table.add_column("Duration", justify="right", width=10)
        table.add_column("Details", width=50)

        for check in report.checks:
            status = "[green]✓ PASS[/green]" if check.passed else "[red]✗ FAIL[/red]"
            duration = f"{check.duration:.2f}s"

            # Create details summary
            details_text = ""
            if check.name == "Ruff":
                issues = check.details.get("check_issues", 0) + check.details.get("format_issues", 0)
                details_text = f"{issues} issues" if issues > 0 else "No issues"
            elif check.name == "MyPy":
                errors = check.details.get("errors", 0)
                details_text = f"{errors} errors" if errors > 0 else "No errors"
            elif check.name == "Tests":
                passed = check.details.get("passed", 0)
                failed = check.details.get("failed", 0)
                coverage = check.details.get("coverage_total", 0)
                details_text = f"{passed} passed, {failed} failed, {coverage}% coverage"

            table.add_row(check.name, status, duration, details_text)

        self.console.print(table)

        # Show errors if any
        failed_checks = [c for c in report.checks if not c.passed]
        if failed_checks and verbose:
            self.console.print("\n[bold red]Error Details:[/bold red]")
            for check in failed_checks:
                self.console.print(Panel(escape(check.error), title=f"{check.name} Error", style="red"))

    def _output_plain_report(self, report: ValidationReport, verbose: bool):
        """Fallback plain text report."""
        print(f"\n{'=' * 60}")
        print(f"VALIDATION REPORT - {report.overall_status}")
        print(f"{'=' * 60}")
        print(f"Duration: {report.total_duration:.2f}s")
        print(f"Branch: {report.git_branch}")
        print(f"Commit: {report.git_commit}")
        print()

        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"{check.name:<12} {status:<8} {check.duration:.2f}s")
            if not check.passed and verbose:
                print(f"  Error: {check.error}")

        print(f"{'=' * 60}")

    def _output_json_report(self, report: ValidationReport, output_file: str | None):
        """Output JSON report."""
        report_data = asdict(report)

        if output_file:
            with open(output_file, "w") as f:
                json.dump(report_data, f, indent=2)
            if self.console and RICH_AVAILABLE:
                self.console.print(f"[green]JSON report saved to: {output_file}[/green]")
        else:
            print(json.dumps(report_data, indent=2))

    def _output_html_report(self, report: ValidationReport, output_file: str | None):
        """Output HTML report."""
        html_content = self._generate_html_report(report)

        filename = output_file or f"validation_report_{report.timestamp.replace(':', '-')}.html"
        with open(filename, "w") as f:
            f.write(html_content)

        if self.console and RICH_AVAILABLE:
            self.console.print(f"[green]HTML report saved to: {filename}[/green]")

    def _generate_html_report(self, report: ValidationReport) -> str:
        """Generate HTML report content."""
        status_color = "#28a745" if report.overall_status == "PASSED" else "#dc3545"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Mistral NER Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: {status_color}; color: white; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .check {{ margin: 10px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .pass {{ border-left: 5px solid #28a745; }}
        .fail {{ border-left: 5px solid #dc3545; }}
        .details {{ margin-top: 10px; font-family: monospace; background: #f8f9fa; padding: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Mistral NER Validation Report</h1>
        <p><strong>Status:</strong> {report.overall_status}</p>
        <p><strong>Duration:</strong> {report.total_duration:.2f}s</p>
        <p><strong>Branch:</strong> {report.git_branch} | <strong>Commit:</strong> {report.git_commit}</p>
        <p><strong>Timestamp:</strong> {report.timestamp}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr><th>Total Checks</th><td>{report.summary["total_checks"]}</td></tr>
            <tr><th>Passed</th><td>{report.summary["passed_checks"]}</td></tr>
            <tr><th>Failed</th><td>{report.summary["failed_checks"]}</td></tr>
            <tr><th>Total Duration</th><td>{report.summary["total_duration"]:.2f}s</td></tr>
        </table>
    </div>

    <h2>Check Results</h2>
"""

        for check in report.checks:
            status_class = "pass" if check.passed else "fail"
            status_text = "PASSED" if check.passed else "FAILED"

            html += f"""
    <div class="check {status_class}">
        <h3>{check.name} - {status_text}</h3>
        <p><strong>Duration:</strong> {check.duration:.2f}s | <strong>Exit Code:</strong> {check.exit_code}</p>
"""

            if check.details:
                html += "<h4>Details:</h4><ul>"
                for key, value in check.details.items():
                    html += f"<li><strong>{key}:</strong> {value}</li>"
                html += "</ul>"

            if check.error and not check.passed:
                html += f'<div class="details"><strong>Error:</strong><br><pre>{check.error}</pre></div>'

            html += "</div>"

        html += """
</body>
</html>
"""
        return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive validation script for Mistral NER project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run all checks
  %(prog)s --ruff-only             # Run only ruff checks
  %(prog)s --mypy-only             # Run only mypy checks
  %(prog)s --tests-only            # Run only tests
  %(prog)s --files src/config.py   # Check specific file
  %(prog)s --dirs src/             # Check specific directory
  %(prog)s --json --output report.json  # Output JSON report
  %(prog)s --html --output report.html  # Output HTML report
  %(prog)s --fix                   # Auto-fix issues where possible
        """,
    )

    # Check selection
    parser.add_argument("--ruff-only", action="store_true", help="Run only Ruff checks")
    parser.add_argument("--mypy-only", action="store_true", help="Run only MyPy checks")
    parser.add_argument("--tests-only", action="store_true", help="Run only tests")

    # Target selection
    parser.add_argument("--files", nargs="+", help="Target specific files")
    parser.add_argument("--dirs", nargs="+", help="Target specific directories")

    # Output options
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--html", action="store_true", help="Output HTML report")
    parser.add_argument("--output", "-o", help="Output file (for JSON/HTML)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    # Action options
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues where possible")

    args = parser.parse_args()

    # Determine which checks to run
    checks = []
    if args.ruff_only:
        checks = ["ruff"]
    elif args.mypy_only:
        checks = ["mypy"]
    elif args.tests_only:
        checks = ["tests"]
    else:
        checks = ["ruff", "mypy", "tests"]

    # Determine output format
    output_format = "console"
    if args.json:
        output_format = "json"
    elif args.html:
        output_format = "html"

    # Handle target selection
    file_selector = FileSelector(args.files, args.dirs)
    target_args = file_selector.get_target_args("all")

    # Run validation
    validator = ValidationScript()

    try:
        report = validator.run_validation(
            checks=checks,
            target_args=target_args,
            fix=args.fix,
            output_format=output_format,
            output_file=args.output,
            verbose=args.verbose and not args.quiet,
        )

        # Exit with appropriate code
        exit_code = 0 if report.overall_status == "PASSED" else 1
        sys.exit(exit_code)

    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            Console().print("\n[red]Validation interrupted by user[/red]")
        else:
            print("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        if RICH_AVAILABLE:
            Console().print(f"[red]Validation failed: {e}[/red]")
        else:
            print(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
