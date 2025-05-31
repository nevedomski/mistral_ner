"""
Tests for scripts/validate.py - comprehensive validation script.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from scripts.validate import (
    BaseRunner,
    CheckResult,
    FileSelector,
    MypyRunner,
    PytestRunner,
    RuffRunner,
    ValidationReport,
    ValidationScript,
    main,
)


class TestFileSelector:
    """Test FileSelector functionality."""

    def test_empty_selector(self):
        """Test FileSelector with no files or dirs."""
        selector = FileSelector()
        assert selector.get_target_args("ruff") == []

    def test_files_only(self):
        """Test FileSelector with files."""
        selector = FileSelector(files=["file1.py", "file2.py"])
        result = selector.get_target_args("ruff")
        assert result == ["file1.py", "file2.py"]

    def test_dirs_only(self):
        """Test FileSelector with directories."""
        selector = FileSelector(dirs=["src/", "tests/"])
        result = selector.get_target_args("ruff")
        assert result == ["src/", "tests/"]

    def test_files_and_dirs(self):
        """Test FileSelector with both files and directories."""
        selector = FileSelector(files=["main.py"], dirs=["src/"])
        result = selector.get_target_args("ruff")
        assert result == ["main.py", "src/"]

    def test_none_values(self):
        """Test FileSelector with None values."""
        selector = FileSelector(files=None, dirs=None)
        assert selector.get_target_args("ruff") == []


class TestCheckResult:
    """Test CheckResult dataclass."""

    def test_check_result_creation(self):
        """Test creating a CheckResult."""
        result = CheckResult(
            name="Test", passed=True, duration=1.5, output="test output", error="", details={"count": 5}, exit_code=0
        )
        assert result.name == "Test"
        assert result.passed is True
        assert result.duration == 1.5
        assert result.details["count"] == 5


class TestValidationReport:
    """Test ValidationReport dataclass."""

    def test_validation_report_creation(self):
        """Test creating a ValidationReport."""
        check_result = CheckResult(name="Test", passed=True, duration=1.0, output="", error="", details={}, exit_code=0)
        report = ValidationReport(
            timestamp="2023-01-01T00:00:00",
            git_commit="abcd1234",
            git_branch="main",
            python_version="3.11.0",
            environment="test",
            total_duration=1.0,
            overall_status="PASSED",
            checks=[check_result],
            summary={"total": 1},
        )
        assert report.overall_status == "PASSED"
        assert len(report.checks) == 1


class TestBaseRunner:
    """Test BaseRunner abstract class."""

    def test_base_runner_not_implemented(self):
        """Test that BaseRunner.run raises NotImplementedError."""
        runner = BaseRunner()
        with pytest.raises(NotImplementedError):
            runner.run()

    def test_base_runner_with_console(self):
        """Test BaseRunner with console."""
        console = MagicMock()
        runner = BaseRunner(console=console)
        assert runner.console is console


class TestRuffRunner:
    """Test RuffRunner functionality."""

    @patch("subprocess.run")
    def test_ruff_runner_success(self, mock_run):
        """Test successful ruff run."""
        # Mock successful subprocess runs
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="All checks passed", stderr=""),
            MagicMock(returncode=0, stdout="All files formatted", stderr=""),
        ]

        runner = RuffRunner()
        result = runner.run(target_args=["test.py"])

        assert result.name == "Ruff"
        assert result.passed is True
        assert result.exit_code == 0
        assert "All checks passed" in result.output

    @patch("subprocess.run")
    def test_ruff_runner_failure(self, mock_run):
        """Test failed ruff run."""
        # Mock failed subprocess runs
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="test.py:1:1: E501 line too long", stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]

        runner = RuffRunner()
        result = runner.run(target_args=["test.py"])

        assert result.name == "Ruff"
        assert result.passed is False
        assert result.exit_code == 1
        assert result.details["check_issues"] == 1

    @patch("subprocess.run")
    def test_ruff_runner_timeout(self, mock_run):
        """Test ruff run timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ruff"], timeout=300)

        runner = RuffRunner()
        result = runner.run()

        assert result.name == "Ruff"
        assert result.passed is False
        assert result.exit_code == 124
        assert "Timeout" in result.error

    @patch("subprocess.run")
    def test_ruff_runner_exception(self, mock_run):
        """Test ruff run with exception."""
        mock_run.side_effect = Exception("Command failed")

        runner = RuffRunner()
        result = runner.run()

        assert result.name == "Ruff"
        assert result.passed is False
        assert result.exit_code == 1
        assert "Command failed" in result.error

    def test_parse_ruff_output_with_issues(self):
        """Test parsing ruff output with issues."""
        runner = RuffRunner()
        check_output = "test.py:1:1: E501 line too long [E501]\ntest.py:2:1: F401 unused import [F401]"
        format_output = "Would reformat: test.py\n1 file would be reformatted"

        details = runner._parse_ruff_output(check_output, format_output)

        assert details["check_issues"] == 2
        assert details["format_issues"] == 1
        assert "E501" in details["rules_violated"]
        assert "F401" in details["rules_violated"]

    def test_parse_ruff_output_clean(self):
        """Test parsing clean ruff output."""
        runner = RuffRunner()
        check_output = ""
        format_output = ""

        details = runner._parse_ruff_output(check_output, format_output)

        assert details["check_issues"] == 0
        assert details["format_issues"] == 0
        assert details["rules_violated"] == []


class TestMypyRunner:
    """Test MypyRunner functionality."""

    @patch("subprocess.run")
    def test_mypy_runner_success(self, mock_run):
        """Test successful mypy run."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Success: no issues found in 1 file", stderr="")

        runner = MypyRunner()
        result = runner.run(target_args=["test.py"])

        assert result.name == "MyPy"
        assert result.passed is True
        assert result.exit_code == 0
        assert "Success" in result.output

    @patch("subprocess.run")
    def test_mypy_runner_failure(self, mock_run):
        """Test failed mypy run."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="test.py:1: error: Name 'x' is not defined [name-defined]", stderr=""
        )

        runner = MypyRunner()
        result = runner.run()

        assert result.name == "MyPy"
        assert result.passed is False
        assert result.exit_code == 1
        assert result.details["errors"] == 1
        assert "name-defined" in result.details["error_codes"]

    def test_parse_mypy_output_with_errors(self):
        """Test parsing mypy output with errors."""
        runner = MypyRunner()
        output = "test.py:1: error: Name 'x' is not defined [name-defined]\ntest.py:2: error: Unused variable [unused-variable]"

        details = runner._parse_mypy_output(output)

        assert details["errors"] == 2
        assert "name-defined" in details["error_codes"]
        assert "unused-variable" in details["error_codes"]

    def test_parse_mypy_output_success(self):
        """Test parsing successful mypy output."""
        runner = MypyRunner()
        output = "Success: no issues found in 5 files processed"

        details = runner._parse_mypy_output(output)

        assert details["errors"] == 0
        assert details["error_codes"] == []


class TestPytestRunner:
    """Test PytestRunner functionality."""

    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open, read_data='{"files": {}}')
    def test_test_runner_success(self, mock_file, mock_run):
        """Test successful test run."""
        mock_run.return_value = MagicMock(returncode=0, stdout="5 passed\nTOTAL     100   20   80%", stderr="")

        runner = PytestRunner()
        result = runner.run()

        assert result.name == "Tests"
        assert result.passed is True
        assert result.exit_code == 0
        assert result.details["passed"] == 5
        assert result.details["coverage_total"] == 80.0

    @patch("subprocess.run")
    def test_test_runner_failure(self, mock_run):
        """Test failed test run."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="3 passed, 2 failed\nTOTAL     100   50   50%", stderr="Some tests failed"
        )

        runner = PytestRunner()
        result = runner.run()

        assert result.name == "Tests"
        assert result.passed is False
        assert result.exit_code == 1
        assert result.details["passed"] == 3
        assert result.details["failed"] == 2

    def test_parse_pytest_output_with_results(self):
        """Test parsing pytest output with results."""
        runner = PytestRunner()
        output = "10 passed, 2 failed, 1 skipped\nTOTAL     200   40   80%"

        details = runner._parse_pytest_output(output)

        assert details["passed"] == 10
        assert details["failed"] == 2
        assert details["skipped"] == 1
        assert details["tests_run"] == 13
        assert details["coverage_total"] == 80.0

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"files": {"test.py": {"summary": {"percent_covered": 95.5}}}}',
    )
    def test_parse_pytest_output_with_coverage_file(self, mock_file):
        """Test parsing pytest output with coverage file."""
        runner = PytestRunner()
        output = "5 passed"

        details = runner._parse_pytest_output(output)

        assert details["coverage_files"]["test.py"] == 95.5


class TestValidationScript:
    """Test ValidationScript main orchestrator."""

    def test_validation_script_init(self):
        """Test ValidationScript initialization."""
        script = ValidationScript()
        assert "ruff" in script.runners
        assert "mypy" in script.runners
        assert "tests" in script.runners

    @patch.object(ValidationScript, "_get_environment_info")
    @patch.object(ValidationScript, "_generate_summary")
    @patch.object(ValidationScript, "_output_report")
    def test_run_validation_success(self, mock_output, mock_summary, mock_env):
        """Test successful validation run."""
        # Setup mocks
        mock_env.return_value = {
            "timestamp": "2023-01-01T00:00:00",
            "git_commit": "abcd1234",
            "git_branch": "main",
            "python_version": "3.11.0",
            "environment": "test",
        }
        mock_summary.return_value = {"total": 1}

        # Mock successful runner
        mock_runner = MagicMock()
        mock_runner.run.return_value = CheckResult(
            name="Test", passed=True, duration=1.0, output="", error="", details={}, exit_code=0
        )

        script = ValidationScript()
        script.runners = {"test": mock_runner}

        report = script.run_validation(checks=["test"])

        assert report.overall_status == "PASSED"
        assert len(report.checks) == 1
        mock_runner.run.assert_called_once()

    @patch.object(ValidationScript, "_get_environment_info")
    @patch.object(ValidationScript, "_generate_summary")
    @patch.object(ValidationScript, "_output_report")
    def test_run_validation_failure(self, mock_output, mock_summary, mock_env):
        """Test failed validation run."""
        # Setup mocks
        mock_env.return_value = {
            "timestamp": "2023-01-01T00:00:00",
            "git_commit": "abcd1234",
            "git_branch": "main",
            "python_version": "3.11.0",
            "environment": "test",
        }
        mock_summary.return_value = {"total": 1}

        # Mock failed runner
        mock_runner = MagicMock()
        mock_runner.run.return_value = CheckResult(
            name="Test", passed=False, duration=1.0, output="", error="Failed", details={}, exit_code=1
        )

        script = ValidationScript()
        script.runners = {"test": mock_runner}

        report = script.run_validation(checks=["test"])

        assert report.overall_status == "FAILED"
        assert len(report.checks) == 1

    @patch("subprocess.run")
    def test_get_environment_info(self, mock_run):
        """Test getting environment information."""
        mock_run.side_effect = [MagicMock(returncode=0, stdout="abcd1234efgh"), MagicMock(returncode=0, stdout="main")]

        script = ValidationScript()
        info = script._get_environment_info()

        assert info["git_commit"] == "abcd1234"
        assert info["git_branch"] == "main"
        assert "timestamp" in info
        assert "python_version" in info

    @patch("subprocess.run")
    def test_get_environment_info_git_failure(self, mock_run):
        """Test getting environment info when git fails."""
        mock_run.side_effect = [MagicMock(returncode=1, stdout=""), MagicMock(returncode=1, stdout="")]

        script = ValidationScript()
        info = script._get_environment_info()

        assert info["git_commit"] == "unknown"
        assert info["git_branch"] == "unknown"

    def test_generate_summary(self):
        """Test generating summary statistics."""
        script = ValidationScript()
        results = [
            CheckResult("Test1", True, 1.0, "", "", {}, 0),
            CheckResult("Test2", False, 2.0, "", "", {}, 1),
        ]

        summary = script._generate_summary(results)

        assert summary["total_checks"] == 2
        assert summary["passed_checks"] == 1
        assert summary["failed_checks"] == 1
        assert summary["total_duration"] == 3.0

    def test_output_json_report(self, tmp_path):
        """Test JSON report output."""
        script = ValidationScript()

        report = ValidationReport(
            timestamp="2023-01-01T00:00:00",
            git_commit="abcd1234",
            git_branch="main",
            python_version="3.11.0",
            environment="test",
            total_duration=1.0,
            overall_status="PASSED",
            checks=[],
            summary={},
        )

        output_file = tmp_path / "report.json"
        script._output_json_report(report, str(output_file))

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["overall_status"] == "PASSED"

    def test_output_html_report(self, tmp_path):
        """Test HTML report output."""
        script = ValidationScript()

        report = ValidationReport(
            timestamp="2023-01-01T00:00:00",
            git_commit="abcd1234",
            git_branch="main",
            python_version="3.11.0",
            environment="test",
            total_duration=1.0,
            overall_status="PASSED",
            checks=[],
            summary={"total_checks": 0, "passed_checks": 0, "failed_checks": 0, "total_duration": 1.0},
        )

        output_file = tmp_path / "report.html"
        script._output_html_report(report, str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "Mistral NER Validation Report" in content
        assert "PASSED" in content


class TestMainFunction:
    """Test main function and CLI argument parsing."""

    @patch("scripts.validate.ValidationScript")
    @patch("sys.argv", ["validate.py", "--ruff-only"])
    def test_main_ruff_only(self, mock_script_class):
        """Test main function with --ruff-only."""
        mock_script = MagicMock()
        mock_script.run_validation.return_value = MagicMock(overall_status="PASSED")
        mock_script_class.return_value = mock_script

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        mock_script.run_validation.assert_called_once()
        args = mock_script.run_validation.call_args
        assert args[1]["checks"] == ["ruff"]

    @patch("scripts.validate.ValidationScript")
    @patch("sys.argv", ["validate.py", "--mypy-only"])
    def test_main_mypy_only(self, mock_script_class):
        """Test main function with --mypy-only."""
        mock_script = MagicMock()
        mock_script.run_validation.return_value = MagicMock(overall_status="PASSED")
        mock_script_class.return_value = mock_script

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        args = mock_script.run_validation.call_args
        assert args[1]["checks"] == ["mypy"]

    @patch("scripts.validate.ValidationScript")
    @patch("sys.argv", ["validate.py", "--tests-only"])
    def test_main_tests_only(self, mock_script_class):
        """Test main function with --tests-only."""
        mock_script = MagicMock()
        mock_script.run_validation.return_value = MagicMock(overall_status="PASSED")
        mock_script_class.return_value = mock_script

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        args = mock_script.run_validation.call_args
        assert args[1]["checks"] == ["tests"]

    @patch("scripts.validate.ValidationScript")
    @patch("sys.argv", ["validate.py"])
    def test_main_all_checks(self, mock_script_class):
        """Test main function with all checks."""
        mock_script = MagicMock()
        mock_script.run_validation.return_value = MagicMock(overall_status="PASSED")
        mock_script_class.return_value = mock_script

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        args = mock_script.run_validation.call_args
        assert args[1]["checks"] == ["ruff", "mypy", "tests"]

    @patch("scripts.validate.ValidationScript")
    @patch("sys.argv", ["validate.py", "--files", "test.py", "main.py"])
    def test_main_with_files(self, mock_script_class):
        """Test main function with file targeting."""
        mock_script = MagicMock()
        mock_script.run_validation.return_value = MagicMock(overall_status="PASSED")
        mock_script_class.return_value = mock_script

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        args = mock_script.run_validation.call_args
        assert args[1]["target_args"] == ["test.py", "main.py"]

    @patch("scripts.validate.ValidationScript")
    @patch("sys.argv", ["validate.py", "--json", "--output", "report.json"])
    def test_main_json_output(self, mock_script_class):
        """Test main function with JSON output."""
        mock_script = MagicMock()
        mock_script.run_validation.return_value = MagicMock(overall_status="PASSED")
        mock_script_class.return_value = mock_script

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        args = mock_script.run_validation.call_args
        assert args[1]["output_format"] == "json"
        assert args[1]["output_file"] == "report.json"

    @patch("scripts.validate.ValidationScript")
    @patch("sys.argv", ["validate.py"])
    def test_main_failure(self, mock_script_class):
        """Test main function with validation failure."""
        mock_script = MagicMock()
        mock_script.run_validation.return_value = MagicMock(overall_status="FAILED")
        mock_script_class.return_value = mock_script

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("scripts.validate.ValidationScript")
    @patch("sys.argv", ["validate.py"])
    def test_main_keyboard_interrupt(self, mock_script_class):
        """Test main function with keyboard interrupt."""
        mock_script = MagicMock()
        mock_script.run_validation.side_effect = KeyboardInterrupt()
        mock_script_class.return_value = mock_script

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 130

    @patch("scripts.validate.ValidationScript")
    @patch("sys.argv", ["validate.py"])
    def test_main_exception(self, mock_script_class):
        """Test main function with general exception."""
        mock_script = MagicMock()
        mock_script.run_validation.side_effect = Exception("Test error")
        mock_script_class.return_value = mock_script

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1


class TestIntegration:
    """Integration tests for the validation script."""

    def test_validation_script_executable(self):
        """Test that the validation script is executable."""
        script_path = Path("scripts/validate.py")
        assert script_path.exists()
        assert script_path.stat().st_mode & 0o111  # Check executable bit

    @patch("subprocess.run")
    def test_end_to_end_validation(self, mock_run):
        """Test end-to-end validation with mocked subprocess calls."""
        # Mock successful tool runs
        mock_run.side_effect = [
            # Git commands for environment info
            MagicMock(returncode=0, stdout="abcd1234"),
            MagicMock(returncode=0, stdout="main"),
            # Ruff check and format
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
            # MyPy
            MagicMock(returncode=0, stdout="Success: no issues found", stderr=""),
            # Pytest
            MagicMock(returncode=0, stdout="5 passed\nTOTAL     100   10   90%", stderr=""),
        ]

        script = ValidationScript()
        report = script.run_validation(checks=["ruff", "mypy", "tests"])

        assert report.overall_status == "PASSED"
        assert len(report.checks) == 3
        assert all(check.passed for check in report.checks)
