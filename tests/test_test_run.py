"""Tests for test_run.py module."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import test_run


class TestRunTest:
    """Test the run_test function."""

    def test_run_test_passing(self):
        """Test run_test with passing test."""
        test_run.passed = 0
        test_run.failed = 0
        test_run.errors = []

        def passing_test():
            assert True

        with patch("builtins.print"):
            test_run.run_test(passing_test, "test_passing")

        assert test_run.passed == 1
        assert test_run.failed == 0
        assert len(test_run.errors) == 0

    def test_run_test_assertion_error(self):
        """Test run_test with assertion error."""
        test_run.passed = 0
        test_run.failed = 0
        test_run.errors = []

        def failing_test():
            raise AssertionError("Test assertion failed")

        with patch("builtins.print"):
            test_run.run_test(failing_test, "test_failing")

        assert test_run.passed == 0
        assert test_run.failed == 1
        assert len(test_run.errors) == 1
        assert test_run.errors[0]["test"] == "test_failing"
        assert test_run.errors[0]["type"] == "assertion"
        assert "Test assertion failed" in test_run.errors[0]["error"]

    def test_run_test_exception(self):
        """Test run_test with general exception."""
        test_run.passed = 0
        test_run.failed = 0
        test_run.errors = []

        def error_test():
            raise ValueError("Test error")

        with patch("builtins.print"):
            test_run.run_test(error_test, "test_error")

        assert test_run.passed == 0
        assert test_run.failed == 1
        assert len(test_run.errors) == 1
        assert test_run.errors[0]["test"] == "test_error"
        assert test_run.errors[0]["type"] == "exception"
        assert "Test error" in test_run.errors[0]["error"]
        assert "traceback" in test_run.errors[0]


class TestTestImports:
    """Test the test_imports function."""

    @patch("test_run.run_test")
    def test_test_imports(self, mock_run_test):
        """Test that test_imports calls run_test for each module."""
        with patch("builtins.print"):
            test_run.test_imports()

        # Should call run_test for each module
        expected_modules = ["src.config", "src.utils", "src.data", "src.model", "src.evaluation", "src.training"]
        assert mock_run_test.call_count == len(expected_modules)

        # Check that each module was tested
        calls = [call[0][1] for call in mock_run_test.call_args_list]
        for module in expected_modules:
            assert f"import {module}" in calls


class TestTestConfig:
    """Test the test_config function."""

    @patch("test_run.run_test")
    def test_test_config(self, mock_run_test):
        """Test that test_config calls run_test for each config test."""
        with patch("builtins.print"):
            test_run.test_config()

        # Should call run_test 3 times (default, data, update)
        assert mock_run_test.call_count == 3

        # Check test names
        calls = [call[0][1] for call in mock_run_test.call_args_list]
        assert "Default configuration" in calls
        assert "Data configuration" in calls
        assert "Configuration update from args" in calls

    def test_default_config_test_function(self):
        """Test the actual default config test logic."""
        # This tests the inner function logic
        from src.config import Config

        config = Config()
        assert config.model.model_name == "mistralai/Mistral-7B-v0.3"
        assert config.model.num_labels == 9
        assert config.model.load_in_8bit
        assert config.data.max_length == 256
        assert config.training.num_train_epochs == 5

    def test_data_config_test_function(self):
        """Test the actual data config test logic."""
        from src.config import DataConfig

        data_config = DataConfig()
        assert len(data_config.label_names) == 9
        assert data_config.id2label[0] == "O"
        assert data_config.label2id["O"] == 0

    def test_config_update_test_function(self):
        """Test the actual config update test logic."""
        from src.config import Config

        config = Config()

        class Args:
            model_name = "test-model"
            learning_rate = 1e-3
            use_wandb = False

        args = Args()
        config.update_from_args(args)
        assert config.model.model_name == "test-model"
        assert config.training.learning_rate == 1e-3
        assert not config.logging.use_wandb


class TestTestUtils:
    """Test the test_utils function."""

    @patch("test_run.run_test")
    def test_test_utils(self, mock_run_test):
        """Test that test_utils calls run_test for each util test."""
        with patch("builtins.print"):
            test_run.test_utils()

        # Should call run_test 2 times
        assert mock_run_test.call_count == 2

        # Check test names
        calls = [call[0][1] for call in mock_run_test.call_args_list]
        assert "Memory estimation" in calls
        assert "Mixed precision detection" in calls

    def test_memory_estimation_test_function(self):
        """Test the actual memory estimation test logic."""
        from src.utils import estimate_memory_usage

        estimate = estimate_memory_usage(
            model_size_gb=14.0, batch_size=4, sequence_length=256, use_8bit=True, use_lora=True
        )
        assert "model_memory_gb" in estimate
        assert "total_memory_gb" in estimate
        assert estimate["model_memory_gb"] < 14.0  # Should be reduced with 8-bit

    def test_mixed_precision_test_function(self):
        """Test the actual mixed precision test logic."""
        from src.utils import detect_mixed_precision_support

        support = detect_mixed_precision_support()
        assert isinstance(support, dict)
        assert "fp16" in support
        assert "bf16" in support
        assert all(isinstance(v, bool) for v in support.values())


class TestTestData:
    """Test the test_data function."""

    @patch("test_run.run_test")
    def test_test_data(self, mock_run_test):
        """Test that test_data calls run_test for each data test."""
        with patch("builtins.print"):
            test_run.test_data()

        # Should call run_test 2 times
        assert mock_run_test.call_count == 2

        # Check test names
        calls = [call[0][1] for call in mock_run_test.call_args_list]
        assert "Sample dataset creation" in calls
        assert "Label list extraction" in calls

    def test_sample_dataset_test_function(self):
        """Test the actual sample dataset test logic."""
        from src.data import create_sample_dataset

        dataset = create_sample_dataset(size=10)
        assert "train" in dataset
        assert "validation" in dataset
        assert "test" in dataset
        assert len(dataset["train"]) == 10
        assert len(dataset["validation"]) == 2

    def test_label_list_test_function(self):
        """Test the actual label list test logic."""
        from src.data import create_sample_dataset, get_label_list

        dataset = create_sample_dataset(size=5)
        labels = get_label_list(dataset)
        assert isinstance(labels, list)
        assert len(labels) == 9
        assert labels[0] == "O"


class TestMain:
    """Test the main function."""

    @patch("test_run.test_data")
    @patch("test_run.test_utils")
    @patch("test_run.test_config")
    @patch("test_run.test_imports")
    @patch("builtins.print")
    def test_main_success(self, mock_print, mock_test_imports, mock_test_config, mock_test_utils, mock_test_data):
        """Test main function with successful tests."""
        test_run.passed = 10
        test_run.failed = 0
        test_run.errors = []

        with pytest.raises(SystemExit) as exc_info:
            test_run.main()

        assert exc_info.value.code == 0

        # Verify all test functions were called
        mock_test_imports.assert_called_once()
        mock_test_config.assert_called_once()
        mock_test_utils.assert_called_once()
        mock_test_data.assert_called_once()

    @patch("test_run.test_data")
    @patch("test_run.test_utils")
    @patch("test_run.test_config")
    @patch("test_run.test_imports")
    @patch("builtins.print")
    def test_main_with_failures(self, mock_print, mock_test_imports, mock_test_config, mock_test_utils, mock_test_data):
        """Test main function with test failures."""
        test_run.passed = 5
        test_run.failed = 2
        test_run.errors = [
            {"test": "test1", "error": "Error 1", "type": "assertion"},
            {"test": "test2", "error": "Error 2", "type": "exception", "traceback": "Traceback info"},
        ]

        with pytest.raises(SystemExit) as exc_info:
            test_run.main()

        assert exc_info.value.code == 1

    @patch("test_run.test_data")
    @patch("test_run.test_utils")
    @patch("test_run.test_config")
    @patch("test_run.test_imports")
    @patch("builtins.print")
    def test_main_prints_summary(
        self, mock_print, mock_test_imports, mock_test_config, mock_test_utils, mock_test_data
    ):
        """Test that main function prints test summary."""
        test_run.passed = 8
        test_run.failed = 1
        test_run.errors = [{"test": "test1", "error": "Error", "type": "assertion"}]

        with pytest.raises(SystemExit):
            test_run.main()

        # Check that summary was printed
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
        summary_found = any("8 passed, 1 failed" in str(call) for call in print_calls)
        assert summary_found

    @patch("test_run.test_data")
    @patch("test_run.test_utils")
    @patch("test_run.test_config")
    @patch("test_run.test_imports")
    @patch("builtins.print")
    def test_main_prints_errors(self, mock_print, mock_test_imports, mock_test_config, mock_test_utils, mock_test_data):
        """Test that main function prints error details."""
        test_run.passed = 0
        test_run.failed = 1
        test_run.errors = [{"test": "failing_test", "error": "Test failed", "type": "assertion"}]

        with pytest.raises(SystemExit):
            test_run.main()

        # Check that error details were printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        error_section_found = any("Errors:" in call for call in print_calls)
        assert error_section_found


class TestGlobalVariables:
    """Test global variable initialization."""

    def test_initial_global_state(self):
        """Test that global variables start in correct state."""
        # These might be modified by other tests, so we just check they exist
        assert hasattr(test_run, "passed")
        assert hasattr(test_run, "failed")
        assert hasattr(test_run, "errors")
        assert isinstance(test_run.errors, list)


class TestModuleStructure:
    """Test module structure and imports."""

    def test_module_docstring(self):
        """Test that module has proper docstring."""
        assert test_run.__doc__ is not None
        assert "Simple test runner" in test_run.__doc__

    def test_path_setup(self):
        """Test that sys.path is modified correctly."""
        # The module should add parent directory to path
        parent_path = str(Path(test_run.__file__).parent)
        assert parent_path in sys.path

    def test_all_functions_exist(self):
        """Test that all expected functions exist."""
        expected_functions = ["run_test", "test_imports", "test_config", "test_utils", "test_data", "main"]

        for func_name in expected_functions:
            assert hasattr(test_run, func_name)
            assert callable(getattr(test_run, func_name))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
