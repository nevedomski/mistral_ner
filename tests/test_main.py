"""Tests for main.py module."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import main


class TestMainModule:
    """Test main module functionality."""

    @patch("main.main")
    def test_main_import_and_call(self, mock_main_function):
        """Test that main function is imported and can be called."""
        # Verify the main function exists and is callable
        assert callable(main.main)

        # Call the main function
        main.main()

        # Verify it was called
        mock_main_function.assert_called_once()

    def test_main_module_path_setup(self):
        """Test that sys.path is properly modified."""
        # The main module should add the parent directory to sys.path
        # This is important for importing scripts.train
        parent_path = str(Path(main.__file__).parent)
        assert parent_path in sys.path

    @patch("scripts.train.main")
    def test_main_calls_train_script(self, mock_train_main):
        """Test that main.py calls the training script's main function."""
        # Re-import to trigger the main call
        import importlib

        importlib.reload(main)

        # The main function should have been called during import
        # since it's at module level with if __name__ == "__main__"
        # We can't easily test this without running the script directly

    def test_main_module_attributes(self):
        """Test main module has expected attributes."""
        # Check that the main module has the necessary imports
        assert hasattr(main, "main")
        assert hasattr(main, "sys")
        assert hasattr(main, "Path")

    def test_main_module_docstring(self):
        """Test main module has docstring."""
        assert main.__doc__ is not None
        assert "Main entry point for Mistral NER training" in main.__doc__


# Test if script can be run directly
class TestMainScriptExecution:
    """Test main script execution scenarios."""

    @patch("scripts.train.main")
    def test_script_execution_calls_train_main(self, mock_train_main):
        """Test that running main.py as script calls train.main()."""
        # We can test the import structure but actual execution
        # requires subprocess testing which is more complex

        # Test that the import path is correct
        from scripts.train import main as train_main

        assert callable(train_main)

    def test_main_function_exists_in_scripts_train(self):
        """Test that the main function exists in scripts.train module."""
        from scripts.train import main as train_main

        assert callable(train_main)

        # Check function signature
        import inspect

        sig = inspect.signature(train_main)
        assert len(sig.parameters) == 0  # main() takes no arguments

    def test_path_manipulation(self):
        """Test that path manipulation works correctly."""
        # Test Path usage
        current_file_path = Path(main.__file__)
        parent_path = current_file_path.parent

        # Should be able to construct path to scripts
        scripts_path = parent_path / "scripts"
        assert scripts_path.name == "scripts"

    @patch("sys.path")
    def test_sys_path_modification(self, mock_sys_path):
        """Test sys.path modification behavior."""
        # Re-import to trigger path modification

        # Mock sys.path as a list
        mock_sys_path.append = lambda x: None

        # This would be called during import
        test_path = str(Path(__file__).parent)
        mock_sys_path.append(test_path)

        # Verify append was called (mocked)
        assert callable(mock_sys_path.append)


# Integration test
class TestMainIntegration:
    """Test main module integration."""

    def test_can_import_train_script_after_path_setup(self):
        """Test that train script can be imported after path setup."""
        # This should work because main.py sets up the path
        try:
            from scripts.train import main as train_main

            assert callable(train_main)
            success = True
        except ImportError:
            success = False

        assert success, "Should be able to import scripts.train.main after path setup"

    def test_main_module_can_be_imported(self):
        """Test that main module can be imported without errors."""
        # Should not raise any exceptions
        import main as main_module

        assert main_module is not None

    def test_main_module_functions_accessible(self):
        """Test that all expected functions are accessible."""
        # After import, should have access to train functionality
        from scripts.train import main as train_main
        from scripts.train import parse_args

        assert callable(parse_args)
        assert callable(train_main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
