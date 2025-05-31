"""Tests for scripts/sync_wandb.py."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add the scripts path to sys.path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from sync_wandb import main, setup_logging


class TestSyncWandB:
    """Test sync_wandb.py script."""

    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging()
        assert logger.name == "sync_wandb"

    @patch("sync_wandb.list_offline_runs")
    @patch("sync_wandb.Path.exists")
    @patch("sys.argv", ["sync_wandb.py", "--list"])
    def test_main_list_no_runs(self, mock_exists, mock_list_runs):
        """Test main function with --list flag and no runs."""
        mock_exists.return_value = True
        mock_list_runs.return_value = []

        main()

        mock_list_runs.assert_called_once_with("./wandb")

    @patch("sync_wandb.list_offline_runs")
    @patch("sync_wandb.Path.exists")
    @patch("sys.argv", ["sync_wandb.py", "--list"])
    def test_main_list_with_runs(self, mock_exists, mock_list_runs):
        """Test main function with --list flag and runs."""
        mock_exists.return_value = True
        mock_list_runs.return_value = [
            {"run_id": "offline-run-123", "size_mb": 10.5},
            {"run_id": "offline-run-456", "size_mb": 20.1},
        ]

        main()

        mock_list_runs.assert_called_once_with("./wandb")

    @patch("sync_wandb.Path.exists")
    @patch("sys.argv", ["sync_wandb.py", "--wandb-dir", "/nonexistent"])
    def test_main_nonexistent_dir(self, mock_exists):
        """Test main function with non-existent directory."""
        mock_exists.return_value = False

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("sync_wandb.list_offline_runs")
    @patch("sync_wandb.sync_offline_run")
    @patch("sync_wandb.Path.exists")
    @patch("sys.argv", ["sync_wandb.py", "--run-id", "offline-run-123"])
    def test_main_sync_specific_run_success(self, mock_exists, mock_sync, mock_list_runs):
        """Test syncing a specific run successfully."""
        mock_exists.return_value = True
        mock_list_runs.return_value = [
            {"run_id": "offline-run-123", "path": "/path/to/run"},
        ]
        mock_sync.return_value = True

        main()

        mock_sync.assert_called_once_with("/path/to/run")

    @patch("sync_wandb.list_offline_runs")
    @patch("sync_wandb.sync_offline_run")
    @patch("sync_wandb.Path.exists")
    @patch("sys.argv", ["sync_wandb.py", "--run-id", "offline-run-123"])
    def test_main_sync_specific_run_failure(self, mock_exists, mock_sync, mock_list_runs):
        """Test syncing a specific run with failure."""
        mock_exists.return_value = True
        mock_list_runs.return_value = [
            {"run_id": "offline-run-123", "path": "/path/to/run"},
        ]
        mock_sync.return_value = False

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("sync_wandb.list_offline_runs")
    @patch("sync_wandb.Path.exists")
    @patch("sys.argv", ["sync_wandb.py", "--run-id", "nonexistent-run"])
    def test_main_sync_nonexistent_run(self, mock_exists, mock_list_runs):
        """Test syncing a non-existent run."""
        mock_exists.return_value = True
        mock_list_runs.return_value = [
            {"run_id": "offline-run-123", "path": "/path/to/run"},
        ]

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("sync_wandb.list_offline_runs")
    @patch("sync_wandb.Path.exists")
    @patch("sys.argv", ["sync_wandb.py", "--dry-run"])
    def test_main_dry_run_all(self, mock_exists, mock_list_runs):
        """Test dry run for all runs."""
        mock_exists.return_value = True
        mock_list_runs.return_value = [
            {"run_id": "offline-run-123", "path": "/path/to/run1"},
            {"run_id": "offline-run-456", "path": "/path/to/run2"},
        ]

        main()

        mock_list_runs.assert_called_once_with("./wandb")

    @patch("sync_wandb.list_offline_runs")
    @patch("sync_wandb.Path.exists")
    @patch("sys.argv", ["sync_wandb.py", "--run-id", "offline-run-123", "--dry-run"])
    def test_main_dry_run_specific(self, mock_exists, mock_list_runs):
        """Test dry run for specific run."""
        mock_exists.return_value = True
        mock_list_runs.return_value = [
            {"run_id": "offline-run-123", "path": "/path/to/run"},
        ]

        main()

        mock_list_runs.assert_called_once_with("./wandb")

    @patch("sync_wandb.list_offline_runs")
    @patch("sync_wandb.sync_all_offline_runs")
    @patch("sync_wandb.Path.exists")
    @patch("sys.argv", ["sync_wandb.py"])
    def test_main_sync_all_success(self, mock_exists, mock_sync_all, mock_list_runs):
        """Test syncing all runs successfully."""
        mock_exists.return_value = True
        mock_list_runs.return_value = [
            {"run_id": "offline-run-123", "path": "/path/to/run1"},
            {"run_id": "offline-run-456", "path": "/path/to/run2"},
        ]
        mock_sync_all.return_value = {
            "synced": ["offline-run-123", "offline-run-456"],
            "failed": [],
        }

        main()

        mock_sync_all.assert_called_once_with("./wandb")

    @patch("sync_wandb.list_offline_runs")
    @patch("sync_wandb.sync_all_offline_runs")
    @patch("sync_wandb.Path.exists")
    @patch("sys.argv", ["sync_wandb.py"])
    def test_main_sync_all_with_failures(self, mock_exists, mock_sync_all, mock_list_runs):
        """Test syncing all runs with some failures."""
        mock_exists.return_value = True
        mock_list_runs.return_value = [
            {"run_id": "offline-run-123", "path": "/path/to/run1"},
            {"run_id": "offline-run-456", "path": "/path/to/run2"},
        ]
        mock_sync_all.return_value = {
            "synced": ["offline-run-123"],
            "failed": ["offline-run-456"],
        }

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("sync_wandb.list_offline_runs")
    @patch("sync_wandb.sync_all_offline_runs")
    @patch("sync_wandb.Path.exists")
    @patch("sys.argv", ["sync_wandb.py"])
    def test_main_sync_all_no_runs_to_sync(self, mock_exists, mock_sync_all, mock_list_runs):
        """Test syncing when no runs need syncing."""
        mock_exists.return_value = True
        mock_list_runs.return_value = [{"run_id": "offline-run-123", "path": "/path/to/run1"}]
        mock_sync_all.return_value = {"synced": [], "failed": []}

        main()

        mock_sync_all.assert_called_once_with("./wandb")
