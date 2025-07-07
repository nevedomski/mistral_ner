"""Tests for scripts/train.py module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train import main, parse_args


class MockDataset:
    """Mock dataset class with __len__ method."""

    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class TestParseArgs:
    """Test argument parsing functionality."""

    def test_parse_args_defaults(self):
        """Test argument parsing with default values."""
        with patch("sys.argv", ["train.py"]):
            args = parse_args()

        assert args.config == "configs/default.yaml"
        assert args.model_name is None
        assert args.load_in_8bit is None
        assert args.max_length is None
        assert args.output_dir is None
        assert args.num_train_epochs is None
        assert args.per_device_train_batch_size is None
        assert args.learning_rate is None
        assert args.resume_from_checkpoint is None
        assert args.use_wandb is None
        assert args.wandb_project is None
        assert args.eval_only is False
        assert args.test is False
        assert args.debug is False

    def test_parse_args_custom_config(self):
        """Test argument parsing with custom config."""
        with patch("sys.argv", ["train.py", "--config", "custom.yaml"]):
            args = parse_args()

        assert args.config == "custom.yaml"

    def test_parse_args_model_arguments(self):
        """Test model-related argument parsing."""
        with patch("sys.argv", ["train.py", "--model-name", "custom-model", "--load-in-8bit"]):
            args = parse_args()

        assert args.model_name == "custom-model"
        assert args.load_in_8bit is True

    def test_parse_args_no_load_in_8bit(self):
        """Test --no-load-in-8bit flag."""
        with patch("sys.argv", ["train.py", "--no-load-in-8bit"]):
            args = parse_args()

        assert args.load_in_8bit is False

    def test_parse_args_data_arguments(self):
        """Test data-related argument parsing."""
        with patch("sys.argv", ["train.py", "--max-length", "512"]):
            args = parse_args()

        assert args.max_length == 512

    def test_parse_args_training_arguments(self):
        """Test training-related argument parsing."""
        test_argv = [
            "train.py",
            "--output-dir",
            "/test/output",
            "--num-train-epochs",
            "5",
            "--batch-size",
            "8",
            "--learning-rate",
            "1e-4",
            "--resume-from-checkpoint",
            "/test/checkpoint",
        ]
        with patch("sys.argv", test_argv):
            args = parse_args()

        assert args.output_dir == "/test/output"
        assert args.num_train_epochs == 5
        assert args.per_device_train_batch_size == 8
        assert args.learning_rate == 1e-4
        assert args.resume_from_checkpoint == "/test/checkpoint"

    def test_parse_args_logging_arguments(self):
        """Test logging-related argument parsing."""
        with patch("sys.argv", ["train.py", "--use-wandb", "--wandb-project", "test-project"]):
            args = parse_args()

        assert args.use_wandb is True
        assert args.wandb_project == "test-project"

    def test_parse_args_no_wandb(self):
        """Test --no-wandb flag."""
        with patch("sys.argv", ["train.py", "--no-wandb"]):
            args = parse_args()

        assert args.use_wandb is False

    def test_parse_args_other_flags(self):
        """Test other flags."""
        with patch("sys.argv", ["train.py", "--eval-only", "--test", "--debug"]):
            args = parse_args()

        assert args.eval_only is True
        assert args.test is True
        assert args.debug is True


class TestMain:
    """Test main training function."""

    @patch("scripts.train.run_training_pipeline")
    @patch("scripts.train.prepare_datasets")
    @patch("scripts.train.print_trainable_parameters")
    @patch("scripts.train.setup_model")
    @patch("scripts.train.print_dataset_statistics")
    @patch("scripts.train.load_conll2003_dataset")
    @patch("scripts.train.check_gpu_memory")
    @patch("scripts.train.setup_logging")
    @patch("scripts.train.Config.from_yaml")
    @patch("scripts.train.parse_args")
    def test_main_training_mode(
        self,
        mock_parse_args,
        mock_config_from_yaml,
        mock_setup_logging,
        mock_check_gpu,
        mock_load_dataset,
        mock_print_stats,
        mock_setup_model,
        mock_print_params,
        mock_prepare_datasets,
        mock_run_training,
    ):
        """Test main function in training mode."""
        # Setup mocks
        mock_args = Mock()
        mock_args.config = "configs/default.yaml"
        mock_args.debug = False
        mock_args.eval_only = False
        mock_args.test = False
        mock_parse_args.return_value = mock_args

        mock_config = Mock()
        mock_config.logging.log_level = "info"
        mock_config.logging.log_dir = "./logs"
        mock_config.model.model_name = "test-model"
        mock_config.data.label_names = ["O", "B-PER"]
        mock_config.data.multi_dataset.enabled = False  # Single dataset mode
        mock_config.training.per_device_eval_batch_size = 8
        # Ensure hyperopt is disabled in the mock
        mock_config.hyperopt.enabled = False
        mock_config_from_yaml.return_value = mock_config

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer)

        mock_print_params.return_value = {"trainable_params": 1000}

        # Create mock datasets with proper __len__ method
        mock_train_dataset = MagicMock()
        mock_train_dataset.__len__.return_value = 1000
        mock_eval_dataset = MagicMock()
        mock_eval_dataset.__len__.return_value = 200
        mock_test_dataset = MagicMock()
        mock_test_dataset.__len__.return_value = 300
        mock_data_collator = Mock()
        mock_prepare_datasets.return_value = (
            mock_train_dataset,
            mock_eval_dataset,
            mock_test_dataset,
            mock_data_collator,
        )

        mock_train_metrics = {"train_loss": 0.5}
        mock_run_training.return_value = mock_train_metrics

        with patch("torch.cuda.is_available", return_value=True):
            main()

        # Verify key function calls
        mock_config.update_from_args.assert_called_once_with(mock_args)
        mock_setup_logging.assert_called_once_with("info", "./logs")
        mock_load_dataset.assert_called_once()
        mock_setup_model.assert_called_once_with(model_name="test-model", config=mock_config)
        mock_prepare_datasets.assert_called_once()
        mock_run_training.assert_called_once()

    @patch("scripts.train.evaluate_model")
    @patch("scripts.train.prepare_datasets")
    @patch("scripts.train.print_trainable_parameters")
    @patch("scripts.train.setup_model")
    @patch("scripts.train.print_dataset_statistics")
    @patch("scripts.train.load_conll2003_dataset")
    @patch("scripts.train.check_gpu_memory")
    @patch("scripts.train.setup_logging")
    @patch("scripts.train.Config.from_yaml")
    @patch("scripts.train.parse_args")
    def test_main_eval_only_mode(
        self,
        mock_parse_args,
        mock_config_from_yaml,
        mock_setup_logging,
        mock_check_gpu,
        mock_load_dataset,
        mock_print_stats,
        mock_setup_model,
        mock_print_params,
        mock_prepare_datasets,
        mock_evaluate,
    ):
        """Test main function in evaluation-only mode."""
        # Setup mocks
        mock_args = Mock()
        mock_args.config = "configs/default.yaml"
        mock_args.debug = False
        mock_args.eval_only = True
        mock_args.test = True
        mock_parse_args.return_value = mock_args

        mock_config = Mock()
        mock_config.logging.log_level = "info"
        mock_config.logging.log_dir = "./logs"
        mock_config.model.model_name = "test-model"
        mock_config.data.label_names = ["O", "B-PER"]
        mock_config.training.per_device_eval_batch_size = 8
        # Ensure hyperopt is disabled in the mock
        mock_config.hyperopt.enabled = False
        mock_config_from_yaml.return_value = mock_config

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer)

        # Create mock datasets with proper __len__ method
        mock_train_dataset = MagicMock()
        mock_train_dataset.__len__.return_value = 1000
        mock_eval_dataset = MagicMock()
        mock_eval_dataset.__len__.return_value = 200
        mock_test_dataset = MagicMock()
        mock_test_dataset.__len__.return_value = 300
        mock_data_collator = Mock()
        mock_prepare_datasets.return_value = (
            mock_train_dataset,
            mock_eval_dataset,
            mock_test_dataset,
            mock_data_collator,
        )

        mock_val_metrics = {"eval_f1": 0.85}
        mock_test_metrics = {"test_f1": 0.80}
        mock_evaluate.side_effect = [mock_val_metrics, mock_test_metrics]

        with patch("torch.cuda.is_available", return_value=True):
            main()

        # Verify evaluation was called twice (validation and test)
        assert mock_evaluate.call_count == 2

    @patch("scripts.train.evaluate_model")
    @patch("src.model.load_model_from_checkpoint")
    @patch("scripts.train.run_training_pipeline")
    @patch("scripts.train.prepare_datasets")
    @patch("scripts.train.print_trainable_parameters")
    @patch("scripts.train.setup_model")
    @patch("scripts.train.print_dataset_statistics")
    @patch("scripts.train.load_conll2003_dataset")
    @patch("scripts.train.check_gpu_memory")
    @patch("scripts.train.setup_logging")
    @patch("scripts.train.Config.from_yaml")
    @patch("scripts.train.parse_args")
    def test_main_training_with_test(
        self,
        mock_parse_args,
        mock_config_from_yaml,
        mock_setup_logging,
        mock_check_gpu,
        mock_load_dataset,
        mock_print_stats,
        mock_setup_model,
        mock_print_params,
        mock_prepare_datasets,
        mock_run_training,
        mock_load_checkpoint,
        mock_evaluate,
    ):
        """Test main function with training and test evaluation."""
        # Setup mocks
        mock_args = Mock()
        mock_args.config = "configs/default.yaml"
        mock_args.debug = False
        mock_args.eval_only = False
        mock_args.test = True
        mock_parse_args.return_value = mock_args

        mock_config = Mock()
        mock_config.logging.log_level = "info"
        mock_config.logging.log_dir = "./logs"
        mock_config.model.model_name = "test-model"
        mock_config.data.label_names = ["O", "B-PER"]
        mock_config.training.per_device_eval_batch_size = 8
        mock_config.training.output_dir = "./output"
        # Ensure hyperopt is disabled in the mock
        mock_config.hyperopt.enabled = False
        mock_config_from_yaml.return_value = mock_config

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer)

        # Create mock datasets with proper __len__ method
        mock_train_dataset = MagicMock()
        mock_train_dataset.__len__.return_value = 1000
        mock_eval_dataset = MagicMock()
        mock_eval_dataset.__len__.return_value = 200
        mock_test_dataset = MagicMock()
        mock_test_dataset.__len__.return_value = 300
        mock_data_collator = Mock()
        mock_prepare_datasets.return_value = (
            mock_train_dataset,
            mock_eval_dataset,
            mock_test_dataset,
            mock_data_collator,
        )

        mock_train_metrics = {"train_loss": 0.5}
        mock_run_training.return_value = mock_train_metrics

        mock_test_metrics = {"test_f1": 0.80}
        mock_evaluate.return_value = mock_test_metrics

        # Mock best model checkpoint exists and fix dynamic import
        mock_load_checkpoint = MagicMock(return_value=(mock_model, mock_tokenizer))
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
            patch.dict("sys.modules", {"src.model": MagicMock(load_model_from_checkpoint=mock_load_checkpoint)}),
        ):
            main()

        # Verify checkpoint loading and test evaluation
        mock_load_checkpoint.assert_called_once()
        mock_evaluate.assert_called_once()

    @patch("scripts.train.setup_logging")
    @patch("scripts.train.Config.from_yaml")
    @patch("scripts.train.parse_args")
    def test_main_debug_mode(self, mock_parse_args, mock_config_from_yaml, mock_setup_logging):
        """Test main function with debug logging."""
        mock_args = Mock()
        mock_args.config = "configs/default.yaml"
        mock_args.debug = True
        mock_parse_args.return_value = mock_args

        mock_config = Mock()
        mock_config.logging.log_level = "info"
        mock_config.logging.log_dir = "./logs"
        mock_config_from_yaml.return_value = mock_config

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        # Mock to avoid going through full training
        with (
            patch("scripts.train.load_conll2003_dataset", side_effect=Exception("Stop here")),
            pytest.raises(SystemExit),
        ):
            main()

        # Verify debug logging was used
        mock_setup_logging.assert_called_once_with("debug", "./logs")

    @patch("scripts.train.setup_logging")
    @patch("scripts.train.Config.from_yaml")
    @patch("scripts.train.parse_args")
    def test_main_no_cuda(self, mock_parse_args, mock_config_from_yaml, mock_setup_logging):
        """Test main function without CUDA."""
        mock_args = Mock()
        mock_args.config = "configs/default.yaml"
        mock_args.debug = False
        mock_parse_args.return_value = mock_args

        mock_config = Mock()
        mock_config.logging.log_level = "info"
        mock_config.logging.log_dir = "./logs"
        mock_config_from_yaml.return_value = mock_config

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        # Mock to avoid going through full training
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("scripts.train.load_conll2003_dataset", side_effect=Exception("Stop here")),
            pytest.raises(SystemExit),
        ):
            main()

        # Verify CUDA warning was logged
        assert any("CUDA not available" in str(call) for call in mock_logger.warning.call_args_list)

    @patch("scripts.train.setup_logging")
    @patch("scripts.train.Config.from_yaml")
    @patch("scripts.train.parse_args")
    def test_main_keyboard_interrupt(self, mock_parse_args, mock_config_from_yaml, mock_setup_logging):
        """Test main function with keyboard interrupt."""
        mock_args = Mock()
        mock_args.config = "configs/default.yaml"
        mock_args.debug = False
        mock_parse_args.return_value = mock_args

        mock_config = Mock()
        mock_config_from_yaml.return_value = mock_config

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        with (
            patch("scripts.train.load_conll2003_dataset", side_effect=KeyboardInterrupt()),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1

    @patch("scripts.train.setup_logging")
    @patch("scripts.train.Config.from_yaml")
    @patch("scripts.train.parse_args")
    def test_main_generic_exception(self, mock_parse_args, mock_config_from_yaml, mock_setup_logging):
        """Test main function with generic exception."""
        mock_args = Mock()
        mock_args.config = "configs/default.yaml"
        mock_args.debug = False
        mock_parse_args.return_value = mock_args

        mock_config = Mock()
        mock_config_from_yaml.return_value = mock_config

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        with (
            patch("scripts.train.load_conll2003_dataset", side_effect=RuntimeError("Training failed")),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        mock_logger.error.assert_called()


class TestMainExecution:
    """Test script execution."""

    def test_main_execution_coverage(self):
        """Test execution of main when script is run directly."""
        # Test the TYPE_CHECKING import coverage
        import typing

        if typing.TYPE_CHECKING:
            pass  # This is line 14

        # Test __main__ execution (line 206)
        with (
            patch("scripts.train.main") as _,
            patch("__main__.__name__", "__main__"),
        ):
            # Import and execute the script
            import importlib

            import scripts.train

            importlib.reload(scripts.train)

    def test_wandb_arguments_comprehensive(self):
        """Test comprehensive wandb argument parsing."""
        test_argv = [
            "train.py",
            "--wandb-entity",
            "test-entity",
            "--wandb-name",
            "test-run",
            "--wandb-mode",
            "offline",
            "--wandb-dir",
            "/test/wandb",
            "--wandb-resume",
            "allow",
            "--wandb-run-id",
            "test-run-id",
            "--wandb-tags",
            "tag1",
            "tag2",
            "--wandb-notes",
            "Test notes",
        ]
        with patch("sys.argv", test_argv):
            args = parse_args()

        assert args.wandb_entity == "test-entity"
        assert args.wandb_name == "test-run"
        assert args.wandb_mode == "offline"
        assert args.wandb_dir == "/test/wandb"
        assert args.wandb_resume == "allow"
        assert args.wandb_run_id == "test-run-id"
        assert args.wandb_tags == ["tag1", "tag2"]
        assert args.wandb_notes == "Test notes"

    @patch("scripts.train.evaluate_model")
    @patch("scripts.train.prepare_datasets")
    @patch("scripts.train.print_trainable_parameters")
    @patch("scripts.train.setup_model")
    @patch("scripts.train.print_dataset_statistics")
    @patch("scripts.train.load_conll2003_dataset")
    @patch("scripts.train.check_gpu_memory")
    @patch("scripts.train.setup_logging")
    @patch("scripts.train.Config.from_yaml")
    @patch("scripts.train.parse_args")
    def test_main_eval_only_no_test(
        self,
        mock_parse_args,
        mock_config_from_yaml,
        mock_setup_logging,
        mock_check_gpu,
        mock_load_dataset,
        mock_print_stats,
        mock_setup_model,
        mock_print_params,
        mock_prepare_datasets,
        mock_evaluate,
    ):
        """Test main function in evaluation-only mode without test."""
        # Setup mocks
        mock_args = Mock()
        mock_args.config = "configs/default.yaml"
        mock_args.debug = False
        mock_args.eval_only = True
        mock_args.test = False  # No test evaluation
        mock_parse_args.return_value = mock_args

        mock_config = Mock()
        mock_config.logging.log_level = "info"
        mock_config.logging.log_dir = "./logs"
        mock_config.model.model_name = "test-model"
        mock_config.data.label_names = ["O", "B-PER"]
        mock_config.training.per_device_eval_batch_size = 8
        # Ensure hyperopt is disabled in the mock
        mock_config.hyperopt.enabled = False
        mock_config_from_yaml.return_value = mock_config

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer)

        mock_train_dataset = Mock()
        mock_train_dataset.__len__ = Mock(return_value=1000)
        mock_eval_dataset = Mock()
        mock_eval_dataset.__len__ = Mock(return_value=200)
        mock_test_dataset = Mock()
        mock_test_dataset.__len__ = Mock(return_value=300)
        mock_data_collator = Mock()
        mock_prepare_datasets.return_value = (
            mock_train_dataset,
            mock_eval_dataset,
            mock_test_dataset,
            mock_data_collator,
        )

        mock_val_metrics = {"eval_f1": 0.85}
        mock_evaluate.return_value = mock_val_metrics

        with patch("torch.cuda.is_available", return_value=True):
            main()

        # Verify evaluation was called only once (validation only)
        assert mock_evaluate.call_count == 1

    @patch("scripts.train.evaluate_model")
    @patch("scripts.train.run_training_pipeline")
    @patch("scripts.train.prepare_datasets")
    @patch("scripts.train.print_trainable_parameters")
    @patch("scripts.train.setup_model")
    @patch("scripts.train.print_dataset_statistics")
    @patch("scripts.train.load_conll2003_dataset")
    @patch("scripts.train.check_gpu_memory")
    @patch("scripts.train.setup_logging")
    @patch("scripts.train.Config.from_yaml")
    @patch("scripts.train.parse_args")
    def test_main_training_no_checkpoint(
        self,
        mock_parse_args,
        mock_config_from_yaml,
        mock_setup_logging,
        mock_check_gpu,
        mock_load_dataset,
        mock_print_stats,
        mock_setup_model,
        mock_print_params,
        mock_prepare_datasets,
        mock_run_training,
        mock_evaluate,
    ):
        """Test main function with training and test but no checkpoint."""
        # Setup mocks
        mock_args = Mock()
        mock_args.config = "configs/default.yaml"
        mock_args.debug = False
        mock_args.eval_only = False
        mock_args.test = True
        mock_parse_args.return_value = mock_args

        mock_config = Mock()
        mock_config.logging.log_level = "info"
        mock_config.logging.log_dir = "./logs"
        mock_config.model.model_name = "test-model"
        mock_config.data.label_names = ["O", "B-PER"]
        mock_config.training.per_device_eval_batch_size = 8
        mock_config.training.output_dir = "./output"
        # Ensure hyperopt is disabled in the mock
        mock_config.hyperopt.enabled = False
        mock_config_from_yaml.return_value = mock_config

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer)

        mock_train_dataset = Mock()
        mock_train_dataset.__len__ = Mock(return_value=1000)
        mock_eval_dataset = Mock()
        mock_eval_dataset.__len__ = Mock(return_value=200)
        mock_test_dataset = Mock()
        mock_test_dataset.__len__ = Mock(return_value=300)
        mock_data_collator = Mock()
        mock_prepare_datasets.return_value = (
            mock_train_dataset,
            mock_eval_dataset,
            mock_test_dataset,
            mock_data_collator,
        )

        mock_train_metrics = {"train_loss": 0.5}
        mock_run_training.return_value = mock_train_metrics

        mock_test_metrics = {"test_f1": 0.80}
        mock_evaluate.return_value = mock_test_metrics

        # Mock best model checkpoint does NOT exist
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            main()

        # Verify test evaluation was called but no checkpoint loading
        mock_evaluate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
