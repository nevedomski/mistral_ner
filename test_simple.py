"""Simple working test to achieve 90% coverage."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from scripts.train import main


class MockDataset:
    """Mock dataset with __len__ method."""

    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class TestScriptsTrainSimple:
    """Simple working tests for main function."""

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
    def test_main_training_simple(
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
        """Test main function training mode."""
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
        mock_config_from_yaml.return_value = mock_config

        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger

        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer)

        mock_print_params.return_value = {"trainable_params": 1000}

        # Create datasets with __len__
        mock_train_dataset = MockDataset(1000)
        mock_eval_dataset = MockDataset(200)
        mock_test_dataset = MockDataset(300)
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

        # Verify key calls
        mock_run_training.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
