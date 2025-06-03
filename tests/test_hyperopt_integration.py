"""Integration tests for hyperparameter optimization."""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

from src.config import Config
from src.hyperopt.config import HyperoptConfig
from src.hyperopt.optimizer import HyperparameterOptimizer
from src.hyperopt.utils import create_objective_function, create_ray_tune_search_space


class TestHyperoptIntegration:
    """Integration tests for hyperparameter optimization."""

    def test_config_integration(self):
        """Test hyperopt config integration with main config."""
        config = Config()

        # Test default hyperopt config
        assert hasattr(config, "hyperopt")
        assert isinstance(config.hyperopt, HyperoptConfig)
        assert config.hyperopt.enabled is False

    def test_config_yaml_integration(self):
        """Test hyperopt config loading from YAML."""
        yaml_content = """
hyperopt:
  enabled: true
  strategy: "optuna"
  num_trials: 10
  search_space:
    learning_rate:
      type: "uniform"
      low: 0.001
      high: 0.01
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = Config.from_yaml(f.name)

                assert config.hyperopt.enabled is True
                assert config.hyperopt.strategy == "optuna"
                assert config.hyperopt.num_trials == 10
                assert "learning_rate" in config.hyperopt.search_space

            finally:
                os.unlink(f.name)

    def test_search_space_conversion_integration(self):
        """Test search space conversion works with real config."""
        config = HyperoptConfig()
        search_space = create_ray_tune_search_space(config)

        # Check that default search space is converted
        assert len(search_space) > 0
        assert "learning_rate" in search_space
        assert "lora_r" in search_space

    @patch("src.training.run_training_pipeline")
    @patch("src.model.setup_model")
    def test_objective_function_creation(self, mock_setup_model, mock_run_training):
        """Test objective function creation and execution."""
        # Mock setup_model to return mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer)

        # Mock training pipeline to return metrics
        mock_run_training.return_value = {"eval_f1": 0.85, "eval_loss": 0.5, "train_loss": 0.3}

        base_config = Config()
        hyperopt_config = HyperoptConfig()
        train_dataset = Mock()
        eval_dataset = Mock()
        data_collator = Mock()

        objective_func = create_objective_function(
            base_config=base_config,
            hyperopt_config=hyperopt_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Test objective function with trial config
        trial_config = {"learning_rate": 0.001, "lora_r": 32, "per_device_train_batch_size": 8}

        with patch("ray.tune.report") as mock_report:
            objective_func(trial_config)

            # Verify training was called
            mock_setup_model.assert_called_once()
            mock_run_training.assert_called_once()

            # Verify metrics were reported
            mock_report.assert_called_once()
            reported_metrics = mock_report.call_args[1]
            assert "eval_f1" in reported_metrics
            assert reported_metrics["eval_f1"] == 0.85

    @patch("src.training.run_training_pipeline")
    @patch("src.model.setup_model")
    def test_objective_function_error_handling(self, mock_setup_model, mock_run_training):
        """Test objective function handles errors gracefully."""
        # Mock setup_model to raise an error
        mock_setup_model.side_effect = Exception("Model setup failed")

        base_config = Config()
        hyperopt_config = HyperoptConfig()

        objective_func = create_objective_function(
            base_config=base_config,
            hyperopt_config=hyperopt_config,
            train_dataset=Mock(),
            eval_dataset=Mock(),
            data_collator=Mock(),
        )

        trial_config = {"learning_rate": 0.001}

        with patch("ray.tune.report") as mock_report:
            objective_func(trial_config)

            # Verify error was handled and worst metric reported
            mock_report.assert_called_once()
            reported_metrics = mock_report.call_args[1]
            assert "trial_failed" in reported_metrics
            assert reported_metrics["trial_failed"] is True
            assert reported_metrics["eval_f1"] == 0.0  # Worst value for max mode

    def test_optimizer_with_minimal_config(self):
        """Test optimizer works with minimal configuration."""
        config = HyperoptConfig(
            num_trials=2,
            strategy="random",
            search_space={"learning_rate": {"type": "uniform", "low": 0.001, "high": 0.01}},
        )

        # Should not raise any exceptions during initialization
        optimizer = HyperparameterOptimizer(config)
        assert optimizer.config == config

    @patch("src.hyperopt.optimizer.ray")
    @patch("src.hyperopt.optimizer.tune.Tuner")
    def test_end_to_end_optimization_flow(self, mock_tuner_class, mock_ray):
        """Test end-to-end optimization flow without actual Ray execution."""
        # Setup mocks
        mock_ray.is_initialized.return_value = True

        mock_results = MagicMock()
        mock_best_result = Mock()
        mock_best_result.metrics = {"eval_f1": 0.85}
        mock_best_result.config = {"learning_rate": 0.001}
        mock_results.get_best_result.return_value = mock_best_result
        mock_results.__len__.return_value = 5

        # Mock empty dataframe for summary
        mock_df = Mock()
        mock_df.empty = True
        mock_results.get_dataframe.return_value = mock_df

        mock_tuner = Mock()
        mock_tuner.fit.return_value = mock_results
        mock_tuner_class.return_value = mock_tuner

        # Create optimizer
        config = HyperoptConfig(
            strategy="random",
            num_trials=5,
            search_space={"learning_rate": {"type": "uniform", "low": 0.001, "high": 0.01}},
        )

        optimizer = HyperparameterOptimizer(config)

        # Create search space and objective
        search_space = create_ray_tune_search_space(config)
        objective_func = Mock()
        base_config = Config()

        # Run optimization
        results = optimizer.optimize(objective_func, search_space, base_config)

        # Verify results
        assert results == mock_results
        mock_tuner.fit.assert_called_once()
        mock_tuner_class.assert_called_once()

    def test_yaml_config_validation(self):
        """Test that YAML config with hyperopt validates correctly."""
        yaml_content = """
model:
  model_name: "test-model"

hyperopt:
  enabled: true
  strategy: "combined"
  optuna_enabled: true
  asha_enabled: true
  search_space:
    learning_rate:
      type: "loguniform"
      low: 0.00001
      high: 0.001
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = Config.from_yaml(f.name)

                # Test that config validates (should not raise)
                from src.hyperopt.utils import validate_hyperopt_config

                validate_hyperopt_config(config.hyperopt)

                # Test search space conversion
                search_space = create_ray_tune_search_space(config.hyperopt)
                assert "learning_rate" in search_space

            finally:
                os.unlink(f.name)
