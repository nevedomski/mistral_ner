"""Tests for hyperparameter optimizer."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import ray.tune as tune

from src.hyperopt.config import HyperoptConfig
from src.hyperopt.optimizer import HyperparameterOptimizer


class TestHyperparameterOptimizer:
    """Test HyperparameterOptimizer class."""

    def test_initialization(self):
        """Test optimizer initialization."""
        config = HyperoptConfig()
        optimizer = HyperparameterOptimizer(config)

        assert optimizer.config == config

    def test_initialization_with_invalid_config(self):
        """Test initialization with invalid config raises error."""
        config = HyperoptConfig(search_space={})  # Empty search space

        with pytest.raises(ValueError, match="search_space cannot be empty"):
            HyperparameterOptimizer(config)

    @patch("src.hyperopt.optimizer.ray")
    def test_initialize_ray_local(self, mock_ray):
        """Test Ray initialization for local cluster."""
        config = HyperoptConfig(ray_address=None)
        optimizer = HyperparameterOptimizer(config)

        mock_ray.is_initialized.return_value = False
        optimizer._initialize_ray()

        mock_ray.init.assert_called_once_with(ignore_reinit_error=True)

    @patch("src.hyperopt.optimizer.ray")
    def test_initialize_ray_remote(self, mock_ray):
        """Test Ray initialization for remote cluster."""
        config = HyperoptConfig(ray_address="ray://localhost:10001")
        optimizer = HyperparameterOptimizer(config)

        mock_ray.is_initialized.return_value = False
        optimizer._initialize_ray()

        mock_ray.init.assert_called_once_with(address="ray://localhost:10001")

    @patch("src.hyperopt.optimizer.ray")
    def test_initialize_ray_already_initialized(self, mock_ray):
        """Test Ray initialization when already initialized."""
        config = HyperoptConfig()
        optimizer = HyperparameterOptimizer(config)

        mock_ray.is_initialized.return_value = True
        optimizer._initialize_ray()

        mock_ray.init.assert_not_called()

    def test_create_optuna_search_tpe(self):
        """Test Optuna search creation with TPE sampler."""
        config = HyperoptConfig(optuna_sampler="TPE")
        optimizer = HyperparameterOptimizer(config)

        search = optimizer._create_optuna_search()

        # Verify it's an OptunaSearch instance
        from ray.tune.search.optuna import OptunaSearch

        assert isinstance(search, OptunaSearch)

    def test_create_optuna_search_cma(self):
        """Test Optuna search creation with CMA sampler."""
        config = HyperoptConfig(optuna_sampler="CMA")
        optimizer = HyperparameterOptimizer(config)

        search = optimizer._create_optuna_search()

        from ray.tune.search.optuna import OptunaSearch

        assert isinstance(search, OptunaSearch)

    def test_create_asha_scheduler(self):
        """Test ASHA scheduler creation."""
        config = HyperoptConfig(asha_max_t=10, asha_grace_period=2, asha_reduction_factor=3)
        optimizer = HyperparameterOptimizer(config)

        scheduler = optimizer._create_asha_scheduler()

        from ray.tune.schedulers import ASHAScheduler

        assert isinstance(scheduler, ASHAScheduler)

    @patch("src.hyperopt.optimizer.tune.Tuner")
    @patch("src.hyperopt.optimizer.ray")
    def test_create_combined_tuner(self, mock_ray, mock_tuner_class):
        """Test combined tuner creation."""
        config = HyperoptConfig(strategy="combined")
        optimizer = HyperparameterOptimizer(config)

        mock_ray.is_initialized.return_value = True
        mock_tuner = Mock()
        mock_tuner_class.return_value = mock_tuner

        objective_func = Mock()
        search_space = {"lr": tune.uniform(0.1, 1.0)}

        tuner = optimizer._create_combined_tuner(objective_func, search_space)

        mock_tuner_class.assert_called_once()
        assert tuner == mock_tuner

    @patch("src.hyperopt.optimizer.tune.Tuner")
    @patch("src.hyperopt.optimizer.ray")
    def test_create_optuna_tuner(self, mock_ray, mock_tuner_class):
        """Test Optuna-only tuner creation."""
        config = HyperoptConfig(strategy="optuna")
        optimizer = HyperparameterOptimizer(config)

        mock_ray.is_initialized.return_value = True
        mock_tuner = Mock()
        mock_tuner_class.return_value = mock_tuner

        objective_func = Mock()
        search_space = {"lr": tune.uniform(0.1, 1.0)}

        tuner = optimizer._create_optuna_tuner(objective_func, search_space)

        mock_tuner_class.assert_called_once()
        assert tuner == mock_tuner

    @patch("src.hyperopt.optimizer.tune.Tuner")
    @patch("src.hyperopt.optimizer.ray")
    def test_create_asha_tuner(self, mock_ray, mock_tuner_class):
        """Test ASHA-only tuner creation."""
        config = HyperoptConfig(strategy="asha")
        optimizer = HyperparameterOptimizer(config)

        mock_ray.is_initialized.return_value = True
        mock_tuner = Mock()
        mock_tuner_class.return_value = mock_tuner

        objective_func = Mock()
        search_space = {"lr": tune.uniform(0.1, 1.0)}

        tuner = optimizer._create_asha_tuner(objective_func, search_space)

        mock_tuner_class.assert_called_once()
        assert tuner == mock_tuner

    @patch("src.hyperopt.optimizer.tune.Tuner")
    @patch("src.hyperopt.optimizer.ray")
    def test_create_random_tuner(self, mock_ray, mock_tuner_class):
        """Test random search tuner creation."""
        config = HyperoptConfig(strategy="random")
        optimizer = HyperparameterOptimizer(config)

        mock_ray.is_initialized.return_value = True
        mock_tuner = Mock()
        mock_tuner_class.return_value = mock_tuner

        objective_func = Mock()
        search_space = {"lr": tune.uniform(0.1, 1.0)}

        tuner = optimizer._create_random_tuner(objective_func, search_space)

        mock_tuner_class.assert_called_once()
        assert tuner == mock_tuner

    def test_unknown_strategy_raises_error(self):
        """Test unknown strategy raises error."""
        # Should raise error during config creation
        with pytest.raises(ValueError, match="Invalid strategy: unknown"):
            HyperoptConfig(strategy="unknown")

    @patch("src.hyperopt.optimizer.ray")
    @patch("src.hyperopt.optimizer.tune.Tuner")
    def test_optimize_combined(self, mock_tuner_class, mock_ray):
        """Test optimize method with combined strategy."""
        config = HyperoptConfig(strategy="combined", num_trials=5)
        optimizer = HyperparameterOptimizer(config)

        # Mock Ray
        mock_ray.is_initialized.return_value = True

        # Mock tuner and results
        mock_results = Mock()
        mock_tuner = Mock()
        mock_tuner.fit.return_value = mock_results
        mock_tuner_class.return_value = mock_tuner

        objective_func = Mock()
        search_space = {"lr": tune.uniform(0.1, 1.0)}
        base_config = Mock()

        results = optimizer.optimize(objective_func, search_space, base_config)

        assert results == mock_results
        mock_tuner.fit.assert_called_once()

    @patch("src.hyperopt.optimizer.Path")
    def test_setup_logging_creates_directory(self, mock_path_class):
        """Test logging setup creates results directory."""
        config = HyperoptConfig(results_dir="./test_results", log_to_file=False)

        # Create mock path instance
        mock_path_instance = Mock()
        mock_path_class.return_value = mock_path_instance

        # Create optimizer which calls _setup_logging in __init__
        HyperparameterOptimizer(config)

        # Verify Path was called with the results directory
        mock_path_class.assert_called_with("./test_results")
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_log_results_summary_with_results(self):
        """Test results summary logging with valid results."""
        config = HyperoptConfig()
        optimizer = HyperparameterOptimizer(config)

        # Mock results
        mock_results = MagicMock()
        mock_best_result = Mock()
        mock_best_result.metrics = {"eval_f1": 0.85}
        mock_best_result.config = {"lr": 0.001, "batch_size": 32}
        mock_results.get_best_result.return_value = mock_best_result
        mock_results.__len__.return_value = 10

        # Mock dataframe
        mock_df = Mock()
        mock_df.empty = False
        mock_df.nlargest.return_value.iterrows.return_value = [(0, {"eval_f1": 0.85}), (1, {"eval_f1": 0.83})]
        mock_results.get_dataframe.return_value = mock_df

        with patch("src.hyperopt.optimizer.logger") as mock_logger:
            optimizer._log_results_summary(mock_results)

            # Verify logging calls
            assert mock_logger.info.call_count >= 3

    def test_log_results_summary_with_error(self):
        """Test results summary logging handles errors gracefully."""
        config = HyperoptConfig()
        optimizer = HyperparameterOptimizer(config)

        # Mock results that raise exception
        mock_results = Mock()
        mock_results.get_best_result.side_effect = Exception("Test error")

        with patch("src.hyperopt.optimizer.logger") as mock_logger:
            optimizer._log_results_summary(mock_results)

            mock_logger.error.assert_called_once()

    def test_save_best_config(self):
        """Test saving best configuration to file."""
        config = HyperoptConfig()
        optimizer = HyperparameterOptimizer(config)

        # Mock results
        mock_results = Mock()
        mock_best_result = Mock()
        mock_best_result.metrics = {"eval_f1": 0.85}
        mock_best_result.config = {"lr": 0.001}
        mock_results.get_best_result.return_value = mock_best_result

        with patch("builtins.open", create=True) as mock_open, patch("yaml.dump") as mock_yaml_dump:
            optimizer.save_best_config(mock_results, "best_config.yaml")

            mock_open.assert_called_once_with("best_config.yaml", "w")
            mock_yaml_dump.assert_called_once()

    def test_context_manager(self):
        """Test context manager functionality."""
        config = HyperoptConfig()

        with patch("src.hyperopt.optimizer.ray") as mock_ray:
            mock_ray.is_initialized.return_value = True

            with HyperparameterOptimizer(config) as optimizer:
                assert isinstance(optimizer, HyperparameterOptimizer)

            mock_ray.shutdown.assert_called_once()

    def test_context_manager_no_ray_shutdown_when_not_initialized(self):
        """Test context manager doesn't shutdown Ray when not initialized."""
        config = HyperoptConfig()

        with patch("src.hyperopt.optimizer.ray") as mock_ray:
            mock_ray.is_initialized.return_value = False

            with HyperparameterOptimizer(config) as optimizer:
                assert isinstance(optimizer, HyperparameterOptimizer)

            mock_ray.shutdown.assert_not_called()
