"""Tests for hyperparameter optimization utilities."""

from unittest.mock import patch

import pytest

from src.hyperopt.config import HyperoptConfig
from src.hyperopt.utils import (
    create_ray_tune_search_space,
    format_search_space_summary,
    validate_hyperopt_config,
)


class TestCreateRayTuneSearchSpace:
    """Test search space conversion to Ray Tune format."""

    def test_uniform_parameter(self):
        """Test uniform parameter conversion."""
        config = HyperoptConfig(search_space={"lr": {"type": "uniform", "low": 0.1, "high": 1.0}})

        search_space = create_ray_tune_search_space(config)

        assert "lr" in search_space
        # Can't directly test tune.uniform object, but verify it exists
        assert search_space["lr"] is not None

    def test_loguniform_parameter(self):
        """Test loguniform parameter conversion."""
        config = HyperoptConfig(search_space={"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}})

        search_space = create_ray_tune_search_space(config)

        assert "lr" in search_space
        assert search_space["lr"] is not None

    def test_choice_parameter(self):
        """Test choice parameter conversion."""
        config = HyperoptConfig(search_space={"batch_size": {"type": "choice", "choices": [16, 32, 64]}})

        search_space = create_ray_tune_search_space(config)

        assert "batch_size" in search_space
        assert search_space["batch_size"] is not None

    def test_int_parameter(self):
        """Test int parameter conversion."""
        config = HyperoptConfig(search_space={"epochs": {"type": "int", "low": 1, "high": 10}})

        search_space = create_ray_tune_search_space(config)

        assert "epochs" in search_space
        assert search_space["epochs"] is not None

    def test_logint_parameter_warning(self):
        """Test logint parameter conversion with warning."""
        config = HyperoptConfig(search_space={"param": {"type": "logint", "low": 1, "high": 100}})

        with patch("src.hyperopt.utils.logger") as mock_logger:
            search_space = create_ray_tune_search_space(config)

            assert "param" in search_space
            mock_logger.warning.assert_called_once()

    def test_unknown_parameter_type(self):
        """Test unknown parameter type raises error."""
        config = HyperoptConfig(search_space={"param": {"type": "unknown", "low": 1, "high": 10}})

        with pytest.raises(ValueError, match="Unknown parameter type"):
            create_ray_tune_search_space(config)

    def test_multiple_parameters(self):
        """Test multiple parameters conversion."""
        config = HyperoptConfig(
            search_space={
                "lr": {"type": "uniform", "low": 0.1, "high": 1.0},
                "batch_size": {"type": "choice", "choices": [16, 32]},
                "epochs": {"type": "int", "low": 1, "high": 5},
            }
        )

        search_space = create_ray_tune_search_space(config)

        assert len(search_space) == 3
        assert all(param in search_space for param in ["lr", "batch_size", "epochs"])


class TestValidateHyperoptConfig:
    """Test hyperopt configuration validation."""

    def test_valid_config(self):
        """Test valid configuration passes validation."""
        config = HyperoptConfig()
        # Should not raise any exception
        validate_hyperopt_config(config)

    def test_empty_search_space(self):
        """Test empty search space raises error."""
        config = HyperoptConfig(search_space={})

        with pytest.raises(ValueError, match="search_space cannot be empty"):
            validate_hyperopt_config(config)

    def test_parameter_missing_type(self):
        """Test parameter missing type field."""
        config = HyperoptConfig(
            search_space={
                "param": {"low": 1, "high": 10}  # Missing 'type'
            }
        )

        with pytest.raises(ValueError, match="missing 'type' field"):
            validate_hyperopt_config(config)

    def test_uniform_parameter_missing_bounds(self):
        """Test uniform parameter missing bounds."""
        config = HyperoptConfig(
            search_space={
                "param": {"type": "uniform", "low": 1}  # Missing 'high'
            }
        )

        with pytest.raises(ValueError, match="missing 'low' or 'high' field"):
            validate_hyperopt_config(config)

    def test_uniform_parameter_invalid_bounds(self):
        """Test uniform parameter with invalid bounds."""
        config = HyperoptConfig(
            search_space={
                "param": {"type": "uniform", "low": 10, "high": 1}  # low > high
            }
        )

        with pytest.raises(ValueError, match="low must be < high"):
            validate_hyperopt_config(config)

    def test_choice_parameter_missing_choices(self):
        """Test choice parameter missing choices."""
        config = HyperoptConfig(
            search_space={
                "param": {"type": "choice"}  # Missing 'choices'
            }
        )

        with pytest.raises(ValueError, match="missing or empty 'choices' field"):
            validate_hyperopt_config(config)

    def test_choice_parameter_empty_choices(self):
        """Test choice parameter with empty choices."""
        config = HyperoptConfig(
            search_space={
                "param": {"type": "choice", "choices": []}  # Empty choices
            }
        )

        with pytest.raises(ValueError, match="missing or empty 'choices' field"):
            validate_hyperopt_config(config)

    def test_unknown_parameter_type_validation(self):
        """Test unknown parameter type in validation."""
        config = HyperoptConfig(search_space={"param": {"type": "invalid", "low": 1, "high": 10}})

        with pytest.raises(ValueError, match="Unknown parameter type"):
            validate_hyperopt_config(config)

    def test_combined_strategy_validation(self):
        """Test combined strategy validation."""
        # Valid combined strategy
        config = HyperoptConfig(strategy="combined", optuna_enabled=True, asha_enabled=True)
        validate_hyperopt_config(config)

        # Invalid combined strategy - should fail at config creation
        with pytest.raises(ValueError, match="Combined strategy requires both"):
            HyperoptConfig(strategy="combined", optuna_enabled=False, asha_enabled=True)

    def test_max_concurrent_validation(self):
        """Test max_concurrent validation."""
        config = HyperoptConfig(max_concurrent=0)

        with pytest.raises(ValueError, match="max_concurrent must be > 0"):
            validate_hyperopt_config(config)

    def test_num_trials_validation(self):
        """Test num_trials validation."""
        config = HyperoptConfig(num_trials=0)

        with pytest.raises(ValueError, match="num_trials must be > 0"):
            validate_hyperopt_config(config)


class TestFormatSearchSpaceSummary:
    """Test search space formatting."""

    def test_format_uniform_parameter(self):
        """Test formatting uniform parameter."""
        search_space = {"lr": {"type": "uniform", "low": 0.1, "high": 1.0}}

        summary = format_search_space_summary(search_space)

        assert "Search Space:" in summary
        assert "lr: uniform(0.1, 1.0)" in summary

    def test_format_loguniform_parameter(self):
        """Test formatting loguniform parameter."""
        search_space = {"lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3}}

        summary = format_search_space_summary(search_space)

        assert "lr: loguniform(1e-05, 0.001)" in summary

    def test_format_choice_parameter(self):
        """Test formatting choice parameter."""
        search_space = {"batch_size": {"type": "choice", "choices": [16, 32, 64]}}

        summary = format_search_space_summary(search_space)

        assert "batch_size: choice([16, 32, 64])" in summary

    def test_format_multiple_parameters(self):
        """Test formatting multiple parameters."""
        search_space = {
            "lr": {"type": "uniform", "low": 0.1, "high": 1.0},
            "batch_size": {"type": "choice", "choices": [16, 32]},
        }

        summary = format_search_space_summary(search_space)

        assert "lr: uniform(0.1, 1.0)" in summary
        assert "batch_size: choice([16, 32])" in summary
        assert summary.count("\n") >= 2  # Multiple lines

    def test_format_unknown_parameter(self):
        """Test formatting unknown parameter type."""
        search_space = {"param": {"type": "unknown", "value": "test"}}

        summary = format_search_space_summary(search_space)

        assert "param:" in summary
        # Should include the raw parameter config
        assert "unknown" in summary
