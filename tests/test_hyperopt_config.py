"""Tests for hyperparameter optimization configuration."""

import pytest

from src.hyperopt.config import HyperoptConfig


class TestHyperoptConfig:
    """Test HyperoptConfig class."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = HyperoptConfig()

        assert config.enabled is False
        assert config.strategy == "combined"
        assert config.num_trials == 50
        assert config.max_concurrent == 4
        assert config.optuna_enabled is True
        assert config.asha_enabled is True
        assert config.metric == "eval_f1"
        assert config.mode == "max"
        assert len(config.search_space) > 0

    def test_strategy_validation(self):
        """Test strategy validation."""
        # Valid strategies
        for strategy in ["optuna", "asha", "combined", "random"]:
            config = HyperoptConfig(strategy=strategy)
            assert config.strategy == strategy

        # Invalid strategy should raise error
        with pytest.raises(ValueError, match="Invalid strategy"):
            HyperoptConfig(strategy="invalid")

    def test_combined_strategy_validation(self):
        """Test combined strategy requires both components enabled."""
        # Should work with both enabled
        config = HyperoptConfig(strategy="combined", optuna_enabled=True, asha_enabled=True)
        assert config.strategy == "combined"

        # Should fail with optuna disabled
        with pytest.raises(ValueError, match="Combined strategy requires"):
            HyperoptConfig(strategy="combined", optuna_enabled=False, asha_enabled=True)

        # Should fail with asha disabled
        with pytest.raises(ValueError, match="Combined strategy requires"):
            HyperoptConfig(strategy="combined", optuna_enabled=True, asha_enabled=False)

    def test_mode_validation(self):
        """Test mode validation."""
        # Valid modes
        for mode in ["max", "min"]:
            config = HyperoptConfig(mode=mode)
            assert config.mode == mode

        # Invalid mode should raise error
        with pytest.raises(ValueError, match="Invalid mode"):
            HyperoptConfig(mode="invalid")

    def test_optuna_sampler_validation(self):
        """Test Optuna sampler validation."""
        # Valid samplers
        for sampler in ["TPE", "CMA", "Random", "GPSampler"]:
            config = HyperoptConfig(optuna_sampler=sampler)
            assert config.optuna_sampler == sampler

        # Invalid sampler should raise error
        with pytest.raises(ValueError, match="Invalid optuna_sampler"):
            HyperoptConfig(optuna_sampler="invalid")

    def test_optuna_pruner_validation(self):
        """Test Optuna pruner validation."""
        # Valid pruners
        for pruner in ["median", "hyperband", "none"]:
            config = HyperoptConfig(optuna_pruner=pruner)
            assert config.optuna_pruner == pruner

        # Invalid pruner should raise error
        with pytest.raises(ValueError, match="Invalid optuna_pruner"):
            HyperoptConfig(optuna_pruner="invalid")

    def test_search_space_structure(self):
        """Test default search space structure."""
        config = HyperoptConfig()

        # Check required parameters exist
        assert "learning_rate" in config.search_space
        assert "lora_r" in config.search_space
        assert "per_device_train_batch_size" in config.search_space

        # Check parameter types
        lr_config = config.search_space["learning_rate"]
        assert lr_config["type"] == "loguniform"
        assert "low" in lr_config
        assert "high" in lr_config

        lora_config = config.search_space["lora_r"]
        assert lora_config["type"] == "choice"
        assert "choices" in lora_config
        assert isinstance(lora_config["choices"], list)

    def test_custom_search_space(self):
        """Test custom search space configuration."""
        custom_space = {
            "learning_rate": {"type": "uniform", "low": 0.001, "high": 0.01},
            "batch_size": {"type": "choice", "choices": [16, 32]},
        }

        config = HyperoptConfig(search_space=custom_space)
        assert config.search_space == custom_space

    def test_resources_per_trial_default(self):
        """Test default resources per trial."""
        config = HyperoptConfig()

        assert "cpu" in config.resources_per_trial
        assert "gpu" in config.resources_per_trial
        assert config.resources_per_trial["cpu"] == 1.0
        assert config.resources_per_trial["gpu"] == 1.0

    def test_timeout_optional(self):
        """Test timeout is optional."""
        config = HyperoptConfig()
        assert config.timeout is None

        config_with_timeout = HyperoptConfig(timeout=3600)
        assert config_with_timeout.timeout == 3600
