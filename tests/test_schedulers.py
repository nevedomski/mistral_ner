"""Tests for learning rate schedulers module."""

import pytest
import torch
import torch.nn as nn
from transformers import TrainingArguments

from src.schedulers import (
    LayerwiseLearningRateScheduler,
    create_advanced_scheduler_config,
    create_layerwise_optimizer_groups,
    create_scheduler,
    get_scheduler_config,
)


class TestCreateScheduler:
    """Test scheduler creation factory."""

    @pytest.fixture
    def optimizer(self):
        """Create a dummy optimizer."""
        model = nn.Linear(10, 5)
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    @pytest.fixture
    def training_args(self):
        """Create dummy training arguments."""
        args = TrainingArguments(
            output_dir="./test",
            warmup_ratio=0.1,
        )
        # Manually set num_training_steps as it's not a constructor parameter
        args.num_training_steps = 1000
        return args

    def test_create_linear_scheduler(self, optimizer, training_args):
        """Test creating linear scheduler."""
        scheduler = create_scheduler(optimizer, "linear", training_args, num_training_steps=1000)
        assert scheduler is not None
        assert hasattr(scheduler, "step")
        assert hasattr(scheduler, "get_last_lr")

    def test_create_cosine_scheduler(self, optimizer, training_args):
        """Test creating cosine scheduler."""
        scheduler = create_scheduler(optimizer, "cosine", training_args, num_training_steps=1000)
        assert scheduler is not None
        # Test scheduler behavior
        initial_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        assert scheduler.get_last_lr()[0] != initial_lr

    def test_create_cosine_with_restarts(self, optimizer, training_args):
        """Test creating cosine with restarts scheduler."""
        scheduler = create_scheduler(
            optimizer,
            "cosine_with_restarts",
            training_args,
            num_training_steps=1000,
            num_cycles=3,
        )
        assert scheduler is not None

    def test_create_polynomial_scheduler(self, optimizer, training_args):
        """Test creating polynomial scheduler."""
        scheduler = create_scheduler(optimizer, "polynomial", training_args, num_training_steps=1000, power=2.0)
        assert scheduler is not None

    def test_create_linear_with_no_warmup(self, optimizer):
        """Test creating linear scheduler with no warmup."""
        training_args = TrainingArguments(
            output_dir="./test",
            warmup_ratio=0.0,
        )
        training_args.num_training_steps = 1000

        scheduler = create_scheduler(optimizer, "linear", training_args, num_training_steps=1000)
        assert scheduler is not None

    def test_create_scheduler_edge_cases(self, optimizer, training_args):
        """Test edge cases for scheduler creation."""
        # Test with different num_cycles for cosine
        scheduler = create_scheduler(optimizer, "cosine", training_args, num_training_steps=1000, num_cycles=2.0)
        assert scheduler is not None

    def test_create_scheduler_with_warmup_steps(self, optimizer, training_args):
        """Test creating scheduler with warmup steps override."""
        training_args.warmup_steps = 100
        training_args.warmup_ratio = None

        scheduler = create_scheduler(optimizer, "linear", training_args, num_training_steps=1000)
        assert scheduler is not None

    def test_create_unknown_scheduler(self, optimizer, training_args):
        """Test creating unknown scheduler raises error."""
        with pytest.raises(ValueError, match="Unsupported scheduler type"):
            create_scheduler(optimizer, "unknown_type", training_args, num_training_steps=1000)


class TestLayerwiseLearningRateScheduler:
    """Test LayerwiseLearningRateScheduler implementation."""

    def test_layerwise_scheduler_init(self):
        """Test LayerwiseLearningRateScheduler initialization."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 30), nn.Linear(30, 5))

        scheduler = LayerwiseLearningRateScheduler(
            model=model,
            base_lr=1e-3,
            decay_factor=0.9,
            num_layers=3,
        )

        assert scheduler.base_lr == 1e-3
        assert scheduler.decay_factor == 0.9
        assert scheduler.num_layers == 3

    def test_layerwise_scheduler_auto_layers(self):
        """Test LayerwiseLearningRateScheduler with auto layer detection."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 10))

        scheduler = LayerwiseLearningRateScheduler(
            model=model,
            base_lr=1e-3,
            decay_factor=0.9,
        )

        # Should auto-detect layers
        assert scheduler.num_layers > 0

    def test_layerwise_scheduler_get_layer_lrs(self):
        """Test LayerwiseLearningRateScheduler get_layer_learning_rates method."""
        model = nn.Linear(10, 5)

        scheduler = LayerwiseLearningRateScheduler(
            model=model,
            base_lr=1e-3,
            decay_factor=0.9,
            num_layers=1,
        )

        lrs = scheduler.get_layer_learning_rates()
        assert isinstance(lrs, dict)
        # Should have learning rates for model parameters
        assert len(lrs) > 0

    def test_layerwise_scheduler_layer_assignment(self):
        """Test layer assignment for parameters."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))

        scheduler = LayerwiseLearningRateScheduler(
            model=model,
            base_lr=1e-3,
            decay_factor=0.8,
            num_layers=2,
        )

        # Get layer assignment
        for name, _ in model.named_parameters():
            layer_idx = scheduler._get_layer_index(name)
            assert 0 <= layer_idx < scheduler.num_layers


class TestSchedulerConfig:
    """Test scheduler configuration functions."""

    def test_get_scheduler_config_cosine(self):
        """Test getting cosine scheduler config."""
        config = get_scheduler_config("cosine_aggressive")
        assert isinstance(config, dict)
        assert config["type"] == "cosine"

    def test_create_advanced_scheduler_config(self):
        """Test creating advanced scheduler config."""
        config = create_advanced_scheduler_config(
            scheduler_type="cosine",
            num_cycles=2.0,
        )
        assert config["type"] == "cosine"
        assert config["num_cycles"] == 2.0

    def test_create_advanced_scheduler_config_with_kwargs(self):
        """Test creating advanced scheduler config with extra kwargs."""
        config = create_advanced_scheduler_config(
            scheduler_type="polynomial",
            power=2.0,
            extra_param="value",
        )
        assert config["type"] == "polynomial"
        assert config["power"] == 2.0
        assert config["extra_param"] == "value"


class TestCreateLayerwiseOptimizerGroups:
    """Test layerwise optimizer group creation."""

    def test_create_layerwise_groups_basic(self):
        """Test creating layerwise optimizer groups."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 30), nn.Linear(30, 5))

        groups = create_layerwise_optimizer_groups(
            model,
            base_lr=1e-3,
            weight_decay=0.01,
            decay_factor=0.9,
        )

        assert len(groups) >= 1  # At least one group created
        # Check that groups have required keys
        for group in groups:
            assert "params" in group
            assert "lr" in group
            assert "weight_decay" in group

    def test_create_layerwise_groups_with_multiple_layers(self):
        """Test creating layerwise groups with multiple layers."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        groups = create_layerwise_optimizer_groups(
            model,
            base_lr=1e-3,
            weight_decay=0.01,
            decay_factor=0.95,
        )

        # Should have groups for parameters
        assert len(groups) >= 1


class TestSchedulerIntegration:
    """Test scheduler integration scenarios."""

    def test_scheduler_with_training_loop(self):
        """Test scheduler in a mini training loop."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        training_args = TrainingArguments(
            output_dir="./test",
            warmup_ratio=0.1,
        )

        scheduler = create_scheduler(optimizer, "cosine", training_args, num_training_steps=100)

        # Simulate training loop
        initial_lr = scheduler.get_last_lr()[0]
        for _ in range(20):
            scheduler.step()

        # LR should have changed
        current_lr = scheduler.get_last_lr()[0]
        assert current_lr != initial_lr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
