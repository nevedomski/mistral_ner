"""Tests for class weighting functionality in Tier 2 improvements."""

from __future__ import annotations

import torch
import torch.nn as nn

from datasets import Dataset
from src.config import Config
from src.losses import (
    WeightedFocalLoss,
    calculate_class_weights,
    compute_class_frequencies,
    create_loss_function,
)
from src.training import create_custom_trainer_class


class TestClassWeighting:
    """Test class weighting functionality."""

    def test_calculate_class_weights_inverse(self) -> None:
        """Test inverse frequency class weight calculation."""
        frequencies = [1000, 100, 10]  # Imbalanced classes
        weights = calculate_class_weights(frequencies, weight_type="inverse")

        # Check that rare classes get higher weights
        assert weights[2] > weights[1] > weights[0]
        # Check normalization (sum should equal number of classes)
        assert abs(sum(weights) - len(weights)) < 0.01

    def test_calculate_class_weights_inverse_sqrt(self) -> None:
        """Test inverse square root class weight calculation."""
        frequencies = [1000, 100, 10]
        weights = calculate_class_weights(frequencies, weight_type="inverse_sqrt")

        # Check that rare classes get higher weights (but less extreme than inverse)
        assert weights[2] > weights[1] > weights[0]
        # Check that weights are less extreme than pure inverse
        inverse_weights = calculate_class_weights(frequencies, weight_type="inverse")
        assert (weights[2] / weights[0]) < (inverse_weights[2] / inverse_weights[0])

    def test_calculate_class_weights_effective(self) -> None:
        """Test effective number based class weight calculation."""
        frequencies = [1000, 100, 10]
        weights = calculate_class_weights(frequencies, weight_type="effective")

        # Check that rare classes get higher weights
        assert weights[2] > weights[1] > weights[0]
        assert abs(sum(weights) - len(weights)) < 0.01

    def test_calculate_class_weights_smoothing(self) -> None:
        """Test smoothing parameter effect."""
        frequencies = [1000, 0, 10]  # One class with zero frequency

        # Without smoothing would cause division by zero
        weights = calculate_class_weights(frequencies, weight_type="inverse", smoothing=1.0)

        # Check no infinite weights
        assert all(w < float("inf") for w in weights)
        assert all(w > 0 for w in weights)

    def test_weighted_focal_loss(self) -> None:
        """Test WeightedFocalLoss initialization and forward pass."""
        num_labels = 9
        class_weights = [0.1, 1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 3.0, 3.0]

        loss_fn = WeightedFocalLoss(
            num_labels=num_labels,
            class_weights=class_weights,
            gamma=2.0,
        )

        # Test forward pass
        batch_size, seq_len = 4, 10
        inputs = torch.randn(batch_size, seq_len, num_labels)
        targets = torch.randint(0, num_labels, (batch_size, seq_len))

        loss = loss_fn(inputs, targets)
        assert loss.item() > 0
        # Note: loss might not have requires_grad if no gradients are needed

    def test_create_loss_function_with_auto_weights(self) -> None:
        """Test loss function creation with automatic weight calculation."""
        num_labels = 9
        class_frequencies = [1000, 100, 100, 50, 50, 75, 75, 25, 25]

        # Test weighted cross entropy
        loss_fn = create_loss_function(
            loss_type="weighted_cross_entropy",
            num_labels=num_labels,
            class_frequencies=class_frequencies,
            auto_weight=True,
            weight_type="inverse",
        )

        assert isinstance(loss_fn, nn.CrossEntropyLoss)
        assert loss_fn.weight is not None
        assert len(loss_fn.weight) == num_labels

        # Test focal loss with auto weights
        focal_loss = create_loss_function(
            loss_type="focal",
            num_labels=num_labels,
            class_frequencies=class_frequencies,
            auto_weight=True,
            weight_type="inverse_sqrt",
        )

        assert hasattr(focal_loss, "alpha")
        assert focal_loss.alpha is not None

    def test_create_loss_function_with_manual_weights(self) -> None:
        """Test loss function creation with manual weight specification."""
        num_labels = 9
        manual_weights = [0.1, 1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 3.0, 3.0]

        loss_fn = create_loss_function(
            loss_type="weighted_cross_entropy",
            num_labels=num_labels,
            class_weights=manual_weights,
        )

        assert isinstance(loss_fn, nn.CrossEntropyLoss)
        assert loss_fn.weight is not None
        assert torch.allclose(loss_fn.weight.cpu(), torch.tensor(manual_weights))

    def test_compute_class_frequencies(self) -> None:
        """Test computing class frequencies from dataset."""
        # Create mock dataset
        data = {
            "labels": [
                [0, 1, 2, -100, 0],  # -100 should be ignored
                [0, 0, 3, 4, 5],
                [0, 1, 1, 2, 2],
            ]
        }
        dataset = Dataset.from_dict(data)

        frequencies = compute_class_frequencies(dataset)

        # Expected counts: 0:5, 1:3, 2:3, 3:1, 4:1, 5:1
        assert frequencies[0] == 5
        assert frequencies[1] == 3
        assert frequencies[2] == 3
        assert frequencies[3] == 1
        assert frequencies[4] == 1
        assert frequencies[5] == 1

    def test_trainer_class_with_class_weights(self) -> None:
        """Test custom trainer class creation with class weights enabled."""
        config = Config()
        config.training.use_class_weights = True
        config.training.class_weight_type = "inverse"
        config.training.loss_type = "focal"

        # Create mock dataset
        data = {"labels": [[0, 1, 2, 0, 0]] * 10}
        train_dataset = Dataset.from_dict(data)

        # Create trainer class
        trainer_class = create_custom_trainer_class(config, train_dataset)

        # Verify it's a valid trainer class
        from transformers import Trainer

        assert issubclass(trainer_class, Trainer)
        assert hasattr(trainer_class, "compute_loss")

    def test_different_weight_types_produce_different_results(self) -> None:
        """Test that different weight types produce different weight distributions."""
        frequencies = [1000, 100, 10, 1]

        inverse_weights = calculate_class_weights(frequencies, weight_type="inverse")
        sqrt_weights = calculate_class_weights(frequencies, weight_type="inverse_sqrt")
        effective_weights = calculate_class_weights(frequencies, weight_type="effective")

        # All should produce different results
        assert inverse_weights != sqrt_weights
        assert inverse_weights != effective_weights
        assert sqrt_weights != effective_weights

        # But all should maintain the same ordering (rare classes get higher weights)
        for weights in [inverse_weights, sqrt_weights, effective_weights]:
            assert weights[3] > weights[2] > weights[1] > weights[0]

    def test_edge_cases(self) -> None:
        """Test edge cases for class weighting."""
        # Single class
        weights = calculate_class_weights([100], weight_type="inverse")
        assert len(weights) == 1
        assert weights[0] == 1.0

        # All equal frequencies
        equal_freq = [100, 100, 100, 100]
        weights = calculate_class_weights(equal_freq, weight_type="inverse")
        # All weights should be equal and sum to number of classes
        assert all(abs(w - 1.0) < 0.01 for w in weights)

        # Very imbalanced case
        imbalanced = [10000, 1]
        weights = calculate_class_weights(imbalanced, weight_type="inverse", smoothing=0.1)
        # Second class should have much higher weight
        assert weights[1] > weights[0] * 100
