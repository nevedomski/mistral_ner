"""Tests for loss functions module."""

import pytest
import torch
import torch.nn as nn

from src.losses import (
    ClassBalancedLoss,
    FocalLoss,
    LabelSmoothingLoss,
    compute_class_frequencies,
    create_loss_function,
)


class TestFocalLoss:
    """Test FocalLoss implementation."""

    def test_focal_loss_init_default(self):
        """Test FocalLoss initialization with default parameters."""
        loss = FocalLoss(num_labels=5)
        assert loss.num_labels == 5
        assert loss.gamma == 2.0
        assert loss.alpha is None
        assert loss.reduction == "mean"
        assert loss.ignore_index == -100

    def test_focal_loss_init_with_alpha_float(self):
        """Test FocalLoss initialization with float alpha."""
        loss = FocalLoss(num_labels=5, alpha=0.25)
        assert loss.alpha is not None
        assert torch.allclose(loss.alpha, torch.ones(5) * 0.25)

    def test_focal_loss_init_with_alpha_list(self):
        """Test FocalLoss initialization with list alpha."""
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        loss = FocalLoss(num_labels=5, alpha=alpha_list)
        assert loss.alpha is not None
        assert torch.allclose(loss.alpha, torch.tensor(alpha_list))

    def test_focal_loss_forward_basic(self):
        """Test FocalLoss forward pass with basic input."""
        loss = FocalLoss(num_labels=3)
        inputs = torch.randn(4, 3)  # batch_size=4, num_labels=3
        targets = torch.tensor([0, 1, 2, 0])

        result = loss(inputs, targets)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0  # scalar output
        assert result.item() > 0  # loss should be positive

    def test_focal_loss_forward_with_ignore_index(self):
        """Test FocalLoss forward pass with ignore_index."""
        loss = FocalLoss(num_labels=3, ignore_index=-100)
        inputs = torch.randn(6, 3)
        targets = torch.tensor([0, 1, -100, 2, -100, 0])

        result = loss(inputs, targets)
        assert isinstance(result, torch.Tensor)
        assert result.item() > 0

    def test_focal_loss_forward_with_alpha(self):
        """Test FocalLoss forward pass with alpha weighting."""
        alpha = [0.5, 1.0, 1.5]
        loss = FocalLoss(num_labels=3, alpha=alpha, gamma=2.0)
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])

        result = loss(inputs, targets)
        assert isinstance(result, torch.Tensor)
        assert result.item() > 0

    def test_focal_loss_reduction_modes(self):
        """Test FocalLoss with different reduction modes."""
        for reduction in ["mean", "sum", "none"]:
            loss = FocalLoss(num_labels=3, reduction=reduction)
            inputs = torch.randn(4, 3)
            targets = torch.tensor([0, 1, 2, 0])

            result = loss(inputs, targets)
            if reduction == "none":
                assert result.shape == (4,)
            else:
                assert result.ndim == 0

    def test_focal_loss_edge_cases(self):
        """Test FocalLoss edge cases."""
        loss = FocalLoss(num_labels=3)

        # All ignored indices
        inputs = torch.randn(3, 3)
        targets = torch.tensor([-100, -100, -100])
        result = loss(inputs, targets)
        assert result.item() == 0.0

        # Empty valid mask
        inputs = torch.randn(0, 3)
        targets = torch.tensor([], dtype=torch.long)
        result = loss(inputs, targets)
        assert result.item() == 0.0

    def test_focal_loss_alpha_device_handling(self):
        """Test focal loss handles alpha device mismatch correctly."""
        num_labels = 9
        alpha = torch.ones(num_labels)
        loss_fn = FocalLoss(num_labels=num_labels, gamma=2.0, alpha=alpha)

        # Create inputs and targets
        inputs = torch.randn(2, 5, num_labels)
        targets = torch.randint(0, num_labels, (2, 5))

        # Test that loss computation works (will test device handling logic)
        loss = loss_fn(inputs, targets)
        assert loss.item() > 0
        # Verify alpha was used (loss should be finite and reasonable)
        assert torch.isfinite(loss)


class TestLabelSmoothingLoss:
    """Test LabelSmoothingLoss implementation."""

    def test_label_smoothing_init(self):
        """Test LabelSmoothingLoss initialization."""
        loss = LabelSmoothingLoss(num_labels=5, smoothing=0.1)
        assert loss.num_labels == 5
        assert loss.smoothing == 0.1
        assert loss.confidence == 0.9
        assert loss.ignore_index == -100

    def test_label_smoothing_forward(self):
        """Test LabelSmoothingLoss forward pass."""
        loss = LabelSmoothingLoss(num_labels=3, smoothing=0.1)
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])

        result = loss(inputs, targets)
        assert isinstance(result, torch.Tensor)
        assert result.item() > 0

    def test_label_smoothing_with_ignore_index(self):
        """Test LabelSmoothingLoss with ignore_index."""
        loss = LabelSmoothingLoss(num_labels=3, smoothing=0.1)
        inputs = torch.randn(5, 3)
        targets = torch.tensor([0, 1, -100, 2, -100])

        result = loss(inputs, targets)
        assert isinstance(result, torch.Tensor)
        assert result.item() > 0

    def test_label_smoothing_edge_cases(self):
        """Test LabelSmoothingLoss edge cases."""
        loss = LabelSmoothingLoss(num_labels=3, smoothing=0.1)

        # All ignored
        inputs = torch.randn(3, 3)
        targets = torch.tensor([-100, -100, -100])
        result = loss(inputs, targets)
        assert result.item() == 0.0


class TestClassBalancedLoss:
    """Test ClassBalancedLoss implementation."""

    def test_class_balanced_init(self):
        """Test ClassBalancedLoss initialization."""
        class_freq = [100, 50, 25, 10]
        loss = ClassBalancedLoss(num_labels=4, class_frequencies=class_freq)
        assert loss.num_labels == 4
        assert loss.beta == 0.9999
        assert loss.base_loss is not None

    def test_class_balanced_forward(self):
        """Test ClassBalancedLoss forward pass."""
        class_freq = [100, 50, 25]
        loss = ClassBalancedLoss(num_labels=3, class_frequencies=class_freq, beta=0.99)
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])

        result = loss(inputs, targets)
        assert isinstance(result, torch.Tensor)
        assert result.item() > 0

    def test_class_balanced_effective_numbers(self):
        """Test effective number calculation with different frequencies."""
        class_freq = [1000, 100, 10]
        loss = ClassBalancedLoss(num_labels=3, class_frequencies=class_freq, beta=0.9)

        # The base loss should be initialized correctly
        assert loss.base_loss is not None
        # For focal loss, check alpha instead of weight
        assert hasattr(loss.base_loss, "alpha")
        assert loss.base_loss.alpha is not None

        # Test with imbalanced data
        inputs = torch.randn(5, 3)
        targets = torch.tensor([0, 0, 1, 2, 2])  # More common classes
        result = loss(inputs, targets)
        assert isinstance(result, torch.Tensor)


class TestCreateLossFunction:
    """Test loss function factory."""

    def test_create_focal_loss(self):
        """Test creating focal loss."""
        loss = create_loss_function("focal", num_labels=5, gamma=3.0, alpha=0.5)
        assert isinstance(loss, FocalLoss)
        assert loss.gamma == 3.0

    def test_create_label_smoothing_loss(self):
        """Test creating label smoothing loss."""
        loss = create_loss_function("label_smoothing", num_labels=5, smoothing=0.2)
        assert isinstance(loss, LabelSmoothingLoss)
        assert loss.smoothing == 0.2

    def test_create_class_balanced_loss(self):
        """Test creating class balanced loss."""
        class_freq = [100, 50, 25]
        loss = create_loss_function("class_balanced", num_labels=3, class_frequencies=class_freq, beta=0.95)
        assert isinstance(loss, ClassBalancedLoss)
        assert loss.beta == 0.95

    def test_create_cross_entropy_loss(self):
        """Test creating standard cross entropy loss."""
        loss = create_loss_function("cross_entropy", num_labels=5)
        assert isinstance(loss, nn.CrossEntropyLoss)

    def test_create_unknown_loss(self):
        """Test creating unknown loss type raises error."""
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            create_loss_function("unknown_loss", num_labels=5)


class TestComputeClassFrequencies:
    """Test class frequency computation."""

    def test_compute_class_frequencies_basic(self):
        """Test basic class frequency computation."""
        # Create mock dataset
        dataset = [
            {"labels": [0, 1, 2, 0]},
            {"labels": [1, 1, 0, 2]},
            {"labels": [2, 2, 1, 0]},
        ]

        frequencies = compute_class_frequencies(dataset, "labels")
        assert len(frequencies) == 3
        assert frequencies[0] == 4  # 0 appears 4 times
        assert frequencies[1] == 4  # 1 appears 4 times
        assert frequencies[2] == 4  # 2 appears 4 times

    def test_compute_class_frequencies_with_ignore_index(self):
        """Test class frequency computation with ignore index (-100 filtered automatically)."""
        dataset = [
            {"labels": [0, 1, -100, 2]},
            {"labels": [1, -100, 0, 2]},
        ]

        frequencies = compute_class_frequencies(dataset, "labels")
        assert len(frequencies) == 3
        assert frequencies[0] == 2
        assert frequencies[1] == 2
        assert frequencies[2] == 2

    def test_compute_class_frequencies_tensor_labels(self):
        """Test class frequency computation with tensor labels."""
        dataset = [
            {"labels": torch.tensor([0, 1, 2])},
            {"labels": torch.tensor([2, 1, 0])},
        ]

        frequencies = compute_class_frequencies(dataset, "labels")
        assert len(frequencies) == 3
        assert all(f == 2 for f in frequencies)

    def test_compute_class_frequencies_auto_num_labels(self):
        """Test class frequency computation with auto num_labels detection."""
        dataset = [
            {"labels": [0, 1, 4]},  # max label is 4
            {"labels": [2, 3, 4]},
        ]

        frequencies = compute_class_frequencies(dataset, "labels")
        assert len(frequencies) == 5  # 0 to 4
        assert frequencies[0] == 1
        assert frequencies[1] == 1
        assert frequencies[2] == 1
        assert frequencies[3] == 1
        assert frequencies[4] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
