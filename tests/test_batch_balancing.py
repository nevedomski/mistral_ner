"""Tests for batch balancing functionality in Tier 2 improvements."""

from __future__ import annotations

import torch

from datasets import Dataset
from src.batch_balancing import (
    BalancedBatchSampler,
    BatchCompositionLogger,
    EntityAwareBatchSampler,
    compute_batch_statistics,
)
from src.config import Config
from src.losses import BatchBalancedFocalLoss, create_loss_function
from src.training import create_custom_trainer_class


class TestBatchBalancing:
    """Test batch balancing functionality."""

    def create_mock_dataset(self, n_samples: int = 100) -> Dataset:
        """Create a mock dataset with imbalanced labels."""
        data = {"labels": []}

        for i in range(n_samples):
            if i < 70:  # 70% all O tokens
                labels = [0] * 10
            elif i < 85:  # 15% with PER entities
                labels = [0, 1, 2, 0, 0, 0, 0, 0, 0, 0]
            elif i < 95:  # 10% with ORG entities
                labels = [0, 0, 0, 3, 4, 0, 0, 0, 0, 0]
            else:  # 5% with LOC entities
                labels = [0, 0, 0, 0, 0, 5, 6, 0, 0, 0]

            data["labels"].append(labels)

        return Dataset.from_dict(data)

    def test_balanced_batch_sampler_initialization(self) -> None:
        """Test BalancedBatchSampler initialization."""
        dataset = self.create_mock_dataset()

        sampler = BalancedBatchSampler(
            dataset=dataset,
            batch_size=8,
            min_positive_ratio=0.3,
        )

        # Check indices were built correctly
        assert len(sampler.positive_indices) == 30  # 30% have entities
        assert len(sampler.negative_indices) == 70  # 70% all O
        assert len(sampler.positive_indices) + len(sampler.negative_indices) == len(dataset)

    def test_balanced_batch_sampler_iteration(self) -> None:
        """Test that balanced sampler creates balanced batches."""
        dataset = self.create_mock_dataset()

        sampler = BalancedBatchSampler(
            dataset=dataset,
            batch_size=10,
            min_positive_ratio=0.3,
        )

        batches = list(sampler)

        # Check that batches maintain minimum positive ratio
        for batch in batches:
            positive_count = sum(1 for idx in batch if idx in sampler.positive_indices)
            assert positive_count >= 3  # At least 30% positive
            assert len(batch) == 10 or not sampler.drop_last

    def test_entity_aware_batch_sampler(self) -> None:
        """Test EntityAwareBatchSampler functionality."""
        dataset = self.create_mock_dataset()

        sampler = EntityAwareBatchSampler(
            dataset=dataset,
            batch_size=8,
        )

        # Check entity indices were built
        assert "negative" in sampler.entity_indices
        assert len(sampler.entity_indices["negative"]) == 70

        # Check that we have some entity groups
        entity_groups = [k for k in sampler.entity_indices if k != "negative"]
        assert len(entity_groups) > 0

        # Test iteration
        batches = list(sampler)
        assert len(batches) > 0

    def test_compute_batch_statistics(self) -> None:
        """Test batch statistics computation."""
        # Create batch with mixed labels
        batch_labels = torch.tensor(
            [
                [0, 1, 2, 0, 0, -100, -100],  # Sample with PER entity
                [0, 0, 3, 4, 0, -100, -100],  # Sample with ORG entity
                [0, 0, 0, 0, 0, -100, -100],  # All O tokens
            ]
        )

        stats = compute_batch_statistics(batch_labels)

        assert stats["total_tokens"] == 15  # Excluding -100
        assert stats["o_tokens"] == 11
        assert stats["entity_tokens"] == 4
        assert abs(stats["entity_ratio"] - 4 / 15) < 0.01
        assert stats["unique_entities"] == 4
        assert 1 in stats["entity_distribution"]
        assert 2 in stats["entity_distribution"]
        assert 3 in stats["entity_distribution"]
        assert 4 in stats["entity_distribution"]

    def test_batch_composition_logger(self) -> None:
        """Test BatchCompositionLogger functionality."""
        logger = BatchCompositionLogger(log_every_n_batches=2)

        # Log some batches
        batch1 = torch.tensor([[0, 1, 2, 0, 0]])
        batch2 = torch.tensor([[0, 0, 0, 0, 0]])
        batch3 = torch.tensor([[3, 4, 0, 0, 0]])

        logger.log_batch(batch1)
        logger.log_batch(batch2)
        logger.log_batch(batch3)

        # Get summary
        summary = logger.get_summary()

        assert summary["total_batches"] == 3
        assert "avg_entity_ratio" in summary
        assert "entity_type_distribution" in summary
        assert summary["entity_type_distribution"][1] == 1  # One B-PER
        assert summary["entity_type_distribution"][2] == 1  # One I-PER

    def test_batch_balanced_focal_loss(self) -> None:
        """Test BatchBalancedFocalLoss computation."""
        num_labels = 9
        loss_fn = BatchBalancedFocalLoss(
            num_labels=num_labels,
            gamma=2.0,
            batch_balance_beta=0.999,
        )

        # Create inputs with imbalanced batch
        batch_size, seq_len = 4, 10
        inputs = torch.randn(batch_size, seq_len, num_labels)

        # Create targets with mostly O tokens
        targets = torch.zeros(batch_size, seq_len, dtype=torch.long)
        targets[0, 2:4] = torch.tensor([1, 2])  # Add some PER entity
        targets[1, 5:7] = torch.tensor([3, 4])  # Add some ORG entity

        loss = loss_fn(inputs, targets)

        assert loss.item() > 0
        assert loss.ndim == 0  # Scalar loss

    def test_batch_balanced_focal_loss_empty_batch(self) -> None:
        """Test BatchBalancedFocalLoss with all ignored tokens."""
        loss_fn = BatchBalancedFocalLoss(num_labels=9)

        inputs = torch.randn(2, 5, 9)
        targets = torch.full((2, 5), -100)  # All ignored

        loss = loss_fn(inputs, targets)
        assert loss.item() == 0.0

    def test_loss_function_factory_with_batch_balanced(self) -> None:
        """Test creating batch-balanced focal loss via factory."""
        loss_fn = create_loss_function(
            loss_type="batch_balanced_focal",
            num_labels=9,
            gamma=2.0,
            batch_balance_beta=0.999,
        )

        assert isinstance(loss_fn, BatchBalancedFocalLoss)
        assert loss_fn.gamma == 2.0
        assert loss_fn.batch_balance_beta == 0.999

    def test_trainer_with_batch_balancing(self) -> None:
        """Test custom trainer with batch balancing enabled."""
        config = Config()
        config.training.use_batch_balancing = True
        config.training.batch_balance_type = "balanced"
        config.training.min_positive_ratio = 0.3

        dataset = self.create_mock_dataset()

        # Create trainer class
        trainer_class = create_custom_trainer_class(config, dataset)

        # Check it has the custom get_train_dataloader method
        assert hasattr(trainer_class, "get_train_dataloader")

    def test_balanced_sampler_edge_cases(self) -> None:
        """Test edge cases for balanced sampler."""
        # Dataset with all negative samples
        data = {"labels": [[0] * 10 for _ in range(50)]}
        dataset = Dataset.from_dict(data)

        sampler = BalancedBatchSampler(
            dataset=dataset,
            batch_size=10,
            min_positive_ratio=0.3,
        )

        assert len(sampler.positive_indices) == 0
        assert len(sampler.negative_indices) == 50

        # Should still create batches even without positive samples
        batches = list(sampler)
        assert len(batches) > 0

    def test_batch_statistics_edge_cases(self) -> None:
        """Test edge cases for batch statistics."""
        # Empty batch
        empty_batch = torch.tensor([])
        stats = compute_batch_statistics(empty_batch)
        assert stats["total_tokens"] == 0
        assert stats["entity_ratio"] == 0.0

        # All ignored tokens
        ignored_batch = torch.full((2, 5), -100)
        stats = compute_batch_statistics(ignored_batch)
        assert stats["total_tokens"] == 0

        # Single token batch
        single_token = torch.tensor([[1]])
        stats = compute_batch_statistics(single_token)
        assert stats["total_tokens"] == 1
        assert stats["entity_tokens"] == 1
        assert stats["entity_ratio"] == 1.0
