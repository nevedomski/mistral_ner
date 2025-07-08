"""Comprehensive tests for samplers module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.datasets.samplers import DistributedMultiDatasetSampler, MultiDatasetSampler


class TestMultiDatasetSampler:
    """Test MultiDatasetSampler functionality."""

    def test_init_basic(self):
        """Test basic initialization."""
        sampler = MultiDatasetSampler([100, 200, 150])

        assert sampler.num_datasets == 3
        assert sampler.total_size == 450
        assert sampler.strategy == "interleave"
        assert sampler.seed == 42
        assert sampler.batch_size == 1
        assert sampler.drop_last is False

    def test_init_with_weights(self):
        """Test initialization with custom weights."""
        weights = [0.5, 0.3, 0.2]
        sampler = MultiDatasetSampler([100, 200, 150], weights=weights)

        # Weights should be normalized
        np.testing.assert_array_almost_equal(sampler.weights, weights)

    def test_init_empty_datasets(self):
        """Test error with empty datasets."""
        with pytest.raises(ValueError, match="At least one dataset is required"):
            MultiDatasetSampler([])

    def test_init_invalid_strategy(self):
        """Test error with invalid strategy."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            MultiDatasetSampler([100, 200], strategy="invalid")

    def test_init_mismatched_weights(self):
        """Test error with mismatched weights."""
        with pytest.raises(ValueError, match="Number of weights.*must match"):
            MultiDatasetSampler([100, 200], weights=[0.5])

    def test_dataset_boundaries(self):
        """Test dataset boundaries calculation."""
        sampler = MultiDatasetSampler([100, 200, 150])

        assert sampler.dataset_boundaries == [0, 100, 300, 450]

    def test_default_weights_proportional(self):
        """Test default weights are proportional to dataset sizes."""
        sampler = MultiDatasetSampler([100, 200, 100])

        expected_weights = np.array([0.25, 0.5, 0.25])
        np.testing.assert_array_almost_equal(sampler.weights, expected_weights)

    def test_interleaved_indices_creation(self):
        """Test interleaved indices creation."""
        sampler = MultiDatasetSampler([10, 10, 10], strategy="interleave", seed=42)
        indices = list(sampler)

        assert len(indices) == 30
        # Check all indices are within bounds
        assert all(0 <= idx < 30 for idx in indices)
        # Check each dataset contributes equally
        dataset_counts = [0, 0, 0]
        for idx in indices:
            if idx < 10:
                dataset_counts[0] += 1
            elif idx < 20:
                dataset_counts[1] += 1
            else:
                dataset_counts[2] += 1
        assert dataset_counts == [10, 10, 10]

    def test_weighted_indices_creation(self):
        """Test weighted indices creation."""
        # Use larger datasets for more stable testing
        sampler = MultiDatasetSampler([1000, 1000, 1000], strategy="weighted", weights=[0.5, 0.3, 0.2], seed=42)
        indices = list(sampler)

        assert len(indices) == 3000

        # Count samples from each dataset
        dataset_counts = [0, 0, 0]
        for idx in indices:
            if idx < 1000:
                dataset_counts[0] += 1
            elif idx < 2000:
                dataset_counts[1] += 1
            else:
                dataset_counts[2] += 1

        # Check approximate distribution (with some tolerance)
        assert 1400 < dataset_counts[0] < 1600  # ~50%
        assert 800 < dataset_counts[1] < 1000  # ~30%
        assert 500 < dataset_counts[2] < 700  # ~20%

    def test_len_without_drop_last(self):
        """Test length calculation without dropping last batch."""
        sampler = MultiDatasetSampler([100, 200, 150], batch_size=32, drop_last=False)

        assert len(sampler) == 450

    def test_len_with_drop_last(self):
        """Test length calculation with dropping last batch."""
        sampler = MultiDatasetSampler([100, 200, 150], batch_size=32, drop_last=True)

        # 450 // 32 = 14, so 14 * 32 = 448
        assert len(sampler) == 448

    def test_set_epoch(self):
        """Test setting epoch for different shuffling."""
        sampler = MultiDatasetSampler([10, 10], seed=42)

        # Get indices for epoch 0
        indices_epoch0 = list(sampler)

        # Set epoch 1
        sampler.set_epoch(1)
        indices_epoch1 = list(sampler)

        # Indices should be different
        assert indices_epoch0 != indices_epoch1

    def test_interleaved_small_datasets(self):
        """Test interleaving with datasets of different sizes."""
        sampler = MultiDatasetSampler([5, 10, 3], strategy="interleave", seed=42)
        indices = list(sampler)

        assert len(indices) == 18
        # All indices should be valid
        assert all(0 <= idx < 18 for idx in indices)

    def test_weighted_with_replacement(self):
        """Test weighted sampling with replacement when dataset exhausted."""
        sampler = MultiDatasetSampler([5, 5], strategy="weighted", weights=[0.9, 0.1], seed=42)
        indices = list(sampler)

        assert len(indices) == 10
        # First dataset should be sampled more
        dataset0_count = sum(1 for idx in indices if idx < 5)
        dataset1_count = sum(1 for idx in indices if 5 <= idx < 10)
        assert dataset0_count > dataset1_count

    def test_reproducibility_with_seed(self):
        """Test reproducibility with same seed."""
        sampler1 = MultiDatasetSampler([100, 200], seed=123)
        sampler2 = MultiDatasetSampler([100, 200], seed=123)

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        assert indices1 == indices2

    def test_different_seeds_different_indices(self):
        """Test different seeds produce different indices."""
        sampler1 = MultiDatasetSampler([100, 200], seed=123)
        sampler2 = MultiDatasetSampler([100, 200], seed=456)

        indices1 = list(sampler1)
        indices2 = list(sampler2)

        assert indices1 != indices2


class TestDistributedMultiDatasetSampler:
    """Test DistributedMultiDatasetSampler functionality."""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.is_available")
    def test_init_no_distributed(self, mock_available, mock_initialized):
        """Test initialization without distributed setup."""
        mock_available.return_value = False
        mock_initialized.return_value = False

        sampler = DistributedMultiDatasetSampler([100, 200])

        assert sampler.num_replicas == 1
        assert sampler.rank == 0
        assert sampler.shuffle is True

    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.is_available")
    def test_init_with_distributed(self, mock_available, mock_initialized, mock_rank, mock_world_size):
        """Test initialization with distributed setup."""
        mock_available.return_value = True
        mock_initialized.return_value = True
        mock_world_size.return_value = 4
        mock_rank.return_value = 2

        sampler = DistributedMultiDatasetSampler([100, 200])

        assert sampler.num_replicas == 4
        assert sampler.rank == 2

    def test_init_explicit_params(self):
        """Test initialization with explicit parameters."""
        sampler = DistributedMultiDatasetSampler([100, 200], num_replicas=4, rank=1, shuffle=False)

        assert sampler.num_replicas == 4
        assert sampler.rank == 1
        assert sampler.shuffle is False

    def test_num_samples_calculation(self):
        """Test samples per replica calculation."""
        sampler = DistributedMultiDatasetSampler(
            [100, 200, 150],  # Total 450
            num_replicas=4,
            rank=0,
        )

        # 450 / 4 = 112.5, rounded up to 113
        assert sampler.num_samples == 113
        assert sampler.total_samples == 452  # 113 * 4

    def test_num_samples_with_drop_last(self):
        """Test samples calculation with drop_last."""
        sampler = DistributedMultiDatasetSampler(
            [100, 200, 150],  # Total 450
            num_replicas=4,
            rank=0,
            drop_last=True,
        )

        # 450 / 4 = 112.5, rounded down to 112
        assert sampler.num_samples == 112

    def test_iter_distributes_samples(self):
        """Test that iteration distributes samples correctly."""
        # Small dataset for easy verification
        sampler = DistributedMultiDatasetSampler(
            [12],  # 12 samples total
            num_replicas=3,
            rank=1,
            shuffle=False,
        )

        indices = list(sampler)

        # Each replica should get 4 samples
        assert len(indices) == 4
        # Rank 1 should get indices 1, 4, 7, 10
        expected = [1, 4, 7, 10]
        assert indices == expected

    def test_iter_with_padding(self):
        """Test iteration when padding is needed."""
        sampler = DistributedMultiDatasetSampler(
            [10],  # 10 samples, not divisible by 3
            num_replicas=3,
            rank=0,
            shuffle=False,
        )

        indices = list(sampler)

        # Should get 4 samples (10/3 rounded up)
        assert len(indices) == 4
        # Last sample should be from padding (cycling)
        assert indices == [0, 3, 6, 9]

    def test_set_epoch_affects_shuffling(self):
        """Test that set_epoch affects shuffling."""
        sampler = DistributedMultiDatasetSampler([100, 200], num_replicas=2, rank=0, shuffle=True, seed=42)

        # Get indices for epoch 0
        sampler.set_epoch(0)
        indices_epoch0 = list(sampler)

        # Get indices for epoch 1
        sampler.set_epoch(1)
        indices_epoch1 = list(sampler)

        # Should be different due to shuffling
        assert indices_epoch0 != indices_epoch1

    def test_no_shuffle_consistent(self):
        """Test that no shuffle produces consistent results."""
        sampler = DistributedMultiDatasetSampler([100, 200], num_replicas=2, rank=0, shuffle=False)

        indices1 = list(sampler)
        indices2 = list(sampler)

        assert indices1 == indices2

    def test_all_ranks_cover_dataset(self):
        """Test that all ranks together cover the full dataset."""
        dataset_sizes = [20, 30]
        all_indices = []

        for rank in range(4):
            sampler = DistributedMultiDatasetSampler(dataset_sizes, num_replicas=4, rank=rank, shuffle=False)
            all_indices.extend(list(sampler))

        # Remove padding
        all_indices = [idx for idx in all_indices if idx < 50]

        # Should have all unique indices from 0 to 49
        assert len(set(all_indices)) == 50
        assert min(all_indices) == 0
        assert max(all_indices) == 49

    def test_weighted_strategy_distributed(self):
        """Test distributed sampler with weighted strategy."""
        sampler = DistributedMultiDatasetSampler(
            [100, 100], num_replicas=2, rank=0, strategy="weighted", weights=[0.8, 0.2], seed=42
        )

        indices = list(sampler)

        # Should get half the samples
        assert len(indices) == 100

        # Count distribution (approximately)
        dataset0_count = sum(1 for idx in indices if idx < 100)
        dataset1_count = sum(1 for idx in indices if 100 <= idx < 200)

        # First dataset should have more samples
        assert dataset0_count > dataset1_count
