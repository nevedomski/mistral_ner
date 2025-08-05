"""Custom samplers for multi-dataset training."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger("mistral_ner")


class MultiDatasetSampler(Sampler[int]):
    """
    Sampler for training on multiple datasets with different mixing strategies.

    This sampler supports two strategies:
    1. 'interleave': Round-robin sampling from each dataset
    2. 'weighted': Probabilistic sampling based on dataset weights

    Compatible with HuggingFace Trainer and supports distributed training.
    """

    def __init__(
        self,
        dataset_sizes: list[int],
        strategy: str = "interleave",
        weights: list[float] | None = None,
        seed: int = 42,
        batch_size: int = 1,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        """
        Initialize the multi-dataset sampler.

        Args:
            dataset_sizes: List of dataset sizes
            strategy: Mixing strategy ('interleave' or 'weighted')
            weights: Dataset weights for weighted sampling (normalized internally)
            seed: Random seed for reproducibility
            batch_size: Batch size for proper epoch length calculation
            drop_last: Whether to drop the last incomplete batch
            shuffle: Whether to shuffle the indices
        """
        self.dataset_sizes = dataset_sizes
        self.num_datasets = len(dataset_sizes)
        self.strategy = strategy
        self.seed = seed
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Validate inputs
        if self.num_datasets == 0:
            raise ValueError("At least one dataset is required")

        if strategy not in ["interleave", "weighted"]:
            raise ValueError(f"Invalid strategy: {strategy}. Must be 'interleave' or 'weighted'")

        # Calculate dataset boundaries for indexing
        self.dataset_boundaries = [0]
        for size in dataset_sizes:
            self.dataset_boundaries.append(self.dataset_boundaries[-1] + size)
        self.total_size = self.dataset_boundaries[-1]

        # Setup weights for weighted sampling
        if weights is not None:
            if len(weights) != self.num_datasets:
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match number of datasets ({self.num_datasets})"
                )
            self.weights = np.array(weights) / np.sum(weights)
        else:
            # Default to dataset size proportional weights
            self.weights = np.array(dataset_sizes) / sum(dataset_sizes)

        # Pre-compute indices
        self._indices: list[int] = []
        self._prepare_indices()

        logger.info(
            f"MultiDatasetSampler initialized with {self.num_datasets} datasets, "
            f"strategy='{self.strategy}', total_size={self.total_size}"
        )

    def _prepare_indices(self) -> None:
        """Pre-compute sampling indices based on strategy."""
        rng = np.random.RandomState(self.seed)

        if self.strategy == "interleave":
            self._indices = self._create_interleaved_indices(rng)
        elif self.strategy == "weighted":
            self._indices = self._create_weighted_indices(rng)

    def _create_interleaved_indices(self, rng: np.random.RandomState) -> list[int]:
        """
        Create indices for round-robin sampling.

        Each dataset is sampled in turn, with smaller datasets cycling
        through their indices multiple times to match larger datasets.
        """
        indices = []

        # Create shuffled indices for each dataset
        dataset_indices = []
        for dataset_idx in range(self.num_datasets):
            start_idx = self.dataset_boundaries[dataset_idx]
            dataset_size = self.dataset_sizes[dataset_idx]

            # Create indices for this dataset (shuffled if needed)
            dataset_idx_array = np.arange(dataset_size) + start_idx
            if self.shuffle:
                rng.shuffle(dataset_idx_array)
            dataset_indices.append(dataset_idx_array)

        # Interleave samples using round-robin
        # Continue until all datasets are exhausted
        dataset_positions = [0] * self.num_datasets
        datasets_exhausted = [False] * self.num_datasets

        while not all(datasets_exhausted):
            for dataset_idx in range(self.num_datasets):
                if not datasets_exhausted[dataset_idx]:
                    pos = dataset_positions[dataset_idx]
                    if pos < len(dataset_indices[dataset_idx]):
                        indices.append(dataset_indices[dataset_idx][pos])
                        dataset_positions[dataset_idx] += 1
                    else:
                        datasets_exhausted[dataset_idx] = True

        return indices

    def _create_weighted_indices(self, rng: np.random.RandomState) -> list[int]:
        """
        Create indices for weighted sampling.

        Samples are drawn from datasets according to their weights,
        with replacement when a dataset is exhausted.
        """
        indices = []

        # Create shuffled indices for each dataset
        dataset_indices = []
        dataset_positions = [0] * self.num_datasets  # Track position in each dataset

        for dataset_idx in range(self.num_datasets):
            start_idx = self.dataset_boundaries[dataset_idx]
            dataset_size = self.dataset_sizes[dataset_idx]

            # Create indices for this dataset (shuffled if needed)
            dataset_idx_array = np.arange(dataset_size) + start_idx
            if self.shuffle:
                rng.shuffle(dataset_idx_array)
            dataset_indices.append(dataset_idx_array)

        # Sample based on weights
        for _ in range(self.total_size):
            # Choose dataset based on weights
            dataset_idx = rng.choice(self.num_datasets, p=self.weights)

            # Get next index from chosen dataset
            pos = dataset_positions[dataset_idx]

            # If we've exhausted this dataset, reshuffle and start over
            if pos >= len(dataset_indices[dataset_idx]):
                start_idx = self.dataset_boundaries[dataset_idx]
                dataset_size = self.dataset_sizes[dataset_idx]
                dataset_indices[dataset_idx] = np.arange(dataset_size) + start_idx
                if self.shuffle:
                    rng.shuffle(dataset_indices[dataset_idx])
                dataset_positions[dataset_idx] = 0
                pos = 0

            indices.append(dataset_indices[dataset_idx][pos])
            dataset_positions[dataset_idx] += 1

        return indices

    def __iter__(self) -> Iterator[int]:
        """Iterate over pre-computed indices."""
        return iter(self._indices)

    def __len__(self) -> int:
        """Return the total number of samples."""
        if self.drop_last:
            return (self.total_size // self.batch_size) * self.batch_size
        return self.total_size

    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for shuffling.

        This is useful for distributed training to ensure different
        shuffling across epochs while maintaining consistency across processes.
        """
        self.seed = self.seed + epoch
        self._prepare_indices()


class DistributedMultiDatasetSampler(MultiDatasetSampler):
    """
    Distributed version of MultiDatasetSampler for multi-GPU training.

    Compatible with torch.nn.parallel.DistributedDataParallel and HuggingFace Trainer.
    """

    def __init__(
        self,
        dataset_sizes: list[int],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        strategy: str = "interleave",
        weights: list[float] | None = None,
        seed: int = 42,
        batch_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        """
        Initialize the distributed multi-dataset sampler.

        Args:
            dataset_sizes: List of dataset sizes
            num_replicas: Number of processes (GPUs)
            rank: Rank of current process
            shuffle: Whether to shuffle (changes seed each epoch)
            strategy: Mixing strategy ('interleave' or 'weighted')
            weights: Dataset weights for weighted sampling
            seed: Random seed
            batch_size: Batch size
            drop_last: Whether to drop last incomplete batch
        """
        # Initialize parent sampler
        super().__init__(dataset_sizes, strategy, weights, seed, batch_size, drop_last, shuffle)

        # Setup distributed parameters
        if num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0

        # Calculate samples per replica
        self.num_samples = self.total_size // self.num_replicas
        if self.total_size % self.num_replicas != 0 and not self.drop_last:
            # Add extra samples to make it evenly divisible
            self.num_samples += 1

        self.total_samples = self.num_samples * self.num_replicas

        logger.info(
            f"DistributedMultiDatasetSampler: rank={self.rank}/{self.num_replicas}, "
            f"samples_per_replica={self.num_samples}"
        )

    def __iter__(self) -> Iterator[int]:
        """Iterate over indices for current replica."""
        if self.shuffle:
            # Deterministically shuffle based on epoch
            self.set_epoch(self.epoch)

        indices = list(super().__iter__())

        # Add extra samples to make it evenly divisible
        if len(indices) < self.total_samples:
            extra = self.total_samples - len(indices)
            indices += indices[:extra]
        else:
            indices = indices[: self.total_samples]

        # Subsample for current replica
        indices = indices[self.rank : self.total_samples : self.num_replicas]

        return iter(indices)

    def __len__(self) -> int:
        """Return the number of samples for current replica."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Set epoch for proper shuffling across epochs.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch
        super().set_epoch(epoch)
