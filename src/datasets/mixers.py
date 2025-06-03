"""Dataset mixing strategies for multi-dataset training."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from datasets import concatenate_datasets, interleave_datasets

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


class DatasetMixer:
    """Handles different strategies for combining multiple datasets."""

    @staticmethod
    def mix(
        datasets: list[DatasetDict], strategy: str = "interleave", weights: list[float] | None = None, seed: int = 42
    ) -> DatasetDict:
        """Mix multiple datasets according to specified strategy.

        Args:
            datasets: List of DatasetDict objects to mix
            strategy: Mixing strategy - "concat", "interleave", or "weighted"
            weights: Optional weights for each dataset (used with interleave/weighted)
            seed: Random seed for reproducibility

        Returns:
            Mixed DatasetDict with train/validation/test splits
        """
        if not datasets:
            raise ValueError("No datasets provided to mix")

        # Set random seeds
        random.seed(seed)
        # Using numpy's legacy seed for compatibility with datasets library
        np.random.seed(seed)  # noqa: NPY002

        # Normalize weights if provided
        if weights:
            weights = DatasetMixer._normalize_weights(weights, len(datasets))

        # Mix each split separately
        mixed_splits = {}
        for split in ["train", "validation", "test"]:
            # Collect this split from all datasets that have it
            split_datasets = []
            split_weights = []

            for i, dataset in enumerate(datasets):
                if split in dataset:
                    split_datasets.append(dataset[split])
                    if weights:
                        split_weights.append(weights[i])

            if not split_datasets:
                continue

            # Normalize weights for this split if needed
            if split_weights:
                split_weights = DatasetMixer._normalize_weights(split_weights, len(split_datasets))

            # Apply mixing strategy
            if strategy == "concat":
                mixed_splits[split] = DatasetMixer._concat_datasets(split_datasets)
            elif strategy == "interleave":
                mixed_splits[split] = DatasetMixer._interleave_datasets(split_datasets, split_weights, seed)
            elif strategy == "weighted":
                mixed_splits[split] = DatasetMixer._weighted_sample(split_datasets, split_weights, seed)
            else:
                raise ValueError(f"Unknown mixing strategy: {strategy}")

        return mixed_splits

    @staticmethod
    def _normalize_weights(weights: list[float], num_datasets: int) -> list[float]:
        """Normalize weights to sum to 1.0."""
        if len(weights) != num_datasets:
            raise ValueError(f"Number of weights ({len(weights)}) doesn't match number of datasets ({num_datasets})")

        total = sum(weights)
        if total == 0:
            return [1.0 / num_datasets] * num_datasets
        return [w / total for w in weights]

    @staticmethod
    def _concat_datasets(datasets: list[Dataset]) -> Dataset:
        """Concatenate datasets sequentially."""
        return concatenate_datasets(datasets)

    @staticmethod
    def _interleave_datasets(datasets: list[Dataset], weights: list[float] | None, seed: int) -> Dataset:
        """Interleave datasets with optional weights."""
        return interleave_datasets(datasets, probabilities=weights, seed=seed, stopping_strategy="all_exhausted")

    @staticmethod
    def _weighted_sample(datasets: list[Dataset], weights: list[float] | None, seed: int) -> Dataset:
        """Create a weighted sample from multiple datasets.

        This creates a new dataset by sampling from each source dataset
        proportionally to the weights.
        """
        if not weights:
            weights = [1.0 / len(datasets)] * len(datasets)

        # Calculate target size (use minimum to ensure all datasets contribute)
        total_size = sum(len(d) for d in datasets)
        target_sizes = [int(total_size * w) for w in weights]

        # Adjust for rounding errors
        size_diff = total_size - sum(target_sizes)
        if size_diff > 0:
            target_sizes[-1] += size_diff

        # Sample from each dataset
        sampled_datasets = []
        for dataset, target_size in zip(datasets, target_sizes, strict=False):
            if target_size > len(dataset):
                # If target is larger than dataset, repeat some samples
                indices = list(range(len(dataset)))
                extra_needed = target_size - len(dataset)
                indices.extend(random.choices(indices, k=extra_needed))
            else:
                # Sample without replacement
                indices = random.sample(range(len(dataset)), target_size)

            sampled = dataset.select(indices)
            sampled_datasets.append(sampled)

        # Concatenate and shuffle
        combined = concatenate_datasets(sampled_datasets)
        return combined.shuffle(seed=seed)

    @staticmethod
    def analyze_mixture(dataset: Dataset, dataset_names: list[str]) -> dict[str, float]:
        """Analyze the composition of a mixed dataset.

        This requires datasets to have a 'dataset_source' field added during mixing.

        Args:
            dataset: Mixed dataset to analyze
            dataset_names: Names of source datasets

        Returns:
            Dictionary mapping dataset names to their proportions
        """
        if "dataset_source" not in dataset.features:
            return {"unknown": 1.0}

        counts: dict[str, float] = {name: 0.0 for name in dataset_names}
        for example in dataset:
            source = example.get("dataset_source", "unknown")
            if source in counts:
                counts[source] += 1.0

        total = sum(counts.values())
        if total == 0:
            return counts

        return {name: count / total for name, count in counts.items()}
