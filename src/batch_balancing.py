"""Batch balancing utilities for handling class imbalance in NER training."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger("mistral_ner")


class BalancedBatchSampler(Sampler[list[int]]):
    """
    Custom sampler that creates balanced batches with better entity representation.

    This sampler ensures each batch contains a minimum number of positive examples
    (non-O tokens) to improve learning on rare entity classes.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        min_positive_ratio: float = 0.3,
        label_column: str = "labels",
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        """
        Initialize balanced batch sampler.

        Args:
            dataset: The dataset to sample from
            batch_size: Size of each batch
            min_positive_ratio: Minimum ratio of samples with entities in each batch
            label_column: Name of the label column in dataset
            drop_last: Whether to drop the last incomplete batch
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_positive_ratio = min_positive_ratio
        self.label_column = label_column
        self.drop_last = drop_last
        self.seed = seed

        # Build indices for positive and negative samples
        self._build_indices()

        logger.info(
            f"Initialized BalancedBatchSampler: "
            f"positive_samples={len(self.positive_indices)}, "
            f"negative_samples={len(self.negative_indices)}, "
            f"min_positive_ratio={min_positive_ratio}"
        )

    def _build_indices(self) -> None:
        """Build separate indices for positive (with entities) and negative (all O) samples."""
        self.positive_indices = []
        self.negative_indices = []

        for idx in range(len(self.dataset)):
            labels = self.dataset[idx][self.label_column]

            # Check if sample contains any non-O tokens (ignoring -100)
            has_entity = any(label > 0 for label in labels if label != -100)

            if has_entity:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)

    def __iter__(self) -> Iterator[list[int]]:
        """Generate balanced batches."""
        # Set random seed for reproducibility
        random.seed(self.seed)

        # Shuffle indices
        positive_indices = self.positive_indices.copy()
        negative_indices = self.negative_indices.copy()
        random.shuffle(positive_indices)
        random.shuffle(negative_indices)

        # Calculate minimum positive samples per batch
        min_positive_per_batch = int(self.batch_size * self.min_positive_ratio)

        batches = []
        pos_idx = 0
        neg_idx = 0

        # Handle edge case: no positive samples
        if len(positive_indices) == 0:
            # Create batches from negative samples only
            for i in range(0, len(negative_indices), self.batch_size):
                batch = negative_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                    batches.append(batch)
        else:
            while pos_idx < len(positive_indices) and neg_idx < len(negative_indices):
                batch = []

                # Add minimum positive samples
                for _ in range(min_positive_per_batch):
                    if pos_idx < len(positive_indices):
                        batch.append(positive_indices[pos_idx])
                        pos_idx += 1
                    else:
                        break

                # Fill rest with mix of positive and negative
                remaining_slots = self.batch_size - len(batch)

                # Randomly mix remaining positive and negative samples
                remaining_pos = positive_indices[pos_idx:]
                remaining_neg = negative_indices[neg_idx:]

                # Combine and shuffle
                combined = []
                if remaining_pos:
                    combined.extend(remaining_pos[: remaining_slots // 2])
                    pos_idx += len(combined)
                if remaining_neg:
                    neg_needed = remaining_slots - len(combined)
                    combined.extend(remaining_neg[:neg_needed])
                    neg_idx += len(combined) - (remaining_slots - neg_needed)

                random.shuffle(combined)
                batch.extend(combined[:remaining_slots])

                if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                    batches.append(batch)

        # Yield batches
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        """Return the number of batches."""
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


class EntityAwareBatchSampler(Sampler[list[int]]):
    """
    Advanced batch sampler that groups samples by entity type distribution.

    This sampler creates batches where each batch focuses on specific entity types,
    helping the model learn better representations for rare entities.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        entity_groups: list[list[int]] | None = None,
        label_column: str = "labels",
        seed: int = 42,
    ) -> None:
        """
        Initialize entity-aware batch sampler.

        Args:
            dataset: The dataset to sample from
            batch_size: Size of each batch
            entity_groups: Groups of entity label IDs to focus on
            label_column: Name of the label column
            seed: Random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.label_column = label_column
        self.seed = seed

        # Default entity groups if not provided
        if entity_groups is None:
            # Group by entity type: PER, ORG, LOC, MISC
            self.entity_groups = [
                [1, 2],  # B-PER, I-PER
                [3, 4],  # B-ORG, I-ORG
                [5, 6],  # B-LOC, I-LOC
                [7, 8],  # B-MISC, I-MISC
            ]
        else:
            self.entity_groups = entity_groups

        self._build_entity_indices()

    def _build_entity_indices(self) -> None:
        """Build indices grouped by dominant entity type."""
        self.entity_indices: dict[int | str, list[int]] = defaultdict(list)
        self.mixed_indices = []  # Samples with multiple entity types

        for idx in range(len(self.dataset)):
            labels = self.dataset[idx][self.label_column]

            # Count entities by group
            entity_counts = {i: 0 for i in range(len(self.entity_groups))}

            for label in labels:
                if label > 0 and label != -100:
                    for group_idx, group_labels in enumerate(self.entity_groups):
                        if label in group_labels:
                            entity_counts[group_idx] += 1
                            break

            # Classify sample
            total_entities = sum(entity_counts.values())
            if total_entities == 0:
                self.entity_indices["negative"].append(idx)
            else:
                # Find dominant entity type
                dominant_groups = [group_idx for group_idx, count in entity_counts.items() if count > 0]

                if len(dominant_groups) == 1:
                    self.entity_indices[dominant_groups[0]].append(idx)
                else:
                    self.mixed_indices.append(idx)

    def __iter__(self) -> Iterator[list[int]]:
        """Generate entity-focused batches."""
        random.seed(self.seed)

        # Create batches focusing on each entity type
        all_batches = []

        # Process each entity group
        for group_idx, indices in self.entity_indices.items():
            if group_idx == "negative":
                continue

            group_indices = indices.copy()
            random.shuffle(group_indices)

            # Mix with some negative and mixed samples
            neg_indices = self.entity_indices["negative"].copy()
            mixed_indices = self.mixed_indices.copy()
            random.shuffle(neg_indices)
            random.shuffle(mixed_indices)

            # Create batches for this entity group
            for i in range(0, len(group_indices), self.batch_size // 2):
                batch = group_indices[i : i + self.batch_size // 2]

                # Add some mixed samples
                if mixed_indices:
                    batch.extend(mixed_indices[: self.batch_size // 4])
                    mixed_indices = mixed_indices[self.batch_size // 4 :]

                # Fill with negative samples
                remaining = self.batch_size - len(batch)
                if neg_indices and remaining > 0:
                    batch.extend(neg_indices[:remaining])
                    neg_indices = neg_indices[remaining:]

                if len(batch) >= self.batch_size // 2:  # At least half full
                    all_batches.append(batch)

        # Shuffle all batches
        random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        """Return estimated number of batches."""
        total_positive = sum(len(indices) for group, indices in self.entity_indices.items() if group != "negative")
        return max(1, total_positive // (self.batch_size // 2))


def compute_batch_statistics(batch_labels: torch.Tensor) -> dict[str, Any]:
    """
    Compute statistics for a batch of labels.

    Args:
        batch_labels: Tensor of shape [batch_size, seq_len]

    Returns:
        Dictionary with batch statistics
    """
    # Flatten and filter valid labels
    valid_labels = batch_labels[batch_labels != -100]

    if len(valid_labels) == 0:
        return {
            "total_tokens": 0,
            "entity_tokens": 0,
            "o_tokens": 0,
            "entity_ratio": 0.0,
            "unique_entities": 0,
            "entity_distribution": {},
        }

    # Count statistics
    total_tokens = len(valid_labels)
    o_tokens = (valid_labels == 0).sum().item()
    entity_tokens = total_tokens - o_tokens

    # Entity distribution
    entity_counts = {}
    for label in valid_labels.unique():
        if label > 0:
            entity_counts[label.item()] = (valid_labels == label).sum().item()

    return {
        "total_tokens": total_tokens,
        "entity_tokens": entity_tokens,
        "o_tokens": o_tokens,
        "entity_ratio": entity_tokens / total_tokens if total_tokens > 0 else 0.0,
        "unique_entities": len(entity_counts),
        "entity_distribution": entity_counts,
    }


class BatchCompositionLogger:
    """Logger for tracking batch composition statistics during training."""

    def __init__(self, log_every_n_batches: int = 100) -> None:
        """Initialize batch composition logger."""
        self.log_every_n_batches = log_every_n_batches
        self.batch_count = 0
        self.cumulative_stats: dict[str, Any] = {
            "total_entity_ratio": 0.0,
            "total_unique_entities": 0,
            "entity_type_counts": defaultdict(int),
        }

    def log_batch(self, batch_labels: torch.Tensor) -> None:
        """Log statistics for a batch."""
        self.batch_count += 1

        stats = compute_batch_statistics(batch_labels)

        # Update cumulative stats
        self.cumulative_stats["total_entity_ratio"] += stats["entity_ratio"]
        self.cumulative_stats["total_unique_entities"] += stats["unique_entities"]

        for entity, count in stats["entity_distribution"].items():
            self.cumulative_stats["entity_type_counts"][entity] += count

        # Log periodically
        if self.batch_count % self.log_every_n_batches == 0:
            avg_entity_ratio = self.cumulative_stats["total_entity_ratio"] / self.batch_count
            logger.info(
                f"Batch {self.batch_count} - "
                f"Avg entity ratio: {avg_entity_ratio:.3f}, "
                f"Current batch entity ratio: {stats['entity_ratio']:.3f}, "
                f"Unique entities in batch: {stats['unique_entities']}"
            )

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        if self.batch_count == 0:
            return {}

        return {
            "total_batches": self.batch_count,
            "avg_entity_ratio": self.cumulative_stats["total_entity_ratio"] / self.batch_count,
            "entity_type_distribution": dict(self.cumulative_stats["entity_type_counts"]),
        }
