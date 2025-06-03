"""Base class for NER dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import DatasetDict


class BaseNERDataset(ABC):
    """Abstract base class for NER dataset loaders.

    Each dataset loader must implement:
    - load(): Load the dataset from source
    - get_label_mapping(): Map dataset labels to unified schema
    - preprocess(): Dataset-specific preprocessing
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize dataset loader with optional config."""
        self.config = config or {}

    @abstractmethod
    def load(self) -> DatasetDict:
        """Load the dataset from source.

        Returns:
            DatasetDict with train/validation/test splits
        """
        pass

    @abstractmethod
    def get_label_mapping(self) -> dict[str, str]:
        """Get mapping from dataset-specific labels to unified schema.

        Returns:
            Dictionary mapping original labels to unified labels
            Example: {"B-person": "B-PER", "I-person": "I-PER"}
        """
        pass

    @abstractmethod
    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Apply dataset-specific preprocessing.

        Args:
            examples: Batch of examples from the dataset

        Returns:
            Preprocessed examples
        """
        pass

    def validate_dataset(self, dataset: DatasetDict) -> None:
        """Validate that dataset has required structure.

        Args:
            dataset: Dataset to validate

        Raises:
            ValueError: If dataset structure is invalid
        """
        required_splits = ["train", "validation", "test"]
        for split in required_splits:
            if split not in dataset:
                # Some datasets might not have test split
                if split == "test":
                    continue
                raise ValueError(f"Dataset missing required split: {split}")

            # Check for required features
            if "tokens" not in dataset[split].features:
                raise ValueError(f"Dataset split '{split}' missing 'tokens' feature")
            if "ner_tags" not in dataset[split].features:
                raise ValueError(f"Dataset split '{split}' missing 'ner_tags' feature")
