"""WNUT-17 dataset loader."""

from __future__ import annotations

import logging
from typing import Any

from datasets import DatasetDict, load_dataset

from ..base import BaseNERDataset

logger = logging.getLogger(__name__)


class WNUTDataset(BaseNERDataset):
    """Loader for WNUT-17 Emerging Entities dataset.

    WNUT-17 focuses on identifying unusual, previously-unseen entities
    in noisy text. It contains 6 entity types:
    - person
    - location
    - corporation
    - product
    - creative-work
    - group
    """

    def load(self) -> DatasetDict:
        """Load WNUT-17 dataset from HuggingFace."""
        logger.info("Loading WNUT-17 dataset...")
        try:
            dataset = load_dataset("wnut_17")

            logger.info(f"Dataset loaded successfully. Keys: {dataset.keys()}")
            logger.info(f"Train size: {len(dataset['train'])}")
            logger.info(f"Validation size: {len(dataset['validation'])}")
            logger.info(f"Test size: {len(dataset['test'])}")

            self.validate_dataset(dataset)
            return dataset
        except Exception as e:
            logger.error(f"Failed to load WNUT-17 dataset: {e}")
            raise

    def get_default_label_mapping(self) -> dict[str, str]:
        """Get label mapping for WNUT-17 to unified schema."""
        return {
            "O": "O",
            # Person
            "B-person": "B-PER",
            "I-person": "I-PER",
            # Location
            "B-location": "B-LOC",
            "I-location": "I-LOC",
            # Corporation/Organization
            "B-corporation": "B-ORG",
            "I-corporation": "I-ORG",
            # Product
            "B-product": "B-PROD",
            "I-product": "I-PROD",
            # Creative work
            "B-creative-work": "B-ART",
            "I-creative-work": "I-ART",
            # Group (map to ORG for simplicity)
            "B-group": "B-ORG",
            "I-group": "I-ORG",
        }

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """WNUT-17 preprocessing.

        WNUT-17 data comes from noisy sources like Twitter, Reddit, etc.
        We might want to handle special tokens or normalize text here.
        """
        # For now, return as-is
        return examples
