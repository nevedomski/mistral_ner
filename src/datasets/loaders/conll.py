"""CoNLL-2003 dataset loader."""

from __future__ import annotations

import logging
from typing import Any

from datasets import DatasetDict, load_dataset

from ..base import BaseNERDataset

logger = logging.getLogger(__name__)


class CoNLLDataset(BaseNERDataset):
    """Loader for CoNLL-2003 NER dataset.

    The CoNLL-2003 dataset contains 4 entity types:
    - PER (Person)
    - ORG (Organization)
    - LOC (Location)
    - MISC (Miscellaneous)
    """

    def load(self) -> DatasetDict:
        """Load CoNLL-2003 dataset from HuggingFace."""
        logger.info("Loading CoNLL-2003 dataset...")
        try:
            dataset = load_dataset("conll2003")
            logger.info(f"Dataset loaded successfully. Keys: {dataset.keys()}")

            # Log sizes only for splits that exist
            if "train" in dataset:
                logger.info(f"Train size: {len(dataset['train'])}")
            if "validation" in dataset:
                logger.info(f"Validation size: {len(dataset['validation'])}")
            if "test" in dataset:
                logger.info(f"Test size: {len(dataset['test'])}")

            self.validate_dataset(dataset)
            return dataset
        except Exception as e:
            logger.error(f"Failed to load CoNLL-2003 dataset: {e}")
            raise

    def get_label_mapping(self) -> dict[str, str]:
        """Get label mapping for CoNLL-2003.

        CoNLL-2003 already uses a compatible schema, so minimal mapping needed.
        """
        return {
            "O": "O",
            "B-PER": "B-PER",
            "I-PER": "I-PER",
            "B-ORG": "B-ORG",
            "I-ORG": "I-ORG",
            "B-LOC": "B-ADDR",  # Map locations to addresses for bank PII
            "I-LOC": "I-ADDR",  # Map locations to addresses for bank PII
            "B-MISC": "B-MISC",
            "I-MISC": "I-MISC",
        }

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """CoNLL-2003 doesn't need special preprocessing."""
        return examples
