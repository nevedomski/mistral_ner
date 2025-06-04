"""WikiNER/WikiANN dataset loader."""

from __future__ import annotations

import logging
from typing import Any

from datasets import DatasetDict, load_dataset

from ..base import BaseNERDataset

logger = logging.getLogger(__name__)


class WikiNERDataset(BaseNERDataset):
    """WikiNER/WikiANN dataset loader.

    WikiNER is a multilingual Named Entity Recognition dataset automatically
    annotated from Wikipedia. It contains 3 entity types: PER, ORG, LOC.

    Dataset: https://huggingface.co/datasets/wikiann
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize WikiNER dataset loader.

        Args:
            config: Optional configuration with:
                - language: Language code (default: "en")
        """
        super().__init__(config)
        self.language = self.config.get("language", "en")

    def load(self) -> DatasetDict:
        """Load WikiNER dataset.

        Returns:
            DatasetDict with train/validation/test splits
        """
        logger.info(f"Loading WikiNER dataset for language: {self.language}")

        try:
            # Load WikiNER/WikiANN dataset
            dataset = load_dataset("wikiann", self.language)

            # WikiNER has tokens and ner_tags already in the right format
            # No preprocessing needed for basic structure

            logger.info(f"WikiNER dataset loaded successfully for {self.language}")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load WikiNER dataset: {e}")
            raise

    def get_label_mapping(self) -> dict[str, str]:
        """Get label mapping from WikiNER to unified schema.

        WikiNER uses IOB2 format with tags:
        - O (0): Outside
        - B-PER (1): Beginning of Person
        - I-PER (2): Inside Person
        - B-ORG (3): Beginning of Organization
        - I-ORG (4): Inside Organization
        - B-LOC (5): Beginning of Location
        - I-LOC (6): Inside Location

        Returns:
            Mapping from WikiNER labels to unified labels
        """
        return {
            "O": "O",
            "B-PER": "B-PER",
            "I-PER": "I-PER",
            "B-ORG": "B-ORG",
            "I-ORG": "I-ORG",
            "B-LOC": "B-LOC",
            "I-LOC": "I-LOC",
        }

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Preprocess WikiNER examples.

        WikiNER is already in the correct format with tokens and ner_tags.

        Args:
            examples: Batch of examples

        Returns:
            Preprocessed examples
        """
        # WikiNER already has the correct structure
        # Just ensure consistency
        return examples
