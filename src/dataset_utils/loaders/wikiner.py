"""WikiNER/WikiANN dataset loader."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic.dataclasses import dataclass

from ..base import BaseDataset

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class WikiNERDataset(BaseDataset):
    """WikiNER/WikiANN dataset loader.

    WikiNER is a multilingual Named Entity Recognition dataset automatically
    annotated from Wikipedia. It contains 3 entity types: PER, ORG, LOC.

    Dataset: https://huggingface.co/datasets/wikiann

    WikiNER uses IOB2 format with tags:
    - O (0): Outside
    - B-PER (1): Beginning of Person
    - I-PER (2): Inside Person
    - B-ORG (3): Beginning of Organization
    - I-ORG (4): Inside Organization
    - B-LOC (5): Beginning of Location
    - I-LOC (6): Inside Location
    """

    name: str = "wikiann"
    developer: str = "unimelb-nlp"
    filter_english: bool = True
    label_mappings: ClassVar[dict[str, str] | None] = {
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

    def _filter_english(self) -> None:
        """Filter the dataset to only include English examples."""
        self.config_args.append("en")
