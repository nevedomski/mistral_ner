"""WNUT-17 dataset loader."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic.dataclasses import dataclass

from ..base import BaseDataset

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class WNUTDataset(BaseDataset):
    """Loader for WNUT-17 Emerging Entities dataset.

    WNUT-17 focuses on identifying unusual, previously-unseen entities
    in noisy text. It contains 6 entity types:
    - person
    - location
    - corporation
    - product
    - creative-work
    - group

    Dataset: https://huggingface.co/datasets/wnut_17
    """

    name: str = "wnut_17"
    developer: str = ""  # No developer prefix for wnut_17
    trust_remote_code: bool = True
    label_mappings: ClassVar[dict[str, str] | None] = {
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
