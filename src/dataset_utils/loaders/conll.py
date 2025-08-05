"""CoNLL-2003 dataset loader."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic.dataclasses import dataclass

from ..base import BaseDataset

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class CoNLLDataset(BaseDataset):
    """Loader for CoNLL-2003 NER dataset.

    The CoNLL-2003 dataset contains 4 entity types:
    - PER (Person)
    - ORG (Organization)
    - LOC (Location)
    - MISC (Miscellaneous)
    """

    name: str = "conll2003"
    developer: str = "eriktks"
    trust_remote_code: bool = True
    label_mappings: ClassVar[dict[str, str] | None] = {
        "O": "O",
        "B-PER": "B-PER",
        "I-PER": "I-PER",
        "B-ORG": "B-ORG",
        "I-ORG": "I-ORG",
        "B-LOC": "B-LOC",  # Default: preserve LOC
        "I-LOC": "I-LOC",  # Default: preserve LOC
        "B-MISC": "B-MISC",
        "I-MISC": "I-MISC",
    }

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """CoNLL-2003 doesn't need special preprocessing."""
        return examples
