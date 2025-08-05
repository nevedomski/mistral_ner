"""Few-NERD dataset loader."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import Field
from pydantic.dataclasses import dataclass

from ..base import BaseDataset

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class FewNERDDataset(BaseDataset):
    """Loader for Few-NERD dataset.

    Few-NERD is a large-scale, fine-grained NER dataset with:
    - 8 coarse-grained types
    - 66 fine-grained types
    - 188,200 sentences

    We'll focus on the coarse-grained types for compatibility.
    """

    name: str = "few-nerd"
    developer: str = "DFKI-SLT"
    # Default to coarse-grained supervised setting
    setting: str = "supervised"
    trust_remote_code: bool = True
    config_args: list[str] = Field(
        default_factory=lambda: ["supervised"],
        description="List of configuration arguments for the Few-NERD dataset loader",
    )
    label_mappings: ClassVar[dict[str, str] | None] = {
        "O": "O",
        # Person
        "B-person": "B-PER",
        "I-person": "I-PER",
        # Location
        "B-location": "B-LOC",
        "I-location": "I-LOC",
        # Organization
        "B-organization": "B-ORG",
        "I-organization": "I-ORG",
        # Miscellaneous becomes MISC
        "B-miscellaneous": "B-MISC",
        "I-miscellaneous": "I-MISC",
        # Other becomes MISC
        "B-other": "B-MISC",
        "I-other": "I-MISC",
        # Building -> Facility
        "B-building": "B-FAC",
        "I-building": "I-FAC",
        # Art
        "B-art": "B-ART",
        "I-art": "I-ART",
        # Product
        "B-product": "B-PROD",
        "I-product": "I-PROD",
        # Event
        "B-event": "B-EVENT",
        "I-event": "I-EVENT",
    }

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Pass through preprocessing.

        Tag scheme conversion is now handled by the generic TagSchemeConverter
        in BaseDataset, so we just return the examples as-is.
        """
        return examples
