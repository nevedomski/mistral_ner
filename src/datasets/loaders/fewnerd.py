"""Few-NERD dataset loader."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from datasets import DatasetDict, load_dataset

from ..base import BaseNERDataset

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FewNERDDataset(BaseNERDataset):
    """Loader for Few-NERD dataset.

    Few-NERD is a large-scale, fine-grained NER dataset with:
    - 8 coarse-grained types
    - 66 fine-grained types
    - 188,200 sentences

    We'll focus on the coarse-grained types for compatibility.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize with config."""
        super().__init__(config)
        # Default to coarse-grained supervised setting
        self.setting = config.get("setting", "supervised") if config else "supervised"
        self.use_fine_grained = config.get("use_fine_grained", False) if config else False

    def load(self) -> DatasetDict:
        """Load Few-NERD dataset from HuggingFace."""
        logger.info(f"Loading Few-NERD dataset (setting: {self.setting})...")
        try:
            # Few-NERD has different configurations
            dataset = load_dataset("DFKI-SLT/few-nerd", self.setting)

            logger.info(f"Dataset loaded successfully. Keys: {dataset.keys()}")
            if "train" in dataset:
                logger.info(f"Train size: {len(dataset['train'])}")
            if "validation" in dataset:
                logger.info(f"Validation size: {len(dataset['validation'])}")
            if "test" in dataset:
                logger.info(f"Test size: {len(dataset['test'])}")

            # Few-NERD uses IO tagging, we need to convert to BIO
            dataset = self._convert_io_to_bio(dataset)

            self.validate_dataset(dataset)
            return dataset
        except Exception as e:
            logger.error(f"Failed to load Few-NERD dataset: {e}")
            raise

    def _convert_io_to_bio(self, dataset: DatasetDict) -> DatasetDict:
        """Convert IO tagging to BIO tagging.

        Few-NERD uses IO tagging (no B- prefix), we need to add it.
        """

        def add_bio_tags(examples: dict[str, Any]) -> dict[str, Any]:
            new_tags = []
            for tags in examples["ner_tags"]:
                bio_tags = []
                prev_tag = "O"

                for tag in tags:
                    # Convert to string label first if needed
                    tag_str = examples.features["ner_tags"].feature.names[tag] if isinstance(tag, int) else tag  # type: ignore[attr-defined]

                    if tag_str == "O":
                        bio_tags.append(tag_str)
                    elif tag_str.startswith("I-"):
                        # Already has I- prefix
                        bio_tags.append(tag_str)
                    elif tag_str.startswith("B-"):
                        # Already has B- prefix
                        bio_tags.append(tag_str)
                    else:
                        # Need to add B- or I- prefix
                        entity_type = tag_str
                        if prev_tag == "O" or not prev_tag.endswith(entity_type):
                            bio_tags.append(f"B-{entity_type}")
                        else:
                            bio_tags.append(f"I-{entity_type}")

                    prev_tag = bio_tags[-1]

                new_tags.append(bio_tags)

            examples["ner_tags"] = new_tags
            return examples

        return dataset.map(add_bio_tags, batched=True)

    def get_default_label_mapping(self) -> dict[str, str]:
        """Get label mapping for Few-NERD coarse-grained types."""
        # Coarse-grained types mapping
        coarse_mapping = {
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

        if self.use_fine_grained:
            # Add fine-grained mappings
            # This would be a much larger mapping table
            logger.warning("Fine-grained mapping not fully implemented")

        return coarse_mapping

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Few-NERD preprocessing."""
        return examples
