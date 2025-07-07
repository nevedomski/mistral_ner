"""OntoNotes 5.0 dataset loader."""

from __future__ import annotations

import logging
from typing import Any

from datasets import DatasetDict, load_dataset

from ..base import BaseNERDataset

logger = logging.getLogger(__name__)


class OntoNotesDataset(BaseNERDataset):
    """Loader for OntoNotes 5.0 NER dataset.

    OntoNotes 5.0 contains 18 entity types:
    - PERSON, NORP (nationalities, religious, political groups)
    - FAC (facilities), ORG (organizations), GPE (geo-political entities)
    - LOC (locations), PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE
    - DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL
    """

    def load(self) -> DatasetDict:
        """Load OntoNotes 5.0 dataset.

        Note: The original OntoNotes requires LDC license.
        We'll use the preprocessed version from HuggingFace.
        """
        logger.info("Loading OntoNotes 5.0 dataset...")
        try:
            # Try to load from HuggingFace
            # There are several versions available
            dataset = load_dataset("tner/ontonotes5")

            logger.info(f"Dataset loaded successfully. Keys: {dataset.keys()}")
            if "train" in dataset:
                logger.info(f"Train size: {len(dataset['train'])}")
            if "validation" in dataset:
                logger.info(f"Validation size: {len(dataset['validation'])}")
            if "test" in dataset:
                logger.info(f"Test size: {len(dataset['test'])}")

            self.validate_dataset(dataset)
            return dataset
        except Exception as e:
            logger.error(f"Failed to load OntoNotes dataset: {e}")
            # Try alternative source
            try:
                logger.info("Trying alternative OntoNotes source...")
                dataset = load_dataset("conll2012_ontonotesv5", "english_v12")
                return self._convert_conll2012_format(dataset)
            except Exception as e2:
                logger.error(f"Alternative source also failed: {e2}")
                raise

    def _convert_conll2012_format(self, dataset: DatasetDict) -> DatasetDict:
        """Convert CoNLL-2012 format to our expected format."""
        # Return the dataset as-is for now - it should already be in the right format
        return dataset

    def get_default_label_mapping(self) -> dict[str, str]:
        """Get default label mapping for OntoNotes to unified schema."""
        return {
            "O": "O",
            # Person
            "B-PERSON": "B-PER",
            "I-PERSON": "I-PER",
            # Organizations
            "B-ORG": "B-ORG",
            "I-ORG": "I-ORG",
            # Locations - default: preserve all location types
            "B-LOC": "B-LOC",
            "I-LOC": "I-LOC",
            "B-GPE": "B-GPE",  # Geo-political entities
            "I-GPE": "I-GPE",
            "B-FAC": "B-FAC",  # Facilities
            "I-FAC": "I-FAC",
            # Temporal
            "B-DATE": "B-DATE",
            "I-DATE": "I-DATE",
            "B-TIME": "B-TIME",
            "I-TIME": "I-TIME",
            # Numeric
            "B-MONEY": "B-MONEY",
            "I-MONEY": "I-MONEY",
            "B-PERCENT": "B-PERCENT",
            "I-PERCENT": "I-PERCENT",
            "B-QUANTITY": "B-QUANT",
            "I-QUANTITY": "I-QUANT",
            "B-ORDINAL": "B-ORD",
            "I-ORDINAL": "I-ORD",
            "B-CARDINAL": "B-CARD_NUM",
            "I-CARDINAL": "I-CARD_NUM",
            # Other
            "B-NORP": "B-NORP",  # Nationalities, religious, political groups
            "I-NORP": "I-NORP",
            "B-PRODUCT": "B-PROD",
            "I-PRODUCT": "I-PROD",
            "B-EVENT": "B-EVENT",
            "I-EVENT": "I-EVENT",
            "B-WORK_OF_ART": "B-ART",
            "I-WORK_OF_ART": "I-ART",
            "B-LAW": "B-LAW",
            "I-LAW": "I-LAW",
            "B-LANGUAGE": "B-LANG",
            "I-LANGUAGE": "I-LANG",
        }

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """OntoNotes-specific preprocessing if needed."""
        # Handle different field names that might appear in OntoNotes
        if "sentences" in examples and "tokens" not in examples:
            examples["tokens"] = examples["sentences"]
        if "named_entities" in examples and "ner_tags" not in examples:
            examples["ner_tags"] = examples["named_entities"]
        return examples
