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

    def get_label_mapping(self) -> dict[str, str]:
        """Get label mapping for OntoNotes to unified schema."""
        return {
            "O": "O",
            # Person
            "B-PERSON": "B-PER",
            "I-PERSON": "I-PER",
            # Organizations
            "B-ORG": "B-ORG",
            "I-ORG": "I-ORG",
            # Locations - all map to ADDR for bank PII
            "B-LOC": "B-ADDR",
            "I-LOC": "I-ADDR",
            "B-GPE": "B-ADDR",  # Cities, countries -> address info
            "I-GPE": "I-ADDR",
            "B-FAC": "B-ADDR",  # Facilities -> address info
            "I-FAC": "I-ADDR",
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
            "B-QUANTITY": "O",  # Not relevant for bank PII
            "I-QUANTITY": "O",
            "B-ORDINAL": "O",   # Not relevant for bank PII
            "I-ORDINAL": "O",
            "B-CARDINAL": "O",  # Not relevant for bank PII
            "I-CARDINAL": "O",
            # Other
            "B-NORP": "B-MISC",  # Nationalities -> misc PII
            "I-NORP": "I-MISC",
            "B-PRODUCT": "B-MISC",  # Products -> misc (or could be ORG)
            "I-PRODUCT": "I-MISC",
            "B-EVENT": "O",     # Not relevant for bank PII
            "I-EVENT": "O",
            "B-WORK_OF_ART": "O",  # Not relevant for bank PII
            "I-WORK_OF_ART": "O",
            "B-LAW": "O",       # Not relevant for bank PII
            "I-LAW": "O",
            "B-LANGUAGE": "O",  # Not relevant for bank PII
            "I-LANGUAGE": "O",
        }

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """OntoNotes-specific preprocessing if needed."""
        # Handle different field names that might appear in OntoNotes
        if "sentences" in examples and "tokens" not in examples:
            examples["tokens"] = examples["sentences"]
        if "named_entities" in examples and "ner_tags" not in examples:
            examples["ner_tags"] = examples["named_entities"]
        return examples
