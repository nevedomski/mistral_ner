"""Mendeley/Isotonic Synthetic PII dataset loader."""

from __future__ import annotations

import logging
from typing import Any

from datasets import DatasetDict, load_dataset

from ..base import BaseNERDataset

logger = logging.getLogger(__name__)


class MendeleyPIIDataset(BaseNERDataset):
    """Mendeley/Isotonic Synthetic PII dataset loader.

    This dataset contains 200k synthetic examples with PII entities
    in multiple languages. It includes various PII types like names,
    addresses, credit cards, usernames, etc.

    Dataset: https://huggingface.co/datasets/Isotonic/pii-masking-200k
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize Mendeley PII dataset loader."""
        super().__init__(config)
        self.filter_english = self.config.get("filter_english", True)

    def load(self) -> DatasetDict:
        """Load Mendeley/Isotonic PII dataset.

        Returns:
            DatasetDict with train/validation/test splits
        """
        logger.info("Loading Mendeley/Isotonic PII Masking dataset...")

        try:
            # Load the dataset
            dataset = load_dataset("Isotonic/pii-masking-200k")

            # Filter for English if requested
            if self.filter_english and "language" in dataset["train"].features:
                logger.info("Filtering for English language data...")
                dataset = dataset.filter(lambda x: x["language"] == "en", desc="Filtering English data")

            # The dataset only has a train split, create validation/test
            if "train" in dataset and "validation" not in dataset:
                logger.info("Creating train/validation/test splits...")

                # Split: 80% train, 10% validation, 10% test
                train_val_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
                val_test = train_val_test["test"].train_test_split(test_size=0.5, seed=42)

                dataset = DatasetDict(
                    {"train": train_val_test["train"], "validation": val_test["train"], "test": val_test["test"]}
                )

            # Preprocess to ensure correct format
            dataset = dataset.map(self.preprocess, batched=True, desc="Preprocessing Mendeley PII dataset")

            logger.info("Mendeley PII dataset loaded successfully")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load Mendeley PII dataset: {e}")
            raise

    def get_label_mapping(self) -> dict[str, str]:
        """Get label mapping from Mendeley to unified schema.

        Returns:
            Mapping from Mendeley labels to unified labels
        """
        return {
            "O": "O",
            # Personal names
            "B-PREFIX": "B-PER",
            "I-PREFIX": "I-PER",
            "B-FIRSTNAME": "B-PER",
            "I-FIRSTNAME": "I-PER",
            "B-LASTNAME": "B-PER",
            "I-LASTNAME": "I-PER",
            # Locations
            "B-CITY": "B-LOC",
            "I-CITY": "I-LOC",
            "B-STATE": "B-LOC",
            "I-STATE": "I-LOC",
            "B-COUNTRY": "B-LOC",
            "I-COUNTRY": "I-LOC",
            "B-STREET": "B-ADDR",
            "I-STREET": "I-ADDR",
            "B-ZIPCODE": "B-ADDR",
            "I-ZIPCODE": "I-ADDR",
            # Financial
            "B-CREDITCARDNUMBER": "B-CARD",
            "I-CREDITCARDNUMBER": "I-CARD",
            "B-CREDITCARDCVV": "B-CARD",
            "I-CREDITCARDCVV": "I-CARD",
            "B-CREDITCARDISSUER": "B-ORG",
            "I-CREDITCARDISSUER": "I-ORG",
            "B-ACCOUNTNUMBER": "B-BANK",
            "I-ACCOUNTNUMBER": "I-BANK",
            # IDs
            "B-SSN": "B-SSN",
            "I-SSN": "I-SSN",
            "B-DRIVERLICENSE": "B-LICENSE",
            "I-DRIVERLICENSE": "I-LICENSE",
            "B-PASSPORT": "B-PASSPORT",
            "I-PASSPORT": "I-PASSPORT",
            # Contact
            "B-EMAIL": "B-EMAIL",
            "I-EMAIL": "I-EMAIL",
            "B-PHONENUMBER": "B-PHONE",
            "I-PHONENUMBER": "I-PHONE",
            # Dates
            "B-DATE": "B-DATE",
            "I-DATE": "I-DATE",
            "B-TIME": "B-TIME",
            "I-TIME": "I-TIME",
            # Other
            "B-USERNAME": "B-MISC",
            "I-USERNAME": "I-MISC",
            "B-JOBTYPE": "B-MISC",
            "I-JOBTYPE": "I-MISC",
            "B-JOBAREA": "B-MISC",
            "I-JOBAREA": "I-MISC",
            "B-URL": "B-MISC",
            "I-URL": "I-MISC",
            "B-IP": "B-MISC",
            "I-IP": "I-MISC",
        }

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Preprocess Mendeley examples.

        Convert from Mendeley format to standard tokens/ner_tags format.

        Args:
            examples: Batch of examples

        Returns:
            Preprocessed examples with tokens and ner_tags
        """
        processed_examples: dict[str, list[Any]] = {"tokens": [], "ner_tags": []}

        for i in range(len(examples["tokenised_text"])):
            # Get tokens
            tokens = examples["tokenised_text"][i]

            # Get BIO labels
            if "bio_labels" in examples and examples["bio_labels"][i]:
                bio_labels = examples["bio_labels"][i]

                # Ensure the labels are in list format
                if isinstance(bio_labels, str):
                    # Try to parse if it's a string representation
                    bio_labels = self._parse_bio_labels(bio_labels, len(tokens))

                # Validate length matches
                if len(bio_labels) != len(tokens):
                    logger.warning(f"Label length mismatch: {len(bio_labels)} labels for {len(tokens)} tokens")
                    bio_labels = ["O"] * len(tokens)
            else:
                # Default to all O tags
                bio_labels = ["O"] * len(tokens)

            processed_examples["tokens"].append(tokens)
            processed_examples["ner_tags"].append(bio_labels)

        return processed_examples

    def _parse_bio_labels(self, bio_labels_str: str, num_tokens: int) -> list[str]:
        """Parse BIO labels from string format.

        Args:
            bio_labels_str: String representation of BIO labels
            num_tokens: Number of tokens

        Returns:
            List of BIO labels
        """
        # Default to all O tags
        bio_labels = ["O"] * num_tokens

        try:
            # Try to parse as list
            import ast

            labels = ast.literal_eval(bio_labels_str)
            if isinstance(labels, list) and len(labels) == num_tokens:
                bio_labels = labels
        except Exception:
            # Try splitting by common delimiters
            for delimiter in [" ", ",", "\t"]:
                parts = bio_labels_str.strip().split(delimiter)
                if len(parts) == num_tokens:
                    bio_labels = parts
                    break

        return bio_labels
