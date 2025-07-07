"""Gretel AI Synthetic PII Finance dataset loader."""

from __future__ import annotations

import json
import logging
from typing import Any

from datasets import DatasetDict, load_dataset

from ..base import BaseNERDataset

logger = logging.getLogger(__name__)


class GretelPIIDataset(BaseNERDataset):
    """Loader for Gretel AI Multilingual Synthetic PII Finance dataset.

    This dataset contains synthetic financial documents with labeled PII:
    - 29 distinct PII types
    - Multiple languages (we filter for English)
    - Financial domain focus
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize with config."""
        super().__init__(config)
        self.language = config.get("language", "English") if config else "English"

    def load(self) -> DatasetDict:
        """Load Gretel PII dataset from HuggingFace."""
        logger.info(f"Loading Gretel PII dataset (language: {self.language})...")
        try:
            dataset = load_dataset("gretelai/synthetic_pii_finance_multilingual")

            # Filter for English only
            if self.language:
                dataset = dataset.filter(lambda x: x["language"] == self.language)

            # Convert to our expected format
            dataset = self._convert_to_ner_format(dataset)

            logger.info(f"Dataset loaded successfully. Keys: {dataset.keys()}")
            if "train" in dataset:
                logger.info(f"Train size: {len(dataset['train'])}")
            if "test" in dataset:
                logger.info(f"Test size: {len(dataset['test'])}")

            self.validate_dataset(dataset)
            return dataset
        except Exception as e:
            logger.error(f"Failed to load Gretel PII dataset: {e}")
            raise

    def _convert_to_ner_format(self, dataset: DatasetDict) -> DatasetDict:
        """Convert Gretel format to standard NER format with tokens and tags.

        Gretel dataset has:
        - generated_text: The full text
        - pii_spans: JSON string with PII annotations
        """

        def convert_example(example: dict[str, Any]) -> dict[str, Any]:
            text = example["generated_text"]

            # Parse PII spans
            try:
                pii_spans = json.loads(example["pii_spans"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pii_spans = []

            # Tokenize text (simple whitespace tokenization)
            tokens = text.split()

            # Create character to token mapping
            char_to_token = {}
            char_pos = 0
            for i, token in enumerate(tokens):
                for j in range(len(token)):
                    char_to_token[char_pos + j] = i
                char_pos += len(token) + 1  # +1 for space

            # Initialize tags
            tags = ["O"] * len(tokens)

            # Apply PII spans
            for span in pii_spans:
                start_char = span.get("start", 0)
                end_char = span.get("end", start_char + 1)
                label = span.get("label", "PII")

                # Map character positions to tokens
                start_token = char_to_token.get(start_char)
                end_token = char_to_token.get(end_char - 1)

                if start_token is not None and end_token is not None:
                    # Apply BIO tagging
                    tags[start_token] = f"B-{label}"
                    for i in range(start_token + 1, end_token + 1):
                        if i < len(tags):
                            tags[i] = f"I-{label}"

            return {"tokens": tokens, "ner_tags": tags, "id": example.get("index", 0)}

        # Convert all splits
        converted_splits = {}

        if "train" in dataset:
            train_data = dataset["train"].map(convert_example)
            # Split train into train/validation (90/10)
            train_size = int(0.9 * len(train_data))
            converted_splits["train"] = train_data.select(range(train_size))
            converted_splits["validation"] = train_data.select(range(train_size, len(train_data)))

        if "test" in dataset:
            converted_splits["test"] = dataset["test"].map(convert_example)

        return DatasetDict(converted_splits)

    def get_default_label_mapping(self) -> dict[str, str]:
        """Get label mapping for Gretel PII types to unified schema."""
        return {
            "O": "O",
            # Personal identifiers
            "B-PERSON": "B-PER",
            "I-PERSON": "I-PER",
            "B-NAME": "B-PER",
            "I-NAME": "I-PER",
            # Financial
            "B-CREDIT_CARD": "B-CARD",
            "I-CREDIT_CARD": "I-CARD",
            "B-BANK_ACCOUNT": "B-BANK",
            "I-BANK_ACCOUNT": "I-BANK",
            "B-IBAN": "B-BANK",
            "I-IBAN": "I-BANK",
            "B-SWIFT": "B-BANK",
            "I-SWIFT": "I-BANK",
            "B-ROUTING_NUMBER": "B-BANK",  # Add routing number mapping
            "I-ROUTING_NUMBER": "I-BANK",
            "B-ACCOUNT_NUMBER": "B-BANK",  # Add account number mapping
            "I-ACCOUNT_NUMBER": "I-BANK",
            # Government IDs
            "B-SSN": "B-SSN",
            "I-SSN": "I-SSN",
            "B-PASSPORT": "B-PASSPORT",
            "I-PASSPORT": "I-PASSPORT",
            "B-LICENSE": "B-LICENSE",
            "I-LICENSE": "I-LICENSE",
            "B-DRIVER_LICENSE": "B-LICENSE",  # Alternative naming
            "I-DRIVER_LICENSE": "I-LICENSE",
            # Contact
            "B-PHONE": "B-PHONE",
            "I-PHONE": "I-PHONE",
            "B-EMAIL": "B-EMAIL",
            "I-EMAIL": "I-EMAIL",
            "B-ADDRESS": "B-ADDR",
            "I-ADDRESS": "I-ADDR",
            # Locations -> ADDR for bank PII
            "B-CITY": "B-ADDR",
            "I-CITY": "I-ADDR",
            "B-STATE": "B-ADDR",
            "I-STATE": "I-ADDR",
            "B-COUNTRY": "B-ADDR",
            "I-COUNTRY": "I-ADDR",
            "B-ZIPCODE": "B-ADDR",
            "I-ZIPCODE": "I-ADDR",
            # Dates and numbers
            "B-DATE": "B-DATE",
            "I-DATE": "I-DATE",
            "B-DOB": "B-DATE",  # Date of birth -> DATE
            "I-DOB": "I-DATE",
            "B-DATE_OF_BIRTH": "B-DATE",  # Alternative naming
            "I-DATE_OF_BIRTH": "I-DATE",
            # Organizations
            "B-COMPANY": "B-ORG",
            "I-COMPANY": "I-ORG",
            "B-ORGANIZATION": "B-ORG",
            "I-ORGANIZATION": "I-ORG",
            # Monetary values
            "B-AMOUNT": "B-MONEY",
            "I-AMOUNT": "I-MONEY",
            "B-CURRENCY": "B-MONEY",
            "I-CURRENCY": "I-MONEY",
            # Generic PII
            "B-PII": "B-MISC",  # Generic PII -> MISC
            "I-PII": "I-MISC",
            "B-USERNAME": "B-MISC",
            "I-USERNAME": "I-MISC",
            "B-PASSWORD": "B-MISC",
            "I-PASSWORD": "I-MISC",
            "B-IP_ADDRESS": "B-MISC",
            "I-IP_ADDRESS": "I-MISC",
        }

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Gretel-specific preprocessing."""
        return examples
