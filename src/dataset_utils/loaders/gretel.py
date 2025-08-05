"""Gretel AI Synthetic PII Finance dataset loader."""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

from pydantic.dataclasses import dataclass

from ..base import BaseDataset

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class GretelPIIDataset(BaseDataset):
    """Loader for Gretel AI Multilingual Synthetic PII Finance dataset.

    This dataset contains synthetic financial documents with labeled PII:
    - 29 distinct PII types
    - Multiple languages (we filter for English)
    - Financial domain focus

    Dataset: https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual
    """

    name: str = "synthetic_pii_finance_multilingual"
    developer: str = "gretelai"
    filter_english: bool = True
    label_mappings: ClassVar[dict[str, str] | None] = {
        "O": "O",
        # Personal names
        "B-name": "B-PER",
        "I-name": "I-PER",
        "B-first_name": "B-PER",
        "I-first_name": "I-PER",
        # Contact information
        "B-email": "B-EMAIL",
        "I-email": "I-EMAIL",
        "B-phone_number": "B-PHONE",
        "I-phone_number": "I-PHONE",
        # Addresses and locations
        "B-street_address": "B-ADDR",
        "I-street_address": "I-ADDR",
        "B-local_latlng": "B-ADDR",
        "I-local_latlng": "I-ADDR",
        # Financial - Credit cards
        "B-credit_card_number": "B-CARD",
        "I-credit_card_number": "I-CARD",
        "B-credit_card_security_code": "B-CARD",
        "I-credit_card_security_code": "I-CARD",
        # Financial - Banking
        "B-iban": "B-BANK",
        "I-iban": "I-BANK",
        "B-bank_routing_number": "B-BANK",
        "I-bank_routing_number": "I-BANK",
        "B-bban": "B-BANK",
        "I-bban": "I-BANK",
        "B-swift_bic_code": "B-BANK",
        "I-swift_bic_code": "I-BANK",
        "B-account_pin": "B-BANK",
        "I-account_pin": "I-BANK",
        # Organizations
        "B-company": "B-ORG",
        "I-company": "I-ORG",
        # Dates and times
        "B-date": "B-DATE",
        "I-date": "I-DATE",
        "B-date_time": "B-DATE",
        "I-date_time": "I-DATE",
        "B-time": "B-TIME",
        "I-time": "I-TIME",
        "B-date_of_birth": "B-DATE",
        "I-date_of_birth": "I-DATE",
        # Employee and customer IDs
        "B-employee_id": "B-MISC",
        "I-employee_id": "I-MISC",
        "B-customer_id": "B-MISC",
        "I-customer_id": "I-MISC",
        # Network and technical
        "B-ipv4": "B-MISC",
        "I-ipv4": "I-MISC",
        "B-ipv6": "B-MISC",
        "I-ipv6": "I-MISC",
        "B-api_key": "B-MISC",
        "I-api_key": "I-MISC",
        # User credentials
        "B-user_name": "B-MISC",
        "I-user_name": "I-MISC",
        "B-password": "B-MISC",
        "I-password": "I-MISC",
    }

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Gretel-specific preprocessing.

        Handle conversion from Gretel format if needed.
        The base class load() will handle language filtering through config_kwargs.
        """
        # If the dataset has the expected format already, return as-is
        if "tokens" in examples and "ner_tags" in examples:
            return examples

        # Otherwise, we need to convert from Gretel format
        processed_examples: dict[str, list[Any]] = {"tokens": [], "ner_tags": []}

        # Handle both single example and batch
        num_examples = len(examples.get("generated_text", [])) if "generated_text" in examples else 0

        for i in range(num_examples):
            text = examples["generated_text"][i]

            # Parse PII spans
            try:
                pii_spans = json.loads(examples["pii_spans"][i])
            except (json.JSONDecodeError, KeyError, TypeError, IndexError):
                pii_spans = []

            # Tokenize text (simple whitespace tokenization)
            tokens = text.split()

            # Create character to token mapping
            char_to_token = {}
            char_pos = 0
            for idx, token in enumerate(tokens):
                for j in range(len(token)):
                    char_to_token[char_pos + j] = idx
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
                    for idx in range(start_token + 1, end_token + 1):
                        if idx < len(tags):
                            tags[idx] = f"I-{label}"

            processed_examples["tokens"].append(tokens)
            processed_examples["ner_tags"].append(tags)

        return processed_examples

    def _filter_english(self) -> None:
        """GretelPII dataset doesn't support language filtering via config."""
        # The dataset is already filtered for English in preprocessing
        pass
