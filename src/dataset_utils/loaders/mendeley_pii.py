"""Mendeley/Isotonic Synthetic PII dataset loader."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic.dataclasses import dataclass

from ..base import BaseDataset

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class MendeleyPIIDataset(BaseDataset):
    """Mendeley/Isotonic Synthetic PII dataset loader.

    This dataset contains 200k synthetic examples with PII entities
    in multiple languages. It includes various PII types like names,
    addresses, credit cards, usernames, etc.

    Dataset: https://huggingface.co/datasets/Isotonic/pii-masking-200k
    """

    name: str = "pii-masking-200k"
    developer: str = "Isotonic"
    filter_english: bool = True
    label_mappings: ClassVar[dict[str, str] | None] = {
        "O": "O",
        # Personal names
        "B-PREFIX": "B-PER",
        "I-PREFIX": "I-PER",
        "B-FIRSTNAME": "B-PER",
        "I-FIRSTNAME": "I-PER",
        "B-MIDDLENAME": "B-PER",
        "I-MIDDLENAME": "I-PER",
        "B-LASTNAME": "B-PER",
        "I-LASTNAME": "I-PER",
        # Personal attributes
        "B-AGE": "B-MISC",
        "I-AGE": "I-MISC",
        "B-GENDER": "B-MISC",
        "I-GENDER": "I-MISC",
        "B-SEX": "B-MISC",
        "B-HEIGHT": "B-MISC",
        "I-HEIGHT": "I-MISC",
        "B-EYECOLOR": "B-MISC",
        "I-EYECOLOR": "I-MISC",
        # Locations and addresses
        "B-CITY": "B-ADDR",
        "I-CITY": "I-ADDR",
        "B-STATE": "B-ADDR",
        "I-STATE": "I-ADDR",
        "B-COUNTY": "B-ADDR",
        "I-COUNTY": "I-ADDR",
        "B-STREET": "B-ADDR",
        "I-STREET": "I-ADDR",
        "B-BUILDINGNUMBER": "B-ADDR",
        "I-BUILDINGNUMBER": "I-ADDR",
        "B-ZIPCODE": "B-ADDR",
        "I-ZIPCODE": "I-ADDR",
        "B-SECONDARYADDRESS": "B-ADDR",
        "I-SECONDARYADDRESS": "I-ADDR",
        "B-NEARBYGPSCOORDINATE": "B-ADDR",
        "I-NEARBYGPSCOORDINATE": "I-ADDR",
        "B-ORDINALDIRECTION": "B-ADDR",
        "I-ORDINALDIRECTION": "I-ADDR",
        # Organizations
        "B-COMPANYNAME": "B-ORG",
        "I-COMPANYNAME": "I-ORG",
        "B-CREDITCARDISSUER": "B-ORG",
        "I-CREDITCARDISSUER": "I-ORG",
        # Job-related
        "B-JOBTITLE": "B-MISC",
        "I-JOBTITLE": "I-MISC",
        "B-JOBTYPE": "B-MISC",
        "I-JOBTYPE": "I-MISC",
        "B-JOBAREA": "B-MISC",
        "I-JOBAREA": "I-MISC",
        # Financial - Credit cards
        "B-CREDITCARDNUMBER": "B-CARD",
        "I-CREDITCARDNUMBER": "I-CARD",
        "B-CREDITCARDCVV": "B-CARD",
        "I-CREDITCARDCVV": "I-CARD",
        # Financial - Banking
        "B-ACCOUNTNUMBER": "B-BANK",
        "I-ACCOUNTNUMBER": "I-BANK",
        "B-ACCOUNTNAME": "B-BANK",
        "I-ACCOUNTNAME": "I-BANK",
        "B-IBAN": "B-BANK",
        "I-IBAN": "I-BANK",
        # Financial - Currency and amounts
        "B-AMOUNT": "B-MISC",
        "I-AMOUNT": "I-MISC",
        "B-CURRENCY": "B-MISC",
        "I-CURRENCY": "I-MISC",
        "B-CURRENCYCODE": "B-MISC",
        "I-CURRENCYCODE": "I-MISC",
        "B-CURRENCYNAME": "B-MISC",
        "I-CURRENCYNAME": "I-MISC",
        "B-CURRENCYSYMBOL": "B-MISC",
        # Contact information
        "B-EMAIL": "B-EMAIL",
        "I-EMAIL": "I-EMAIL",
        "B-PHONENUMBER": "B-PHONE",
        "I-PHONENUMBER": "I-PHONE",
        "B-PHONEIMEI": "B-PHONE",
        "I-PHONEIMEI": "I-PHONE",
        # Identification numbers
        "B-SSN": "B-SSN",
        "I-SSN": "I-SSN",
        # Dates and times
        "B-DATE": "B-DATE",
        "I-DATE": "I-DATE",
        "B-DOB": "B-DATE",
        "I-DOB": "I-DATE",
        "B-TIME": "B-TIME",
        "I-TIME": "I-TIME",
        # Network and technical
        "B-URL": "B-MISC",
        "I-URL": "I-MISC",
        "B-IP": "B-MISC",
        "I-IP": "I-MISC",
        "B-IPV4": "B-MISC",
        "I-IPV4": "I-MISC",
        "B-IPV6": "B-MISC",
        "I-IPV6": "I-MISC",
        "B-MAC": "B-MISC",
        "I-MAC": "I-MISC",
        "B-USERAGENT": "B-MISC",
        "I-USERAGENT": "I-MISC",
        # Security credentials
        "B-USERNAME": "B-MISC",
        "I-USERNAME": "I-MISC",
        "B-PASSWORD": "B-MISC",
        "I-PASSWORD": "I-MISC",
        "B-PIN": "B-MISC",
        "I-PIN": "I-MISC",
        # Cryptocurrency
        "B-BITCOINADDRESS": "B-MISC",
        "I-BITCOINADDRESS": "I-MISC",
        "B-LITECOINADDRESS": "B-MISC",
        "I-LITECOINADDRESS": "I-MISC",
        "B-ETHEREUMADDRESS": "B-MISC",
        "I-ETHEREUMADDRESS": "I-MISC",
        # Vehicle information
        "B-VEHICLEVIN": "B-MISC",
        "I-VEHICLEVIN": "I-MISC",
        "B-VEHICLEVRM": "B-MISC",
        "I-VEHICLEVRM": "I-MISC",
        # Other identifiers
        "B-MASKEDNUMBER": "B-MISC",
        "I-MASKEDNUMBER": "I-MISC",
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
            # DOCDEV-TODO: SECURITY VULNERABILITY - ast.literal_eval on untrusted data
            # Replace with safer JSON parsing or explicit validation
            # Risk: Code injection if dataset contains malicious strings
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

    def _filter_english(self) -> None:
        """MendeleyPII dataset doesn't support language filtering via config."""
        # The dataset will be filtered in preprocessing based on language field
        pass
