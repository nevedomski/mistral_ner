"""AI4Privacy PII Masking dataset loader."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic.dataclasses import dataclass

from ..base import BaseDataset

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class AI4PrivacyDataset(BaseDataset):
    """AI4Privacy PII Masking dataset loader.

    AI4Privacy dataset contains 65k examples with extensive PII entity types
    including names, addresses, credit cards, emails, and many more.

    Dataset: https://huggingface.co/datasets/ai4privacy/pii-masking-65k
    """

    name: str = "pii-masking-65k"
    developer: str = "ai4privacy"
    label_mappings: ClassVar[dict[str, str] | None] = {
        "O": "O",
        # Personal names
        "B-PREFIX": "B-PER",
        "I-PREFIX": "I-PER",
        "B-SUFFIX": "B-PER",
        "I-SUFFIX": "I-PER",
        "B-FIRSTNAME": "B-PER",
        "I-FIRSTNAME": "I-PER",
        "B-MIDDLENAME": "B-PER",
        "I-MIDDLENAME": "I-PER",
        "B-LASTNAME": "B-PER",
        "I-LASTNAME": "I-PER",
        "B-FULLNAME": "B-PER",
        "I-FULLNAME": "I-PER",
        # Organizations
        "B-COMPANY_NAME": "B-ORG",
        "I-COMPANY_NAME": "I-ORG",
        # Locations - all map to ADDR for bank PII
        "B-CITY": "B-ADDR",
        "I-CITY": "I-ADDR",
        "B-STATE": "B-ADDR",
        "I-STATE": "I-ADDR",
        "B-STREET": "B-ADDR",
        "I-STREET": "I-ADDR",
        "B-BUILDINGNUMBER": "B-ADDR",
        "I-BUILDINGNUMBER": "I-ADDR",
        "B-COUNTY": "B-ADDR",
        "I-COUNTY": "I-ADDR",
        "B-ZIPCODE": "B-ADDR",
        "I-ZIPCODE": "I-ADDR",
        "B-SECONDARYADDRESS": "B-ADDR",
        "I-SECONDARYADDRESS": "I-ADDR",
        # Financial
        "B-CREDITCARDNUMBER": "B-CARD",
        "I-CREDITCARDNUMBER": "I-CARD",
        "B-CREDITCARDCVV": "B-CARD",
        "I-CREDITCARDCVV": "I-CARD",
        "B-CREDITCARDISSUER": "B-ORG",  # Card issuer is org
        "I-CREDITCARDISSUER": "I-ORG",
        "B-ACCOUNTNUMBER": "B-BANK",
        "I-ACCOUNTNUMBER": "I-BANK",
        "B-IBAN": "B-BANK",
        "I-IBAN": "I-BANK",
        "B-BIC": "B-BANK",
        "I-BIC": "I-BANK",
        "B-AMOUNT": "B-MONEY",
        "I-AMOUNT": "I-MONEY",
        "B-CURRENCY": "B-MONEY",
        "I-CURRENCY": "I-MONEY",
        "B-CURRENCYCODE": "B-MONEY",
        "I-CURRENCYCODE": "I-MONEY",
        "B-CURRENCYSYMBOL": "B-MONEY",
        "I-CURRENCYSYMBOL": "I-MONEY",
        "B-CURRENCYNAME": "B-MONEY",
        "I-CURRENCYNAME": "I-MONEY",
        # Contact info
        "B-EMAIL": "B-EMAIL",
        "I-EMAIL": "I-EMAIL",
        "B-PHONEIMEI": "B-PHONE",  # Phone IMEI -> phone
        "I-PHONEIMEI": "I-PHONE",
        "B-PHONE_NUMBER": "B-PHONE",
        "I-PHONE_NUMBER": "I-PHONE",
        # IDs
        "B-SSN": "B-SSN",
        "I-SSN": "I-SSN",
        "B-VEHICLEVIN": "B-MISC",  # VIN -> MISC
        "I-VEHICLEVIN": "I-MISC",
        "B-VEHICLEVRM": "B-MISC",
        "I-VEHICLEVRM": "I-MISC",
        # Dates and times
        "B-DATE": "B-DATE",
        "I-DATE": "I-DATE",
        "B-TIME": "B-TIME",
        "I-TIME": "I-TIME",
        "B-ACCOUNTNAME": "B-PER",  # Account holder name
        "I-ACCOUNTNAME": "I-PER",
        # Other
        "B-USERNAME": "B-MISC",
        "I-USERNAME": "I-MISC",
        "B-PASSWORD": "B-MISC",
        "I-PASSWORD": "I-MISC",
        "B-IP": "B-MISC",
        "I-IP": "I-MISC",
        "B-IPV4": "B-MISC",
        "I-IPV4": "I-MISC",
        "B-IPV6": "B-MISC",
        "I-IPV6": "I-MISC",
        "B-MAC": "B-MISC",
        "I-MAC": "I-MISC",
        "B-URL": "B-MISC",
        "I-URL": "I-MISC",
        "B-BITCOINADDRESS": "B-MISC",
        "I-BITCOINADDRESS": "I-MISC",
        "B-LITECOINADDRESS": "B-MISC",
        "I-LITECOINADDRESS": "I-MISC",
        "B-ETHEREUMADDRESS": "B-MISC",
        "I-ETHEREUMADDRESS": "I-MISC",
        "B-JOBAREA": "B-MISC",
        "I-JOBAREA": "I-MISC",
        "B-JOBTYPE": "B-MISC",
        "I-JOBTYPE": "I-MISC",
        "B-JOBTITLE": "B-MISC",
        "I-JOBTITLE": "I-MISC",
        "B-JOBDESCRIPTOR": "B-MISC",
        "I-JOBDESCRIPTOR": "I-MISC",
        "B-SEX": "B-MISC",
        "I-SEX": "I-MISC",
        "B-USERAGENT": "B-MISC",
        "I-USERAGENT": "I-MISC",
        "B-MASKEDNUMBER": "B-MISC",
        "I-MASKEDNUMBER": "I-MISC",
        "B-PIN": "B-MISC",
        "I-PIN": "I-MISC",
    }

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Preprocess AI4Privacy examples.

        Convert from AI4Privacy format to standard tokens/ner_tags format.

        Args:
            examples: Batch of examples

        Returns:
            Preprocessed examples with tokens and ner_tags
        """
        processed_examples: dict[str, list[Any]] = {"tokens": [], "ner_tags": []}

        for i in range(len(examples["tokenised_unmasked_text"])):
            tokens = examples["tokenised_unmasked_text"][i]

            # Extract BIO labels from token_entity_labels
            bio_labels: list[str] = []
            if "token_entity_labels" in examples and examples["token_entity_labels"][i]:
                # Parse the entity labels
                entity_labels = examples["token_entity_labels"][i]

                # Convert string representation to actual labels
                bio_labels = self._parse_entity_labels(entity_labels, len(tokens)) if isinstance(entity_labels, str) else entity_labels
            else:
                # Default to all O tags
                bio_labels = ["O"] * len(tokens)

            processed_examples["tokens"].append(tokens)
            processed_examples["ner_tags"].append(bio_labels)

        return processed_examples

    def _parse_entity_labels(self, entity_labels_str: str, num_tokens: int) -> list[str]:
        """Parse entity labels from string format.

        Args:
            entity_labels_str: String representation of entity labels
            num_tokens: Number of tokens

        Returns:
            List of BIO labels
        """
        # Default to all O tags
        bio_labels = ["O"] * num_tokens

        # Implement parsing logic based on actual format
        # This is a placeholder - adjust based on actual data format
        try:
            # DOCDEV-TODO: SECURITY VULNERABILITY - ast.literal_eval on untrusted data
            # Replace with safer JSON parsing or explicit validation
            # Risk: Code injection if dataset contains malicious strings
            # Try to parse as list of labels
            import ast

            labels = ast.literal_eval(entity_labels_str)
            if isinstance(labels, list) and len(labels) == num_tokens:
                bio_labels = labels
        except Exception:
            pass

        return bio_labels
