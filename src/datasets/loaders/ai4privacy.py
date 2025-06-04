"""AI4Privacy PII Masking dataset loader."""

from __future__ import annotations

import logging
from typing import Any

from datasets import DatasetDict, load_dataset

from ..base import BaseNERDataset

logger = logging.getLogger(__name__)


class AI4PrivacyDataset(BaseNERDataset):
    """AI4Privacy PII Masking dataset loader.

    AI4Privacy dataset contains 65k examples with extensive PII entity types
    including names, addresses, credit cards, emails, and many more.

    Dataset: https://huggingface.co/datasets/ai4privacy/pii-masking-65k
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize AI4Privacy dataset loader."""
        super().__init__(config)

    def load(self) -> DatasetDict:
        """Load AI4Privacy dataset.

        Returns:
            DatasetDict with train/validation/test splits
        """
        logger.info("Loading AI4Privacy PII Masking dataset...")

        try:
            # Load the dataset
            dataset = load_dataset("ai4privacy/pii-masking-65k")

            # The dataset only has a train split, we need to create validation/test splits
            if "train" in dataset and "validation" not in dataset:
                logger.info("Creating train/validation/test splits...")

                # Split: 80% train, 10% validation, 10% test
                train_val_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
                val_test = train_val_test["test"].train_test_split(test_size=0.5, seed=42)

                dataset = DatasetDict(
                    {"train": train_val_test["train"], "validation": val_test["train"], "test": val_test["test"]}
                )

            # Preprocess to ensure correct format
            dataset = dataset.map(self.preprocess, batched=True, desc="Preprocessing AI4Privacy dataset")

            logger.info("AI4Privacy dataset loaded successfully")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load AI4Privacy dataset: {e}")
            raise

    def get_label_mapping(self) -> dict[str, str]:
        """Get label mapping from AI4Privacy to unified schema.

        AI4Privacy has many PII entity types that map to our unified schema.

        Returns:
            Mapping from AI4Privacy labels to unified labels
        """
        return {
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
            # Organizations
            "B-COMPANY_NAME": "B-ORG",
            "I-COMPANY_NAME": "I-ORG",
            "B-COMPANYNAME": "B-ORG",
            "I-COMPANYNAME": "I-ORG",
            # Locations
            "B-CITY": "B-LOC",
            "I-CITY": "I-LOC",
            "B-STATE": "B-LOC",
            "I-STATE": "I-LOC",
            "B-COUNTRY": "B-LOC",
            "I-COUNTRY": "I-LOC",
            "B-STREET": "B-ADDR",
            "I-STREET": "I-ADDR",
            "B-BUILDINGNUMBER": "B-ADDR",
            "I-BUILDINGNUMBER": "I-ADDR",
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
            "B-IBAN": "B-BANK",
            "I-IBAN": "I-BANK",
            "B-BIC": "B-BANK",
            "I-BIC": "I-BANK",
            # Contact info
            "B-EMAIL": "B-EMAIL",
            "I-EMAIL": "I-EMAIL",
            "B-PHONEIMEI": "B-PHONE",
            "I-PHONEIMEI": "I-PHONE",
            "B-PHONENUMBER": "B-PHONE",
            "I-PHONENUMBER": "I-PHONE",
            # IDs
            "B-SSN": "B-SSN",
            "I-SSN": "I-SSN",
            "B-DRIVERLICENSE": "B-LICENSE",
            "I-DRIVERLICENSE": "I-LICENSE",
            "B-PASSPORT": "B-PASSPORT",
            "I-PASSPORT": "I-PASSPORT",
            # Dates and times
            "B-DATE": "B-DATE",
            "I-DATE": "I-DATE",
            "B-TIME": "B-TIME",
            "I-TIME": "I-TIME",
            # Other
            "B-USERNAME": "B-MISC",
            "I-USERNAME": "I-MISC",
            "B-PASSWORD": "B-MISC",
            "I-PASSWORD": "I-MISC",
            "B-IP": "B-MISC",
            "I-IP": "I-MISC",
            "B-URL": "B-MISC",
            "I-URL": "I-MISC",
            "B-JOBAREA": "B-MISC",
            "I-JOBAREA": "I-MISC",
            "B-JOBTITLE": "B-MISC",
            "I-JOBTITLE": "I-MISC",
            "B-JOBDESCRIPTOR": "B-MISC",
            "I-JOBDESCRIPTOR": "I-MISC",
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
            bio_labels = []
            if "token_entity_labels" in examples and examples["token_entity_labels"][i]:
                # Parse the entity labels
                entity_labels = examples["token_entity_labels"][i]

                # Convert string representation to actual labels
                if isinstance(entity_labels, str):
                    # Parse string format if needed
                    bio_labels = self._parse_entity_labels(entity_labels, len(tokens))
                else:
                    bio_labels = entity_labels
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
            # Try to parse as list of labels
            import ast

            labels = ast.literal_eval(entity_labels_str)
            if isinstance(labels, list) and len(labels) == num_tokens:
                bio_labels = labels
        except Exception:
            pass

        return bio_labels
