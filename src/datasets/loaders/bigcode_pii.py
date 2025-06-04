"""BigCode PII dataset loader."""

from __future__ import annotations

import logging
from typing import Any

from datasets import DatasetDict, load_dataset

from ..base import BaseNERDataset

logger = logging.getLogger(__name__)


class BigCodePIIDataset(BaseNERDataset):
    """BigCode PII dataset loader.

    BigCode PII dataset is designed for detecting personally identifiable
    information in source code. It's part of the StarCoder project.

    Note: This dataset may require authentication/special access on HuggingFace.

    Datasets:
    - https://huggingface.co/datasets/bigcode/the-stack-pii (gated)
    - https://huggingface.co/datasets/bigcode/bigcode-pii-pjj (gated)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize BigCode PII dataset loader.

        Args:
            config: Optional configuration with:
                - dataset_name: Which BigCode dataset to use
                - use_auth_token: HuggingFace auth token for gated datasets
        """
        super().__init__(config)
        self.dataset_name = self.config.get("dataset_name", "bigcode/the-stack-pii")
        self.use_auth_token = self.config.get("use_auth_token", None)

    def load(self) -> DatasetDict:
        """Load BigCode PII dataset.

        Returns:
            DatasetDict with train/validation/test splits
        """
        logger.info(f"Loading BigCode PII dataset: {self.dataset_name}")

        if not self.use_auth_token:
            logger.warning(
                "BigCode PII datasets are gated. You may need to provide a HuggingFace "
                "auth token via 'use_auth_token' config parameter. "
                "Visit https://huggingface.co/settings/tokens to get your token."
            )

        try:
            # Try to load the dataset
            dataset = load_dataset(self.dataset_name, use_auth_token=self.use_auth_token)

            # Check if we need to create splits
            if isinstance(dataset, DatasetDict):
                if "train" in dataset and "validation" not in dataset:
                    # Create validation/test splits
                    dataset = self._create_splits(dataset)
            else:
                # Single split, create train/val/test
                dataset = self._create_splits({"train": dataset})

            # Preprocess to ensure correct format
            dataset = dataset.map(self.preprocess, batched=True, desc="Preprocessing BigCode PII dataset")

            logger.info("BigCode PII dataset loaded successfully")
            return dataset

        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                logger.error(
                    f"Authentication failed for BigCode PII dataset. "
                    f"Please provide a valid HuggingFace token with access to: {self.dataset_name}"
                )
            else:
                logger.error(f"Failed to load BigCode PII dataset: {e}")
            raise

    def _create_splits(self, dataset: dict[str, Any]) -> DatasetDict:
        """Create train/validation/test splits from a single dataset.

        Args:
            dataset: Dictionary with at least a 'train' key

        Returns:
            DatasetDict with train/validation/test splits
        """
        logger.info("Creating train/validation/test splits...")

        train_data = dataset["train"]

        # Split: 80% train, 10% validation, 10% test
        train_val_test = train_data.train_test_split(test_size=0.2, seed=42)
        val_test = train_val_test["test"].train_test_split(test_size=0.5, seed=42)

        return DatasetDict(
            {"train": train_val_test["train"], "validation": val_test["train"], "test": val_test["test"]}
        )

    def get_label_mapping(self) -> dict[str, str]:
        """Get label mapping from BigCode to unified schema.

        BigCode PII focuses on code-specific PII like keys, tokens, emails, etc.

        Returns:
            Mapping from BigCode labels to unified labels
        """
        return {
            "O": "O",
            # Personal identifiers in code
            "B-NAME": "B-PER",
            "I-NAME": "I-PER",
            "B-EMAIL": "B-EMAIL",
            "I-EMAIL": "I-EMAIL",
            "B-USERNAME": "B-MISC",
            "I-USERNAME": "I-MISC",
            # Secrets and keys
            "B-KEY": "B-MISC",
            "I-KEY": "I-MISC",
            "B-PASSWORD": "B-MISC",
            "I-PASSWORD": "I-MISC",
            "B-API_KEY": "B-MISC",
            "I-API_KEY": "I-MISC",
            "B-SECRET": "B-MISC",
            "I-SECRET": "I-MISC",
            "B-TOKEN": "B-MISC",
            "I-TOKEN": "I-MISC",
            # URLs and IPs
            "B-URL": "B-MISC",
            "I-URL": "I-MISC",
            "B-IP": "B-MISC",
            "I-IP": "I-MISC",
            "B-IP_ADDRESS": "B-MISC",
            "I-IP_ADDRESS": "I-MISC",
            # IDs
            "B-ID": "B-MISC",
            "I-ID": "I-MISC",
            "B-UUID": "B-MISC",
            "I-UUID": "I-MISC",
            # Paths (might contain usernames)
            "B-PATH": "B-MISC",
            "I-PATH": "I-MISC",
            "B-FILEPATH": "B-MISC",
            "I-FILEPATH": "I-MISC",
        }

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Preprocess BigCode examples.

        Convert from BigCode format to standard tokens/ner_tags format.

        Args:
            examples: Batch of examples

        Returns:
            Preprocessed examples with tokens and ner_tags
        """
        processed_examples: dict[str, list[Any]] = {"tokens": [], "ner_tags": []}

        # The exact format depends on the specific BigCode dataset
        # This is a general implementation that should work with most formats

        # Check for different possible field names
        token_fields = ["tokens", "input_tokens", "text_tokens", "code_tokens"]
        label_fields = ["labels", "ner_tags", "entity_labels", "pii_labels"]

        tokens_key = None
        labels_key = None

        # Find the actual field names
        for field in token_fields:
            if field in examples:
                tokens_key = field
                break

        for field in label_fields:
            if field in examples:
                labels_key = field
                break

        if not tokens_key:
            # Try to tokenize from text/code field
            if "text" in examples:
                for text in examples["text"]:
                    tokens = text.split()  # Simple whitespace tokenization
                    processed_examples["tokens"].append(tokens)
                    processed_examples["ner_tags"].append(["O"] * len(tokens))
            elif "code" in examples:
                for code in examples["code"]:
                    tokens = code.split()  # Simple whitespace tokenization
                    processed_examples["tokens"].append(tokens)
                    processed_examples["ner_tags"].append(["O"] * len(tokens))
            else:
                raise ValueError("Could not find tokens or text field in BigCode dataset")
        else:
            # Use existing tokens
            for i in range(len(examples[tokens_key])):
                tokens = examples[tokens_key][i]

                labels = examples[labels_key][i] if labels_key and labels_key in examples else ["O"] * len(tokens)

                processed_examples["tokens"].append(tokens)
                processed_examples["ner_tags"].append(labels)

        return processed_examples
