"""BigCode PII dataset loader."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import Field
from pydantic.dataclasses import dataclass

from ..base import BaseDataset

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class BigCodePIIDataset(BaseDataset):
    """BigCode PII dataset loader.

    BigCode PII dataset is designed for detecting personally identifiable
    information in source code. It's part of the StarCoder project.

    Note: This dataset may require authentication/special access on HuggingFace.

    Datasets:
    - https://huggingface.co/datasets/bigcode/the-stack-pii (gated)
    - https://huggingface.co/datasets/bigcode/bigcode-pii-pjj (gated)
    """

    name: str = "the-stack-pii"
    developer: str = "bigcode"
    config_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"use_auth_token": True}, description="Configuration settings for the BigCode PII dataset loader"
    )
    label_mappings: ClassVar[dict[str, str] | None] = {
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
