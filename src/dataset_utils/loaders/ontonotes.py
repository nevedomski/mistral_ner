"""OntoNotes 5.0 dataset loader."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any, ClassVar

import requests
from pydantic import Field
from pydantic.dataclasses import dataclass

from ..base import BaseDataset

# Setup logging
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_ontonotes_id2label() -> dict[str, str]:
    """Fetch OntoNotes id2label mapping from the official dataset label.json.

    Returns a dictionary mapping integer IDs to entity labels.
    This is cached to avoid repeated network requests.
    """
    # DOCDEV-TODO: Add persistent file-based cache for id2label mapping to avoid network requests across runs
    # DOCDEV-TODO: SECURITY VULNERABILITY - Unvalidated HTTP requests
    # Add URL validation, certificate verification, and timeout limits
    # Risk: SSRF attacks, DNS poisoning, network-based vulnerabilities
    # Use raw URL to get the actual JSON content
    url = "https://huggingface.co/datasets/tner/ontonotes5/raw/main/dataset/label.json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        label_data = json.loads(response.text)
        # Convert string keys to integers
        return {str(v): str(k) for k, v in label_data.items()}
    except Exception as e:
        logger.error(f"Failed to fetch OntoNotes id2label mapping: {e}")
        # Fallback mapping based on known OntoNotes structure
        return {
            "0": "O",
            "1": "B-CARDINAL",
            "2": "B-DATE",
            "3": "I-DATE",
            "4": "B-PERSON",
            "5": "I-PERSON",
            "6": "B-NORP",
            "7": "B-GPE",
            "8": "I-GPE",
            "9": "B-LAW",
            "10": "I-LAW",
            "11": "B-ORG",
            "12": "I-ORG",
            "13": "B-PERCENT",
            "14": "I-PERCENT",
            "15": "B-ORDINAL",
            "16": "B-MONEY",
            "17": "I-MONEY",
            "18": "B-WORK_OF_ART",
            "19": "I-WORK_OF_ART",
            "20": "B-FAC",
            "21": "B-TIME",
            "22": "I-CARDINAL",
            "23": "B-LOC",
            "24": "B-QUANTITY",
            "25": "I-QUANTITY",
            "26": "I-NORP",
            "27": "I-LOC",
            "28": "B-PRODUCT",
            "29": "I-TIME",
            "30": "B-EVENT",
            "31": "I-EVENT",
            "32": "I-FAC",
            "33": "B-LANGUAGE",
            "34": "I-PRODUCT",
            "35": "I-ORDINAL",
            "36": "I-LANGUAGE",
        }


def id2label_(tag_lists: dict[str, list[list[str]]]) -> dict[str, list[list[str]]]:
    """Convert lists of integer tags to lists of string labels."""
    id2label_map = get_ontonotes_id2label()
    result = []
    for tags in tag_lists["ner_tags"]:
        result.append([id2label_map.get(tag, f"UNKNOWN_{tag}") for tag in tags])
    return {"ner_tags": result}


@dataclass
class OntoNotesDataset(BaseDataset):
    """Loader for OntoNotes 5.0 NER dataset.

    OntoNotes 5.0 contains 18 entity types:
    - PERSON, NORP (nationalities, religious, political groups)
    - FAC (facilities), ORG (organizations), GPE (geo-political entities)
    - LOC (locations), PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE
    - DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL

    Note: The original OntoNotes requires LDC license.
    We'll use the preprocessed version from HuggingFace.
    """

    name: str = "ontonotes5"
    developer: str = "tner"
    rename_features: dict[str, str] = Field(default_factory=lambda: {"tags": "ner_tags"}, description="Rename 'tags' column to 'ner_tags'")
    label_mappings: ClassVar[dict[str, str] | None] = {
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

    id2label: Callable[[dict[str, list[list[str]]]], dict[str, list[list[str]]]] | None = Field(default=id2label_)

    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """OntoNotes-specific preprocessing.

        The 'tags' to 'ner_tags' renaming is handled by rename_features.
        This method is for any additional preprocessing if needed.
        """
        # OntoNotes data is already in the correct format after renaming
        return examples
