"""Unified label schema and mapping utilities for multi-dataset NER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UnifiedLabelSchema:
    """Unified label schema supporting all dataset entity types.

    This schema covers:
    - Traditional NER entities (PER, ORG, LOC)
    - Extended entities (DATE, MONEY, EVENT, etc.)
    - PII-specific entities (CARD, SSN, PHONE, etc.)
    """

    # Core entities (present in most datasets)
    PERSON: str = "PER"
    ORGANIZATION: str = "ORG"
    LOCATION: str = "LOC"
    MISCELLANEOUS: str = "MISC"

    # Extended entities (OntoNotes, Few-NERD)
    DATE: str = "DATE"
    TIME: str = "TIME"
    MONEY: str = "MONEY"
    PERCENT: str = "PERCENT"
    FACILITY: str = "FAC"
    GPE: str = "GPE"  # Geopolitical entity
    PRODUCT: str = "PROD"
    EVENT: str = "EVENT"
    WORK_OF_ART: str = "ART"
    LANGUAGE: str = "LANG"
    QUANTITY: str = "QUANT"
    ORDINAL: str = "ORD"
    CARDINAL: str = "CARD_NUM"  # Not credit card
    NORP: str = "NORP"  # Nationalities, religious, political groups

    # PII-specific entities
    CREDIT_CARD: str = "CARD"
    SSN: str = "SSN"
    PHONE: str = "PHONE"
    EMAIL: str = "EMAIL"
    ADDRESS: str = "ADDR"
    BANK_ACCOUNT: str = "BANK"
    PASSPORT: str = "PASSPORT"
    DRIVER_LICENSE: str = "LICENSE"

    # Few-NERD specific
    ANIMAL: str = "ANIM"
    BIOLOGICAL: str = "BIO"
    CELESTIAL_BODY: str = "CELEST"
    DISEASE: str = "DIS"
    FOOD: str = "FOOD"
    INSTRUMENT: str = "INST"
    MEDIA: str = "MEDIA"
    PLANT: str = "PLANT"
    MYTHOLOGICAL: str = "MYTH"
    VEHICLE: str = "VEHI"

    @classmethod
    def get_bio_labels(cls) -> list[str]:
        """Get all labels in BIO format including O tag."""
        bio_labels = []

        # Get all entity types from class attributes
        for attr, value in cls.__dict__.items():
            if not attr.startswith("_") and isinstance(value, str):
                bio_labels.extend([f"B-{value}", f"I-{value}"])

        # Sort the BIO labels but keep O at the front
        sorted_bio_labels = sorted(set(bio_labels))
        return ["O", *sorted_bio_labels]


class LabelMapper:
    """Maps dataset-specific labels to unified schema."""

    def __init__(self, unified_labels: list[str] | None = None):
        """Initialize label mapper.

        Args:
            unified_labels: List of unified labels to use. If None, uses default schema.
        """
        self.unified_labels = unified_labels or UnifiedLabelSchema.get_bio_labels()
        self.label2id = {label: i for i, label in enumerate(self.unified_labels)}
        self.id2label = {i: label for i, label in enumerate(self.unified_labels)}

    def map_labels(
        self, examples: dict[str, Any], label_mapping: dict[str, str], label_field: str = "ner_tags"
    ) -> dict[str, Any]:
        """Map dataset-specific labels to unified schema.

        Args:
            examples: Batch of examples with labels
            label_mapping: Mapping from original to unified labels
            label_field: Name of the label field in examples

        Returns:
            Examples with mapped labels
        """
        # Get original label names if available
        label_data = examples[label_field]
        original_labels = None

        # Try to get feature names if this is a HuggingFace dataset with features
        if hasattr(label_data, "feature") and hasattr(label_data.feature, "names"):
            original_labels = label_data.feature.names
        # Also try checking if examples has features metadata
        elif hasattr(examples, "features") and label_field in examples.features:
            feature = examples.features[label_field]
            if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
                original_labels = feature.feature.names

        mapped_tags = []
        for tags in examples[label_field]:
            mapped_example = []
            for tag in tags:
                # Convert tag ID to label name if needed
                tag_name = original_labels[tag] if original_labels and isinstance(tag, int) else tag

                # Map to unified label
                unified_tag = label_mapping.get(tag_name, tag_name)

                # Convert to ID in unified schema
                mapped_id = self.label2id[unified_tag] if unified_tag in self.label2id else self.label2id["O"]

                mapped_example.append(mapped_id)
            mapped_tags.append(mapped_example)

        examples[label_field] = mapped_tags
        return examples

    def get_label_distribution(self, dataset: list[dict[str, Any]]) -> dict[str, int]:
        """Get distribution of labels in a dataset.

        Args:
            dataset: Dataset to analyze

        Returns:
            Dictionary mapping labels to counts
        """
        label_counts = {label: 0 for label in self.unified_labels}

        for example in dataset:
            for tag_id in example["ner_tags"]:
                label = self.id2label.get(tag_id, "O")
                label_counts[label] += 1

        return label_counts
