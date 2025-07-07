"""Base class for NER dataset loaders."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from datasets import DatasetDict

logger = logging.getLogger("mistral_ner.datasets")


class BaseNERDataset(ABC):
    """Abstract base class for NER dataset loaders.

    Each dataset loader must implement:
    - load(): Load the dataset from source
    - get_label_mapping(): Map dataset labels to unified schema
    - preprocess(): Dataset-specific preprocessing
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize dataset loader with optional config."""
        self.config = config or {}
        self._label_mapping: dict[str, str] | None = None

    @abstractmethod
    def load(self) -> DatasetDict:
        """Load the dataset from source.

        Returns:
            DatasetDict with train/validation/test splits
        """
        pass

    def get_label_mapping(self) -> dict[str, str]:
        """Get mapping from dataset-specific labels to unified schema.

        Returns:
            Dictionary mapping original labels to unified labels
            Example: {"B-person": "B-PER", "I-person": "I-PER"}
        """
        # If mapping already loaded, return it
        if self._label_mapping is not None:
            return self._label_mapping

        # Try to load from config
        if "label_mapping" in self.config:
            self._label_mapping = self._load_label_mapping(self.config["label_mapping"])
            return self._label_mapping

        # Fall back to default mapping
        self._label_mapping = self.get_default_label_mapping()
        return self._label_mapping

    @abstractmethod
    def get_default_label_mapping(self) -> dict[str, str]:
        """Get default label mapping for this dataset.

        This method should be implemented by each dataset loader
        to provide the default/fallback mapping when no config is provided.

        Returns:
            Dictionary mapping original labels to unified labels
        """
        pass

    @abstractmethod
    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Apply dataset-specific preprocessing.

        Args:
            examples: Batch of examples from the dataset

        Returns:
            Preprocessed examples
        """
        pass

    def _load_label_mapping(self, mapping_config: dict[str, str] | str) -> dict[str, str]:
        """Load label mapping from various sources.

        Args:
            mapping_config: Can be:
                - str: Path to YAML file or profile name
                - dict: Direct mapping dictionary

        Returns:
            Dictionary mapping original labels to unified labels
        """
        if isinstance(mapping_config, dict):
            # Direct mapping provided
            return mapping_config

        if isinstance(mapping_config, str):
            # Check if it's a profile name
            if mapping_config.startswith("profile:"):
                profile_name = mapping_config.replace("profile:", "").strip()
                return self._load_mapping_profile(profile_name)

            # Otherwise treat as file path
            return self._load_mapping_file(mapping_config)

        raise ValueError(f"Invalid mapping config type: {type(mapping_config)}")

    def _load_mapping_profile(self, profile_name: str) -> dict[str, str]:
        """Load mapping from a predefined profile.

        Args:
            profile_name: Name of the profile (e.g., "bank_pii")

        Returns:
            Dictionary mapping for this dataset from the profile
        """
        try:
            from src.datasets.mapping_profiles import MappingProfiles

            profile = MappingProfiles.get_profile(profile_name)
            dataset_name = self.__class__.__name__.replace("Dataset", "").lower()

            # Try various name variations
            name_variations = [
                dataset_name,
                dataset_name.replace("_", ""),
                self.config.get("dataset_name", ""),
            ]

            for name in name_variations:
                if name in profile:
                    logger.info(f"Loaded mapping profile '{profile_name}' for dataset '{name}'")
                    dataset_mapping = profile[name]
                    assert isinstance(dataset_mapping, dict)  # Type narrowing for mypy
                    return dataset_mapping

            raise ValueError(f"No mapping found for dataset in profile '{profile_name}'")

        except ImportError as e:
            raise ImportError(f"Could not import MappingProfiles: {e}") from e

    def _load_mapping_file(self, file_path: str) -> dict[str, str]:
        """Load mapping from a YAML file.

        Args:
            file_path: Path to YAML file containing the mapping

        Returns:
            Dictionary mapping original labels to unified labels
        """
        path = Path(file_path)
        if not path.is_absolute():
            # Check in configs/mappings directory first
            config_path = Path("configs/mappings") / path
            if config_path.exists():
                path = config_path

        if not path.exists():
            raise FileNotFoundError(f"Mapping file not found: {path}")

        with open(path) as f:
            mapping = yaml.safe_load(f)

        if not isinstance(mapping, dict):
            raise ValueError(f"Invalid mapping file format: expected dict, got {type(mapping)}")

        logger.info(f"Loaded label mapping from file: {path}")
        return mapping

    def validate_dataset(self, dataset: DatasetDict) -> None:
        """Validate that dataset has required structure.

        Args:
            dataset: Dataset to validate

        Raises:
            ValueError: If dataset structure is invalid
        """
        required_splits = ["train", "validation", "test"]
        for split in required_splits:
            if split not in dataset:
                # Some datasets might not have test split
                if split == "test":
                    continue
                raise ValueError(f"Dataset missing required split: {split}")

            # Check for required features
            if "tokens" not in dataset[split].features:
                raise ValueError(f"Dataset split '{split}' missing 'tokens' feature")
            if "ner_tags" not in dataset[split].features:
                raise ValueError(f"Dataset split '{split}' missing 'ner_tags' feature")
