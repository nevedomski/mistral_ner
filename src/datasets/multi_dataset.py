"""Multi-dataset loading and interleaving functionality."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from datasets import Dataset, interleave_datasets

from .label_mapper import LabelMapper
from .registry import DatasetRegistry

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger("mistral_ner")


class MultiDatasetLoader:
    """
    Wrapper for HuggingFace interleave_datasets with NER-specific features.

    This class handles loading multiple NER datasets, applying label mappings,
    and creating interleaved datasets for multi-dataset training.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the multi-dataset loader.

        Args:
            config: Configuration object containing multi-dataset settings
        """
        self.config = config
        self.multi_config = config.data.multi_dataset
        self.datasets: list[Dataset] = []
        self.dataset_names = self.multi_config.dataset_names
        self.weights = self.multi_config.dataset_weights
        self.strategy = self.multi_config.mixing_strategy
        self.label_mapper = LabelMapper(config.data.label_names)
        self.registry = DatasetRegistry()

        # Validate configuration
        self._validate_config()

        logger.info(
            f"MultiDatasetLoader initialized with {len(self.dataset_names)} datasets, strategy='{self.strategy}'"
        )

    def _validate_config(self) -> None:
        """Validate multi-dataset configuration."""
        if not self.dataset_names:
            raise ValueError("No datasets specified in multi_dataset.dataset_names")

        # Validate weights if provided
        if self.weights is not None:
            if len(self.weights) != len(self.dataset_names):
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match number of datasets ({len(self.dataset_names)})"
                )

            # Normalize weights
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]
        else:
            # Equal weights by default
            self.weights = [1.0 / len(self.dataset_names)] * len(self.dataset_names)

        # Validate strategy
        if self.strategy not in ["interleave", "weighted"]:
            raise ValueError(f"Invalid mixing strategy: {self.strategy}. Must be 'interleave' or 'weighted'")

    def _get_label_mapping(self, dataset_name: str) -> dict[str, str]:
        """Get label mapping for a specific dataset."""
        # Check if custom mapping is provided in config
        if self.multi_config.label_mappings and dataset_name in self.multi_config.label_mappings:
            mapping = self.multi_config.label_mappings[dataset_name]
            if isinstance(mapping, str):
                # It's a reference to a mapping profile
                from .mapping_profiles import MappingProfiles

                profile_name = mapping.upper()
                if hasattr(MappingProfiles, profile_name):
                    profile = getattr(MappingProfiles, profile_name)
                    mapping_dict = profile.get(dataset_name, {})
                    return dict(mapping_dict)  # Ensure it's a proper dict
                else:
                    logger.warning(f"Unknown mapping profile: {mapping}")
                    return {}
            return mapping

        # Check if a mapping profile is specified
        if self.multi_config.label_mapping_profile:
            from .mapping_profiles import MappingProfiles

            profile_name = self.multi_config.label_mapping_profile.upper()
            if hasattr(MappingProfiles, profile_name):
                profile = getattr(MappingProfiles, profile_name)
                mapping_dict = profile.get(dataset_name, {})
                return dict(mapping_dict)  # Ensure it's a proper dict
            else:
                logger.warning(f"Unknown mapping profile: {self.multi_config.label_mapping_profile}")
                return {}

        # Default: identity mapping (assume labels already match unified schema)
        return {}

    def load_and_prepare_datasets(
        self, split: str = "train", max_samples_per_dataset: int | None = None
    ) -> list[Dataset]:
        """
        Load individual datasets and apply label mappings.

        Args:
            split: Dataset split to load (train/validation/test)
            max_samples_per_dataset: Maximum samples per dataset (for memory management)

        Returns:
            List of prepared datasets
        """
        self.datasets = []

        for dataset_name in self.dataset_names:
            logger.info(f"Loading dataset: {dataset_name}")

            try:
                # Get loader from registry
                loader = self.registry.get_loader(dataset_name, config={"split": split})

                # Load the dataset
                dataset = loader.load()

                # Apply max samples limit if specified
                max_samples = max_samples_per_dataset or self.multi_config.max_samples_per_dataset
                if max_samples is not None and len(dataset) > max_samples:
                    logger.info(f"Limiting {dataset_name} from {len(dataset)} to {max_samples} samples")
                    dataset = dataset.select(range(max_samples))

                # Apply label mapping
                logger.info(f"Applying label mapping for {dataset_name}")

                # Get label mapping for this dataset
                label_mapping = self._get_label_mapping(dataset_name)

                # Map the dataset labels
                # Create a closure to capture label_mapping
                def map_function(examples: dict[str, Any], mapping: dict[str, str] = label_mapping) -> dict[str, Any]:
                    return self.label_mapper.map_labels(examples, mapping)

                mapped_dataset = dataset.map(map_function, batched=True, desc=f"Mapping labels for {dataset_name}")

                self.datasets.append(mapped_dataset)
                logger.info(f"Successfully loaded {dataset_name}: {len(mapped_dataset)} samples")

            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                raise

        return self.datasets

    def create_interleaved_dataset(self, datasets: list[Dataset] | None = None, seed: int | None = None) -> Dataset:
        """
        Create interleaved dataset using HuggingFace native functionality.

        Args:
            datasets: Optional list of datasets to interleave (uses self.datasets if None)
            seed: Random seed for reproducibility

        Returns:
            Interleaved dataset
        """
        datasets_to_interleave = datasets or self.datasets

        if not datasets_to_interleave:
            raise ValueError("No datasets loaded. Call load_and_prepare_datasets first.")

        # Use provided seed or from config
        seed = seed if seed is not None else self.config.training.seed

        # Log dataset statistics
        total_samples = sum(len(d) for d in datasets_to_interleave)
        logger.info(f"Creating interleaved dataset from {len(datasets_to_interleave)} datasets")
        logger.info(f"Total samples across all datasets: {total_samples}")
        if self.weights is not None:
            for i, (name, dataset) in enumerate(zip(self.dataset_names, datasets_to_interleave, strict=False)):
                logger.info(f"  {name}: {len(dataset)} samples (weight: {self.weights[i]:.2f})")

        # Create interleaved dataset based on strategy
        if self.strategy == "interleave":
            # Round-robin interleaving
            interleaved = interleave_datasets(
                datasets_to_interleave,
                probabilities=None,  # Equal probability for round-robin
                seed=seed,
                stopping_strategy="all_exhausted",  # Continue until all datasets exhausted
            )
            logger.info("Created interleaved dataset with round-robin strategy")

        elif self.strategy == "weighted":
            # Weighted sampling based on provided weights
            interleaved = interleave_datasets(
                datasets_to_interleave,
                probabilities=self.weights,
                seed=seed,
                stopping_strategy="first_exhausted",  # Stop when first dataset exhausted
            )
            logger.info(f"Created interleaved dataset with weighted strategy: {self.weights}")

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return interleaved

    def create_train_eval_datasets(
        self, train_split: str = "train", eval_split: str = "validation", max_samples_per_dataset: int | None = None
    ) -> tuple[Dataset, Dataset]:
        """
        Create both training and evaluation interleaved datasets.

        Args:
            train_split: Split name for training data
            eval_split: Split name for evaluation data
            max_samples_per_dataset: Maximum samples per dataset

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        # Load and prepare training datasets
        logger.info(f"Loading training datasets (split='{train_split}')")
        train_datasets = self.load_and_prepare_datasets(
            split=train_split, max_samples_per_dataset=max_samples_per_dataset
        )
        train_interleaved = self.create_interleaved_dataset(train_datasets)

        # Load and prepare evaluation datasets
        logger.info(f"Loading evaluation datasets (split='{eval_split}')")
        eval_datasets = self.load_and_prepare_datasets(
            split=eval_split, max_samples_per_dataset=max_samples_per_dataset
        )
        eval_interleaved = self.create_interleaved_dataset(eval_datasets)

        return train_interleaved, eval_interleaved

    def get_dataset_statistics(self) -> dict[str, Any]:
        """
        Get statistics about loaded datasets.

        Returns:
            Dictionary with dataset statistics
        """
        if not self.datasets:
            return {"error": "No datasets loaded"}

        stats: dict[str, Any] = {
            "num_datasets": len(self.datasets),
            "total_samples": sum(len(d) for d in self.datasets),
            "strategy": self.strategy,
            "weights": self.weights,
            "datasets": {},
        }

        for name, dataset in zip(self.dataset_names, self.datasets, strict=False):
            stats["datasets"][name] = {
                "num_samples": len(dataset),
                "features": list(dataset.features.keys()),
                "label_distribution": self._get_label_distribution(dataset),
            }

        return stats

    def _get_label_distribution(self, dataset: Dataset) -> dict[str, int]:
        """Get label distribution for a dataset."""
        from collections import Counter

        all_labels = []
        for example in dataset:
            # Check for both possible label field names
            label_field = "ner_tags" if "ner_tags" in example else "labels"
            if label_field in example:
                labels = example[label_field]
                # Filter out special tokens (-100)
                valid_labels = [label for label in labels if label != -100]
                all_labels.extend(valid_labels)

        label_counts = Counter(all_labels)

        # Convert to label names if possible
        if hasattr(self.config.data, "id2label") and self.config.data.id2label:
            return {
                self.config.data.id2label.get(label_id, f"LABEL_{label_id}"): count
                for label_id, count in label_counts.items()
            }

        return dict(label_counts)
