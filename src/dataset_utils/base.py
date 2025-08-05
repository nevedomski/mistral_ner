"""Base class for NER dataset loaders."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from datasets import ClassLabel, Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Sequence, Value, load_dataset
from pydantic import Field, computed_field
from pydantic.dataclasses import dataclass

from .converter import TagSchemeConverter

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class BaseDataset(ABC):
    """Abstract base class for NER dataset loaders.

    This class defines the interface and common attributes for all NER dataset loaders.
    Subclasses must implement:
      - load(): Load the dataset from its source and return a DatasetDict with splits.
      - preprocess(): Apply dataset-specific preprocessing to batches of examples.

    Attributes:
        name (str): Name of the dataset.
        weight (float): Weight for this dataset in multi-dataset training (0 to 1).
        filter_english (bool): Whether to filter for English-only samples.
        max_samples_per_dataset (int | None): Maximum number of samples to use from the dataset.
        label_mappings (dict[str, str] | None): Optional mapping from dataset labels to unified schema.
        id2label (Callable[[list[int]], list[str]] | None): Optional callable to convert integer tags to string labels.
    """

    name: str = ""
    developer: str = ""
    weight: float = Field(ge=0, le=1, default=1.0, description="Weight for this dataset in multi-dataset training")
    filter_english: bool = False
    max_samples_per_dataset: int | None = None
    trust_remote_code: bool = False
    label_mappings: ClassVar[dict[str, str] | None] = None
    id2label: Callable[[dict[str, list[list[str]]]], dict[str, list[list[str]]]] | None = None
    config_args: list[Any] = Field(default_factory=list, description="List of configuration arguments for the dataset loader")
    config_kwargs: dict[str, Any] = Field(default_factory=dict, description="Configuration settings for the dataset loader")
    rename_features: dict[str, str] = Field(default_factory=dict, description="Mapping of old feature names to new feature names")
    output_tagging_scheme: str = Field(default="BIO", description="Output tagging scheme (IO, IOB, IOB2, BIO, BIOES, BILOU)")

    @computed_field
    @property
    def full_name(self) -> str:
        """Return the full name of the dataset in the format 'developer/name'."""
        return f"{self.developer}/{self.name}" if self.developer else self.name

    def load_from_hub(self) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
        """Load the dataset from its source and return splits.

        Returns:
           DatasetDict: A dictionary with 'train', 'validation', and optionally 'test' splits.
        """
        # DOCDEV-NOTE: This method handles HuggingFace dataset loading with authentication and language filtering
        # DOCDEV-TODO: Add persistent cache to avoid re-downloading/processing datasets across runs
        logger.info(f"Loading {self.full_name} dataset from Hugging Face Hub...")

        if self.filter_english:
            logger.info("Filtering for English language data...")
            self._filter_english()

        if self.config_kwargs.get("use_auth_token"):
            logger.warning(
                f"{self.full_name} dataset are gated. You may need to provide a HuggingFace "
                "auth token "
                "Visit https://huggingface.co/settings/tokens to get your token."
            )

        try:
            # Load the dataset
            dataset = load_dataset(f"{self.full_name}", *self.config_args, trust_remote_code=self.trust_remote_code, **self.config_kwargs)

            logger.info(f"{self.full_name} dataset loaded from Hub or cache successfully")

            return dataset

        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                logger.error(
                    f"Authentication failed for {self.full_name} dataset. "
                    f"Please provide a valid HuggingFace token with access to: {self.full_name}"
                )
            else:
                logger.error(f"Failed to load {self.full_name} dataset: {e}")
            raise

    def split_into_train_validation_test(self, dataset: Dataset | DatasetDict) -> DatasetDict:
        """Split the dataset into train, validation, and test sets.

        Args:
            dataset (Dataset | DatasetDict): The dataset to split.

        Returns:
            DatasetDict: A dictionary with 'train', 'validation', and 'test' splits.
        """
        # DOCDEV-NOTE: Automatic split creation with 80/10/10 ratio using seed=42 for reproducibility
        if not (isinstance(dataset, DatasetDict) and all(split in dataset for split in ["train", "validation", "test"])):
            if isinstance(dataset, Dataset):
                logger.info("No train/test/validation splits found.")
                logger.info("Creating train/test/validation splits...")
                train_test = dataset.train_test_split(test_size=0.2, seed=42)
                test_val = train_test["test"].train_test_split(test_size=0.5, seed=42)
                dataset = DatasetDict({"train": train_test["train"], "validation": test_val["train"], "test": test_val["test"]})
            else:
                if "test" not in dataset and "validation" not in dataset:
                    logger.info("Creating test and validation split...")
                    train_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
                    test_val = train_test["test"].train_test_split(test_size=0.5, seed=42)
                    dataset = DatasetDict({"train": train_test["train"], "validation": test_val["train"], "test": test_val["test"]})
                elif "test" not in dataset:
                    logger.info("Creating test split...")
                    val_test = dataset["validation"].train_test_split(test_size=0.5, seed=42)
                    dataset = DatasetDict({"train": dataset["train"], "validation": val_test["train"], "test": val_test["test"]})
                else:
                    logger.info("Creating validation split...")
                    test_val = dataset["test"].train_test_split(test_size=0.5, seed=42)
                    dataset = DatasetDict({"train": dataset["train"], "validation": test_val["train"], "test": test_val["test"]})
        return dataset

    def relabel_to_class_label(self, dataset: DatasetDict | Dataset) -> DatasetDict:
        """Convert dataset labels to ClassLabel format.

        Args:
            dataset (DatasetDict): The dataset to convert.

        Returns:
            DatasetDict: The dataset with ClassLabel features.
        """
        dataset = dataset if isinstance(dataset, DatasetDict) else DatasetDict({"train": dataset})
        all_class_labels = all(isinstance(dataset[split].features["ner_tags"].feature, ClassLabel) for split in dataset)
        if all_class_labels:
            return dataset

        # Check if we have integer tags that need conversion
        has_integer_tags = all(
            isinstance(dataset[split].features["ner_tags"].feature, Value)
            and (
                dataset[split].features["ner_tags"].feature.dtype
                in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"]
            )
            for split in dataset
        )
        if has_integer_tags:
            if self.id2label is None:
                raise ValueError("Dataset has integer labels. id2label must be provided to convert integer tags to ClassLabel format")
            # DOCDEV-TODO: SECURITY VULNERABILITY - Missing input validation for id2label function
            # Add validation to ensure id2label is safe callable and validate its parameters
            # Risk: Potential arbitrary code execution or data corruption
            # DOCDEV-NOTE: Convert integer tags to string tags format using id2label
            # Apply the transformation permanently using map instead of lazy transforms
            dataset = dataset.cast_column("ner_tags", Sequence(Value("string")))
            for split in dataset:
                dataset[split] = dataset[split].map(self.id2label, batched=True, desc=f"Converting integer tags to strings for {split}")

        dataset = self._set_class_labels(dataset, column="ner_tags")
        logger.info("Converted dataset 'ner_tags' labels to ClassLabel format.")
        return dataset

    def load(self) -> DatasetDict:
        """Load the dataset from its source and return splits.

        Returns:
           DatasetDict: A dictionary with 'train', 'validation', and optionally 'test' splits.
        """

        dataset = self.load_from_hub()

        # DOCDEV-TODO: Implement proper IterableDataset support for streaming large datasets
        if isinstance(dataset, IterableDataset):
            raise NotImplementedError(
                f"{self.full_name} dataset is an IterableDataset with no train / test / validation splits. "
                "Please implement a custom loading method for IterableDatasets."
            )

        elif isinstance(dataset, IterableDatasetDict) and not all(split in dataset for split in ["train", "validation", "test"]):
            raise NotImplementedError(
                f"{self.full_name} dataset is an IterableDatasetDict with no train / test / validation splits. "
                "Please implement a custom loading method for IterableDatasetDicts."
            )

        elif isinstance(dataset, IterableDatasetDict):
            raise NotImplementedError(
                f"{self.full_name} dataset is an IterableDatasetDict with train / test / validation splits. Not implemented yet"
            )

        dataset = self.split_into_train_validation_test(dataset)
        logger.info(f"{self.full_name} dataset dictionary created successfully")

        # DOCDEV-TODO: MEMORY LEAK RISK - Large datasets loaded entirely into memory
        # Risk: OOM crashes with large datasets (Mendeley: 200k, AI4Privacy: 65k examples)
        # Implement proper memory management, streaming support, and cleanup mechanisms
        # Preprocess to ensure correct format
        dataset = dataset.map(self.preprocess, batched=True)
        logger.info(f"{self.full_name} dataset preprocessed successfully")

        # Apply feature renaming if specified
        if self.rename_features:
            logger.info(f"Renaming features: {self.rename_features}")
            dataset = dataset.rename_columns(self.rename_features)

        dataset = self.relabel_to_class_label(dataset)

        # DOCDEV-NOTE: Commented out IterableDataset conversion - revisit for memory optimization
        # dataset = IterableDatasetDict({split: dataset[split].to_iterable_dataset(num_shards=1024) for split in dataset})

        # Apply tag scheme conversion to ensure output format
        logger.info(f"Converting tags to {self.output_tagging_scheme} scheme")

        # Add updated single dataset tags
        dataset = self._convert_tag_scheme(dataset)

        # Apply label mappings to create unified tags
        # DOCDEV-NOTE: Unified tags enable cross-dataset consistency - all datasets map to same entity types
        dataset = self._apply_label_mappings(dataset)

        self.validate_dataset(dataset)
        logger.info(f"Validated {self.full_name} dataset successfully")
        self.dataset = dataset

        return dataset

    @abstractmethod
    def preprocess(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Apply dataset-specific preprocessing to a batch of examples.

        This method should convert the dataset examples into a standard format
        with 'tokens' and 'ner_tags' fields.

        Args:
            examples (dict[str, Any]): Batch of examples from the dataset.

        Returns:
            dict[str, Any]: Preprocessed examples.
        """
        pass

    def _filter_english(self) -> None:
        """Filter the dataset to only include English examples."""
        self.config_kwargs["language"] = "en"

    def _convert_tag_scheme(self, dataset: DatasetDict) -> DatasetDict:
        """Convert NER tags to the specified output tagging scheme.

        This method handles both integer and string tags, converting them to the desired tagging scheme.
        Args:
            examples: Batch of examples with ner_tags

        Returns:
            Examples with converted tags
        """
        # DOCDEV-NOTE: Tag conversion flow - original tags -> string tags -> converted tags -> ClassLabel format
        class_label = dataset["train"].features["ner_tags"].feature
        for split in dataset:
            if "ner_tags" not in dataset[split].features:
                raise ValueError(f"Split '{split}' does not contain 'ner_tags' feature")

            # Get tags for conversion
            tags = dataset[split]["ner_tags"]
            tags = [class_label.int2str(tag) for tag in tags]

            # Handle class label tags
            dataset[split] = dataset[split].add_column(
                "single_set_tags", TagSchemeConverter.convert(tags, to_scheme=self.output_tagging_scheme)
            )

        dataset = self._set_class_labels(dataset, column="single_set_tags")

        return dataset

    def _apply_label_mappings(self, dataset: DatasetDict) -> DatasetDict:
        """Apply label mappings to convert tags to unified schema.

        This method maps single_set_tags to single_set_unified_tags using
        the label_mappings defined in each loader.

        Args:
            dataset: Dataset with single_set_tags column

        Returns:
            Dataset with added single_set_unified_tags column
        """
        # DOCDEV-NOTE: Label mapping flow - single_set_tags -> apply mappings -> single_set_unified_tags
        if not self.label_mappings:
            # If no mappings defined, unified tags are same as converted tags
            logger.info("No label mappings defined, unified tags will be same as converted tags")
            for split in dataset:
                dataset[split] = dataset[split].add_column("single_set_unified_tags", dataset[split]["single_set_tags"])
        else:
            # Apply mappings
            class_label = dataset["train"].features["single_set_tags"].feature

            def map_to_unified(examples):
                """Map tags to unified schema using label mappings."""
                unified_tags = []
                for tags in examples["single_set_tags"]:
                    # Convert int tags to strings, apply mapping, handle unmapped
                    string_tags = [class_label.int2str(tag) for tag in tags]
                    # DOCDEV-NOTE: Tags not in mappings are preserved unchanged (defensive approach)
                    unified = [self.label_mappings.get(tag, tag) for tag in string_tags]
                    unified_tags.append(unified)
                return {"single_set_unified_tags": unified_tags}

            for split in dataset:
                dataset[split] = dataset[split].map(map_to_unified, batched=True, desc=f"Applying label mappings for {split}")

        # Convert to ClassLabel format
        dataset = self._set_class_labels(dataset, column="single_set_unified_tags")
        logger.info("Applied label mappings to create unified tags")

        return dataset

    def _set_class_labels(self, dataset: DatasetDict, column: str) -> DatasetDict:
        """Set the class labels for the dataset."""
        names = sorted(list(set(name for split in dataset for row in dataset[split][column] for name in row)))
        # DOCDEV-NOTE: Always put "O" tag first in ClassLabel for consistency
        if "O" in names:
            names.remove("O")
            names = ["O", *names]
        class_labels = ClassLabel(num_classes=len(names), names=names)
        for split in dataset:
            dataset[split] = dataset[split].map(
                lambda example: {column: [class_labels.str2int(tag) for tag in example]}, input_columns=column, batched=True
            )
            dataset[split] = dataset[split].cast_column(column, Sequence(class_labels))
        return dataset

    def validate_dataset(self, dataset: DatasetDict) -> None:
        """Validate that the dataset has the required splits and features.

        Args:
            dataset (DatasetDict): Dataset to validate.

        Raises:
            ValueError: If a required split or feature is missing.
        """
        # DOCDEV-NOTE: Three-level tag hierarchy for maximum flexibility:
        # 1. ner_tags: Original dataset tags (e.g., "person", "PERSON", "B-PER")
        # 2. single_set_tags: Tags converted to consistent scheme (e.g., all to BIO)
        # 3. single_set_unified_tags: Tags mapped to unified schema (e.g., all person variants to "B-PER")
        required_splits = ["train", "validation", "test"]
        for split in required_splits:
            if split not in dataset:
                # Some datasets might not have test split
                # if split == "test":
                #     continue
                raise ValueError(f"Dataset missing required split: {split}")

            # Check for required features
            # DOCDEV-NOTE: single_set_tags contains the converted tags for single-dataset training
            # DOCDEV-NOTE: single_set_unified_tags contains tags mapped to unified schema
            for feature in [
                "tokens",
                "single_set_tags",
                "single_set_unified_tags",
                "ner_tags",
            ]:
                if feature not in dataset[split].features:
                    raise ValueError(f"Dataset split '{split}' missing '{feature}' feature")

            # Check for required features types
            for feature in ["single_set_tags", "single_set_unified_tags", "ner_tags"]:
                if not isinstance(dataset[split][feature].features.feature, ClassLabel):
                    raise ValueError(f"Dataset split '{split}' feature '{feature}' must be of type List[ClassLabel]")
