"""Data loading and processing for Mistral NER fine-tuning."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from datasets import Dataset, DatasetDict
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Setup logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .config import Config


def validate_dataset(
    dataset: DatasetDict,
    config: Config,
    expected_labels: list[str],
    label_field: str = "multi_set_unified_tags",
    required_keys: tuple[str, ...] = ("train", "validation", "test"),
    required_features: tuple[str, ...] = ("tokens", "multi_set_unified_tags"),
) -> None:
    """Validate the dataset structure and labels."""

    for key in required_keys:
        if key not in dataset:
            raise ValueError(f"Dataset missing required split: {key}")

    # Check dataset features
    for split in required_keys:
        for feature in required_features:
            if feature not in dataset[split].features:
                raise ValueError(f"Dataset split '{split}' missing required '{feature}' feature")

    # Validate label consistency
    label_feature = dataset["train"].features[label_field]
    if hasattr(label_feature, "feature") and hasattr(label_feature.feature, "names"):
        dataset_labels = label_feature.feature.names
        if sorted(dataset_labels) != sorted(expected_labels):
            logger.warning(f"Label mismatch. Expected: {expected_labels}, Got: {dataset_labels}")


def tokenize_and_align_labels(
    examples: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
    label_all_tokens: bool = False,
    tag_field: str = "multi_set_unified_tags",
) -> dict[str, Any]:
    """
    Tokenize text and align labels with tokens.

    Args:
        examples: Batch of examples from the dataset
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        label_all_tokens: Whether to label all subword tokens or just the first

    Returns:
        Dictionary with tokenized inputs and aligned labels
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,
        max_length=max_length,
        is_split_into_words=True,
        return_tensors="pt",
    )

    labels = []
    for i, label in enumerate(examples[tag_field]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        # DOCDEV-TODO: Add validation for edge cases (empty word_ids, label length mismatch)
        # Risk: Silent data corruption if word_idx exceeds label bounds
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 label
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word gets the label
                label_ids.append(label[word_idx])
            else:
                # For other tokens of the word
                label_ids.append(label[word_idx] if label_all_tokens else -100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs  # type: ignore[no-any-return]


def create_data_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorForTokenClassification:
    """Create a data collator for token classification."""

    return DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, max_length=tokenizer.model_max_length, label_pad_token_id=-100
    )


def prepare_datasets(
    tokenizer: PreTrainedTokenizerBase, config: Config, dataset: DatasetDict
) -> tuple[Dataset, Dataset, Dataset, DataCollatorForTokenClassification]:
    """
    Prepare datasets for training.

    Args:
        tokenizer: Tokenizer to use
        config: Configuration object
        dataset: Optional pre-loaded dataset

    Returns:
        Tuple of (train_dataset, eval_dataset, test_dataset, data_collator)
    """

    # Validate dataset
    validate_dataset(dataset, config, config.data.label_names)

    # Create tokenization function with fixed parameters
    tokenize_fn: Callable[[dict[str, Any]], dict[str, Any]] = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        label_all_tokens=config.data.label_all_tokens,
        tag_field="multi_set_unified_tags",
    )

    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names, desc="Tokenizing")

    # Create data collator
    data_collator = create_data_collator(tokenizer)

    return (tokenized_datasets["train"], tokenized_datasets["validation"], tokenized_datasets["test"], data_collator)


def print_dataset_statistics(
    dataset: DatasetDict, tokenizer: PreTrainedTokenizerBase | None = None, tag_field: str = "multi_set_unified_tags"
) -> None:
    """Print statistics about the dataset."""
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    for split_name, split_data in dataset.items():
        split_name = str(split_name)
        print(f"\n{split_name.upper()} Split:")
        print(f"  Number of examples: {len(split_data)}")

        if "tokens" in split_data.features:
            # Calculate average sentence length
            lengths = [len(tokens) for tokens in split_data["tokens"]]
            avg_length = sum(lengths) / len(lengths)
            max_length = max(lengths)
            min_length = min(lengths)

            print(f"  Average tokens per example: {avg_length:.2f}")
            print(f"  Max tokens: {max_length}")
            print(f"  Min tokens: {min_length}")

        if tag_field in split_data.features:
            # Count entities using direct column access
            entity_counts: dict[int, int] = {}

            # Get all tags from the column at once
            all_tags = split_data[tag_field]  # This gets the entire column

            for tag_sequence in all_tags:  # Each tag_sequence is a list of tags for one example
                for tag in tag_sequence:
                    entity_counts[tag] = entity_counts.get(tag, 0) + 1

            print("  Entity distribution:")
            features = split_data.features[tag_field]
            if hasattr(features, "feature") and hasattr(features.feature, "names"):
                label_names = features.feature.names
                for tag_id, count in sorted(entity_counts.items()):
                    if tag_id < len(label_names):
                        print(f"    {label_names[tag_id]}: {count}")

    print("=" * 50 + "\n")


# def prepare_multi_datasets(
#     tokenizer: PreTrainedTokenizerBase, config: Config
# ) -> tuple[Dataset, Dataset, Dataset, DataCollatorForTokenClassification]:
#     """
#     Prepare multiple datasets for training.

#     Args:
#         tokenizer: Tokenizer to use
#         config: Configuration object with multi-dataset settings

#     Returns:
#         Tuple of (train_dataset, eval_dataset, test_dataset, data_collator)
#     """
#
#     logger.info("Preparing multi-dataset training...")

#     # Initialize components
#     registry = DatasetRegistry()
#     # label_mapper = LabelMapper(config.data.label_names)

#     # Load all requested datasets
#     all_datasets = []

#     for i, dataset in enumerate(config.data.datasets):
#         logger.info(f"Loading dataset {i + 1}/{len(config.data.datasets)}: {dataset.name}")

#         # # Get dataset-specific config
#         # dataset_config: dict[str, Any] = {
#         #     "language": (config.data.multi_dataset.filter_english and "English") or None,
#         #     "dataset_name": dataset_name,  # Used for profile lookup
#         # }

#         # Add label mapping configuration if available
#         if config.data.multi_dataset.label_mapping_profile:
#             # Use profile-based mapping
#             dataset_config["label_mapping"] = f"profile:{config.data.multi_dataset.label_mapping_profile}"
#         elif dataset_name in config.data.multi_dataset.label_mappings:
#             # Use dataset-specific mapping from config
#             mapping = config.data.multi_dataset.label_mappings[dataset_name]
#             # mapping can be either a dict or a string (file path)
#             dataset_config["label_mapping"] = mapping

#         # Load dataset with config
#         loader = registry.get_loader(dataset_name, dataset_config)
#         dataset = loader.load()

#         # Get label mapping for this dataset (will use config if provided)
#         label_mapping = loader.get_label_mapping()

#         # Apply label mapping to unify schema
#         logger.info(f"Mapping labels for {dataset_name}")
#         dataset = dataset.map(
#             lambda x, mapping=label_mapping: label_mapper.map_labels(x, mapping),
#             batched=True,
#             desc=f"Mapping labels for {dataset_name}",
#         )

#         # Apply dataset-specific preprocessing
#         dataset = dataset.map(loader.preprocess, batched=True, desc=f"Preprocessing {dataset_name}")

#         # Add dataset source identifier
#         dataset = dataset.map(lambda x, name=dataset_name: {"dataset_source": name}, desc=f"Adding source for {dataset_name}")

#         # Limit samples if configured
#         if config.data.multi_dataset.max_samples_per_dataset:
#             max_samples = config.data.multi_dataset.max_samples_per_dataset
#             for split in dataset:
#                 if len(dataset[split]) > max_samples:
#                     dataset[split] = dataset[split].select(range(max_samples))
#                     logger.info(f"Limited {split} split to {max_samples} samples")

#         all_datasets.append(dataset)

#     # Mix datasets according to strategy
#     logger.info(f"Mixing datasets using strategy: {config.data.multi_dataset.mixing_strategy}")
#     mixed_dataset = DatasetMixer.mix(
#         all_datasets,
#         strategy=config.data.multi_dataset.mixing_strategy,
#         weights=config.data.multi_dataset.dataset_weights,
#         seed=config.training.data_seed,
#     )

#     # Print statistics
#     logger.info("Mixed dataset statistics:")
#     for split, data in mixed_dataset.items():
#         logger.info(f"  {split}: {len(data)} examples")

#     # Create tokenization function
#     tokenize_fn: Callable[[dict[str, Any]], dict[str, Any]] = partial(
#         tokenize_and_align_labels,
#         tokenizer=tokenizer,
#         max_length=config.data.max_length,
#         label_all_tokens=config.data.label_all_tokens,
#         tag_field="multi_set_unified_tags",
#     )

#     # Tokenize mixed dataset
#     logger.info("Tokenizing mixed dataset...")
#     tokenized_datasets = mixed_dataset.map(tokenize_fn, batched=True, remove_columns=mixed_dataset["train"].column_names, desc="Tokenizing")

#     # Create data collator
#     data_collator = create_data_collator(tokenizer)

#     # Handle missing test split
#     test_dataset = tokenized_datasets.get("test", tokenized_datasets["validation"])

#     return (tokenized_datasets["train"], tokenized_datasets["validation"], test_dataset, data_collator)
