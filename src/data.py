"""Data loading and processing for Mistral NER fine-tuning."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from transformers import PreTrainedTokenizerBase

from datasets import Dataset, DatasetDict, load_dataset

if TYPE_CHECKING:
    from transformers import DataCollatorForTokenClassification

    from .config import Config

# Import multi-dataset components
try:
    from .datasets import DatasetMixer, DatasetRegistry, LabelMapper

    MULTI_DATASET_AVAILABLE = True
except ImportError:
    MULTI_DATASET_AVAILABLE = False

logger = logging.getLogger("mistral_ner")


def load_conll2003_dataset() -> DatasetDict:
    """Load the CoNLL-2003 dataset."""
    try:
        logger.info("Loading CoNLL-2003 dataset...")
        dataset = load_dataset("conll2003")
        logger.info(f"Dataset loaded successfully. Keys: {dataset.keys()}")
        logger.info(f"Train size: {len(dataset['train'])}")
        logger.info(f"Validation size: {len(dataset['validation'])}")
        logger.info(f"Test size: {len(dataset['test'])}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load CoNLL-2003 dataset: {e}")
        raise


def validate_dataset(dataset: DatasetDict, expected_labels: list[str]) -> None:
    """Validate the dataset structure and labels."""
    required_keys = ["train", "validation", "test"]
    for key in required_keys:
        if key not in dataset:
            raise ValueError(f"Dataset missing required split: {key}")

    # Check dataset features
    for split in required_keys:
        if "tokens" not in dataset[split].features:
            raise ValueError(f"Dataset split '{split}' missing 'tokens' feature")
        if "ner_tags" not in dataset[split].features:
            raise ValueError(f"Dataset split '{split}' missing 'ner_tags' feature")

    # Validate label consistency
    label_feature = dataset["train"].features["ner_tags"]
    if hasattr(label_feature, "feature") and hasattr(label_feature.feature, "names"):
        dataset_labels = label_feature.feature.names
        if dataset_labels != expected_labels:
            logger.warning(f"Label mismatch. Expected: {expected_labels}, Got: {dataset_labels}")


def tokenize_and_align_labels(
    examples: dict[str, Any], tokenizer: PreTrainedTokenizerBase, max_length: int = 256, label_all_tokens: bool = False
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
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

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
    from transformers import DataCollatorForTokenClassification

    return DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, max_length=tokenizer.model_max_length, label_pad_token_id=-100
    )


def prepare_datasets(
    tokenizer: PreTrainedTokenizerBase, config: Config, dataset: DatasetDict | None = None
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
    # Check if multi-dataset mode is enabled
    if config.data.multi_dataset.enabled:
        if not MULTI_DATASET_AVAILABLE:
            raise ImportError("Multi-dataset components not available. Check installation.")
        return prepare_multi_datasets(tokenizer, config)

    # Single dataset mode (backward compatibility)
    # Load dataset if not provided
    if dataset is None:
        dataset = load_conll2003_dataset()

    # Validate dataset
    validate_dataset(dataset, config.data.label_names)

    # Create tokenization function with fixed parameters
    tokenize_fn: Callable[[dict[str, Any]], dict[str, Any]] = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        label_all_tokens=config.data.label_all_tokens,
    )

    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset["train"].column_names, desc="Tokenizing"
    )

    # Create data collator
    data_collator = create_data_collator(tokenizer)

    return (tokenized_datasets["train"], tokenized_datasets["validation"], tokenized_datasets["test"], data_collator)


def create_sample_dataset(size: int = 100, config: Config | None = None) -> DatasetDict:
    """Create a small sample dataset for testing."""

    if config is None:
        label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    else:
        label_names = config.data.label_names

    # Use label_names for validation but not required for the sample generation
    _ = label_names

    def generate_sample(idx: int) -> dict[str, Any]:
        tokens = ["This", "is", "a", "test", "sentence", "with", "John", "Smith", "in", "New", "York", "."]
        ner_tags = [0, 0, 0, 0, 0, 0, 1, 2, 0, 5, 6, 0]  # Example tags

        return {"id": str(idx), "tokens": tokens, "ner_tags": ner_tags}

    # Generate samples
    train_samples = [generate_sample(i) for i in range(size)]
    val_samples = [generate_sample(i + size) for i in range(size // 5)]
    test_samples = [generate_sample(i + size + size // 5) for i in range(size // 5)]

    # Create datasets
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_samples),
            "validation": Dataset.from_list(val_samples),
            "test": Dataset.from_list(test_samples),
        }
    )

    return dataset_dict


def get_label_list(dataset: DatasetDict) -> list[str]:
    """Extract label list from dataset."""
    features = dataset["train"].features["ner_tags"]
    if hasattr(features, "feature") and hasattr(features.feature, "names"):
        return features.feature.names  # type: ignore[no-any-return]
    else:
        # Fallback to default CoNLL-2003 labels
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def print_dataset_statistics(dataset: DatasetDict, tokenizer: PreTrainedTokenizerBase | None = None) -> None:
    """Print statistics about the dataset."""
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    for split_name, split_data in dataset.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Number of examples: {len(split_data)}")

        if "tokens" in split_data.features:
            # Calculate average sentence length
            lengths = [len(ex["tokens"]) for ex in split_data]
            avg_length = sum(lengths) / len(lengths)
            max_length = max(lengths)
            min_length = min(lengths)

            print(f"  Average tokens per example: {avg_length:.2f}")
            print(f"  Max tokens: {max_length}")
            print(f"  Min tokens: {min_length}")

        if "ner_tags" in split_data.features:
            # Count entities
            entity_counts: dict[int, int] = {}
            for ex in split_data:
                for tag in ex["ner_tags"]:
                    entity_counts[tag] = entity_counts.get(tag, 0) + 1

            print("  Entity distribution:")
            features = split_data.features["ner_tags"]
            if hasattr(features, "feature") and hasattr(features.feature, "names"):
                label_names = features.feature.names
                for tag_id, count in sorted(entity_counts.items()):
                    if tag_id < len(label_names):
                        print(f"    {label_names[tag_id]}: {count}")

    print("=" * 50 + "\n")


def prepare_multi_datasets(
    tokenizer: PreTrainedTokenizerBase, config: Config
) -> tuple[Dataset, Dataset, Dataset, DataCollatorForTokenClassification]:
    """
    Prepare multiple datasets for training.

    Args:
        tokenizer: Tokenizer to use
        config: Configuration object with multi-dataset settings

    Returns:
        Tuple of (train_dataset, eval_dataset, test_dataset, data_collator)
    """
    logger.info("Preparing multi-dataset training...")

    # Initialize components
    registry = DatasetRegistry()
    label_mapper = LabelMapper(config.data.label_names)

    # Load all requested datasets
    all_datasets = []
    dataset_names = config.data.multi_dataset.dataset_names

    for i, dataset_name in enumerate(dataset_names):
        logger.info(f"Loading dataset {i + 1}/{len(dataset_names)}: {dataset_name}")

        # Get dataset-specific config if needed
        dataset_config = {
            "language": (config.data.multi_dataset.filter_english and "English") or None,
        }

        # Load dataset
        loader = registry.get_loader(dataset_name, dataset_config)
        dataset = loader.load()

        # Get label mapping for this dataset
        label_mapping = loader.get_label_mapping()

        # Apply label mapping to unify schema
        logger.info(f"Mapping labels for {dataset_name}")
        dataset = dataset.map(
            lambda x, mapping=label_mapping: label_mapper.map_labels(x, mapping),
            batched=True,
            desc=f"Mapping labels for {dataset_name}",
        )

        # Apply dataset-specific preprocessing
        dataset = dataset.map(loader.preprocess, batched=True, desc=f"Preprocessing {dataset_name}")

        # Add dataset source identifier
        dataset = dataset.map(
            lambda x, name=dataset_name: {"dataset_source": name}, desc=f"Adding source for {dataset_name}"
        )

        # Limit samples if configured
        if config.data.multi_dataset.max_samples_per_dataset:
            max_samples = config.data.multi_dataset.max_samples_per_dataset
            for split in dataset:
                if len(dataset[split]) > max_samples:
                    dataset[split] = dataset[split].select(range(max_samples))
                    logger.info(f"Limited {split} split to {max_samples} samples")

        all_datasets.append(dataset)

    # Mix datasets according to strategy
    logger.info(f"Mixing datasets using strategy: {config.data.multi_dataset.mixing_strategy}")
    mixed_dataset = DatasetMixer.mix(
        all_datasets,
        strategy=config.data.multi_dataset.mixing_strategy,
        weights=config.data.multi_dataset.dataset_weights,
        seed=config.training.data_seed,
    )

    # Print statistics
    logger.info("Mixed dataset statistics:")
    for split, data in mixed_dataset.items():
        logger.info(f"  {split}: {len(data)} examples")

    # Create tokenization function
    tokenize_fn: Callable[[dict[str, Any]], dict[str, Any]] = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        label_all_tokens=config.data.label_all_tokens,
    )

    # Tokenize mixed dataset
    logger.info("Tokenizing mixed dataset...")
    tokenized_datasets = mixed_dataset.map(
        tokenize_fn, batched=True, remove_columns=mixed_dataset["train"].column_names, desc="Tokenizing"
    )

    # Create data collator
    data_collator = create_data_collator(tokenizer)

    # Handle missing test split
    test_dataset = tokenized_datasets.get("test", tokenized_datasets["validation"])

    return (tokenized_datasets["train"], tokenized_datasets["validation"], test_dataset, data_collator)
