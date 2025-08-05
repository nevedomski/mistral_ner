"""Multi-dataset configuration for Mistral NER fine-tuning."""

from __future__ import annotations

from typing import Any, Literal

from datasets import DatasetDict
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from .dataset_utils.base import BaseDataset
# DOCDEV-TODO: CONFIGURATION COUPLING ISSUE - Hardcoded dataset imports
# Risk: Tight coupling makes it difficult to add new datasets without modifying core files
# Solution: Implement plugin-based registry system for dynamic dataset discovery
from .dataset_utils.loaders.ai4privacy import AI4PrivacyDataset
from .dataset_utils.loaders.bigcode_pii import BigCodePIIDataset
from .dataset_utils.loaders.conll import CoNLLDataset
from .dataset_utils.loaders.fewnerd import FewNERDDataset
from .dataset_utils.loaders.gretel import GretelPIIDataset
from .dataset_utils.loaders.mendeley_pii import MendeleyPIIDataset
from .dataset_utils.loaders.ontonotes import OntoNotesDataset
from .dataset_utils.loaders.wikiner import WikiNERDataset
from .dataset_utils.loaders.wnut import WNUTDataset

# DOCDEV-TODO: Replace hardcoded dictionary with dynamic registry pattern
datasets = {
    "ai4privacy": AI4PrivacyDataset,
    "bigcode_pii": BigCodePIIDataset,
    "conll": CoNLLDataset,
    "fewnerd": FewNERDDataset,
    "gretel": GretelPIIDataset,
    "mendeley": MendeleyPIIDataset,
    "ontonotes": OntoNotesDataset,
    "wikiner": WikiNERDataset,
    "wnut": WNUTDataset,
}


def prepare_datasets(config: dict[str, Any]) -> dict[str, BaseDataset]:
    """Prepare datasets from configuration.

    Args:
        config: Dictionary mapping dataset names to their configuration parameters.

    Returns:
        Dictionary mapping dataset names to instantiated dataset loaders.

    Raises:
        ValueError: If dataset name is unknown or configuration is invalid.
    """
    prepared_datasets: dict[str, BaseDataset] = {}

    # Validate all dataset names first
    for name in config:
        if name not in datasets:
            raise ValueError(f"Unknown dataset: '{name}'. Available datasets: {sorted(list(datasets.keys()))}")

    # Instantiate each dataset with its config
    for name, dataset_config in config.items():
        if not isinstance(dataset_config, dict):
            raise ValueError(f"Invalid dataset configuration for '{name}': {dataset_config}. Configuration must be a dictionary.")
        try:
            prepared_datasets[name] = datasets[name](**dataset_config)
        except Exception as e:
            raise ValueError(f"Failed to initialize dataset '{name}': {e!s}") from e

    return prepared_datasets


@dataclass
class MultiDatasetConfig:
    """Configuration for multi-dataset training in Mistral NER.

    This class manages the configuration for training on multiple NER datasets
    simultaneously. It handles dataset loading, mixing strategies, and common
    preprocessing parameters.

    The datasets are loaded dynamically based on configuration, with support for:
    - AI4Privacy PII dataset
    - BigCode PII dataset
    - CoNLL-2003 NER dataset
    - Few-NERD fine-grained NER dataset
    - Gretel Synthetic PII dataset
    - Mendeley/Isotonic PII dataset
    - OntoNotes 5.0 dataset
    - WikiNER/WikiANN multilingual dataset
    - WNUT-17 Emerging Entities dataset

    Attributes:
        datasets (dict[str, BaseDataset]): Dictionary mapping dataset names to their
            configured loader instances. Populated via prepare_datasets() based on config.

        max_length (int): Maximum sequence length for tokenization (default: 256).
            Sequences longer than this will be truncated.

        label_all_tokens (bool): Whether to assign labels to all tokens (default: False).
            If False, only the first subword of each word gets a label.

        return_entity_level_metrics (bool): Whether to compute entity-level F1 scores
            in addition to token-level metrics (default: True).

        mixing_strategy (str): How to combine multiple datasets - 'concat', 'interleave',
            or 'weighted' (default: 'interleave').
            - 'concat': Concatenate all datasets sequentially
            - 'interleave': Alternate between datasets during training
            - 'weighted': Sample from datasets based on their weights

    Example Configuration:
        ```yaml
        data:
          datasets:
            conll:
              filter_english: true
              max_samples_per_dataset: 10000
            ontonotes:
              filter_english: true
              weight: 2.0  # Give OntoNotes 2x weight
          mixing_strategy: weighted
          max_length: 128
        ```

    Note:
        - Dataset weights are specified within each dataset's configuration
        - Datasets are loaded lazily when first accessed
        - All datasets are automatically converted to a unified tagging scheme
    """

    datasets: dict[str, BaseDataset] = Field(default_factory=dict, description="Dictionary of datasets to be used for training.")
    max_length: int = 256
    label_all_tokens: bool = False
    return_entity_level_metrics: bool = True
    # Mixing strategy
    mixing_strategy: Literal["concat", "interleave", "weighted"] = "interleave"
    num_labels: int | None = None

    def mix_datasets(self) -> DatasetDict:
        """Mix datasets based on the configured mixing strategy."""
        # DOCDEV-TODO: Critical incomplete feature - multi-dataset training is broken
        # This is the core advertised feature but only supports single dataset
        # Needs implementation of mixers.py, multi_dataset.py, registry.py, samplers.py
        if self.mixing_strategy == "concat":
            raise NotImplementedError("Concatenation strategy not yet implemented.")
        elif self.mixing_strategy == "interleave":
            pass
        elif self.mixing_strategy == "weighted":
            raise NotImplementedError("Weighted sampling strategy not yet implemented.")
        else:
            raise ValueError(f"Unknown mixing strategy: {self.mixing_strategy}")

        dataset = next(iter(self.datasets.values())).load()

        # def add_multi_set_tags(examples: dict[str, Any]) -> dict[str, Any]:
        #     examples["multi_set_unified_tags"] = examples["single_set_unified_tags"]
        #     return examples

        # unified_dataset = dataset.map(add_multi_set_tags)
        for split in dataset:
            dataset[split] = dataset[split].add_column("multi_set_unified_tags", dataset[split]["single_set_unified_tags"])  # type: ignore[call-arg, unused-ignore]
            dataset[split] = dataset[split].cast_column("multi_set_unified_tags", dataset[split].features["single_set_unified_tags"])

        self.num_labels = dataset["train"].features["multi_set_unified_tags"].feature.num_classes
        self.str2int = dataset["train"].features["multi_set_unified_tags"].feature.str2int
        self.int2str = dataset["train"].features["multi_set_unified_tags"].feature.int2str
        self.label_names = dataset["train"].features["multi_set_unified_tags"].feature.names
        self.label2id = {label: i for i, label in enumerate(dataset["train"].features["multi_set_unified_tags"].feature.names)}
        self.id2label = {i: label for i, label in enumerate(dataset["train"].features["multi_set_unified_tags"].feature.names)}
        return dataset

    @field_validator("datasets", mode="before")
    @classmethod
    def validate_datasets(cls, value: dict[str, Any]) -> dict[str, BaseDataset]:
        """Validate and prepare datasets from configuration."""
        if not isinstance(value, dict):
            raise ValueError("Datasets configuration must be a dictionary mapping dataset names to their configurations.")
        return prepare_datasets(value)
