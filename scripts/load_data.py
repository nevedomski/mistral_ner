#!/usr/bin/env python3
"""Main training script for Mistral NER fine-tuning."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from datasets import DatasetDict

if TYPE_CHECKING:
    pass


from src.config import Config

# Setup logging
logger = logging.getLogger(__name__)


def load_data(config: Config) -> DatasetDict:
    """Main training function."""

    logger.info("Starting Dataset Loading")

    try:
        # for i, dataset_name in enumerate(config.data.datasets):
        #     logger.info(f"Loading dataset {i + 1}/{len(config.data.datasets)}: {dataset_name}")
        #     dataset = config.data.datasets[dataset_name].load()
        #     logger.info(f"Dataset {dataset_name} loaded successfully with {len(dataset)} samples.")
        unified_dataset = config.data.mix_datasets()
        logger.info(f"Mixed datasets by {config.data.mixing_strategy}.")
        return unified_dataset

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)
