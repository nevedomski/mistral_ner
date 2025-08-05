#!/usr/bin/env python3
"""Main entry point for Mistral NER training."""

# Import and run the training script
# sys.path.append(str(Path(__file__).parent))
import logging
from typing import Any, cast

import hydra
from omegaconf import OmegaConf

from scripts import load_data, run_training
from src.config import Config

# Setup logging
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="bank_pii")
def main(cfg: Config) -> None:
    """Get the main configuration object.

    This function retrieves the main configuration object, which can be used
    to access all sub-configurations and their attributes.

    Returns:
        Config: The main configuration object containing all settings.
    """
    # Load the configuration from the YAML files
    yaml_config = OmegaConf.to_object(cfg)

    if isinstance(yaml_config, dict) and all(isinstance(k, str) for k in yaml_config):
        yaml_config = cast(dict[str, Any], yaml_config)
        # Convert DictConfig to Config dataclass
        config = Config(**yaml_config)

    else:
        raise TypeError("Expected a dictionary configuration object.")
    logger.info("Configuration loaded successfully")

    unified_dataset = load_data(config)
    if unified_dataset is None:
        logger.error("No dataset loaded. Exiting.")
        return

    logger.info(
        f"Loaded {len(unified_dataset)} samples from {len(config.data.datasets)} datasets with mixing strategy '{config.data.mixing_strategy}'."
    )

    run_training(unified_dataset, config)
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
