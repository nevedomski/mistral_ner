"""Configuration management for Mistral NER fine-tuning."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import Field, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

from .config_datasets import MultiDatasetConfig
from .config_hyperopt import OptimizationConfig
from .config_logging import LoggingConfig
from .config_model import ModelConfig
from .config_training import TrainingConfig

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Main configuration class for Mistral NER fine-tuning.

    This is the root configuration that combines all sub-configurations for a complete
    training setup. It orchestrates model settings, dataset configuration, training
    hyperparameters, logging, and hyperparameter optimization.

    Attributes:
        model (ModelConfig): Model and LoRA configuration settings.
            Controls model loading, quantization, and LoRA parameters.

        data (MultiDatasetConfig): Dataset configuration for multi-dataset training.
            Manages dataset loading, mixing strategies, and preprocessing.

        training (TrainingConfig): Training hyperparameters and settings.
            Controls learning rate, batch size, loss functions, etc.

        logging (LoggingConfig): Logging and experiment tracking configuration.
            Manages local logging and Weights & Biases integration.

        hyperopt (OptimizationConfig): Hyperparameter optimization settings.
            Controls Optuna/ASHA optimization if enabled.

    The configuration supports multiple ways of initialization:
    1. Default values (suitable for quick experiments)
    2. Hydra YAML configuration files
    3. Programmatic instantiation with custom values

    Example:
        ```python
        # Using defaults
        config = Config()

        # Custom configuration
        config = Config(
            model=ModelConfig(load_in_4bit=True),
            training=TrainingConfig(num_train_epochs=10),
            data=MultiDatasetConfig(mixing_strategy="weighted"),
        )
        ```

    Note:
        The setup_wandb validator automatically manages WandB configuration
        based on logging settings, ensuring consistency between logging.wandb_mode
        and training.report_to fields.
    """

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: MultiDatasetConfig = Field(default_factory=MultiDatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    hyperopt: OptimizationConfig = Field(default_factory=OptimizationConfig)

    @model_validator(mode="after")
    def setup_wandb(self) -> Config:
        """Setup WandB based on configuration."""
        # DOCDEV-TODO: Configuration coupling - cross-module state modification
        # This validator modifies training.report_to based on logging settings
        # Creates tight coupling between configuration sections
        # Consider using factory pattern or separate configuration resolution phase
        if not (self.logging.use_wandb and self.logging.wandb_mode != "disabled") and "wandb" in self.training.report_to:
            self.training.report_to.remove("wandb")
        return self

    def to_yaml(self, path: str | Path) -> None:
        """Convert the configuration to a YAML string."""
        with open(path, "w") as f:
            config_dict = TypeAdapter(type(self)).dump_python(self)
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @model_validator(mode="after")
    def validate_training_config(self) -> Config:
        """Validate training configuration for consistency."""
        # Validate mixed precision settings
        if self.training.fp16 and self.training.bf16:
            raise ValueError("Cannot enable both fp16 and bf16 simultaneously. Please set either fp16=True OR bf16=True, not both.")

        # Validate eval and save strategies match
        if self.training.eval_strategy != self.training.save_strategy:
            logger.warning(  # type: ignore[unreachable]
                f"eval_strategy ({self.training.eval_strategy}) and save_strategy ({self.training.save_strategy}) "
                "don't match. This may lead to unexpected behavior."
            )

        # Validate batch size vs gradient accumulation
        effective_batch_size = self.training.per_device_train_batch_size * self.training.gradient_accumulation_steps
        if effective_batch_size > 128:
            logger.warning(
                f"Effective batch size ({effective_batch_size}) is very large. This may lead to memory issues or training instability."
            )

        # Validate loss function parameters
        if self.training.label_smoothing_loss and self.training.label_smoothing <= 0:
            raise ValueError("label_smoothing must be > 0 when using label_smoothing loss")

        if self.training.focal_loss and self.training.focal_gamma < 0:
            raise ValueError("focal_gamma must be >= 0 for focal loss")

        if self.training.class_balanced_loss and not (0 < self.training.class_balanced_beta < 1):
            raise ValueError("class_balanced_beta must be between 0 and 1")

        return self
