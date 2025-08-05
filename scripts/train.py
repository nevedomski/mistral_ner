#!/usr/bin/env python3
"""Main training script for Mistral NER fine-tuning."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from datasets import DatasetDict

if TYPE_CHECKING:
    pass

import logging

from src.config import Config
from src.data import prepare_datasets, print_dataset_statistics
from src.evaluation import evaluate_model
from src.model import setup_model
from src.training import run_training_pipeline
from src.utils import check_gpu_memory, print_trainable_parameters

# Setup logging
logger = logging.getLogger(__name__)


def run_training(dataset: DatasetDict, config: Config) -> None:
    """Main training function."""

    logger.info(f"Starting {config.model.model_name} fine-tuning")

    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be slow on CPU.")
    else:
        gpu_info = check_gpu_memory()
        logger.info(f"GPU memory status: {gpu_info}")

    try:
        logger.info(f"Datasets loaded and will be used: {config.data.datasets.keys()}")
        logger.info(f"Mixing strategy: {config.data.mixing_strategy}")
        weights = {name: items.weight for name, items in config.data.datasets.items()}
        logger.info(f"Dataset weights: {weights}")
        print_dataset_statistics(dataset)

        # Setup model and tokenizer
        if not config.model.model_name:
            raise ValueError("Model name must be specified in the configuration.")
        logger.info("Setting up model and tokenizer...")
        model, tokenizer = setup_model(model_name=config.model.model_name, config=config)

        # Print trainable parameters
        trainable_params = print_trainable_parameters(model)
        logger.info(f"Model setup complete. Trainable parameters: {trainable_params}")

        # Prepare datasets
        logger.info("Preparing datasets...")
        train_dataset, eval_dataset, test_dataset, data_collator = prepare_datasets(tokenizer=tokenizer, config=config, dataset=dataset)

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

        # Check if hyperparameter optimization is enabled
        if config.hyperopt.enabled:
            # DOCDEV-TODO: Validate and fix hyperparameter optimization logic
            pass  # Placeholder for hyperparameter optimization logic
            # logger.info("Hyperparameter optimization enabled - running optimization...")

            # from src.hyperopt import HyperparameterOptimizer, create_objective_function
            # from src.hyperopt.utils import create_ray_tune_search_space

            # # Create search space
            # search_space = create_ray_tune_search_space(config.hyperopt)

            # # Create objective function
            # objective_func = create_objective_function(
            #     base_config=config,
            #     hyperopt_config=config.hyperopt,
            #     train_dataset=train_dataset,
            #     eval_dataset=eval_dataset,
            #     data_collator=data_collator,
            # )

            # # Run optimization
            # with HyperparameterOptimizer(config.hyperopt) as optimizer:
            #     results = optimizer.optimize(
            #         objective_func=objective_func,
            #         search_space=search_space,
            #         base_config=config,
            #     )

            #     # Get best configuration and run final training
            #     best_result = results.get_best_result(config.hyperopt.metric, config.hyperopt.mode)
            #     logger.info(f"Best hyperparameters found: {best_result.config}")

            #     # Update config with best hyperparameters and run final training
            #     for param_name, param_value in (best_result.config or {}).items():
            #         if param_name == "learning_rate":
            #             config.training.learning_rate = param_value
            #         elif param_name == "lora_r":
            #             config.model.lora_r = param_value
            #         elif param_name == "per_device_train_batch_size":
            #             config.training.per_device_train_batch_size = param_value
            #         elif param_name == "warmup_ratio":
            #             config.training.warmup_ratio = param_value
            #         elif param_name == "weight_decay":
            #             config.training.weight_decay = param_value

            #     logger.info("Running final training with best hyperparameters...")

            #     # Re-setup model with best hyperparameters
            #     model, tokenizer = setup_model(model_name=config.model.model_name, config=config)

            #     # Re-enable WandB for final training
            #     config.logging.use_wandb = True
            #     config.training.report_to = ["wandb"]

            #     train_metrics = run_training_pipeline(
            #         model=model,
            #         tokenizer=tokenizer,
            #         train_dataset=train_dataset,
            #         eval_dataset=eval_dataset,
            #         data_collator=data_collator,
            #         config=config,
            #     )

            #     logger.info(f"Final training completed. Metrics: {train_metrics}")
        else:
            # Regular training mode
            logger.info("Starting training pipeline...")

            # Run training
            train_metrics = run_training_pipeline(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                config=config,
            )

            logger.info(f"Training completed. Final metrics: {train_metrics}")

        # Evaluate on test set if requested
        if config.training.evaluate_on_test_set:
            logger.info("Evaluating on test set...")

            # Load best model if available
            best_model_path = Path(config.training.output_dir) / "checkpoint-best"
            if best_model_path.exists():
                logger.info(f"Loading best model from {best_model_path}")
                from src.model import load_model_from_checkpoint

                model, tokenizer = load_model_from_checkpoint(str(best_model_path), config)

            test_metrics = evaluate_model(
                model=model,
                eval_dataset=test_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                label_names=config.data.label_names,
                batch_size=config.training.per_device_eval_batch_size,
            )
            logger.info(f"Test metrics: {test_metrics}")

        logger.info("Script completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)
