#!/usr/bin/env python3
"""Main training script for Mistral NER fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from argparse import Namespace

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.data import load_conll2003_dataset, prepare_datasets, print_dataset_statistics
from src.evaluation import evaluate_model
from src.model import setup_model
from src.training import run_training_pipeline
from src.utils import check_gpu_memory, print_trainable_parameters, setup_logging


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Mistral-7B for Named Entity Recognition")

    # Config file
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML file")

    # Model arguments
    parser.add_argument("--model-name", type=str, help="Model name or path (overrides config)")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit (overrides config)")
    parser.add_argument(
        "--no-load-in-8bit", dest="load_in_8bit", action="store_false", help="Don't load model in 8-bit"
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (overrides config)")
    parser.add_argument(
        "--no-load-in-4bit", dest="load_in_4bit", action="store_false", help="Don't load model in 4-bit"
    )

    # Data arguments
    parser.add_argument("--max-length", type=int, help="Maximum sequence length (overrides config)")

    # Training arguments
    parser.add_argument("--output-dir", type=str, help="Output directory (overrides config)")
    parser.add_argument("--num-train-epochs", type=int, help="Number of training epochs (overrides config)")
    parser.add_argument(
        "--batch-size", type=int, dest="per_device_train_batch_size", help="Training batch size (overrides config)"
    )
    parser.add_argument("--learning-rate", type=float, help="Learning rate (overrides config)")
    parser.add_argument("--resume-from-checkpoint", type=str, help="Resume training from checkpoint")

    # Logging arguments
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, help="WandB project name (overrides config)")
    parser.add_argument("--wandb-entity", type=str, help="WandB entity name (overrides config)")
    parser.add_argument("--wandb-name", type=str, help="WandB run name (overrides config)")
    parser.add_argument(
        "--wandb-mode", type=str, choices=["online", "offline", "disabled"], help="WandB mode (overrides config)"
    )
    parser.add_argument("--wandb-dir", type=str, help="WandB directory for offline runs (overrides config)")
    parser.add_argument(
        "--wandb-resume",
        type=str,
        choices=["allow", "must", "never", "auto"],
        help="WandB resume strategy (overrides config)",
    )
    parser.add_argument("--wandb-run-id", type=str, help="WandB run ID for resuming (overrides config)")
    parser.add_argument("--wandb-tags", type=str, nargs="+", help="WandB tags (overrides config)")
    parser.add_argument("--wandb-notes", type=str, help="WandB notes (overrides config)")

    # Other arguments
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation, no training")
    parser.add_argument("--test", action="store_true", help="Run evaluation on test set after training")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.set_defaults(load_in_8bit=None, load_in_4bit=None, use_wandb=None)

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    # Update config from command line arguments
    config.update_from_args(args)

    # Setup logging
    log_level = "debug" if args.debug else config.logging.log_level
    logger = setup_logging(log_level, config.logging.log_dir)

    logger.info("Starting Mistral NER fine-tuning")
    logger.info(f"Configuration loaded from: {args.config}")

    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be slow on CPU.")
    else:
        gpu_info = check_gpu_memory()
        logger.info(f"GPU memory status: {gpu_info}")

    try:
        # Check if multi-dataset mode is enabled
        dataset = None
        if config.data.multi_dataset.enabled:
            logger.info("Multi-dataset mode enabled. Will load multiple datasets...")
            logger.info(f"Datasets to load: {config.data.multi_dataset.dataset_names}")
            logger.info(f"Mixing strategy: {config.data.multi_dataset.mixing_strategy}")
            logger.info(f"Dataset weights: {config.data.multi_dataset.dataset_weights}")
        else:
            # Load single dataset
            logger.info("Loading dataset...")
            dataset = load_conll2003_dataset()
            print_dataset_statistics(dataset)

        # Setup model and tokenizer
        logger.info("Setting up model and tokenizer...")
        model, tokenizer = setup_model(model_name=config.model.model_name, config=config)

        # Print trainable parameters
        trainable_params = print_trainable_parameters(model)
        logger.info(f"Model setup complete. Trainable parameters: {trainable_params}")

        # Prepare datasets
        logger.info("Preparing datasets...")
        train_dataset, eval_dataset, test_dataset, data_collator = prepare_datasets(
            tokenizer=tokenizer, config=config, dataset=dataset
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

        if args.eval_only:
            # Evaluation only mode
            logger.info("Running evaluation only...")

            # Evaluate on validation set
            logger.info("Evaluating on validation set...")
            val_metrics = evaluate_model(
                model=model,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                label_names=config.data.label_names,
                batch_size=config.training.per_device_eval_batch_size,
            )
            logger.info(f"Validation metrics: {val_metrics}")

            # Evaluate on test set
            if args.test:
                logger.info("Evaluating on test set...")
                test_metrics = evaluate_model(
                    model=model,
                    eval_dataset=test_dataset,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    label_names=config.data.label_names,
                    batch_size=config.training.per_device_eval_batch_size,
                )
                logger.info(f"Test metrics: {test_metrics}")

        else:
            # Check if hyperparameter optimization is enabled
            if hasattr(config, "hyperopt") and hasattr(config.hyperopt, "enabled") and config.hyperopt.enabled:
                logger.info("Hyperparameter optimization enabled - running optimization...")

                from src.hyperopt import HyperparameterOptimizer, create_objective_function
                from src.hyperopt.utils import create_ray_tune_search_space

                # Create search space
                search_space = create_ray_tune_search_space(config.hyperopt)

                # Create objective function
                objective_func = create_objective_function(
                    base_config=config,
                    hyperopt_config=config.hyperopt,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=data_collator,
                )

                # Run optimization
                with HyperparameterOptimizer(config.hyperopt) as optimizer:
                    results = optimizer.optimize(
                        objective_func=objective_func,
                        search_space=search_space,
                        base_config=config,
                    )

                    # Get best configuration and run final training
                    best_result = results.get_best_result(config.hyperopt.metric, config.hyperopt.mode)
                    logger.info(f"Best hyperparameters found: {best_result.config}")

                    # Update config with best hyperparameters and run final training
                    for param_name, param_value in (best_result.config or {}).items():
                        if param_name == "learning_rate":
                            config.training.learning_rate = param_value
                        elif param_name == "lora_r":
                            config.model.lora_r = param_value
                        elif param_name == "per_device_train_batch_size":
                            config.training.per_device_train_batch_size = param_value
                        elif param_name == "warmup_ratio":
                            config.training.warmup_ratio = param_value
                        elif param_name == "weight_decay":
                            config.training.weight_decay = param_value

                    logger.info("Running final training with best hyperparameters...")

                    # Re-setup model with best hyperparameters
                    model, tokenizer = setup_model(model_name=config.model.model_name, config=config)

                    # Re-enable WandB for final training
                    config.logging.use_wandb = True
                    config.training.report_to = ["wandb"]

                    train_metrics = run_training_pipeline(
                        model=model,
                        tokenizer=tokenizer,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=data_collator,
                        config=config,
                    )

                    logger.info(f"Final training completed. Metrics: {train_metrics}")
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
            if args.test:
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


if __name__ == "__main__":
    main()
