#!/usr/bin/env python3
"""Hyperparameter optimization script for Mistral NER fine-tuning."""

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
from src.hyperopt import HyperparameterOptimizer, create_objective_function
from src.hyperopt.utils import create_ray_tune_search_space, format_search_space_summary
from src.utils import check_gpu_memory, setup_logging


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for Mistral NER fine-tuning")

    # Config file
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML file")

    # Hyperoptimization strategy
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["optuna", "asha", "combined", "random"],
        help="Optimization strategy (overrides config)",
    )

    # Trial settings
    parser.add_argument("--num-trials", type=int, help="Number of trials to run (overrides config)")
    parser.add_argument("--max-concurrent", type=int, help="Maximum concurrent trials (overrides config)")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds (overrides config)")

    # Metric settings
    parser.add_argument("--metric", type=str, help="Metric to optimize (overrides config)")
    parser.add_argument("--mode", type=str, choices=["max", "min"], help="Optimization mode (overrides config)")

    # Ray settings
    parser.add_argument("--ray-address", type=str, help="Ray cluster address (overrides config)")

    # Output settings
    parser.add_argument("--results-dir", type=str, help="Results directory (overrides config)")
    parser.add_argument("--study-name", type=str, help="Study name (overrides config)")

    # Control flags
    parser.add_argument("--save-best-config", type=str, help="Save best configuration to file")
    parser.add_argument("--resume", action="store_true", help="Resume previous study")
    parser.add_argument("--dry-run", action="store_true", help="Show configuration and exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def update_hyperopt_config_from_args(config: Config, args: Namespace) -> None:
    """Update hyperopt configuration from command line arguments."""
    if args.strategy:
        config.hyperopt.strategy = args.strategy
    if args.num_trials:
        config.hyperopt.num_trials = args.num_trials
    if args.max_concurrent:
        config.hyperopt.max_concurrent = args.max_concurrent
    if args.timeout:
        config.hyperopt.timeout = args.timeout
    if args.metric:
        config.hyperopt.metric = args.metric
    if args.mode:
        config.hyperopt.mode = args.mode
    if args.ray_address:
        config.hyperopt.ray_address = args.ray_address
    if args.results_dir:
        config.hyperopt.results_dir = args.results_dir
    if args.study_name:
        config.hyperopt.study_name = args.study_name


def print_optimization_summary(config: Config) -> None:
    """Print optimization configuration summary."""
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION CONFIGURATION")
    print("=" * 80)
    print(f"Strategy: {config.hyperopt.strategy}")
    print(f"Number of trials: {config.hyperopt.num_trials}")
    print(f"Max concurrent: {config.hyperopt.max_concurrent}")
    print(f"Metric: {config.hyperopt.metric} ({config.hyperopt.mode})")
    print(f"Results directory: {config.hyperopt.results_dir}")
    print(f"Study name: {config.hyperopt.study_name}")

    if config.hyperopt.timeout:
        print(f"Timeout: {config.hyperopt.timeout}s")

    if config.hyperopt.ray_address:
        print(f"Ray address: {config.hyperopt.ray_address}")

    # Print strategy-specific settings
    if config.hyperopt.strategy in ["optuna", "combined"]:
        print(f"Optuna sampler: {config.hyperopt.optuna_sampler}")
        print(f"Optuna pruner: {config.hyperopt.optuna_pruner}")

    if config.hyperopt.strategy in ["asha", "combined"]:
        print(f"ASHA max_t: {config.hyperopt.asha_max_t}")
        print(f"ASHA grace period: {config.hyperopt.asha_grace_period}")
        print(f"ASHA reduction factor: {config.hyperopt.asha_reduction_factor}")

    # Print search space
    print("\n" + format_search_space_summary(config.hyperopt.search_space))
    print("=" * 80)


def main() -> None:
    """Main function."""
    args = parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    # Update from command line arguments
    update_hyperopt_config_from_args(config, args)

    # Enable hyperopt
    config.hyperopt.enabled = True

    # Setup logging
    log_level = "debug" if args.debug else "info"
    logger = setup_logging(log_level, config.logging.log_dir)

    logger.info("Starting hyperparameter optimization")
    logger.info(f"Configuration loaded from: {args.config}")

    # Print optimization summary
    print_optimization_summary(config)

    if args.dry_run:
        logger.info("Dry run mode - exiting")
        return

    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Optimization will be slow on CPU.")
    else:
        gpu_info = check_gpu_memory()
        logger.info(f"GPU memory status: {gpu_info}")

    try:
        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_conll2003_dataset()
        print_dataset_statistics(dataset)

        # Prepare datasets (using base config for tokenizer setup)
        logger.info("Preparing datasets...")
        from src.model import setup_model

        _, tokenizer = setup_model(config.model.model_name, config)

        train_dataset, eval_dataset, test_dataset, data_collator = prepare_datasets(
            tokenizer=tokenizer, config=config, dataset=dataset
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")

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
        logger.info("Starting hyperparameter optimization...")

        with HyperparameterOptimizer(config.hyperopt) as optimizer:
            results = optimizer.optimize(
                objective_func=objective_func,
                search_space=search_space,
                base_config=config,
            )

            # Save best configuration if requested
            if args.save_best_config:
                optimizer.save_best_config(results, args.save_best_config)

            logger.info("Hyperparameter optimization completed successfully!")

            # Print final summary
            try:
                best_result = results.get_best_result(config.hyperopt.metric, config.hyperopt.mode)
                print("\n" + "=" * 80)
                print("OPTIMIZATION COMPLETED")
                print("=" * 80)
                best_metrics = best_result.metrics or {}
                print(f"Best {config.hyperopt.metric}: {best_metrics.get(config.hyperopt.metric, 0.0):.4f}")
                print("Best hyperparameters:")
                for param, value in (best_result.config or {}).items():
                    print(f"  {param}: {value}")
                print("=" * 80)
            except Exception as e:
                logger.error(f"Error displaying final results: {e}")

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Optimization failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
