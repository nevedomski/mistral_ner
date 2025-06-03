"""Utility functions for hyperparameter optimization."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import ray.tune as tune

if TYPE_CHECKING:
    from src.config import Config

    from .config import HyperoptConfig

logger = logging.getLogger(__name__)


def create_objective_function(
    base_config: Config,
    hyperopt_config: HyperoptConfig,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: Any,
) -> Callable[[dict[str, Any]], None]:
    """Create Ray Tune objective function wrapping existing training pipeline.

    Args:
        base_config: Base configuration to modify with trial hyperparameters
        hyperopt_config: Hyperparameter optimization configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator for batching

    Returns:
        Objective function for Ray Tune
    """

    def objective(trial_config: dict[str, Any]) -> None:
        """Objective function for a single trial."""
        # Import here to avoid circular imports
        from src.model import setup_model
        from src.training import run_training_pipeline

        # Create a copy of base config and update with trial hyperparameters
        config = copy.deepcopy(base_config)

        # Update configuration with trial hyperparameters
        for param_name, param_value in trial_config.items():
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
            elif param_name == "lora_alpha":
                config.model.lora_alpha = param_value
            elif param_name == "lora_dropout":
                config.model.lora_dropout = param_value
            elif param_name == "num_train_epochs":
                config.training.num_train_epochs = param_value
            else:
                logger.warning(f"Unknown hyperparameter: {param_name}")

        try:
            # Set up model and tokenizer for this trial
            model, tokenizer = setup_model(config.model.model_name, config)

            # For hyperparameter optimization, we want to track progress over epochs
            # Set evaluation strategy to epoch for proper reporting
            config.training.eval_strategy = "epoch"
            config.training.save_strategy = "epoch"
            config.training.logging_steps = 1

            # Disable WandB for individual trials to avoid clutter
            config.logging.use_wandb = False
            config.training.report_to = []

            # Run training pipeline
            metrics = run_training_pipeline(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                config=config,
            )

            # Report final metrics to Ray Tune
            tune.report(
                **{
                    hyperopt_config.metric: metrics.get(hyperopt_config.metric, 0.0),
                    "eval_loss": metrics.get("eval_loss", float("inf")),
                    "train_loss": metrics.get("train_loss", float("inf")),
                    "training_iteration": config.training.num_train_epochs,
                }
            )

        except Exception as e:
            logger.error(f"Trial failed with error: {e}")
            # Report failure with worst possible metric value
            worst_value = 0.0 if hyperopt_config.mode == "max" else float("inf")
            tune.report(
                **{
                    hyperopt_config.metric: worst_value,
                    "eval_loss": float("inf"),
                    "train_loss": float("inf"),
                    "training_iteration": 1,
                    "trial_failed": True,
                }
            )

    return objective


def create_ray_tune_search_space(hyperopt_config: HyperoptConfig) -> dict[str, Any]:
    """Convert hyperopt search space to Ray Tune format.

    Args:
        hyperopt_config: Hyperparameter optimization configuration

    Returns:
        Ray Tune compatible search space dictionary
    """
    search_space = {}

    for param_name, param_config in hyperopt_config.search_space.items():
        param_type = param_config["type"]

        if param_type == "uniform":
            search_space[param_name] = tune.uniform(param_config["low"], param_config["high"])
        elif param_type == "loguniform":
            search_space[param_name] = tune.loguniform(param_config["low"], param_config["high"])
        elif param_type == "choice":
            search_space[param_name] = tune.choice(param_config["choices"])
        elif param_type == "int":
            search_space[param_name] = tune.randint(param_config["low"], param_config["high"])
        elif param_type == "logint":
            # Ray Tune doesn't have logint, use loguniform and cast to int
            search_space[param_name] = tune.loguniform(param_config["low"], param_config["high"])
            logger.warning(f"Using loguniform for logint parameter {param_name}")
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return search_space


def validate_hyperopt_config(hyperopt_config: HyperoptConfig) -> None:
    """Validate hyperparameter optimization configuration.

    Args:
        hyperopt_config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required fields
    if not hyperopt_config.search_space:
        raise ValueError("search_space cannot be empty")

    # Validate search space parameters
    for param_name, param_config in hyperopt_config.search_space.items():
        param_type = param_config.get("type")
        if not param_type:
            raise ValueError(f"Parameter {param_name} missing 'type' field")

        if param_type in ["uniform", "loguniform", "int", "logint"]:
            if "low" not in param_config or "high" not in param_config:
                raise ValueError(f"Parameter {param_name} missing 'low' or 'high' field")
            if param_config["low"] >= param_config["high"]:
                raise ValueError(f"Parameter {param_name}: low must be < high")

        elif param_type == "choice":
            if "choices" not in param_config or not param_config["choices"]:
                raise ValueError(f"Parameter {param_name} missing or empty 'choices' field")

        else:
            raise ValueError(f"Unknown parameter type for {param_name}: {param_type}")

    # Validate strategy-specific settings
    if hyperopt_config.strategy == "combined" and (
        not hyperopt_config.optuna_enabled or not hyperopt_config.asha_enabled
    ):
        raise ValueError("Combined strategy requires both optuna_enabled and asha_enabled")

    # Validate resource allocation
    if hyperopt_config.max_concurrent <= 0:
        raise ValueError("max_concurrent must be > 0")

    if hyperopt_config.num_trials <= 0:
        raise ValueError("num_trials must be > 0")


def format_search_space_summary(search_space: dict[str, Any]) -> str:
    """Format search space for logging.

    Args:
        search_space: Search space configuration

    Returns:
        Formatted string summary
    """
    lines = ["Search Space:"]
    for param_name, param_config in search_space.items():
        param_type = param_config.get("type", "unknown")

        if param_type in ["uniform", "loguniform"]:
            lines.append(f"  {param_name}: {param_type}({param_config['low']}, {param_config['high']})")
        elif param_type == "choice":
            choices_str = ", ".join(map(str, param_config["choices"]))
            lines.append(f"  {param_name}: choice([{choices_str}])")
        else:
            lines.append(f"  {param_name}: {param_config}")

    return "\n".join(lines)
