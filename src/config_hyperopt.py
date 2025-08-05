"""Hyperparameter optimization configuration for Mistral NER fine-tuning."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass


def _default_search_space() -> dict[str, ContinuousChoiceConfig | IntChoiceConfig]:
    """Default search space for hyperparameter optimization."""
    return {
        # Learning rate optimization
        "learning_rate": ContinuousChoiceConfig(type="loguniform", low=0.00001, high=0.001),
        # LoRA rank optimization
        "lora_r": IntChoiceConfig(type="choice", choices=[8, 16, 32, 64]),
        # Batch size optimization
        "per_device_train_batch_size": IntChoiceConfig(type="choice", choices=[4, 8, 16]),
        # Warmup ratio optimization
        "warmup_ratio": ContinuousChoiceConfig(type="uniform", low=0.0, high=0.1),
        # Weight decay optimization
        "weight_decay": ContinuousChoiceConfig(type="loguniform", low=0.0001, high=0.1),
    }


@dataclass
class ContinuousChoiceConfig:
    """Configuration for continuous float value optimization.

    Attributes:
        type (str): Type of distribution for continuous float value.
        low (float): Lower bound for the continuous float value.
        high (float): Upper bound for the continuous float value.
    """

    type: Literal["loguniform", "uniform"] = "loguniform"
    low: float = 0.00001
    high: float = 0.001


@dataclass
class IntChoiceConfig:
    """Configuration for integer choice optimization.

    Attributes:
        type (str): Type of optimization for integer choice.
        choices (list[int]): Possible values for integer choice.
    """

    type: Literal["choice"] = "choice"
    choices: list[int] = Field(default_factory=list)


@dataclass
class ResourceConfig:
    """Configuration for resource allocation in optimization trials.

    Attributes:
        cpu (float): Number of CPU cores allocated per trial.
        gpu (float): Number of GPU cores allocated per trial.
    """

    cpu: float = 1.0
    gpu: float = 1.0


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization of Mistral NER models.

    This class manages hyperparameter optimization settings using Optuna (Bayesian optimization)
    and ASHA (Asynchronous Successive Halving Algorithm) for efficient trial scheduling.
    Supports distributed optimization through Ray if configured.

    The optimization process searches for optimal values of:
    - Learning rate (log-uniform distribution)
    - LoRA rank (discrete choices: 8, 16, 32, 64)
    - Batch size (discrete choices: 4, 8, 16)
    - Warmup ratio (uniform distribution)
    - Weight decay (log-uniform distribution)

    Attributes:
        enabled (bool): Whether hyperparameter optimization is enabled.
        strategy (str): Optimization strategy - 'optuna', 'asha', 'combined', or 'random'.
        num_trials (int): Total number of trials to run.
        max_concurrent (int): Maximum number of concurrent trials.
        timeout (int): Maximum time in seconds for optimization (default: 2 hours).

        optuna_enabled (bool): Enable Optuna Bayesian optimization.
        asha_enabled (bool): Enable ASHA early stopping.

        optuna_sampler (str): Optuna sampling algorithm - 'TPE', 'CMA', 'Random', or 'GPSampler'.
        optuna_pruner (str): Optuna pruning strategy - 'median', 'hyperband', or 'none'.
        optuna_n_startup_trials (int): Number of random trials before optimization begins.
        optuna_n_warmup_steps (int): Training steps before pruning is enabled.

        asha_max_t (int): Maximum epochs/resources per trial in ASHA.
        asha_grace_period (int): Minimum epochs before a trial can be stopped.
        asha_reduction_factor (int): Factor by which trials are reduced in each round.
        asha_brackets (int): Number of brackets in ASHA algorithm.

        metric (str): Metric to optimize (default: 'eval_f1').
        mode (str): Optimization mode - 'max' or 'min'.

        search_space (dict): Hyperparameter search space definitions.
        study_name (str): Name for the optimization study.
        storage_url (str | None): Database URL for study persistence (None = in-memory).

        ray_address (str | None): Ray cluster address for distributed optimization.
        resources_per_trial (ResourceConfig): CPU/GPU allocation per trial.

        log_to_file (bool): Whether to log optimization progress to file.
        results_dir (str): Directory to save optimization results.
        checkpoint_freq (int): Frequency of checkpointing trials.
    """

    # 🎯 HYPERPARAMETER OPTIMIZATION CONFIGURATION
    # Main control
    enabled: bool = False
    strategy: Literal["optuna", "asha", "combined", "random"] = "combined"

    # Trial settings
    num_trials: int = 50
    max_concurrent: int = 4
    timeout: int = 7200  # 2 hours

    # Component toggles
    optuna_enabled: bool = True
    asha_enabled: bool = True

    # Optuna settings (Bayesian optimization)
    optuna_sampler: Literal["TPE", "CMA", "Random", "GPSampler"] = "TPE"
    optuna_pruner: Literal["median", "hyperband", "none"] = "median"
    optuna_n_startup_trials: int = 10
    optuna_n_warmup_steps: int = 0

    # ASHA settings (Hyperband early stopping)
    asha_max_t: int = 3  # Maximum epochs for trials
    asha_grace_period: int = 1  # Minimum epochs before stopping
    asha_reduction_factor: int = 3
    asha_brackets: int = 1

    # Optimization metrics
    metric: Literal["eval_f1"] = "eval_f1"
    mode: Literal["max", "min"] = "max"

    # 🔍 SEARCH SPACE DEFINITION
    search_space: dict[str, ContinuousChoiceConfig | IntChoiceConfig] = Field(default_factory=_default_search_space)

    # Storage and persistence
    study_name: str = "mistral_ner_combined_optimization"
    storage_url: str | None = None  # Use in-memory storage

    # Distributed settings
    ray_address: str | None = None  # Use local Ray cluster
    resources_per_trial: ResourceConfig = Field(default_factory=lambda: ResourceConfig(cpu=1.0, gpu=1.0))

    # Logging and results
    log_to_file: bool = True
    results_dir: str = "./hyperopt_results"
    checkpoint_freq: int = 10
