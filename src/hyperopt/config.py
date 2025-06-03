"""Configuration for hyperparameter optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchSpaceConfig:
    """Configuration for a single hyperparameter in the search space."""

    type: str  # "uniform", "loguniform", "choice", "int", "logint"
    low: float | None = None
    high: float | None = None
    choices: list[Any] | None = None
    log: bool = False


@dataclass
class HyperoptConfig:
    """Configuration for hyperparameter optimization."""

    # Main strategy control
    enabled: bool = False
    strategy: str = "combined"  # "optuna", "asha", "combined", "random"

    # Trial settings
    num_trials: int = 50
    max_concurrent: int = 4
    timeout: int | None = None  # seconds

    # Component toggles
    optuna_enabled: bool = True
    asha_enabled: bool = True

    # Optuna settings (Bayesian optimization)
    optuna_sampler: str = "TPE"  # "TPE", "CMA", "Random", "GPSampler"
    optuna_pruner: str = "median"  # "median", "hyperband", "none"
    optuna_n_startup_trials: int = 10
    optuna_n_warmup_steps: int = 0

    # ASHA settings (Hyperband early stopping)
    asha_max_t: int = 100
    asha_grace_period: int = 10
    asha_reduction_factor: int = 3
    asha_brackets: int = 1

    # Metrics
    metric: str = "eval_f1"
    mode: str = "max"  # "max" or "min"

    # Search space definition
    search_space: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
            "lora_r": {"type": "choice", "choices": [8, 16, 32, 64]},
            "per_device_train_batch_size": {"type": "choice", "choices": [4, 8, 16]},
            "warmup_ratio": {"type": "uniform", "low": 0.0, "high": 0.1},
            "weight_decay": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
        }
    )

    # Storage and persistence
    study_name: str = "mistral_ner_hyperopt"
    storage_url: str | None = None  # e.g., "sqlite:///hyperopt.db"

    # Distributed settings
    ray_address: str | None = None  # "auto" for local cluster
    resources_per_trial: dict[str, float] = field(default_factory=lambda: {"cpu": 1.0, "gpu": 1.0})

    # Logging and results
    log_to_file: bool = True
    results_dir: str = "./hyperopt_results"
    checkpoint_freq: int = 10  # Save every N trials

    # Early stopping for trials
    min_resource: int | None = None  # Minimum epochs before stopping
    max_resource: int | None = None  # Maximum epochs to run

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.strategy not in ["optuna", "asha", "combined", "random"]:
            raise ValueError(f"Invalid strategy: {self.strategy}")

        if self.strategy == "combined" and not (self.optuna_enabled and self.asha_enabled):
            raise ValueError("Combined strategy requires both optuna_enabled and asha_enabled to be True")

        if self.mode not in ["max", "min"]:
            raise ValueError(f"Invalid mode: {self.mode}")

        if self.optuna_sampler not in ["TPE", "CMA", "Random", "GPSampler"]:
            raise ValueError(f"Invalid optuna_sampler: {self.optuna_sampler}")

        if self.optuna_pruner not in ["median", "hyperband", "none"]:
            raise ValueError(f"Invalid optuna_pruner: {self.optuna_pruner}")
