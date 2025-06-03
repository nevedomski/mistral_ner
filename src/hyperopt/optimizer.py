"""Main hyperparameter optimizer implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ray
import ray.tune as tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

if TYPE_CHECKING:
    from src.config import Config

    from .config import HyperoptConfig

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Unified hyperparameter optimization with flexible strategies.

    Supports multiple optimization strategies:
    - optuna: Pure Bayesian optimization using Optuna
    - asha: Pure early stopping using ASHA scheduler
    - combined: Optuna search + ASHA scheduling (recommended)
    - random: Random search baseline
    """

    def __init__(self, hyperopt_config: HyperoptConfig) -> None:
        """Initialize the optimizer.

        Args:
            hyperopt_config: Hyperparameter optimization configuration
        """
        self.config = hyperopt_config
        self._validate_config()
        self._setup_logging()

    def optimize(
        self,
        objective_func: Any,
        search_space: dict[str, Any],
        base_config: Config,
    ) -> tune.ResultGrid:
        """Run hyperparameter optimization.

        Args:
            objective_func: Objective function to optimize
            search_space: Ray Tune search space
            base_config: Base configuration for logging

        Returns:
            Ray Tune results
        """
        logger.info(f"Starting hyperparameter optimization with strategy: {self.config.strategy}")
        logger.info(f"Number of trials: {self.config.num_trials}")
        logger.info(f"Max concurrent trials: {self.config.max_concurrent}")

        # Initialize Ray if not already done
        self._initialize_ray()

        # Create tuner based on strategy
        if self.config.strategy == "combined":
            tuner = self._create_combined_tuner(objective_func, search_space)
        elif self.config.strategy == "optuna":
            tuner = self._create_optuna_tuner(objective_func, search_space)
        elif self.config.strategy == "asha":
            tuner = self._create_asha_tuner(objective_func, search_space)
        elif self.config.strategy == "random":
            tuner = self._create_random_tuner(objective_func, search_space)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

        # Run optimization
        logger.info("Starting optimization...")
        results = tuner.fit()

        # Log results summary
        self._log_results_summary(results)

        return results

    def _create_combined_tuner(self, objective_func: Any, search_space: dict[str, Any]) -> tune.Tuner:
        """Create tuner with combined Optuna + ASHA strategy."""
        logger.info("Creating combined Optuna + ASHA tuner")

        # Set up Optuna search algorithm
        optuna_search = self._create_optuna_search()
        search_alg = ConcurrencyLimiter(optuna_search, max_concurrent=self.config.max_concurrent)

        # Set up ASHA scheduler
        scheduler = self._create_asha_scheduler()

        return tune.Tuner(
            objective_func,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=self.config.num_trials,
                metric=self.config.metric,
                mode=self.config.mode,
            ),
            run_config=tune.RunConfig(
                name=self.config.study_name,
                storage_path=self.config.results_dir,
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=self.config.checkpoint_freq,
                    checkpoint_at_end=True,
                ),
                failure_config=tune.FailureConfig(max_failures=3),
                log_to_file=self.config.log_to_file,
            ),
            param_space=search_space,
        )

    def _create_optuna_tuner(self, objective_func: Any, search_space: dict[str, Any]) -> tune.Tuner:
        """Create tuner with Optuna-only strategy."""
        logger.info("Creating Optuna-only tuner")

        optuna_search = self._create_optuna_search()
        search_alg = ConcurrencyLimiter(optuna_search, max_concurrent=self.config.max_concurrent)

        return tune.Tuner(
            objective_func,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                num_samples=self.config.num_trials,
                metric=self.config.metric,
                mode=self.config.mode,
            ),
            run_config=tune.RunConfig(
                name=self.config.study_name,
                storage_path=self.config.results_dir,
                log_to_file=self.config.log_to_file,
            ),
            param_space=search_space,
        )

    def _create_asha_tuner(self, objective_func: Any, search_space: dict[str, Any]) -> tune.Tuner:
        """Create tuner with ASHA-only strategy."""
        logger.info("Creating ASHA-only tuner")

        scheduler = self._create_asha_scheduler()

        return tune.Tuner(
            objective_func,
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=self.config.num_trials,
                metric=self.config.metric,
                mode=self.config.mode,
            ),
            run_config=tune.RunConfig(
                name=self.config.study_name,
                storage_path=self.config.results_dir,
                log_to_file=self.config.log_to_file,
            ),
            param_space=search_space,
        )

    def _create_random_tuner(self, objective_func: Any, search_space: dict[str, Any]) -> tune.Tuner:
        """Create tuner with random search strategy."""
        logger.info("Creating random search tuner")

        return tune.Tuner(
            objective_func,
            tune_config=tune.TuneConfig(
                num_samples=self.config.num_trials,
                metric=self.config.metric,
                mode=self.config.mode,
            ),
            run_config=tune.RunConfig(
                name=self.config.study_name,
                storage_path=self.config.results_dir,
                log_to_file=self.config.log_to_file,
            ),
            param_space=search_space,
        )

    def _create_optuna_search(self) -> OptunaSearch:
        """Create Optuna search algorithm."""
        return OptunaSearch(
            metric=self.config.metric,
            mode=self.config.mode,
        )

    def _create_asha_scheduler(self) -> ASHAScheduler:
        """Create ASHA scheduler."""
        return ASHAScheduler(
            time_attr="training_iteration",
            metric=self.config.metric,
            mode=self.config.mode,
            max_t=self.config.asha_max_t,
            grace_period=self.config.asha_grace_period,
            reduction_factor=self.config.asha_reduction_factor,
            brackets=self.config.asha_brackets,
        )

    def _initialize_ray(self) -> None:
        """Initialize Ray cluster."""
        if not ray.is_initialized():
            if self.config.ray_address:
                logger.info(f"Connecting to Ray cluster at: {self.config.ray_address}")
                ray.init(address=self.config.ray_address)
            else:
                logger.info("Initializing local Ray cluster")
                ray.init(ignore_reinit_error=True)

    def _validate_config(self) -> None:
        """Validate configuration."""
        from .utils import validate_hyperopt_config

        validate_hyperopt_config(self.config)

    def _setup_logging(self) -> None:
        """Setup logging for optimization."""
        # Create results directory
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

        # Setup file logging if enabled
        if self.config.log_to_file:
            log_file = Path(self.config.results_dir) / "hyperopt.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def _log_results_summary(self, results: tune.ResultGrid) -> None:
        """Log summary of optimization results."""
        try:
            best_result = results.get_best_result(self.config.metric, self.config.mode)

            logger.info("=" * 60)
            logger.info("HYPERPARAMETER OPTIMIZATION COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Total trials completed: {len(results)}")
            best_metrics = best_result.metrics or {}
            best_config = best_result.config or {}
            logger.info(f"Best {self.config.metric}: {best_metrics.get(self.config.metric, 0.0):.4f}")
            logger.info("Best hyperparameters:")

            for param, value in best_config.items():
                logger.info(f"  {param}: {value}")

            # Log top 5 results
            logger.info("\nTop 5 results:")
            df = results.get_dataframe()
            if not df.empty:
                # Sort by metric
                ascending = self.config.mode == "min"
                top_5 = df.nlargest(5, self.config.metric) if not ascending else df.nsmallest(5, self.config.metric)

                for i, (_, row) in enumerate(top_5.iterrows(), 1):
                    metric_value = row[self.config.metric]
                    logger.info(f"  {i}. {self.config.metric}={metric_value:.4f}")

            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error generating results summary: {e}")

    def save_best_config(self, results: tune.ResultGrid, output_path: str) -> None:
        """Save best configuration to file.

        Args:
            results: Ray Tune results
            output_path: Path to save best config
        """
        try:
            best_result = results.get_best_result(self.config.metric, self.config.mode)

            # Save as YAML
            import yaml

            with open(output_path, "w") as f:
                yaml.dump(
                    {
                        "best_metric_value": (best_result.metrics or {}).get(self.config.metric, 0.0),
                        "best_config": best_result.config or {},
                        "hyperopt_config": {
                            "strategy": self.config.strategy,
                            "num_trials": self.config.num_trials,
                            "metric": self.config.metric,
                            "mode": self.config.mode,
                        },
                    },
                    f,
                    default_flow_style=False,
                )

            logger.info(f"Best configuration saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error saving best config: {e}")

    def __enter__(self) -> HyperparameterOptimizer:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cleanup Ray."""
        if ray.is_initialized():
            ray.shutdown()
